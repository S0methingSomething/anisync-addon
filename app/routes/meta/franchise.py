"""Franchise mode: assemble one meta whose videos span a whole AniList prequel/sequel chain.

Nuvio renders the Seasons chip row only when a meta's videos span two or more seasons.
Per-entry metas emit a single season, so opt-in franchise mode appends validated sibling
entries' videos (each keeping its own entry-relative Kitsu video id and real season/episode
coordinates) to the requested entry's videos. Trackers, streams and the subtitle reverse-map
keep working because every video id stays entry-relative; the requested entry's own
coordinates are untouched, so there is no new progress fork.

Every sibling is validated with the exact same machinery as the requested entry
(resolve_entry_season_mapping) and fails closed per entry: a sibling whose own season
mapping does not validate is excluded, never guessed. A partial franchise always beats a
wrong one. Accepted tradeoff: a true root-season sibling whose AniZip data lacks absolute
episode numbers is also excluded, exactly like an unvalidated requested entry.
"""

import asyncio
import datetime
import logging

from app.api.anilist import get_media_relations
from app.lib.id_resolver import bulk_resolve_to_kitsu
from app.routes.meta.fetchers import (
    clean_imdb_id,
    fetch_anizp_metadata,
    fetch_cinemeta_metadata,
    fetch_kitsu_meta,
)
from app.routes.meta.stremio import map_kitsu_to_stremio, resolve_entry_season_mapping

FRANCHISE_CACHE_TTL_DAYS = 7
FRANCHISE_MAX_ENTRIES = 8
_FRANCHISE_BFS_LIMIT = 12
_FRANCHISE_TIMEOUT_SECONDS = 15
_CHAIN_NEGATIVE_TTL_MINUTES = 30

_CHAIN_RELATION_TYPES = ("PREQUEL", "SEQUEL")
# MANGA/NOVEL/ONE_SHOT are excluded so the BFS never spends a GraphQL round-trip on
# nodes the ANIME-typed query would return as null anyway.
_EXCLUDED_FORMATS = ("MOVIE", "MUSIC", "MANGA", "NOVEL", "ONE_SHOT")


def _start_date_tuple(media: dict) -> tuple[int, int, int]:
    start = media.get("startDate") or {}
    try:
        return (int(start.get("year") or 0), int(start.get("month") or 0), int(start.get("day") or 0))
    except (TypeError, ValueError):
        return (0, 0, 0)


def order_franchise_media(start_id: int, media_by_id: dict[int, dict]) -> list[int]:
    """Return franchise member ids ordered chronologically, including start_id.

    Membership is collected by walking PREQUEL/SEQUEL edges (movies, music videos and
    adult entries are never traversed into); ordering is by AniList start date then id,
    which is robust against messy franchise graphs with multiple prequel/sequel branches.
    The result is capped at FRANCHISE_MAX_ENTRIES with a window anchored on start_id:
    earlier seasons (predecessors) are preferred, later seasons fill the remainder, so
    requesting a late entry never truncates away the requested entry itself.
    """
    if start_id not in media_by_id:
        return []

    candidates = []
    for media_id, media in media_by_id.items():
        if (media.get("format") or "").upper() in _EXCLUDED_FORMATS:
            continue
        if media.get("isAdult"):
            continue
        candidates.append((_start_date_tuple(media), media_id))
    # Undated members (missing/malformed startDate) sort as (0, 0, 0); push them last
    # so they can never distort the chronological window anchoring.
    candidates.sort(key=lambda item: (item[0] == (0, 0, 0), item[0], item[1]))
    full_order = [media_id for _, media_id in candidates]
    if start_id not in full_order:
        return []
    index = full_order.index(start_id)
    before = min(index, FRANCHISE_MAX_ENTRIES - 1)
    after = FRANCHISE_MAX_ENTRIES - 1 - before
    lo = index - before
    hi = min(len(full_order), index + 1 + after)
    return full_order[lo:hi]


def merge_franchise_videos(base_videos: list, extra_videos: list) -> list:
    """Merge sibling videos into the requested entry's list, sorted by (season, episode).

    The sort is stable and base videos come first, so on equal coordinates the requested
    entry keeps priority.
    """
    merged = list(base_videos) + list(extra_videos)
    return sorted(merged, key=lambda v: (v.get("season") or 1, v.get("episode") or 1))


def sibling_collides(taken_coords: set, sib_coords: list) -> bool:
    """True when any sibling display coordinate is already claimed by an accepted entry.

    A collision would merge two entries into one Nuvio progress key (progress is keyed on
    (content_id, season, episode)), so the whole sibling is excluded — atomic fail-closed,
    requested entry wins. Validated siblings should never collide; this is the final guard
    against overlapping provider mappings.
    """
    return any(coords in taken_coords for coords in sib_coords)


def _cache_doc(col, cache_key: str, fields: dict, now, ttl: datetime.timedelta) -> None:
    """Upsert a cache doc keyed on anilist_id with a merged TTL stamp."""
    try:
        col.update_one(
            {"anilist_id": cache_key},
            {
                "$set": {
                    "anilist_id": cache_key,
                    **fields,
                    "expires_at": now + ttl,
                    "updated_at": now,
                }
            },
            upsert=True,
        )
    except Exception as e:
        logging.error("Failed to write franchise_chain_cache for %s: %s", cache_key, e)


async def _walk_franchise_chain(start_id: int) -> list[int]:
    """BFS the AniList prequel/sequel graph starting at start_id, then order members."""
    media_by_id: dict[int, dict] = {}
    queue = [start_id]
    while queue and len(media_by_id) < _FRANCHISE_BFS_LIMIT:
        media_id = queue.pop(0)
        if media_id in media_by_id:
            continue
        try:
            media = await get_media_relations(None, media_id)
        except Exception as e:
            logging.warning("Franchise chain: relations fetch failed for anilist=%s: %s", media_id, e)
            continue
        if not media:
            continue
        media_by_id[int(media.get("id") or media_id)] = media
        for edge in (media.get("relations") or {}).get("edges") or []:
            if (edge.get("relationType") or "").upper() not in _CHAIN_RELATION_TYPES:
                continue
            node = edge.get("node") or {}
            try:
                node_id = int(node.get("id"))
            except (TypeError, ValueError):
                continue
            if (node.get("format") or "").upper() in _EXCLUDED_FORMATS:
                continue
            if node.get("isAdult"):
                continue
            if node_id not in media_by_id:
                queue.append(node_id)
    return order_franchise_media(start_id, media_by_id)


async def fetch_franchise_chain(anilist_id: int) -> list[str]:
    """Return the ordered AniList ids of the franchise chain (requested id included).

    Cached in Mongo for a week; sibling membership rarely changes, and per-sibling data
    (Kitsu/AniZip/Cinemeta) stays cached on its own shorter TTLs so airing updates flow.
    """
    from app.services.db import db

    cache_key = str(anilist_id)
    col = db.get_collection("franchise_chain_cache")
    now = datetime.datetime.utcnow()
    try:
        cached = col.find_one({"anilist_id": cache_key})
        if cached and cached.get("expires_at") > now:
            return cached.get("chain") or []
    except Exception as e:
        logging.error("Failed to read franchise_chain_cache for %s: %s", cache_key, e)

    try:
        chain = await _walk_franchise_chain(anilist_id)
    except Exception as e:
        logging.warning("Franchise chain walk failed for anilist=%s: %s", anilist_id, e)
        # Negative-cache the failure briefly so an upstream outage costs one stalled
        # request instead of one stalled request per uncached meta.
        _cache_doc(col, cache_key, {"chain": []}, now, datetime.timedelta(minutes=_CHAIN_NEGATIVE_TTL_MINUTES))
        return []

    # Empty walks (no siblings found) are also negative-cached briefly: they re-walk
    # otherwise, paying the full GraphQL cost on every request.
    _cache_doc(
        col,
        cache_key,
        {"chain": [str(x) for x in chain]},
        now,
        datetime.timedelta(days=FRANCHISE_CACHE_TTL_DAYS)
        if chain
        else datetime.timedelta(minutes=_CHAIN_NEGATIVE_TTL_MINUTES),
    )
    return [str(x) for x in chain]


async def fetch_is_likely_sequel(anilist_id: int) -> bool | None:
    """Return True when AniList shows a PREQUEL relation for this media, None on failure.

    Authoritative sequel signal for enrichment gating: any prequel edge (even to a
    movie or a different format) proves the entry continues earlier work. Cached in the
    franchise_chain_cache collection under a `sig:` key for a week; failures are
    negative-cached briefly so an AniList outage costs one lookup instead of one per
    meta request.
    """
    from app.services.db import db

    cache_key = f"sig:{anilist_id}"
    col = db.get_collection("franchise_chain_cache")
    now = datetime.datetime.utcnow()
    try:
        cached = col.find_one({"anilist_id": cache_key})
        if cached and cached.get("expires_at") > now:
            is_sequel = cached.get("is_sequel")
            return is_sequel if isinstance(is_sequel, bool) else None
    except Exception as e:
        logging.error("Failed to read franchise_chain_cache for %s: %s", cache_key, e)

    try:
        media = await get_media_relations(None, anilist_id)
        edges = (media or {}).get("relations", {}).get("edges") or []
        has_prequel = any((edge.get("relationType") or "").upper() == "PREQUEL" for edge in edges)
    except Exception as e:
        logging.warning("Sequel signal fetch failed for anilist=%s: %s", anilist_id, e)
        _cache_doc(col, cache_key, {"is_sequel": None}, now, datetime.timedelta(minutes=_CHAIN_NEGATIVE_TTL_MINUTES))
        return None

    _cache_doc(col, cache_key, {"is_sequel": has_prequel}, now, datetime.timedelta(days=FRANCHISE_CACHE_TTL_DAYS))
    return has_prequel


async def build_franchise_videos(
    user_id: str,
    anilist_id: str,
    kitsu_id: str,
    title_language: str,
    effective_provs: dict,
    show_filler: bool,
    show_watched: bool,
    loop,
    base_videos: list,
) -> tuple[list, bool]:
    """Validate every sibling in the franchise chain and return (merged_videos, has_personalized).

    Siblings whose own season mapping does not validate are excluded entirely (fail closed
    per entry). The requested entry's videos are never modified, only re-sorted into place.
    """
    try:
        async with asyncio.timeout(_FRANCHISE_TIMEOUT_SECONDS):
            return await _build_franchise_videos(
                user_id,
                anilist_id,
                kitsu_id,
                title_language,
                effective_provs,
                show_filler,
                show_watched,
                loop,
                base_videos,
            )
    except TimeoutError:
        logging.warning("Franchise assembly timed out for anilist=%s; returning base meta", anilist_id)
        return base_videos, False
    except Exception as e:
        logging.error("Franchise assembly failed for anilist=%s: %s", anilist_id, e, exc_info=True)
        return base_videos, False


async def _build_franchise_videos(
    user_id: str,
    anilist_id: str,
    kitsu_id: str,
    title_language: str,
    effective_provs: dict,
    show_filler: bool,
    show_watched: bool,
    loop,
    base_videos: list,
) -> tuple[list, bool]:
    from app.services.db import get_user_anime_meta_status, get_user_watch_progress

    chain = await fetch_franchise_chain(int(anilist_id))
    sibling_al_ids = [x for x in chain if x != str(anilist_id)]
    if not sibling_al_ids:
        return base_videos, False

    resolved = await bulk_resolve_to_kitsu(anilist_ids=sibling_al_ids)

    sibling_jobs = []
    for sib_al in sibling_al_ids:
        sib_kitsu = resolved.get(f"anilist:{sib_al}")
        if not sib_kitsu or sib_kitsu == kitsu_id:
            continue
        sibling_jobs.append((sib_al, sib_kitsu))
    if not sibling_jobs:
        return base_videos, False

    async def fetch_sibling(sib_al: str, sib_kitsu: str):
        kitsu_task = fetch_kitsu_meta(sib_kitsu)
        anizp_task = fetch_anizp_metadata(anilist_id=sib_al)
        results = await asyncio.gather(kitsu_task, anizp_task, return_exceptions=True)
        kitsu_data = results[0] if not isinstance(results[0], Exception) else {}
        anizp_data = results[1] if not isinstance(results[1], Exception) else {}
        return kitsu_data, anizp_data

    fetched = await asyncio.gather(*(fetch_sibling(al, kid) for al, kid in sibling_jobs))

    # Cinemeta is the mapping-validation oracle for each sibling; fetch every needed one
    # concurrently so cold-cache assemblies stay inside the timeout budget.
    async def sibling_cinemeta(sib_kitsu_data: dict, sib_anizp: dict):
        sib_attrs = (sib_kitsu_data.get("data") or {}).get("attributes") or {}
        if (sib_attrs.get("subtype") or "tv").lower() == "movie":
            return {}
        sib_imdb = clean_imdb_id((sib_anizp.get("mappings") or {}).get("imdb_id"))
        if not sib_imdb:
            return {}
        sib_status = (sib_attrs.get("status") or "").lower()
        is_releasing = sib_status in ["current", "releasing", "unreleased", "not_yet_released"]
        return await fetch_cinemeta_metadata(sib_imdb, "series", is_releasing=is_releasing)

    cinemetas = await asyncio.gather(*(sibling_cinemeta(kd, az) for kd, az in fetched))

    extra_videos = []
    has_personalized = False
    taken_coords = {(v.get("season") or 1, v.get("episode") or 1) for v in base_videos}

    mapper_jobs = []
    for (sib_al, sib_kitsu), (sib_kitsu_data, sib_anizp), cinemeta_data in zip(
        sibling_jobs, fetched, cinemetas, strict=True
    ):
        sib_attrs = (sib_kitsu_data.get("data") or {}).get("attributes") or {}
        if (sib_attrs.get("subtype") or "tv").lower() == "movie":
            continue
        sib_mal = str((sib_anizp.get("mappings") or {}).get("mal_id") or "") or None

        sib_watched = 0
        canonical_total = None
        if show_watched or show_filler:
            sib_status_info = get_user_anime_meta_status(user_id, mal_id=sib_mal, anilist_id=sib_al)
            if sib_status_info:
                sib_watched = sib_status_info.get("progress") or 0
                canonical_total = sib_status_info.get("total_episodes")
            else:
                sib_watched = get_user_watch_progress(user_id, mal_id=sib_mal, anilist_id=sib_al)
        if show_watched and sib_watched > 0:
            has_personalized = True

        sib_episodes_data = [item for item in (sib_kitsu_data.get("included") or []) if item.get("type") == "episodes"]
        _, sib_map = resolve_entry_season_mapping(
            sib_attrs, sib_episodes_data, sib_anizp, cinemeta_data, canonical_total
        )
        if not sib_map:
            logging.info("Franchise mode: excluding sibling anilist=%s (season mapping did not validate)", sib_al)
            continue

        mapper_jobs.append(
            (
                sib_al,
                sib_kitsu_data,
                f"kitsu:{sib_kitsu}",
                {
                    "anizp_data": sib_anizp,
                    "mal_id": sib_mal,
                    "show_filler_tags": show_filler,
                    "loop": loop,
                    "cinemeta_data": cinemeta_data,
                    "show_watched_tags": show_watched,
                    "watched_progress": sib_watched,
                    "canonical_total_episodes": canonical_total,
                    "title_language": title_language,
                    "episodes_provider": effective_provs.get("episodes", "anizp"),
                    "backdrop_provider": effective_provs.get("backdrop", "fanart"),
                    "poster_provider": effective_provs.get("poster", "anilist"),
                },
            )
        )

    if not mapper_jobs:
        return base_videos, has_personalized

    sib_metas = await asyncio.gather(
        *(asyncio.to_thread(map_kitsu_to_stremio, kd, mid, **kw) for _, kd, mid, kw in mapper_jobs),
        return_exceptions=True,
    )

    for (sib_al, _, _, _), sib_meta in zip(mapper_jobs, sib_metas, strict=True):
        if isinstance(sib_meta, Exception):
            logging.warning("Franchise mode: sibling mapper failed for anilist=%s: %s", sib_al, sib_meta)
            continue
        sib_videos = sib_meta.get("videos") or []
        if not sib_videos:
            continue
        sib_coords = [(v.get("season") or 1, v.get("episode") or 1) for v in sib_videos]
        if sibling_collides(taken_coords, sib_coords):
            logging.info(
                "Franchise mode: excluding sibling anilist=%s (coordinate collision with accepted entries)", sib_al
            )
            continue
        taken_coords.update(sib_coords)
        extra_videos.extend(sib_videos)

    if not extra_videos:
        return base_videos, has_personalized
    return merge_franchise_videos(base_videos, extra_videos), has_personalized
