import asyncio
import re
from datetime import UTC, date, datetime

from .fetchers import clean_imdb_id

_YOUTUBE_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
_MAPPING_DATE_TOLERANCE_DAYS = 45


def _parse_episode_int(value) -> int | None:
    """Return the value as an int when it is a usable AniZip number, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _extract_youtube_trailer_id(attributes: dict) -> str | None:
    """Return the Kitsu entry's YouTube trailer id when it is a well-formed video id."""
    yt_id = attributes.get("youtubeVideoId")
    if not isinstance(yt_id, str):
        return None
    yt_id = yt_id.strip()
    if not _YOUTUBE_ID_RE.fullmatch(yt_id):
        return None
    return yt_id


def _parse_air_date(value) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _determine_total_episodes(
    attributes: dict, episodes_data: list, anizp_episodes: dict, canonical_total_episodes: int | None
) -> int:
    kitsu_ep_count = attributes.get("episodeCount") or 0

    max_kitsu_ep = 0
    if episodes_data:
        ep_nums = []
        for x in episodes_data:
            raw_n = x.get("attributes", {}).get("number")
            if raw_n is not None:
                try:
                    ep_nums.append(int(raw_n))
                except (ValueError, TypeError):
                    pass
        if ep_nums:
            max_kitsu_ep = max(ep_nums)

    max_anizp_ep = 0
    if anizp_episodes:
        try:
            max_anizp_ep = max([int(k) for k in anizp_episodes.keys() if k.isdigit()])
        except ValueError:
            pass

    total_ep = max(kitsu_ep_count, max_kitsu_ep, max_anizp_ep)
    # AniZip mappings sometimes cover a whole parent series; when its numeric keys fully
    # cover the entry's own episode count and AniZip is the sole driver of the excess,
    # cap at the entry count so phantom episodes are never emitted.
    if (
        kitsu_ep_count
        and max_anizp_ep > max_kitsu_ep
        and total_ep > kitsu_ep_count
        and total_ep == max_anizp_ep
        and all(str(n) in anizp_episodes for n in range(1, kitsu_ep_count + 1))
    ):
        total_ep = kitsu_ep_count
    if total_ep == 0:
        total_ep = canonical_total_episodes if (canonical_total_episodes and canonical_total_episodes > 0) else 12
    elif canonical_total_episodes and canonical_total_episodes > 0:
        total_ep = min(total_ep, canonical_total_episodes)
    return total_ep


def _resolve_season_mapping(
    anizp_episodes: dict, cinemeta_data: dict | None, total_ep: int, extrapolate_unaired: bool = False
) -> dict[int, tuple[int, int]] | None:
    """Map entry-relative episodes to root-show (season, episode) coordinates, or None.

    AniZip `absoluteEpisodeNumber` indexes the flattened main-season episode list of
    the root show (as described by Cinemeta/TMDB). AniZip's `seasonNumber`/`episodeNumber`
    are TVDB-shaped and must NOT be used directly. The mapping is accepted only when it
    is complete for every episode, indexes monotonically, produces unique coordinates,
    and agrees on air dates with at least one verifiable date pair for multi-episode
    entries; any failure rejects the whole entry (atomic, fail closed to legacy
    season-1 display). For airing entries, an episode that has not aired yet may lack
    an absolute number; with extrapolate_unaired it is projected as the previous
    absolute + 1, while a missing absolute on an already-aired episode still rejects.
    """
    if total_ep <= 0 or not isinstance(anizp_episodes, dict) or not anizp_episodes:
        return None
    cinemeta_videos = (cinemeta_data or {}).get("videos")
    if not isinstance(cinemeta_videos, list) or not cinemeta_videos:
        return None

    coord_set = set()
    released_dates = {}
    for video in cinemeta_videos:
        try:
            coords = (int(video.get("season")), int(video.get("episode")))
        except (TypeError, ValueError):
            continue
        if coords[0] < 1 or coords[1] < 1:
            continue
        coord_set.add(coords)
        parsed = _parse_air_date(video.get("released"))
        if parsed and coords not in released_dates:
            released_dates[coords] = parsed
    if not coord_set:
        return None
    ordered = sorted(coord_set)

    mapping = {}
    prev_abs = 0
    for ep_num in range(1, total_ep + 1):
        anizp_ep = anizp_episodes.get(str(ep_num)) or {}
        abs_num = _parse_episode_int(anizp_ep.get("absoluteEpisodeNumber"))
        if abs_num is None:
            anizp_date = _parse_air_date(anizp_ep.get("airdate"))
            if not extrapolate_unaired or (anizp_date is not None and anizp_date <= date.today()):
                return None
            abs_num = prev_abs + 1
        if not 1 <= abs_num <= len(ordered):
            return None
        if abs_num <= prev_abs:
            return None
        prev_abs = abs_num
        mapping[ep_num] = ordered[abs_num - 1]

    if len(set(mapping.values())) != len(mapping):
        return None

    checked_dates = 0
    for ep_num, coords in mapping.items():
        anizp_date = _parse_air_date((anizp_episodes.get(str(ep_num)) or {}).get("airdate"))
        cinemeta_date = released_dates.get(coords)
        if anizp_date and cinemeta_date:
            checked_dates += 1
            if abs((anizp_date - cinemeta_date).days) > _MAPPING_DATE_TOLERANCE_DAYS:
                return None
    if not checked_dates and total_ep > 1:
        return None

    return mapping


def map_kitsu_to_stremio(
    kitsu_data: dict,
    meta_id: str,
    anizp_data: dict = None,
    mal_id: str = None,
    show_filler_tags: bool = False,
    loop=None,
    cinemeta_data: dict = None,
    show_watched_tags: bool = False,
    watched_progress: int = 0,
    canonical_total_episodes: int | None = None,
    title_language: str = "english",
    episodes_provider: str = "anizp",
    backdrop_provider: str = "fanart",
    poster_provider: str = "anilist",
) -> dict:
    data = kitsu_data.get("data", {})
    if not data:
        return {}

    video_base = f"kitsu:{data.get('id')}"

    attributes = data.get("attributes", {})
    titles = attributes.get("titles", {})

    if title_language == "english":
        title = (
            titles.get("en")
            or titles.get("en_us")
            or attributes.get("canonicalTitle")
            or titles.get("en_jp")
            or "Unknown Title"
        )
    elif title_language == "romaji":
        title = titles.get("en_jp") or attributes.get("canonicalTitle") or titles.get("en") or "Unknown Title"
    elif title_language == "japanese":
        title = titles.get("ja_jp") or titles.get("en_jp") or attributes.get("canonicalTitle") or "Unknown Title"
    else:  # default / canonical
        title = attributes.get("canonicalTitle") or titles.get("en") or titles.get("en_jp") or "Unknown Title"
    synopsis = attributes.get("synopsis", "")

    # Media Type
    subtype = (attributes.get("subtype") or "tv").lower()
    media_type = "movie" if subtype == "movie" else "series"
    anime_status = (attributes.get("status") or "").lower()

    included = kitsu_data.get("included") or []
    episodes_data = [item for item in included if item.get("type") == "episodes"]
    anizp_episodes = anizp_data.get("episodes", {}) if anizp_data else {}

    is_movie = subtype == "movie"
    is_likely_sequel = False
    real_season_map = None
    total_ep = 0
    if not is_movie:
        canonical_title = (attributes.get("canonicalTitle") or "").lower()
        is_likely_sequel = any(
            marker in canonical_title
            for marker in [
                "season 2",
                "season 3",
                "season 4",
                "season 5",
                "2nd season",
                "3rd season",
                "4th season",
                "part 2",
                "part 3",
                "part 4",
                "cour 2",
            ]
        )
        total_ep = _determine_total_episodes(attributes, episodes_data, anizp_episodes, canonical_total_episodes)
        real_season_map = _resolve_season_mapping(
            anizp_episodes,
            cinemeta_data,
            total_ep,
            extrapolate_unaired=anime_status in ("current", "upcoming", "not_yet_released"),
        )

    # Cinemeta data describes the root franchise show. It may only enrich an entry whose
    # episode coordinates are validated (real_season_map) or that is safely the root
    # itself; an unvalidated sequel must not inherit root-show enrichment.
    use_cinemeta = real_season_map is not None or not is_likely_sequel

    anizp_images = anizp_data.get("images", []) if anizp_data else []
    anizp_fanart = None
    anizp_poster = None
    anizp_logo = None
    for img in anizp_images:
        if img.get("coverType") == "Fanart" and not anizp_fanart:
            anizp_fanart = img.get("url")
        elif img.get("coverType") == "Poster" and not anizp_poster:
            anizp_poster = img.get("url")
        elif img.get("coverType") in ["Clearlogo", "Logo"] and not anizp_logo:
            img_url = img.get("url") or ""
            # Filter out TheTVDB square icons/avatars (which AniZip mistakenly tags as Clearlogo)
            if "/icons/" not in img_url and "/icon/" not in img_url:
                anizp_logo = img_url

    poster_data = attributes.get("posterImage") or {}
    kitsu_poster = poster_data.get("original") or poster_data.get("large") or poster_data.get("medium") or ""
    if poster_provider == "kitsu":
        poster = kitsu_poster or anizp_poster or ""
    else:
        poster = anizp_poster or kitsu_poster or ""
    cover_data = attributes.get("coverImage") or {}
    kitsu_cover = cover_data.get("original") or cover_data.get("large") or cover_data.get("medium")

    if backdrop_provider == "kitsu":
        background = (
            kitsu_cover
            or anizp_fanart
            or (cinemeta_data.get("background") if cinemeta_data and use_cinemeta else "")
            or poster
        )
    else:  # fanart (default)
        background = (
            (cinemeta_data.get("background") if cinemeta_data and use_cinemeta else None)
            or anizp_fanart
            or kitsu_cover
            or poster
        )

    imdb_id = clean_imdb_id(anizp_data.get("mappings", {}).get("imdb_id") if anizp_data else None)

    logo = anizp_logo
    if not logo and cinemeta_data and use_cinemeta:
        logo = cinemeta_data.get("logo")
    if not logo and imdb_id and use_cinemeta:
        logo = f"https://images.metahub.space/logo/medium/{imdb_id}/img"

    average_rating = attributes.get("averageRating")
    rating = str(round(float(average_rating) / 10.0, 1)) if average_rating else None

    # Release info (Year)
    start_date = attributes.get("startDate")
    end_date = attributes.get("endDate")
    release_info = ""
    if start_date:
        release_info = start_date[:4]
        if end_date:
            release_info += f"-{end_date[:4]}"
        else:
            release_info += "-"

    # Videos / Episodes List
    videos = []

    if subtype == "movie":
        videos.append(
            {
                "id": video_base,
                "title": title,
                "episode": 1,
                "season": 1,
                "released": start_date + "T00:00:00Z" if start_date else None,
                "overview": synopsis,
                "thumbnail": background or poster,
            }
        )
    else:
        # Create a mapping of episode number to Kitsu episode data for O(1) lookup
        kitsu_ep_map = {}
        for ep in episodes_data:
            num = ep.get("attributes", {}).get("number")
            if num:
                try:
                    kitsu_ep_map[int(num)] = ep
                except (ValueError, TypeError):
                    pass

        # Build Cinemeta episode thumbnail and release date maps: (season, episode) -> value
        cinemeta_ep_thumbs = {}
        cinemeta_ep_released = {}
        if cinemeta_data and isinstance(cinemeta_data.get("videos"), list):
            for v in cinemeta_data["videos"]:
                s = v.get("season")
                e = v.get("episode")
                thumb = v.get("thumbnail")
                rel = v.get("released")
                if s is not None and e is not None:
                    try:
                        s_int = int(s)
                        e_int = int(e)
                        if thumb:
                            cinemeta_ep_thumbs[(s_int, e_int)] = thumb
                        if rel:
                            cinemeta_ep_released[(s_int, e_int)] = rel
                    except (ValueError, TypeError):
                        pass

        for i in range(1, total_ep + 1):
            ep_num = i
            kitsu_ep = kitsu_ep_map.get(i)
            anizp_ep = anizp_episodes.get(str(i)) or {}

            if real_season_map is not None:
                # Validated root-show coordinates; the tracker-facing video id stays entry-relative.
                season, episode = real_season_map[ep_num]
            else:
                season = 1
                episode = ep_num

            # Extract attributes from Kitsu episode if available
            attrs = kitsu_ep.get("attributes", {}) if kitsu_ep else {}

            if title_language == "english":
                ep_title = (
                    anizp_ep.get("title", {}).get("en")
                    or attrs.get("canonicalTitle")
                    or anizp_ep.get("title", {}).get("x-jat")
                    or f"Episode {ep_num}"
                )
            elif title_language == "romaji":
                ep_title = (
                    anizp_ep.get("title", {}).get("x-jat")
                    or attrs.get("canonicalTitle")
                    or anizp_ep.get("title", {}).get("en")
                    or f"Episode {ep_num}"
                )
            elif title_language == "japanese":
                ep_title = (
                    anizp_ep.get("title", {}).get("ja")
                    or anizp_ep.get("title", {}).get("x-jat")
                    or attrs.get("canonicalTitle")
                    or f"Episode {ep_num}"
                )
            else:  # default / canonical
                ep_title = (
                    attrs.get("canonicalTitle")
                    or anizp_ep.get("title", {}).get("en")
                    or anizp_ep.get("title", {}).get("x-jat")
                    or f"Episode {ep_num}"
                )
            # Match Cinemeta episode thumbnail and release date safely
            cinemeta_thumb = None
            cin_released = None
            if use_cinemeta and (cinemeta_ep_thumbs or cinemeta_ep_released):
                if real_season_map is not None:
                    cinemeta_thumb = cinemeta_ep_thumbs.get((season, episode))
                    cin_released = cinemeta_ep_released.get((season, episode))
                elif not is_likely_sequel:
                    cinemeta_thumb = cinemeta_ep_thumbs.get((1, ep_num))
                    cin_released = cinemeta_ep_released.get((1, ep_num))

            released = attrs.get("airdate") or anizp_ep.get("airdate") or cin_released

            # If the anime or episode is unreleased (air date in future), Cinemeta's Metahub
            # thumbnail link is pre-computed and guaranteed to 404 on Metahub/TMDB.
            # Suppress cinemeta_thumb so it falls back to backdrop (background) or real preview stills.
            if cinemeta_thumb:
                if anime_status in ["unreleased", "not_yet_released"]:
                    cinemeta_thumb = None
                elif released:
                    try:
                        date_str = str(released).strip()[:10].replace("/", "-")
                        ep_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        now_date = datetime.now(UTC).date()
                        if ep_date > now_date:
                            cinemeta_thumb = None
                    except (ValueError, TypeError):
                        pass

            if episodes_provider == "kitsu":
                overview = attrs.get("synopsis") or anizp_ep.get("overview") or anizp_ep.get("summary") or ""
                thumbnail = (
                    (attrs.get("thumbnail") or {}).get("original")
                    or (attrs.get("thumbnail") or {}).get("large")
                    or anizp_ep.get("image")
                    or cinemeta_thumb
                    or background
                )
            else:  # anizp, mal, or default
                overview = anizp_ep.get("overview") or anizp_ep.get("summary") or attrs.get("synopsis") or ""
                thumbnail = (
                    anizp_ep.get("image")
                    or (attrs.get("thumbnail") or {}).get("original")
                    or (attrs.get("thumbnail") or {}).get("large")
                    or cinemeta_thumb
                    or background
                )

            # Check filler status
            is_filler = False
            if mal_id and show_filler_tags:
                from app.services.db import get_jikan_filler_cache

                cached = get_jikan_filler_cache(mal_id, ep_num)
                if cached is not None:
                    is_filler = cached
                else:
                    from app.routes.catalog import background_fetch_and_cache_filler, currently_fetching_pairs

                    pair = (str(mal_id), ep_num)
                    if pair not in currently_fetching_pairs:
                        currently_fetching_pairs.add(pair)
                        if loop and loop.is_running():
                            try:
                                asyncio.run_coroutine_threadsafe(
                                    background_fetch_and_cache_filler(mal_id, ep_num), loop
                                )
                            except Exception:
                                pass

            if is_filler:
                ep_title = f"[Filler] {ep_title}"
            if show_watched_tags and ep_num <= watched_progress:
                ep_title = f"[Watched] {ep_title}"

            released_str = None
            if released and isinstance(released, str) and released.strip():
                released = released.strip()
                if "T" not in released:
                    released_str = released + "T00:00:00Z"
                else:
                    released_str = released

            videos.append(
                {
                    "id": f"{video_base}:{ep_num}",
                    "title": ep_title,
                    "episode": episode,
                    "season": season,
                    "released": released_str,
                    "overview": overview,
                    "thumbnail": thumbnail,
                }
            )

    genres = ["Anime"]
    if use_cinemeta and cinemeta_data and cinemeta_data.get("genres"):
        for g in cinemeta_data["genres"]:
            if g not in genres:
                genres.append(g)

    links = []
    if use_cinemeta and cinemeta_data and "links" in cinemeta_data:
        for link in cinemeta_data["links"]:
            if link.get("category") == "Cast":
                links.append(link)

    meta_obj = {
        "id": meta_id,
        "name": title,
        "type": media_type,
        "poster": poster,
        "background": background,
        "imdbRating": rating,
        "releaseInfo": release_info,
        "description": synopsis,
        "videos": videos,
        "genres": genres,
        "links": links,
    }
    if logo:
        meta_obj["logo"] = logo
    if imdb_id and use_cinemeta:
        meta_obj["imdb_id"] = imdb_id

    trailer_yt_id = _extract_youtube_trailer_id(attributes)
    if trailer_yt_id:
        meta_obj["trailers"] = [{"source": trailer_yt_id, "type": "Trailer"}]
        meta_obj["trailerStreams"] = [{"title": title, "ytId": trailer_yt_id}]

    return meta_obj
