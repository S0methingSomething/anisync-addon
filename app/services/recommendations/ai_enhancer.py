import json
import logging
import re
from pathlib import Path

from app.services.http import get_client

logger = logging.getLogger(__name__)

PROMPT_FILE_PATH = Path(__file__).parent / "prompts" / "recommendation_prompt.md"
_CACHED_PROMPT_TEMPLATE: str | None = None

DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
FALLBACK_OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
]
FALLBACK_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
]


def _load_prompt_template() -> str:
    """Load prompt template from the dedicated markdown file with in-memory caching."""
    global _CACHED_PROMPT_TEMPLATE
    if _CACHED_PROMPT_TEMPLATE is not None:
        return _CACHED_PROMPT_TEMPLATE

    try:
        if PROMPT_FILE_PATH.exists():
            _CACHED_PROMPT_TEMPLATE = PROMPT_FILE_PATH.read_text(encoding="utf-8")
            return _CACHED_PROMPT_TEMPLATE
    except Exception as e:
        logger.error("Failed to read prompt markdown file at %s: %s", PROMPT_FILE_PATH, e)

    # Fallback template if file reading fails
    return (
        "You are an advanced anime recommendation assistant.\n\n"
        "Based on the user's anime watch history:\n{history_text}\n\n"
        "And this list of candidate anime recommendations:\n{candidates_text}\n\n"
        "For each candidate that is relevant, write a personalized, engaging 1-sentence description "
        "explaining why the user would like it based on their history (referencing specific anime they watched "
        "when appropriate). Keep descriptions concise (under 150 characters).\n\n"
        "Return your response as a JSON object mapping the exact candidate title to its personalized description:\n"
        '{{\n  "Anime Title 1": "Description...",\n  "Anime Title 2": "Description..."\n}}\n\n'
        "Return only the raw JSON object."
    )


def _extract_json_object(raw_text: str) -> dict:
    """Extract a JSON object from text, handling markdown fences and stray prose."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)


def _build_prompt_inputs(
    sorted_user_history: list[dict],
    candidate_lists: dict[str, list[dict]],
) -> tuple[dict[str, dict], str]:
    """Compile candidates and watch history with rich metadata into the prompt text."""
    candidates_by_name: dict[str, dict] = {}
    for item_list in candidate_lists.values():
        for item in item_list[:8]:
            if item.get("name") and item["name"] not in candidates_by_name:
                candidates_by_name[item["name"]] = item

    if not candidates_by_name:
        return {}, ""

    history_lines = []
    for show in sorted_user_history[:15]:
        title = show.get("title") or "Unknown"
        list_name = show.get("list_name") or show.get("status") or "watched"
        rating_val = show.get("rating")
        rating_str = f"rated {rating_val}/10" if rating_val else "no rating"
        genres = show.get("genres") or []
        genre_str = f" [Genres: {', '.join(genres[:4])}]" if genres else ""
        history_lines.append(f'- {title} (Category: "{list_name}", {rating_str}){genre_str}')

    candidate_lines = []
    for name, item in candidates_by_name.items():
        genres = item.get("genres") or []
        genre_part = f"Genres: {', '.join(genres[:4])}" if genres else ""
        score_val = item.get("score")
        score_part = f"Score: {score_val}/10" if score_val else ""
        inspired_by = item.get("inspired_by_titles") or []
        insp_part = f"Inspired by: {', '.join(inspired_by[:2])}" if inspired_by else ""

        meta_parts = [p for p in [genre_part, score_part, insp_part] if p]
        meta_str = f" [{' | '.join(meta_parts)}]" if meta_parts else ""

        synopsis = (item.get("synopsis") or "").strip()
        premise_str = ""
        if synopsis:
            first_sentence = synopsis.split(". ")[0].strip()
            if len(first_sentence) > 120:
                first_sentence = first_sentence[:117] + "..."
            premise_str = f" — {first_sentence}"

        candidate_lines.append(f"- {name}{meta_str}{premise_str}")

    history_text = "\n".join(history_lines)
    candidates_text = "\n".join(candidate_lines)

    template = _load_prompt_template()
    prompt = template.format(history_text=history_text, candidates_text=candidates_text)
    return candidates_by_name, prompt


async def _call_openrouter(api_key: str, model: str | None, prompt: str) -> dict[str, str] | None:
    """Execute OpenRouter chat completion across chosen model and fallback models."""
    if not api_key:
        return None

    chosen_model = (model or "").strip() or DEFAULT_OPENROUTER_MODEL
    models_to_try = [chosen_model]
    for fallback in FALLBACK_OPENROUTER_MODELS:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://anisync.stream",
        "X-Title": "AniSync",
    }

    client = get_client()
    for current_model in models_to_try:
        payload = {
            "model": current_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an anime recommendation assistant. "
                        "You must respond ONLY with a raw JSON object mapping titles to descriptions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=18,
            )

            # Fallback if a specific model on OpenRouter does not support response_format
            if resp.status_code == 400 and "response_format" in resp.text:
                payload.pop("response_format", None)
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=18,
                )

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                result = _extract_json_object(content)
                logger.info("Successfully generated AI recommendations via OpenRouter (%s)", current_model)
                return result

            logger.warning(
                "OpenRouter model %s returned status %s: %s, attempting next fallback...",
                current_model,
                resp.status_code,
                resp.text[:160],
            )
        except Exception as err:
            logger.warning("OpenRouter model %s failed (%s), attempting next fallback...", current_model, err)

    return None


async def _call_gemini(api_key: str, prompt: str) -> dict[str, str] | None:
    """Execute Google Gemini generateContent across fallback flash models."""
    if not api_key:
        return None

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    client = get_client()

    for model in FALLBACK_GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            resp = await client.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                result = _extract_json_object(content)
                logger.info("Successfully generated AI recommendations via Gemini (%s)", model)
                return result

            logger.warning(
                "Gemini model %s returned status %s, attempting next fallback...",
                model,
                resp.status_code,
            )
        except Exception as err:
            logger.warning("Gemini model %s failed (%s), attempting next fallback...", model, err)

    return None


def _apply_ai_explanations(
    candidate_lists: dict[str, list[dict]],
    ai_explanations: dict[str, str],
) -> dict[str, list[dict]]:
    """Prepend AI explanations to candidates and boost them to the front of their rows."""

    def enhance_list(items):
        enhanced = []
        others = []
        for item in items:
            name = item.get("name")
            if name in ai_explanations:
                item_copy = item.copy()
                ai_desc = str(ai_explanations[name]).strip()
                syn = item_copy.get("synopsis") or ""
                item_copy["description"] = f"{ai_desc}  \n\n{syn}" if syn else ai_desc
                enhanced.append(item_copy)
            else:
                others.append(item)
        return enhanced + others

    return {key: enhance_list(items) for key, items in candidate_lists.items()}


async def enhance_recommendations_with_ai(
    user: dict,
    sorted_user_history: list[dict],
    candidate_lists: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """
    Main entrypoint: Enhance recommendations with the user's chosen AI provider
    (OpenRouter or Gemini) with cross-provider fallback if enabled.
    """
    openrouter_key = (user.get("openrouter_api_key") or "").strip()
    gemini_key = (user.get("gemini_api_key") or "").strip()
    openrouter_model = (user.get("openrouter_model") or "").strip() or DEFAULT_OPENROUTER_MODEL
    enable_fallback = user.get("ai_enable_fallback", True)

    provider = user.get("ai_provider")
    if not provider:
        # Auto-detect if provider not explicitly set
        if openrouter_key:
            provider = "openrouter"
        elif gemini_key:
            provider = "gemini"
        else:
            return candidate_lists

    if provider == "none":
        return candidate_lists

    candidates_by_name, prompt = _build_prompt_inputs(sorted_user_history, candidate_lists)
    if not candidates_by_name or not prompt:
        return candidate_lists

    explanations: dict[str, str] | None = None

    if provider == "openrouter":
        if openrouter_key:
            explanations = await _call_openrouter(openrouter_key, openrouter_model, prompt)
        if not explanations and enable_fallback and gemini_key:
            logger.info("OpenRouter failed or unconfigured; falling back to Gemini provider...")
            explanations = await _call_gemini(gemini_key, prompt)

    elif provider == "gemini":
        if gemini_key:
            explanations = await _call_gemini(gemini_key, prompt)
        if not explanations and enable_fallback and openrouter_key:
            logger.info("Gemini failed or unconfigured; falling back to OpenRouter provider...")
            explanations = await _call_openrouter(openrouter_key, openrouter_model, prompt)

    if explanations:
        return _apply_ai_explanations(candidate_lists, explanations)

    return candidate_lists


async def enhance_recommendations_with_openrouter(
    api_key: str,
    model: str | None,
    sorted_user_history: list[dict],
    candidate_lists: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Direct OpenRouter invocation wrapper."""
    fake_user = {
        "ai_provider": "openrouter",
        "openrouter_api_key": api_key,
        "openrouter_model": model,
        "ai_enable_fallback": False,
    }
    return await enhance_recommendations_with_ai(fake_user, sorted_user_history, candidate_lists)


async def enhance_recommendations_with_gemini(
    gemini_api_key: str,
    sorted_user_history: list[dict],
    candidate_lists: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Direct Gemini invocation wrapper for legacy callers."""
    # If an OpenRouter key was passed to this function, route to OpenRouter
    if gemini_api_key and gemini_api_key.startswith("sk-or-"):
        return await enhance_recommendations_with_openrouter(
            gemini_api_key,
            DEFAULT_OPENROUTER_MODEL,
            sorted_user_history,
            candidate_lists,
        )

    fake_user = {
        "ai_provider": "gemini",
        "gemini_api_key": gemini_api_key,
        "ai_enable_fallback": False,
    }
    return await enhance_recommendations_with_ai(fake_user, sorted_user_history, candidate_lists)
