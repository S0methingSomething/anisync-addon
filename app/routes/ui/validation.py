import asyncio

from quart import request

from app.routes.utils import rate_limit

from .blueprint import ui_bp


async def check_openrouter_api_key_valid(api_key: str) -> tuple[bool, str]:
    """Verify an OpenRouter API key using their key details endpoint."""
    import httpx

    if not api_key:
        return False, "Key cannot be empty"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://anisync.stream",
        "X-Title": "AniSync",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://openrouter.ai/api/v1/key", headers=headers)
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                label = data.get("label")
                msg = f"API key verified ({label})" if label else "API key verified"
                return True, msg
            if resp.status_code == 401:
                return False, "Invalid OpenRouter API key"

            try:
                err_data = resp.json().get("error", {})
                msg = err_data.get("message", f"HTTP {resp.status_code}")
            except Exception:
                msg = f"HTTP {resp.status_code}"
            return False, msg
    except Exception as e:
        return False, f"Validation failed: {str(e)}"


async def check_gemini_api_key_valid(api_key: str) -> tuple[bool, str]:
    import httpx

    if not api_key:
        return False, "Key cannot be empty"

    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
    ]
    payload = {"contents": [{"parts": [{"text": "Hello, respond with OK if you read this."}]}]}
    last_error = "Invalid API key"

    async with httpx.AsyncClient(timeout=10) as client:
        for model in models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    return True, "API key verified"

                err_data = {}
                try:
                    err_data = resp.json().get("error", {})
                    last_error = err_data.get("message", "Invalid API key")
                except Exception:
                    last_error = f"HTTP {resp.status_code}"

                # Early-exit if the API key is fundamentally invalid to avoid cascading through all fallback models
                error_details = err_data.get("details", [])
                if any(isinstance(d, dict) and d.get("reason") == "API_KEY_INVALID" for d in error_details):
                    return False, last_error
                if resp.status_code in (400, 403) and "API key" in last_error:
                    return False, last_error
            except Exception as e:
                last_error = f"Validation failed: {str(e)}"

    return False, last_error


@ui_bp.route("/openrouter/validation", methods=["POST"])
@rate_limit(limit=10, period_seconds=60)
async def validate_openrouter_key():
    data = await request.get_json() or {}
    api_key = data.get("api_key", "").strip()
    is_valid, msg = await check_openrouter_api_key_valid(api_key)
    if is_valid:
        return {"status": "success", "message": msg}
    return {"status": "error", "message": msg}, 400


# In-memory cache for OpenRouter models list (1 hour TTL)
_OPENROUTER_MODELS_CACHE = {"timestamp": 0.0, "data": []}


@ui_bp.route("/api/models/openrouter", methods=["GET"])
async def get_openrouter_models():
    """Fetch OpenRouter models with optional free-only filtering and in-memory caching."""
    import time

    import httpx

    free_only = request.args.get("free_only", "").lower() in ("true", "1", "yes")
    now = time.time()

    if now - _OPENROUTER_MODELS_CACHE["timestamp"] > 3600 or not _OPENROUTER_MODELS_CACHE["data"]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://openrouter.ai/api/v1/models")
                if resp.status_code == 200:
                    raw_data = resp.json().get("data", [])
                    parsed = []
                    for m in raw_data:
                        mid = m.get("id") or ""
                        pricing = m.get("pricing") or {}
                        try:
                            prompt_price = float(pricing.get("prompt") or 0.0)
                            comp_price = float(pricing.get("completion") or 0.0)
                        except (ValueError, TypeError):
                            prompt_price, comp_price = 1.0, 1.0

                        is_free = mid.endswith(":free") or (prompt_price == 0.0 and comp_price == 0.0)
                        parsed.append(
                            {
                                "id": mid,
                                "name": m.get("name") or mid,
                                "context_length": m.get("context_length"),
                                "is_free": is_free,
                            }
                        )
                    _OPENROUTER_MODELS_CACHE["data"] = parsed
                    _OPENROUTER_MODELS_CACHE["timestamp"] = now
        except Exception as e:
            if not _OPENROUTER_MODELS_CACHE["data"]:
                return {"status": "error", "message": f"Failed to fetch models: {str(e)}"}, 500

    models = _OPENROUTER_MODELS_CACHE["data"]
    if free_only:
        models = [m for m in models if m.get("is_free")]

    return {"status": "success", "models": models}


@ui_bp.route("/api/models/gemini", methods=["POST"])
@rate_limit(limit=10, period_seconds=60)
async def get_gemini_models():
    """Query available Gemini models using the user's provided API key."""
    import httpx

    data = await request.get_json() or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return {"status": "error", "message": "API key required"}, 400

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                raw_models = resp.json().get("models", [])
                models = []
                for m in raw_models:
                    methods = m.get("supportedGenerationMethods") or []
                    if "generateContent" in methods:
                        m_name = m.get("name", "").replace("models/", "")
                        models.append(
                            {
                                "id": m_name,
                                "name": m.get("displayName") or m_name,
                            }
                        )
                return {"status": "success", "models": models}
            return {"status": "error", "message": f"HTTP {resp.status_code}"}, resp.status_code
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


@ui_bp.route("/gemini/validation", methods=["POST"])
@rate_limit(limit=10, period_seconds=60)
async def validate_gemini_key():
    data = await request.get_json() or {}
    api_key = data.get("api_key", "").strip()
    is_valid, msg = await check_gemini_api_key_valid(api_key)
    if is_valid:
        return {"status": "success", "message": msg}
    return {"status": "error", "message": msg}, 400


@ui_bp.route("/rpdb/validation", methods=["POST"])
@rate_limit(limit=10, period_seconds=60)
async def validate_rpdb_key():
    data = await request.get_json() or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return {"status": "error", "message": "Key cannot be empty"}, 400
    try:
        from app.services.poster import validate_rpdb_api_key

        is_valid = await validate_rpdb_api_key(api_key)
        if is_valid:
            return {"status": "success", "message": "API key verified ✓"}
        else:
            return {"status": "error", "message": "Invalid RPDB API key"}, 400
    except Exception as e:
        return {"status": "error", "message": f"Validation failed: {str(e)}"}, 500


@ui_bp.route("/top_poster/validation", methods=["POST"])
@rate_limit(limit=10, period_seconds=60)
async def validate_top_poster_key():
    data = await request.get_json() or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return {"status": "error", "message": "Key cannot be empty"}, 400
    try:
        from app.services.poster import validate_top_poster_api_key

        is_valid = await validate_top_poster_api_key(api_key)
        if is_valid:
            return {"status": "success", "message": "API key verified ✓"}
        else:
            return {"status": "error", "message": "Invalid TOP Posters API key"}, 400
    except Exception as e:
        return {"status": "error", "message": f"Validation failed: {str(e)}"}, 500


@ui_bp.route("/health")
@rate_limit(limit=60, period_seconds=60)
async def health():
    from app.services.db import db

    try:
        # Verify MongoDB is reachable
        await asyncio.to_thread(db.command, "ping")
        return {"status": "healthy", "database": "connected"}, 200
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}, 500
