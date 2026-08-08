"""Minimal async OpenRouter chat client.

Deliberately not routed through langchain: the query path needs one POST with
a system and a user message, and going direct keeps the failure modes legible
- an OpenRouter error message reaches the user verbatim instead of being
wrapped in a provider abstraction.
"""
import json

import aiohttp


class OpenRouterError(RuntimeError):
    pass


async def chat(settings, system_prompt: str, user_prompt: str) -> str:
    """Send one prompt to OpenRouter and return the assistant's reply."""
    url = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter uses these for attribution on free models.
        "HTTP-Referer": "https://github.com/owaismohammad/gurrt",
        "X-Title": "gurrt",
    }
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    # Optional: omit it entirely and the model uses its own limit.
    max_tokens = getattr(settings, "MAX_OUTPUT_TOKENS", None)
    if max_tokens:
        payload["max_tokens"] = max_tokens

    timeout = aiohttp.ClientTimeout(total=settings.LLM_TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise OpenRouterError(
                    f"OpenRouter returned HTTP {resp.status} for "
                    f"{settings.LLM_MODEL}: {body[:500]}"
                )
    # Parse the text already read, rather than resp.json(), which re-reads and
    # rejects anything not labelled application/json.
    try:
        data = json.loads(body)
    except ValueError:
        raise OpenRouterError(f"Unreadable response: {body[:500]}")

    # Free models can answer 200 with an error object rather than choices.
    if "error" in data and not data.get("choices"):
        raise OpenRouterError(str(data["error"])[:500])

    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterError(f"No choices in response: {str(data)[:500]}")

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        finish = choices[0].get("finish_reason")
        raise OpenRouterError(f"Empty reply (finish_reason={finish}).")
    return content
