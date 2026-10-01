from __future__ import annotations


def normalize_openrouter_base_url(url: str | None) -> str:
    """Normalize OpenRouter base URL to the /api/v1 form.

    Examples:
      https://openrouter.ai/api    -> https://openrouter.ai/api/v1
      https://openrouter.ai/api/   -> https://openrouter.ai/api/v1
      https://openrouter.ai/api/v1 -> https://openrouter.ai/api/v1
    """
    if not url:
        return "https://openrouter.ai/api/v1"

    normalized = url.strip().rstrip("/")
    if normalized.endswith("/api"):
        return normalized + "/v1"
    return normalized


def normalize_openrouter_base_url_for_openclaw(url: str | None) -> str:
    return normalize_openrouter_base_url(url)


def normalize_openrouter_base_url_for_claudecode(url: str | None) -> str:
    if not url:
        return "https://openrouter.ai/api"

    normalized = url.strip().rstrip("/")
    if normalized.endswith("/api/v1"):
        return normalized[: -len("/v1")]
    return normalized
