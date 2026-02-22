from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .settings import settings


@dataclass
class WebSearchResult:
    title: str
    snippet: str
    url: str
    display_url: str = ""


def _freshness_from_days(days: int | None) -> str | None:
    if days is None:
        return None
    try:
        d = int(days)
    except Exception:
        return None
    if d <= 2:
        return "Day"
    if d <= 7:
        return "Week"
    if d <= 31:
        return "Month"
    return None


def bing_search(
    query: str,
    *,
    top_k: int = 5,
    market: str | None = None,
    safe_search: str | None = None,
    freshness_days: int | None = None,
    timeout_seconds: float = 8.0,
) -> list[WebSearchResult]:
    """
    Bing Web Search API (summary-only).
    Returns a small list of results {title, snippet, url}.

    NOTE: This function does not fetch page contents.
    """
    q = (query or "").strip()
    if not q:
        return []
    if not settings.bing_search_api_key:
        return []

    k = int(top_k or 0)
    k = max(1, min(10, k))

    endpoint = (settings.bing_search_endpoint or "").strip() or "https://api.bing.microsoft.com/v7.0/search"
    mkt = (market or settings.bing_search_market or "zh-CN").strip() or "zh-CN"
    ss = (safe_search or settings.bing_search_safe_search or "Moderate").strip() or "Moderate"
    freshness = _freshness_from_days(freshness_days)

    headers = {
        "Ocp-Apim-Subscription-Key": settings.bing_search_api_key,
    }
    params: dict[str, Any] = {
        "q": q,
        "count": k,
        "mkt": mkt,
        "safeSearch": ss,
        "textDecorations": False,
        "textFormat": "Raw",
    }
    if freshness:
        params["freshness"] = freshness

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(endpoint, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return []

    items = (((data or {}).get("webPages") or {}).get("value") or []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        return []

    out: list[WebSearchResult] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = str(it.get("name") or "").strip()
        snippet = str(it.get("snippet") or "").strip()
        url = str(it.get("url") or "").strip()
        display_url = str(it.get("displayUrl") or "").strip()
        if not url or not (title or snippet):
            continue
        out.append(WebSearchResult(title=title[:200], snippet=snippet[:800], url=url[:800], display_url=display_url[:200]))
        if len(out) >= k:
            break
    return out

