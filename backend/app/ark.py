from __future__ import annotations

import re
from typing import Any, Iterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .settings import settings


class ArkClient:
    """
    Volcengine Ark (火山方舟) OpenAI-compatible API client for chat generation.
    Default base_url: https://ark.cn-beijing.volces.com/api/v3
    """

    def __init__(self) -> None:
        if not settings.ark_api_key:
            raise RuntimeError("Missing ARK_API_KEY in .env")
        self._http = httpx.Client(timeout=httpx.Timeout(180.0))

    def close(self) -> None:
        self._http.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.ark_api_key}",
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def chat_generate(self, messages: list[dict[str, Any]]) -> str:
        base = settings.ark_base_url.rstrip("/")
        url = base + "/chat/completions"
        body = {
            "model": settings.ark_chat_model,
            "messages": messages,
            "temperature": float(settings.ark_temperature),
        }
        resp = self._http.post(url, headers=self._headers(), json=body)
        resp.raise_for_status()
        payload = resp.json()
        content = (
            (((payload.get("choices") or [{}])[0].get("message") or {}).get("content"))
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ark response: {payload}")
        return _strip_code_fences(content.strip())

    def chat_stream(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        base = settings.ark_base_url.rstrip("/")
        url = base + "/chat/completions"
        body = {
            "model": settings.ark_chat_model,
            "messages": messages,
            "temperature": float(settings.ark_temperature),
            "stream": True,
        }

        with self._http.stream("POST", url, headers=self._headers(), json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                s = line.strip()
                if not s.startswith("data:"):
                    continue
                data = s[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    payload = json_loads(data)
                except Exception:
                    continue
                delta = (
                    (((payload.get("choices") or [{}])[0].get("delta") or {}).get("content"))
                    if isinstance(payload, dict)
                    else None
                )
                if isinstance(delta, str) and delta:
                    yield delta


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    end = t.rfind("```")
    if end <= 0:
        return t
    inner = t[3:end]
    inner = re.sub(r"^\w+\n", "", inner).strip()
    return inner


def json_loads(text: str) -> Any:
    import json

    return json.loads(text)

