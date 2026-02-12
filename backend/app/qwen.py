from __future__ import annotations

import base64
import json
import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .settings import settings


class DashScopeClient:
    def __init__(self) -> None:
        if not settings.dashscope_api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY in .env")
        self._http = httpx.Client(timeout=httpx.Timeout(90.0))
        self._base = "https://dashscope.aliyuncs.com"

    def close(self) -> None:
        self._http.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base}/api/v1/services/embeddings/text-embedding/text-embedding"
        body = {"model": settings.dashscope_embedding_model, "input": {"texts": texts}}
        resp = self._http.post(url, headers=self._headers(), json=body)
        _raise_for_status_with_body(resp)
        payload = resp.json()
        out = payload.get("output") or {}
        embeddings = out.get("embeddings") or []
        vecs: list[list[float]] = []
        for e in embeddings:
            vec = e.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError(f"Unexpected embedding format: {payload}")
            vecs.append([float(x) for x in vec])
        if len(vecs) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: got {len(vecs)} want {len(texts)}")
        return vecs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def vision_describe(self, image_bytes: bytes, mime_type: str = "image/png") -> dict[str, Any]:
        url = f"{self._base}/api/v1/services/aigc/multimodal-generation/generation"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"

        prompt = (
            "你是一个严谨的企业技术支持助手。请只输出严格 JSON，不要输出任何其他字符。\n"
            "要求：不确定就留空或不填，不允许编造。key_text 控制在 5~15 条。\n"
            "输出 JSON schema：\n"
            "{"
            '"type":"string",'
            '"key_text":["..."],'
            '"key_values":{"error_code":"...","request_id":"..."},'
            '"what_it_proves":["..."],'
            '"next_action":["..."],'
            '"confidence":0.0'
            "}"
        )

        body = {
            "model": settings.dashscope_vision_model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": data_url},
                            {"text": prompt},
                        ],
                    }
                ]
            },
            "parameters": {"result_format": "message"},
        }
        resp = self._http.post(url, headers=self._headers(), json=body)
        _raise_for_status_with_body(resp)
        payload = resp.json()
        output = payload.get("output") or {}
        msg = output.get("choices", [{}])[0].get("message") or {}
        content = msg.get("content")
        text = _extract_message_text(content)
        if not text:
            raise RuntimeError(f"Unexpected vision response: {payload}")
        text = _strip_code_fences(text.strip())
        return _normalize_vision(_parse_json_object(text))


def _parse_json_object(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"Vision output is not JSON: {t[:2000]}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Vision output JSON is not object: {t[:2000]}")
    return obj


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


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        return "\n".join(parts).strip()
    return ""


def _raise_for_status_with_body(resp: httpx.Response) -> None:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = resp.text
        except Exception:
            body = "<no body>"
        body = (body or "")[:2000]
        raise RuntimeError(f"DashScope HTTP {resp.status_code}: {body}") from e


def _normalize_vision(obj: dict[str, Any]) -> dict[str, Any]:
    def _str(x: Any) -> str:
        return x.strip() if isinstance(x, str) else ""

    def _str_list(x: Any) -> list[str]:
        if not isinstance(x, list):
            return []
        out: list[str] = []
        for item in x:
            s = _str(item)
            if s and s not in out:
                out.append(s)
        return out

    def _kv(x: Any) -> dict[str, str]:
        if not isinstance(x, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in x.items():
            ks = _str(k)
            vs = _str(v)
            if ks and vs:
                out[ks] = vs
        return out

    t = _str(obj.get("type"))
    key_text = _str_list(obj.get("key_text"))[:15]
    what_it_proves = _str_list(obj.get("what_it_proves"))[:15]
    next_action = _str_list(obj.get("next_action"))[:15]
    key_values = _kv(obj.get("key_values"))

    conf_raw = obj.get("confidence")
    try:
        conf = float(conf_raw)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return {
        "type": t,
        "key_text": key_text,
        "key_values": key_values,
        "what_it_proves": what_it_proves,
        "next_action": next_action,
        "confidence": conf,
    }

