from __future__ import annotations

import json
import re
from typing import Any


def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    end = t.rfind("```")
    if end <= 0:
        return t
    inner = t[3:end]
    inner = re.sub(r"^\w+\n", "", inner).strip()
    return inner


def parse_json_array(text: str) -> list[Any]:
    t = strip_code_fences(text).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    m = re.search(r"\[.*\]", t, flags=re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    return obj if isinstance(obj, list) else []


def parse_json_object(text: str) -> dict[str, Any]:
    t = strip_code_fences(text).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}

