from __future__ import annotations

import time
from dataclasses import dataclass

from .db import AppConfig, SessionLocal


@dataclass
class _Cache:
    loaded_at: float = 0.0
    values: dict[str, str] | None = None


_cache = _Cache()
_TTL_SECONDS = 3.0


def invalidate_cache() -> None:
    _cache.values = None
    _cache.loaded_at = 0.0


def _load_all() -> dict[str, str]:
    db = SessionLocal()
    try:
        try:
            rows = db.query(AppConfig).all()
        except Exception:
            return {}
        out: dict[str, str] = {}
        for r in rows:
            k = str(r.key or "").strip()
            if not k:
                continue
            out[k] = str(r.value or "")
        return out
    finally:
        db.close()


def _get_all_cached() -> dict[str, str]:
    now = time.time()
    if _cache.values is not None and (now - _cache.loaded_at) < _TTL_SECONDS:
        return _cache.values
    vals = _load_all()
    _cache.values = vals
    _cache.loaded_at = now
    return vals


def get_str(key: str, default: str = "") -> str:
    vals = _get_all_cached()
    if key in vals:
        return vals[key]
    return default


def _parse_bool(v: str) -> bool | None:
    t = (v or "").strip().lower()
    if t in {"1", "true", "yes", "y", "on"}:
        return True
    if t in {"0", "false", "no", "n", "off"}:
        return False
    return None


def get_bool(key: str, default: bool) -> bool:
    v = get_str(key, "")
    b = _parse_bool(v)
    return default if b is None else b


def set_value(key: str, value: str) -> None:
    k = (key or "").strip()
    if not k:
        raise ValueError("key is required")
    db = SessionLocal()
    try:
        row = db.get(AppConfig, k)
        if not row:
            row = AppConfig(key=k, value="")
        row.value = value
        db.add(row)
        db.commit()
        invalidate_cache()
    finally:
        db.close()


def set_bool(key: str, value: bool) -> None:
    set_value(key, "true" if value else "false")
