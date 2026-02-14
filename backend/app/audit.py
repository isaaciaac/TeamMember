from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import Request
from sqlalchemy import text

from .db import AuditLog, User

_ADVISORY_LOCK_ID = 903184222

_REDACT_KEYS = re.compile(r"(password|secret|token|api[_-]?key|client_secret|authorization)", flags=re.I)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _canon_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return ""


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if _REDACT_KEYS.search(ks):
                out[ks] = "***REDACTED***"
                continue
            out[ks] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj[:100]]
    if isinstance(obj, str):
        s = obj
        if len(s) > 4000:
            return s[:4000] + "...(truncated)"
        return s
    return obj


def _actor_label(user: User) -> str:
    name = str(getattr(user, "name", "") or "").strip()
    phone = str(getattr(user, "phone", "") or "").strip()
    email = str(getattr(user, "email", "") or "").strip()
    if name and phone:
        return f"{name}<{phone}>"
    if name and email:
        return f"{name}<{email}>"
    return name or phone or email or str(getattr(user, "id", "") or "")[:8]


def append_audit_log(
    db,
    *,
    actor: User,
    action: str,
    entity_type: str = "",
    entity_id: str = "",
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    request: Request | None = None,
) -> None:
    """
    Append-only audit logging for admin actions.
    Designed to be called inside the same DB transaction as the admin mutation.
    """

    db.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": _ADVISORY_LOCK_ID})

    prev = (
        db.query(AuditLog.event_hash)
        .order_by(AuditLog.created_at.desc())
        .limit(1)
        .scalar()
    )
    prev_hash = str(prev or "")

    created_at = _utcnow()
    method = ""
    path = ""
    ip = ""
    ua = ""
    if request is not None:
        try:
            method = str(request.method or "")
        except Exception:
            method = ""
        try:
            path = str(request.url.path or "")
        except Exception:
            path = ""
        try:
            ip = str(getattr(getattr(request, "client", None), "host", "") or "")
        except Exception:
            ip = ""
        try:
            ua = str(request.headers.get("user-agent") or "")
        except Exception:
            ua = ""

    before_obj = _redact(before or {})
    after_obj = _redact(after or {})
    before_json = _canon_json(before_obj)
    after_json = _canon_json(after_obj)

    payload = {
        "created_at": created_at.isoformat(),
        "actor_user_id": str(actor.id),
        "action": str(action or ""),
        "entity_type": str(entity_type or ""),
        "entity_id": str(entity_id or ""),
        "request_method": method,
        "request_path": path,
        "request_ip": ip,
        "user_agent": ua,
        "before": before_obj,
        "after": after_obj,
    }
    payload_json = _canon_json(payload)
    event_hash = hashlib.sha256((prev_hash + "\n" + payload_json).encode("utf-8")).hexdigest()

    row = AuditLog(
        actor_user_id=str(actor.id),
        actor_label=_actor_label(actor),
        action=str(action or "")[:120],
        entity_type=str(entity_type or "")[:60],
        entity_id=str(entity_id or "")[:120],
        request_method=method[:10],
        request_path=path[:300],
        request_ip=ip[:80],
        user_agent=ua[:500],
        before_json=before_json,
        after_json=after_json,
        prev_hash=prev_hash[:80],
        event_hash=event_hash[:80],
        created_at=created_at,
    )
    db.add(row)

