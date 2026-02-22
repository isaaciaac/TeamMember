from __future__ import annotations

import json
from typing import Any

from .db import AiTraceRun


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return ""


def create_ai_trace_run(
    db,
    *,
    thread_id: str,
    actor_user_id: str,
    user_message_id: str,
    router: dict[str, Any] | None = None,
    web_search: dict[str, Any] | None = None,
    decompose: dict[str, Any] | None = None,
    subagent: list[dict[str, Any]] | None = None,
) -> AiTraceRun:
    row = AiTraceRun(
        thread_id=str(thread_id),
        actor_user_id=str(actor_user_id),
        user_message_id=str(user_message_id),
        assistant_message_id="",
        router_json=_safe_json_dumps(router or {}),
        web_search_json=_safe_json_dumps(web_search or {}),
        decompose_json=_safe_json_dumps(decompose or {}),
        subagent_json=_safe_json_dumps(subagent or []),
        error="",
    )
    db.add(row)
    return row


def update_ai_trace_run(
    db,
    row: AiTraceRun,
    *,
    assistant_message_id: str | None = None,
    error: str | None = None,
) -> AiTraceRun:
    if assistant_message_id is not None:
        row.assistant_message_id = str(assistant_message_id)
    if error is not None:
        row.error = str(error or "")
    db.add(row)
    return row

