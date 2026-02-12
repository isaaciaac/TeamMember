from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from .ark import ArkClient
from .config import thread_state_cooldown_seconds, thread_state_enabled, thread_state_window_msgs
from .db import ChatMessage, SessionLocal, ThreadState
from .utils import parse_json_object


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json_loads(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


def get_thread_state(db, thread_id: str) -> dict[str, Any]:
    """
    Fast path: read-only thread semantic state for routing/prompting.
    """
    tid = str(thread_id or "").strip()
    if not tid:
        return {"open_issues": [], "entropy": {}}

    row = db.get(ThreadState, tid)
    if not row:
        return {"open_issues": [], "entropy": {}}

    issues_raw = _safe_json_loads(str(row.open_issues_json or "[]"), [])
    entropy_raw = _safe_json_loads(str(row.entropy_json or "{}"), {})
    issues: list[dict[str, Any]] = issues_raw if isinstance(issues_raw, list) else []
    entropy: dict[str, Any] = entropy_raw if isinstance(entropy_raw, dict) else {}
    return {"open_issues": issues, "entropy": entropy, "updated_at": row.updated_at.isoformat(), "last_analyzed_message_id": row.last_analyzed_message_id}


def background_refresh_thread_state(thread_id: str) -> None:
    """
    Refresh thread-level semantic state (open loops + topic entropy) in background.
    Fail-closed: errors must not affect main request flow.
    """
    if not thread_state_enabled():
        return
    tid = str(thread_id or "").strip()
    if not tid:
        return

    db = SessionLocal()
    ark: ArkClient | None = None
    try:
        latest = (
            db.query(ChatMessage.id)
            .filter(ChatMessage.thread_id == tid)
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
            .first()
        )
        if not latest:
            return
        latest_msg_id = str(latest[0] or "").strip()
        if not latest_msg_id:
            return

        now_utc = _utcnow()
        row = db.get(ThreadState, tid)
        if row:
            if str(row.last_analyzed_message_id or "") == latest_msg_id:
                return
            try:
                if (now_utc - row.updated_at) < timedelta(seconds=int(thread_state_cooldown_seconds())):
                    return
            except Exception:
                # If timestamps are inconsistent, continue (better to refresh than to get stuck).
                pass

        limit = int(thread_state_window_msgs())
        msg_rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == tid)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        msg_rows.reverse()

        convo_lines: list[str] = []
        for m in msg_rows:
            role = str(m.role or "").strip().lower()
            content = str(m.content or "").strip()
            if not content:
                continue
            if role == "user":
                uid = str(m.user_id or "")[:8]
                convo_lines.append(f"USER({uid}): {content}")
            elif role == "assistant":
                convo_lines.append(f"ASSISTANT: {content}")
            else:
                convo_lines.append(f"SYSTEM: {content}")
        convo_text = "\n".join(convo_lines).strip()
        if not convo_text:
            return

        prev_open: list[Any] = []
        if row and str(row.open_issues_json or "").strip():
            prev_open = _safe_json_loads(str(row.open_issues_json or "[]"), [])
        if not isinstance(prev_open, list):
            prev_open = []
        prev_open_txt = json.dumps(prev_open, ensure_ascii=False)[:6000] if prev_open else "[]"

        prompt = (
            "你是一个线程状态分析器，维护两类状态：\n"
            "1) open_issues：当前未闭合事项（open loops）。\n"
            "2) entropy：主题收敛程度（topic entropy），用于决定是否要反问收敛范围。\n"
            "\n"
            "硬规则：\n"
            "- 只基于输入对话与已有 open_issues，不允许编造事实或个人信息。\n"
            "- 输出必须是严格 JSON，不要输出任何额外文字。\n"
            "- open_issues 最多 8 条；每条包含：summary（短句）、status（open|closed）、needed（最多 5 条）。\n"
            "- entropy 输出：level（low|mid|high）、score（0~1）、clarify（仅在 high 时给 1~3 个问题，否则 []）、reason（短句）。\n"
            "- 如果信息不足：open_issues 输出 []，entropy.level=low。\n"
            "\n"
            "输出 schema：\n"
            "{"
            "\"open_issues\":[{\"summary\":\"...\",\"status\":\"open\",\"needed\":[\"...\"]}],"
            "\"entropy\":{\"level\":\"low\",\"score\":0.1,\"clarify\":[],\"reason\":\"...\"}"
            "}"
        )

        ark = ArkClient()
        out = ark.chat_generate(
            [
                {"role": "system", "content": "You are a strict JSON generator."},
                {
                    "role": "user",
                    "content": prompt + "\n\n已有 open_issues（供你更新/关闭）：\n" + prev_open_txt + "\n\n最近对话：\n" + convo_text[-12000:],
                },
            ]
        )
        obj = parse_json_object(out)

        issues: list[dict[str, Any]] = []
        raw_issues = obj.get("open_issues")
        if isinstance(raw_issues, list):
            for x in raw_issues:
                if not isinstance(x, dict):
                    continue
                summary = str(x.get("summary") or "").strip()
                status = str(x.get("status") or "").strip().lower()
                if status not in {"open", "closed"}:
                    status = "open"
                needed_raw = x.get("needed")
                needed: list[str] = []
                if isinstance(needed_raw, list):
                    for it in needed_raw:
                        s = str(it or "").strip()
                        if s and s not in needed:
                            needed.append(s)
                if len(summary) < 2:
                    continue
                issues.append({"summary": summary[:240], "status": status, "needed": needed[:5]})
                if len(issues) >= 8:
                    break

        ent: dict[str, Any] = {}
        raw_ent = obj.get("entropy")
        if isinstance(raw_ent, dict):
            level = str(raw_ent.get("level") or "").strip().lower()
            if level not in {"low", "mid", "high"}:
                level = "low"
            score_raw = raw_ent.get("score")
            try:
                score = float(score_raw)
            except Exception:
                score = 0.0
            score = max(0.0, min(1.0, score))
            clarify_raw = raw_ent.get("clarify")
            clarify: list[str] = []
            if isinstance(clarify_raw, list):
                for q in clarify_raw:
                    s = str(q or "").strip()
                    if s and s not in clarify:
                        clarify.append(s)
            clarify = clarify[:3] if level == "high" else []
            reason = str(raw_ent.get("reason") or "").strip()[:240]
            ent = {"level": level, "score": score, "clarify": clarify, "reason": reason}

        if not row:
            row = ThreadState(thread_id=tid, open_issues_json="[]", entropy_json="{}", last_analyzed_message_id="")
        row.open_issues_json = json.dumps(issues, ensure_ascii=False)
        row.entropy_json = json.dumps(ent, ensure_ascii=False)
        row.last_analyzed_message_id = latest_msg_id
        db.add(row)
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        return
    finally:
        try:
            if ark is not None:
                ark.close()
        except Exception:
            pass
        db.close()

