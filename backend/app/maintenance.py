from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import text

from .ai_trace_learning import build_trace_insight, persist_trace_insight
from .config import ai_trace_enabled, ai_trace_retention_days, decision_profile_enabled, decision_profile_refresh_hours, proactive_timezone
from .db import AiTraceInsight, SessionLocal, User, UserDecisionProfile, engine
from .memory import background_refresh_decision_profile

_LOG = logging.getLogger(__name__)

_STARTED = False
_START_LOCK = threading.Lock()

# A fixed advisory lock id to prevent duplicate maintenance runs when multiple processes exist.
_LOCK_DECISION_PROFILE = 903184221
_LOCK_AI_TRACE_CLEANUP = 903184222
_LOCK_AI_TRACE_INSIGHTS = 903184223

# "Idle time" maintenance target (local timezone).
_RUN_AT_HOUR = 3
_RUN_AT_MINUTE = 15


def start_maintenance_thread() -> None:
    global _STARTED
    with _START_LOCK:
        if _STARTED:
            return
        t = threading.Thread(target=_maintenance_loop, name="tm_maintenance", daemon=True)
        t.start()
        _STARTED = True
        _LOG.info("maintenance thread started")


def _get_tz() -> timezone:
    name = (proactive_timezone() or "").strip()
    if not name:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except Exception:
        pass
    if name.lower() in {"asia/shanghai", "asia/chongqing", "prc", "cst"}:
        return timezone(timedelta(hours=8))
    m = re.match(r"^([+-])(\d{2}):(\d{2})$", name)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        hh = int(m.group(2))
        mm = int(m.group(3))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return timezone(sign * timedelta(hours=hh, minutes=mm))
    return timezone.utc


def _next_local_run(now_local: datetime, *, hour: int, minute: int) -> datetime:
    cand = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cand <= now_local:
        cand = cand + timedelta(days=1)
    return cand


def _maintenance_loop() -> None:
    while True:
        try:
            tz = _get_tz()
            now_local = datetime.now(tz)
            next_run = _next_local_run(now_local, hour=_RUN_AT_HOUR, minute=_RUN_AT_MINUTE)
            sleep_s = (next_run - now_local).total_seconds()
            if sleep_s > 0:
                time.sleep(sleep_s)
            _run_decision_profile_maintenance()
            _cleanup_ai_traces()
            _generate_ai_trace_insights()
            # Avoid accidental double-run if system clock jumps.
            time.sleep(2.0)
        except Exception:
            # Never let the maintenance thread crash the process.
            time.sleep(60.0)


def _run_decision_profile_maintenance() -> None:
    if not decision_profile_enabled():
        return

    # Take a global advisory lock so only one worker performs the job.
    conn = engine.connect()
    try:
        got = conn.execute(text("SELECT pg_try_advisory_lock(:id)"), {"id": _LOCK_DECISION_PROFILE}).scalar()
        if not got:
            return

        refresh_hours = int(decision_profile_refresh_hours())
        threshold = datetime.now(timezone.utc) - timedelta(hours=refresh_hours)

        db = SessionLocal()
        try:
            due_rows = (
                db.query(User.id)
                .outerjoin(UserDecisionProfile, UserDecisionProfile.user_id == User.id)
                .filter((UserDecisionProfile.user_id.is_(None)) | (UserDecisionProfile.updated_at < threshold))
                .all()
            )
            due_user_ids = [str(r[0]) for r in due_rows if str(r[0] or "").strip()]
        finally:
            db.close()

        if not due_user_ids:
            return

        _LOG.info("decision_profile maintenance: due_users=%s refresh_hours=%s", len(due_user_ids), refresh_hours)
        for uid in due_user_ids:
            try:
                background_refresh_decision_profile(uid)
            except Exception:
                # Per-user failures are tolerated; keep going.
                continue
    finally:
        try:
            conn.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": _LOCK_DECISION_PROFILE})
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def _cleanup_ai_traces() -> None:
    days = int(ai_trace_retention_days() or 0)
    days = max(1, min(365, days))
    threshold = datetime.now(timezone.utc) - timedelta(days=days)

    conn = engine.connect()
    try:
        got = conn.execute(text("SELECT pg_try_advisory_lock(:id)"), {"id": _LOCK_AI_TRACE_CLEANUP}).scalar()
        if not got:
            return
        res = conn.execute(text("DELETE FROM ai_trace_runs WHERE created_at < :th"), {"th": threshold})
        try:
            deleted = int(getattr(res, "rowcount", 0) or 0)
        except Exception:
            deleted = 0
        if deleted:
            _LOG.info("ai_trace cleanup: deleted=%s retention_days=%s", deleted, days)
    finally:
        try:
            conn.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": _LOCK_AI_TRACE_CLEANUP})
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def _generate_ai_trace_insights() -> None:
    if not ai_trace_enabled():
        return

    conn = engine.connect()
    try:
        got = conn.execute(text("SELECT pg_try_advisory_lock(:id)"), {"id": _LOCK_AI_TRACE_INSIGHTS}).scalar()
        if not got:
            return

        db = SessionLocal()
        try:
            # Only generate once per ~day.
            latest = db.query(AiTraceInsight).order_by(AiTraceInsight.created_at.desc()).limit(1).first()
            if latest is not None:
                try:
                    if (datetime.now(timezone.utc) - latest.created_at) < timedelta(hours=20):
                        return
                except Exception:
                    pass

            insight = build_trace_insight(db, window_days=7, min_traces=30)
            if not insight:
                return

            persist_trace_insight(db, insight)
            db.commit()

            # Keep insights roughly aligned with trace retention window (defense in depth).
            days = int(ai_trace_retention_days() or 30)
            days = max(7, min(365, days))
            threshold = datetime.now(timezone.utc) - timedelta(days=days)
            db.query(AiTraceInsight).filter(AiTraceInsight.created_at < threshold).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()
    finally:
        try:
            conn.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": _LOCK_AI_TRACE_INSIGHTS})
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
