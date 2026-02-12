from __future__ import annotations

from .runtime_config import get_bool, get_str
from .settings import settings


def rag_policy() -> str:
    v = get_str("rag_policy", settings.rag_policy)
    v = (v or "").strip().lower()
    if v not in {"auto", "force_on", "force_off"}:
        return (settings.rag_policy or "auto").strip().lower()
    return v


def rag_top_k() -> int:
    raw = get_str("rag_top_k", str(settings.rag_top_k))
    try:
        v = int(raw)
    except Exception:
        v = int(settings.rag_top_k)
    return max(0, min(50, v))


def rag_max_context() -> int:
    raw = get_str("rag_max_context", str(settings.rag_max_context))
    try:
        v = int(raw)
    except Exception:
        v = int(settings.rag_max_context)
    return max(0, min(50, v))


def rag_teaching_score_boost() -> float:
    raw = get_str("rag_teaching_score_boost", str(settings.rag_teaching_score_boost))
    try:
        v = float(raw)
    except Exception:
        v = float(settings.rag_teaching_score_boost)
    return max(0.0, min(1.0, v))


def rag_teaching_candidates() -> int:
    raw = get_str("rag_teaching_candidates", str(settings.rag_teaching_candidates))
    try:
        v = int(raw)
    except Exception:
        v = int(settings.rag_teaching_candidates)
    return max(0, min(200, v))


def memory_enabled() -> bool:
    return get_bool("memory_enabled", bool(settings.memory_enabled))


def memory_top_k() -> int:
    raw = get_str("memory_top_k", str(settings.memory_top_k))
    try:
        v = int(raw)
    except Exception:
        v = int(settings.memory_top_k)
    return max(0, min(20, v))


def topic_guard_enabled() -> bool:
    return get_bool("topic_guard_enabled", bool(settings.topic_guard_enabled))


def topic_allowed_topics() -> str:
    v = get_str("topic_allowed_topics", str(settings.topic_allowed_topics))
    return (v or "").strip()


def proactive_enabled() -> bool:
    return get_bool("proactive_enabled", bool(settings.proactive_enabled))


def proactive_min_user_msgs() -> int:
    raw = get_str("proactive_min_user_msgs", str(settings.proactive_min_user_msgs))
    try:
        v = int(raw)
    except Exception:
        v = int(settings.proactive_min_user_msgs)
    return max(0, v)


def proactive_weekday_only() -> bool:
    return get_bool("proactive_weekday_only", bool(settings.proactive_weekday_only))


def proactive_work_start() -> str:
    return (get_str("proactive_work_start", str(settings.proactive_work_start)) or "").strip()


def proactive_work_end() -> str:
    return (get_str("proactive_work_end", str(settings.proactive_work_end)) or "").strip()


def proactive_timezone() -> str:
    return (get_str("proactive_timezone", str(settings.proactive_timezone)) or "").strip()


def thread_state_enabled() -> bool:
    return get_bool("thread_state_enabled", bool(settings.thread_state_enabled))


def thread_state_window_msgs() -> int:
    raw = get_str("thread_state_window_msgs", str(settings.thread_state_window_msgs))
    try:
        v = int(raw)
    except Exception:
        v = int(settings.thread_state_window_msgs)
    return max(10, min(200, v))


def thread_state_cooldown_seconds() -> int:
    raw = get_str("thread_state_cooldown_seconds", str(settings.thread_state_cooldown_seconds))
    try:
        v = int(raw)
    except Exception:
        v = int(settings.thread_state_cooldown_seconds)
    return max(10, min(3600, v))


def decision_profile_enabled() -> bool:
    return get_bool("decision_profile_enabled", bool(settings.decision_profile_enabled))


def decision_profile_refresh_hours() -> int:
    raw = get_str("decision_profile_refresh_hours", str(settings.decision_profile_refresh_hours))
    try:
        v = int(raw)
    except Exception:
        v = int(settings.decision_profile_refresh_hours)
    return max(1, min(24 * 30, v))


def persona_disclosure_enabled() -> bool:
    return get_bool("persona_disclosure_enabled", bool(settings.persona_disclosure_enabled))
