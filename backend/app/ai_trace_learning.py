from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from .config import (
    agent_decompose_bias,
    agent_decompose_policy,
    rag_max_context,
    rag_policy,
    rag_top_k,
    web_search_enabled,
    web_search_max_queries,
    web_search_top_k,
)
from .db import AiTraceInsight, AiTraceRun


def _safe_json_loads(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


def build_trace_insight(
    db: Session,
    *,
    window_days: int = 7,
    min_traces: int = 30,
) -> dict[str, Any] | None:
    """
    Level-2 learning: analyze recent ai_trace_runs and propose parameter suggestions.

    Hard rule: MUST NOT auto-apply any config changes.
    """
    wd = int(window_days or 0)
    wd = max(1, min(30, wd))
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=wd)

    rows = (
        db.query(AiTraceRun.router_json, AiTraceRun.web_search_json, AiTraceRun.decompose_json, AiTraceRun.error)
        .filter(AiTraceRun.created_at >= start)
        .order_by(AiTraceRun.created_at.desc())
        .all()
    )
    trace_count = len(rows)
    if trace_count < int(min_traces or 0):
        return None

    # Current effective config snapshot (for suggestions).
    cfg = {
        "rag_policy": rag_policy(),
        "rag_top_k": rag_top_k(),
        "rag_max_context": rag_max_context(),
        "web_search_enabled": web_search_enabled(),
        "web_search_top_k": web_search_top_k(),
        "web_search_max_queries": web_search_max_queries(),
        "agent_decompose_policy": agent_decompose_policy(),
        "agent_decompose_bias": agent_decompose_bias(),
    }

    # Aggregate stats.
    decompose_on = 0
    router_use_rag = 0
    router_use_web = 0
    actual_web_used = 0
    need_clarify = 0
    errors = 0
    complexity_scores: list[int] = []
    uncertainty_scores: list[int] = []

    for router_json, web_json, decomp_json, err in rows:
        router = _safe_json_loads(str(router_json or "{}"), {})
        web = _safe_json_loads(str(web_json or "{}"), {})
        decomp = _safe_json_loads(str(decomp_json or "{}"), {})

        if isinstance(decomp, dict) and bool(decomp.get("enabled")):
            decompose_on += 1

        if isinstance(router, dict):
            if bool(router.get("use_rag")):
                router_use_rag += 1
            if bool(router.get("use_web")):
                router_use_web += 1
            if bool(router.get("need_clarification")):
                need_clarify += 1
            try:
                cs = int(router.get("complexity_score"))
                if 1 <= cs <= 5:
                    complexity_scores.append(cs)
            except Exception:
                pass
            try:
                us = int(router.get("uncertainty_score"))
                if 1 <= us <= 5:
                    uncertainty_scores.append(us)
            except Exception:
                pass

        used = False
        if isinstance(web, dict):
            qs = web.get("queries")
            if isinstance(qs, list) and len(qs) > 0:
                used = True
            else:
                results = web.get("results")
                if isinstance(results, dict) and any(bool(v) for v in results.values()):
                    used = True
        if used:
            actual_web_used += 1

        if str(err or "").strip():
            errors += 1

    def _rate(n: int) -> float:
        return 0.0 if trace_count <= 0 else float(n) / float(trace_count)

    avg_complexity = (sum(complexity_scores) / len(complexity_scores)) if complexity_scores else None
    avg_uncertainty = (sum(uncertainty_scores) / len(uncertainty_scores)) if uncertainty_scores else None

    stats = {
        "window_days": wd,
        "trace_count": trace_count,
        "decompose_rate": _rate(decompose_on),
        "router_use_rag_rate": _rate(router_use_rag),
        "router_use_web_rate": _rate(router_use_web),
        "actual_web_used_rate": _rate(actual_web_used),
        "need_clarification_rate": _rate(need_clarify),
        "error_rate": _rate(errors),
        "avg_complexity_score": avg_complexity,
        "avg_uncertainty_score": avg_uncertainty,
        "config_snapshot": cfg,
    }

    suggestions: list[dict[str, Any]] = []

    # 1) Decompose bias tuning.
    try:
        cur_bias = int(cfg.get("agent_decompose_bias") or 0)
    except Exception:
        cur_bias = 0

    decompose_rate = float(stats["decompose_rate"] or 0.0)
    if avg_complexity is not None and decompose_rate > 0.75 and avg_complexity <= 3.0:
        new_bias = max(0, cur_bias - 10)
        if new_bias != cur_bias:
            suggestions.append(
                {
                    "id": "agent_decompose_bias_down",
                    "title": "子Agent 触发偏多（建议降低拆分偏好）",
                    "why": "近期开启拆分比例较高，但平均复杂度评分偏低，可能让简单问题也进入深度流程。",
                    "evidence": {"decompose_rate": decompose_rate, "avg_complexity_score": avg_complexity},
                    "recommend_changes": [{"key": "agent_decompose_bias", "value": str(new_bias)}],
                }
            )
    if avg_complexity is not None and decompose_rate < 0.15 and avg_complexity >= 4.0:
        new_bias = min(100, cur_bias + 10)
        if new_bias != cur_bias:
            suggestions.append(
                {
                    "id": "agent_decompose_bias_up",
                    "title": "子Agent 触发偏少（建议提高拆分偏好）",
                    "why": "近期复杂问题比例偏高，但拆分触发偏少，可能导致复杂问题直接进入单轮回答。",
                    "evidence": {"decompose_rate": decompose_rate, "avg_complexity_score": avg_complexity},
                    "recommend_changes": [{"key": "agent_decompose_bias", "value": str(new_bias)}],
                }
            )

    # 2) Web search tuning.
    router_web_rate = float(stats["router_use_web_rate"] or 0.0)
    actual_web_rate = float(stats["actual_web_used_rate"] or 0.0)
    if router_web_rate > 0.30 and actual_web_rate < 0.05 and not bool(cfg.get("web_search_enabled")):
        suggestions.append(
            {
                "id": "web_search_enable",
                "title": "路由建议经常用 Web Search，但当前基本没生效",
                "why": "路由器经常给出 use_web=true，但实际搜索命中很少；通常是 Web Search 未开启或未配置 Key。",
                "evidence": {"router_use_web_rate": router_web_rate, "actual_web_used_rate": actual_web_rate},
                "recommend_changes": [{"key": "web_search_enabled", "value": "true"}],
                "notes": "开启后仍需要配置 BING_SEARCH_API_KEY（环境变量）才能真正生效。",
            }
        )
    try:
        cur_qmax = int(cfg.get("web_search_max_queries") or 0)
    except Exception:
        cur_qmax = 2
    if actual_web_rate > 0.40 and cur_qmax < 3:
        suggestions.append(
            {
                "id": "web_search_max_queries_up",
                "title": "Web Search 使用较多（建议提高每次查询条数上限）",
                "why": "当 Web Search 经常被触发时，适度增加每次查询条数能提高覆盖面（成本也会增加）。",
                "evidence": {"actual_web_used_rate": actual_web_rate, "current_max_queries": cur_qmax},
                "recommend_changes": [{"key": "web_search_max_queries", "value": str(min(5, cur_qmax + 1))}],
            }
        )

    # 3) RAG recall tuning (very conservative).
    try:
        cur_rag_k = int(cfg.get("rag_top_k") or 0)
    except Exception:
        cur_rag_k = 10
    rag_rate = float(stats["router_use_rag_rate"] or 0.0)
    if rag_rate > 0.60 and cur_rag_k < 8:
        suggestions.append(
            {
                "id": "rag_top_k_up",
                "title": "RAG 使用频繁（建议提高 TopK 召回）",
                "why": "当路由频繁选择 RAG 时，较低的 TopK 可能造成召回不足（会增加上下文与成本）。",
                "evidence": {"router_use_rag_rate": rag_rate, "current_rag_top_k": cur_rag_k},
                "recommend_changes": [{"key": "rag_top_k", "value": str(10)}],
            }
        )

    return {
        "window_start": start,
        "window_end": now,
        "trace_count": trace_count,
        "stats": stats,
        "suggestions": suggestions,
    }


def persist_trace_insight(db: Session, insight: dict[str, Any]) -> AiTraceInsight:
    row = AiTraceInsight(
        window_start=insight["window_start"],
        window_end=insight["window_end"],
        trace_count=int(insight.get("trace_count") or 0),
        stats_json=json.dumps(insight.get("stats") or {}, ensure_ascii=False),
        suggestions_json=json.dumps(insight.get("suggestions") or [], ensure_ascii=False),
    )
    db.add(row)
    db.flush()
    return row


def serialize_trace_insight(row: AiTraceInsight) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "window_start": row.window_start.isoformat(),
        "window_end": row.window_end.isoformat(),
        "trace_count": int(row.trace_count or 0),
        "stats": _safe_json_loads(str(row.stats_json or "{}"), {}),
        "suggestions": _safe_json_loads(str(row.suggestions_json or "[]"), []),
        "created_at": row.created_at.isoformat(),
    }

