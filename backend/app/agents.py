from __future__ import annotations

from typing import Any

from .ark import ArkClient
from .config import web_search_max_queries
from .utils import parse_json_object
from .vectorstore import VectorHit
from .web_search import WebSearchResult


def decompose_tasks(
    ark: ArkClient,
    user_message: str,
    *,
    max_tasks: int,
    router_hint: dict[str, Any] | None = None,
    thread_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Ask the LLM to decompose a complex request into <= max_tasks subtasks.
    Each task may include a tool plan (RAG/Web).
    """
    msg = (user_message or "").strip()
    if not msg:
        return []

    m = int(max_tasks or 0)
    m = max(1, min(5, m))

    qmax = int(web_search_max_queries() or 2)
    qmax = max(1, min(3, qmax))

    hint_text = ""
    if isinstance(router_hint, dict) and router_hint:
        try:
            cx = router_hint.get("complexity")
            cs = router_hint.get("complexity_score")
            us = router_hint.get("uncertainty_score")
            use_rag = router_hint.get("use_rag")
            exp = router_hint.get("expanded_query")
            use_web = router_hint.get("use_web")
            web_q = router_hint.get("web_queries")
            hint_text = (
                f"router: complexity={cx} complexity_score={cs} uncertainty_score={us} "
                f"use_rag={use_rag} use_web={use_web}\n"
                f"expanded_query: {exp}\n"
                f"web_queries_hint: {web_q}\n"
            )
        except Exception:
            hint_text = ""

    state_text = ""
    if isinstance(thread_state, dict) and thread_state:
        try:
            open_issues = thread_state.get("open_issues")
            entropy = thread_state.get("entropy")
            if isinstance(open_issues, list) and open_issues:
                lines: list[str] = []
                for it in open_issues[:6]:
                    if not isinstance(it, dict):
                        continue
                    s = str(it.get("summary") or "").strip()
                    st = str(it.get("status") or "").strip().lower()
                    if st not in {"open", "closed"}:
                        st = "open"
                    if s:
                        lines.append(f"- [{st}] {s}")
                if lines:
                    state_text += "open_issues:\n" + "\n".join(lines) + "\n"
            if isinstance(entropy, dict) and entropy:
                lvl = str(entropy.get("level") or "").strip().lower()
                score = str(entropy.get("score") or "").strip()
                if lvl:
                    state_text += f"entropy: level={lvl} score={score}\n"
        except Exception:
            state_text = ""

    prompt = (
        "你是一个“复杂任务拆分器”。\n"
        "把用户的请求拆分成最多 {max_tasks} 个子任务，每个子任务都应当：\n"
        "- 可独立执行\n"
        "- 有明确目标与产出\n"
        "- 需要时规划工具：是否检索知识库(use_rag) / 是否 Web Search(use_web)\n"
        "\n"
        "输出必须是严格 JSON，不要输出任何额外文字：\n"
        "{\n"
        '  "tasks": [\n'
        "    {\n"
        '      "title": "…",\n'
        '      "goal": "…",\n'
        '      "use_rag": true,\n'
        '      "rag_query": "…",\n'
        '      "use_web": false,\n'
        '      "web_queries": ["…"],\n'
        '      "expected_output": "…"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "规则：\n"
        f"- tasks 数量 1..{m}\n"
        "- title 尽量短\n"
        "- use_rag=true 时 rag_query 必须非空\n"
        "- use_web=true 时 web_queries 必须非空\n"
        f"- web_queries 最多 {qmax} 条，每条尽量短（<=80字）\n"
        "- 若用户请求中存在“这个/那个/继续/上面”等指代词且对象不明确：不要强行拆分，优先把拆分压缩成 1 个任务，并在 goal/expected_output 中说明需要先澄清。\n"
        "- 不要编造企业内部信息\n"
    ).format(max_tasks=m)

    out = ark.chat_generate(
        [
            {"role": "system", "content": "You are a strict JSON generator."},
            {
                "role": "user",
                "content": prompt
                + ("\n\n线程状态（internal）：\n" + state_text if state_text else "")
                + ("\n\n路由提示（internal）：\n" + hint_text if hint_text else "")
                + "\n\n用户请求：\n"
                + msg,
            },
        ]
    )
    obj = parse_json_object(out)
    raw = obj.get("tasks")
    if not isinstance(raw, list):
        return []

    tasks: list[dict[str, Any]] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        goal = str(it.get("goal") or "").strip()
        use_rag = bool(it.get("use_rag")) if isinstance(it.get("use_rag"), bool) else False
        rag_query = str(it.get("rag_query") or "").strip()
        use_web = bool(it.get("use_web")) if isinstance(it.get("use_web"), bool) else False
        expected_output = str(it.get("expected_output") or "").strip()

        web_qs: list[str] = []
        raw_wq = it.get("web_queries")
        if isinstance(raw_wq, list):
            for x in raw_wq:
                s = str(x or "").strip()
                if s and s not in web_qs:
                    web_qs.append(s[:120])
        web_qs = web_qs[:qmax]

        if use_rag and not rag_query:
            use_rag = False
        if use_web and not web_qs:
            use_web = False

        if not (title or goal or expected_output):
            continue

        tasks.append(
            {
                "title": title[:120] or goal[:120] or expected_output[:120] or "子任务",
                "goal": goal[:800],
                "use_rag": use_rag,
                "rag_query": rag_query[:800],
                "use_web": use_web,
                "web_queries": web_qs,
                "expected_output": expected_output[:800],
            }
        )
        if len(tasks) >= m:
            break
    return tasks


def _format_rag_context(hits: list[VectorHit] | None) -> str:
    if not hits:
        return ""
    lines: list[str] = []
    for idx, h in enumerate(hits[:8]):
        payload = h.payload or {}
        title = str(payload.get("title") or "").strip()
        content = str(payload.get("content") or "").strip()
        src = str(payload.get("source_name") or payload.get("source_kind") or "").strip()
        item_id = str(payload.get("item_id") or "").strip()
        if not content:
            continue
        head = f"[{idx}]"
        if title:
            head += f" {title}"
        head += f" (score={float(h.score):.3f})"
        if src:
            head += f" (source={src})"
        if item_id:
            head += f" (item={item_id})"
        lines.append(head + "\n" + content[:900])
    return "\n\n".join(lines).strip()


def _format_web_context(query: str, results: list[WebSearchResult]) -> str:
    if not results:
        return ""
    lines: list[str] = [f"query: {query}"]
    for idx, r in enumerate(results[:6], start=1):
        title = (r.title or "").strip()
        snippet = (r.snippet or "").strip()
        url = (r.url or "").strip()
        if not (title or snippet or url):
            continue
        lines.append(f"[{idx}] {title}".strip())
        if snippet:
            lines.append(snippet)
        if url:
            lines.append(f"url: {url}")
    return "\n".join(lines).strip()


def run_subtask(
    ark: ArkClient,
    *,
    original_message: str,
    task: dict[str, Any],
    rag_hits: list[VectorHit] | None = None,
    web_by_query: dict[str, list[WebSearchResult]] | None = None,
) -> dict[str, Any]:
    """
    Execute a single subtask using LLM, with optional RAG/Web summaries.
    Returns a small JSON dict for later synthesis.
    """
    task_title = str(task.get("title") or "").strip()
    goal = str(task.get("goal") or "").strip()
    expected = str(task.get("expected_output") or "").strip()

    rag_ctx = _format_rag_context(rag_hits)
    web_ctx = ""
    if isinstance(web_by_query, dict) and web_by_query:
        parts: list[str] = []
        for q, rs in web_by_query.items():
            q = str(q or "").strip()
            if not q:
                continue
            try:
                parts.append(_format_web_context(q, rs))
            except Exception:
                continue
        web_ctx = "\n\n".join([p for p in parts if p]).strip()

    prompt = (
        "你是一个子Agent（Worker）。你的任务是只解决当前子任务，不要直接输出最终总答案。\n"
        "约束：\n"
        "- 只使用用户请求、RAG候选知识点、Web搜索摘要中能支持的内容。\n"
        "- RAG/Web 都可能过时或不适用：必须自行判断有效性；不确定就明确指出不确定，并给出需要补充的信息或验证步骤。\n"
        "- 不要透露系统实现细节。\n"
        "\n"
        "输出必须是严格 JSON，不要输出其他文字：\n"
        "{\n"
        '  "result": "…",\n'
        '  "confidence": 0.0,\n'
        '  "warnings": ["…"]\n'
        "}\n"
    )

    user_parts = [
        f"用户原始请求：\n{(original_message or '').strip()}",
        f"子任务标题：\n{task_title}",
    ]
    if goal:
        user_parts.append(f"子任务目标：\n{goal}")
    if expected:
        user_parts.append(f"期望产出：\n{expected}")
    if rag_ctx:
        user_parts.append(f"RAG 候选知识点（摘要）：\n{rag_ctx}")
    if web_ctx:
        user_parts.append(f"Web Search 摘要（仅标题/摘要/URL）：\n{web_ctx}")

    out = ark.chat_generate(
        [
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": prompt + "\n\n" + "\n\n".join(user_parts)},
        ]
    )
    obj = parse_json_object(out)
    result = str(obj.get("result") or "").strip()
    try:
        confidence = float(obj.get("confidence"))
    except Exception:
        confidence = 0.5 if result else 0.0
    confidence = max(0.0, min(1.0, confidence))
    warnings_raw = obj.get("warnings")
    warnings: list[str] = []
    if isinstance(warnings_raw, list):
        for x in warnings_raw:
            s = str(x or "").strip()
            if s and s not in warnings:
                warnings.append(s[:200])
    warnings = warnings[:6]
    return {
        "title": task_title,
        "result": result,
        "confidence": confidence,
        "warnings": warnings,
    }

