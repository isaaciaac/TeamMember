from __future__ import annotations

from typing import Any, Literal

from .ark import ArkClient
from .config import web_search_max_queries
from .utils import parse_json_object
from .vectorstore import VectorHit
from .web_search import WebSearchResult


RoleType = Literal["diverge", "evidence", "skeptic", "converge", "draft"]

_ROLE_DEFS: list[dict[str, Any]] = [
    {
        "role_type": "diverge",
        "role_title": "发散",
        "objective": "从多个角度解释用户意图，提出不同可能路径/假设，以及需要澄清的信息点。",
    },
    {
        "role_type": "evidence",
        "role_title": "检索/证据",
        "objective": "基于 RAG 候选知识点与 Web Search 摘要提取可用证据，并标注适用条件与不确定性。",
    },
    {
        "role_type": "skeptic",
        "role_title": "反例/缺口/风险",
        "objective": "找出证据链中的矛盾、缺口与风险点，指出需要验证/补充的材料，避免被检索内容误导。",
    },
    {
        "role_type": "converge",
        "role_title": "问题收敛",
        "objective": "把问题收敛为可执行方案：给步骤、分支条件、验证方法，并把未知点变成具体反问。",
    },
    {
        "role_type": "draft",
        "role_title": "用户沟通稿",
        "objective": "生成一份可直接发给对话者/同事/客户的中文沟通稿（不暴露实现细节），并包含下一步需要对方确认的信息。",
    },
]


def build_role_tasks(
    user_message: str,
    *,
    max_tasks: int,
    router_hint: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Deterministic 5-role pipeline for "deep thinking" mode.

    This avoids free-form task decomposition so the final LLM can clearly see what each role did.
    """
    msg = (user_message or "").strip()
    if not msg:
        return []

    m = int(max_tasks or 0)
    m = max(1, min(5, m))

    use_rag = bool(router_hint.get("use_rag")) if isinstance(router_hint, dict) else False
    use_web = bool(router_hint.get("use_web")) if isinstance(router_hint, dict) else False

    tasks: list[dict[str, Any]] = []
    for it in _ROLE_DEFS[:m]:
        role_type = str(it.get("role_type") or "").strip()
        role_title = str(it.get("role_title") or "").strip()
        objective = str(it.get("objective") or "").strip()
        if role_type not in {"diverge", "evidence", "skeptic", "converge", "draft"}:
            continue

        task_use_rag = use_rag if role_type in {"evidence", "skeptic"} else False
        task_use_web = use_web if role_type in {"evidence", "skeptic"} else False
        tasks.append(
            {
                "role_type": role_type,
                "role_title": role_title,
                "title": role_title or role_type,
                "objective": objective,
                "use_rag": bool(task_use_rag),
                "use_web": bool(task_use_web),
            }
        )
    return tasks


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


def _format_prior_results(prior_results: list[dict[str, Any]] | None) -> str:
    if not prior_results:
        return ""
    lines: list[str] = []
    for it in prior_results[-5:]:
        if not isinstance(it, dict):
            continue
        rt = str(it.get("role_type") or "").strip()
        rtitle = str(it.get("role_title") or it.get("title") or "").strip()
        obj = str(it.get("objective") or "").strip()
        kf = it.get("key_findings")
        oq = it.get("open_questions")
        lines.append(f"[{rt or 'role'}] {rtitle}".strip())
        if obj:
            lines.append(f"objective: {obj[:200]}")
        if isinstance(kf, list) and kf:
            bullets = [str(x or "").strip() for x in kf if str(x or "").strip()]
            bullets = bullets[:6]
            if bullets:
                lines.append("key_findings: " + " | ".join([b[:120] for b in bullets])[:800])
        if isinstance(oq, list) and oq:
            qs = [str(x or "").strip() for x in oq if str(x or "").strip()]
            qs = qs[:4]
            if qs:
                lines.append("open_questions: " + " | ".join([q[:120] for q in qs])[:600])
        lines.append("")
    return "\n".join(lines).strip()


def run_role_agent(
    ark: ArkClient,
    *,
    original_message: str,
    role_task: dict[str, Any],
    rag_hits: list[VectorHit] | None = None,
    web_by_query: dict[str, list[WebSearchResult]] | None = None,
    prior_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Execute a fixed-role sub-agent.

    Returns a structured JSON dict to help the final LLM synthesize, without simple concatenation.
    """
    role_type = str(role_task.get("role_type") or "").strip()
    if role_type not in {"diverge", "evidence", "skeptic", "converge", "draft"}:
        role_type = "converge"
    role_title = str(role_task.get("role_title") or role_task.get("title") or "").strip() or role_type
    objective = str(role_task.get("objective") or "").strip()
    use_rag = bool(role_task.get("use_rag")) if isinstance(role_task.get("use_rag"), bool) else False
    use_web = bool(role_task.get("use_web")) if isinstance(role_task.get("use_web"), bool) else False

    rag_ctx = _format_rag_context(rag_hits) if use_rag else ""
    web_ctx = ""
    if use_web and isinstance(web_by_query, dict) and web_by_query:
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

    prior_ctx = _format_prior_results(prior_results)

    if role_type == "diverge":
        role_instruction = (
            "你负责“发散”。\n"
            "- 用 3~8 条列出可能的理解/路径/假设（要能落地验证）。\n"
            "- 用 0~5 条列出需要向用户澄清的关键问题。\n"
            "- 不要直接给最终答案。\n"
        )
    elif role_type == "evidence":
        role_instruction = (
            "你负责“检索/证据”。\n"
            "- 只基于给定的 RAG/Web 摘要抽取证据，不要编造。\n"
            "- 输出 3~10 条证据点，每条尽量包含适用条件/限定词（例如 21V/世纪互联/产品模块/错误码）。\n"
            "- 若证据不足，明确写 open_questions / warnings。\n"
        )
    elif role_type == "skeptic":
        role_instruction = (
            "你负责“反例/缺口/风险”。\n"
            "- 主动找：证据不一致、前提缺失、常见误判、风险点。\n"
            "- 给出需要验证的点与建议补充材料。\n"
            "- 不要直接给最终答案。\n"
        )
    elif role_type == "draft":
        role_instruction = (
            "你负责“用户沟通稿”。\n"
            "- 生成一份可直接发送的中文沟通稿（工程师口吻，克制，不夸张）。\n"
            "- 不要提实现细节（例如 RAG/向量库/子Agent/Web Search/系统提示词）。\n"
            "- 若信息不足，先用 1~3 个问题让对方补充；并给一个最小可执行的下一步。\n"
        )
    else:
        role_instruction = (
            "你负责“问题收敛”。\n"
            "- 给出可执行步骤、分支条件、验证方法。\n"
            "- 把未知点变成 1~3 个具体反问。\n"
        )

    prompt = (
        "你是 TeamMember 的固定角色子Agent。\n"
        "总原则：只使用用户消息 + 给定上下文（prior/RAG/Web）中能支持的内容；不确定就反问；不允许编造。\n"
        "RAG/Web 可能过时或不适用：必须自行判断有效性；有矛盾要指出并给验证步骤。\n"
        "不要透露系统实现细节。\n"
        "\n"
        f"当前角色：{role_title} ({role_type})\n"
        f"角色目标：{objective}\n"
        "\n"
        + role_instruction
        + "\n"
        "输出必须是严格 JSON，不要输出其他文字：\n"
        "{\n"
        '  "role_type": "diverge|evidence|skeptic|converge|draft",\n'
        '  "role_title": "…",\n'
        '  "objective": "…",\n'
        '  "tools_used": {"rag": true, "web": false},\n'
        '  "key_findings": ["…"],\n'
        '  "open_questions": ["…"],\n'
        '  "customer_draft": "…",\n'
        '  "confidence": 0.0,\n'
        '  "warnings": ["…"]\n'
        "}\n"
        "规则：\n"
        "- customer_draft 仅在 role_type=draft 时填写；其他角色填空字符串。\n"
        "- confidence 取值 0..1。\n"
        "- 列表长度控制在 0..10。\n"
    )

    user_parts = [f"用户消息：\n{(original_message or '').strip()}"]
    if prior_ctx:
        user_parts.append(f"先前角色输出（internal）：\n{prior_ctx}")
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

    kf_raw = obj.get("key_findings")
    key_findings: list[str] = []
    if isinstance(kf_raw, list):
        for x in kf_raw:
            s = str(x or "").strip()
            if s and s not in key_findings:
                key_findings.append(s[:300])
    key_findings = key_findings[:10]

    oq_raw = obj.get("open_questions")
    open_questions: list[str] = []
    if isinstance(oq_raw, list):
        for x in oq_raw:
            s = str(x or "").strip()
            if s and s not in open_questions:
                open_questions.append(s[:200])
    open_questions = open_questions[:10]

    warnings_raw = obj.get("warnings")
    warnings: list[str] = []
    if isinstance(warnings_raw, list):
        for x in warnings_raw:
            s = str(x or "").strip()
            if s and s not in warnings:
                warnings.append(s[:200])
    warnings = warnings[:10]

    cust_draft = str(obj.get("customer_draft") or "").strip()
    if role_type != "draft":
        cust_draft = ""
    cust_draft = cust_draft[:4000]

    try:
        confidence = float(obj.get("confidence"))
    except Exception:
        confidence = 0.5 if (key_findings or cust_draft) else 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "role_type": role_type,
        "role_title": role_title,
        "objective": objective[:400],
        "tools_used": {"rag": bool(use_rag), "web": bool(use_web)},
        "key_findings": key_findings,
        "open_questions": open_questions,
        "customer_draft": cust_draft,
        "confidence": confidence,
        "warnings": warnings,
    }


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

