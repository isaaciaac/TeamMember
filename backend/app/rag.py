from __future__ import annotations

from typing import Any

from .ark import ArkClient
from .config import (
    agent_decompose_bias,
    agent_decompose_policy,
    agent_max_subtasks,
    rag_policy,
    rag_teaching_candidates,
    rag_teaching_score_boost,
    rag_top_k,
    web_search_enabled,
    web_search_max_queries,
)
from .qwen import DashScopeClient
from .settings import settings
from .utils import parse_json_object
from .vectorstore import VectorHit, VectorStore


def _rerank_order(ark: ArkClient, query: str, hits: list[VectorHit], top_k: int) -> list[int]:
    docs: list[str] = []
    for idx, h in enumerate(hits):
        payload = h.payload or {}
        src_kind = str(payload.get("source_kind") or "").strip().lower()
        title = str(payload.get("title") or "").strip()
        content = str(payload.get("content") or "").strip()
        text = content[:900]
        head = f"[{idx}]"
        if src_kind:
            head += f" ({src_kind})"
        if title:
            docs.append(f"{head} {title}\n{text}")
        else:
            docs.append(f"{head} {text}")
    prompt = (
        "你是检索重排器。给定查询和候选知识点，请按相关性从高到低输出候选索引。\n"
        "要求：只输出严格 JSON，不要其他文字。\n"
        "输出格式：{\"order\":[0,2,1]}。\n"
        "规则：\n"
        f"- 最多输出 {max(1, int(top_k))} 个索引\n"
        "- 仅输出候选列表中存在的索引\n"
        "- 不确定就保守排序，不要编造新索引\n"
        "- 若候选标注 (teaching) ，代表人工纠错/沉淀过的知识点；在相关性相近时可略优先。\n"
    )
    out = ark.chat_generate(
        [
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": prompt + "\n\n查询：\n" + query + "\n\n候选：\n" + "\n\n".join(docs)},
        ]
    )
    obj = parse_json_object(out)
    raw = obj.get("order")
    if not isinstance(raw, list):
        return []
    order: list[int] = []
    for x in raw:
        try:
            i = int(x)
        except Exception:
            continue
        if i < 0 or i >= len(hits) or i in order:
            continue
        order.append(i)
        if len(order) >= top_k:
            break
    return order


def decide_rag(ark: ArkClient, user_message: str, *, thread_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    意图展开器 + 工具路由器（RAG/Web/Decompose）。
    返回：
    {
      "complexity": "simple|complex",
      "complexity_score": 1..5,
      "uncertainty_score": 1..5,
      "need_clarification": bool,
      "clarify": [str],
      "use_rag": bool,
      "expanded_query": str,
      "use_web": bool,
      "web_queries": [str],
      "suggest_decompose": bool,
      "reason": str
    }
    """
    policy = rag_policy()
    web_enabled = bool(web_search_enabled())
    web_q_max = int(web_search_max_queries() or 2)
    web_q_max = max(1, min(5, web_q_max))
    decomp_policy = agent_decompose_policy()
    decomp_bias = int(agent_decompose_bias() or 0)
    decomp_bias = max(0, min(100, decomp_bias))
    _max_tasks = int(agent_max_subtasks() or 5)

    state_text = ""
    if isinstance(thread_state, dict) and thread_state:
        # Keep it small and safe for prompt injection.
        try:
            open_issues = thread_state.get("open_issues")
            entropy = thread_state.get("entropy")
            if isinstance(open_issues, list) and open_issues:
                open_lines: list[str] = []
                for it in open_issues[:6]:
                    if not isinstance(it, dict):
                        continue
                    s = str(it.get("summary") or "").strip()
                    st = str(it.get("status") or "").strip().lower()
                    if st not in {"open", "closed"}:
                        st = "open"
                    if not s:
                        continue
                    open_lines.append(f"- [{st}] {s}")
                if open_lines:
                    state_text += "open_issues:\n" + "\n".join(open_lines) + "\n"
            if isinstance(entropy, dict) and entropy:
                lvl = str(entropy.get("level") or "").strip().lower()
                score = str(entropy.get("score") or "").strip()
                if lvl:
                    state_text += f"entropy: level={lvl} score={score}\n"
        except Exception:
            state_text = ""

    prompt = (
        "你是一个意图展开器 + 工具路由器（Intent Expander + Tool Router）。\n"
        "给定用户问题，你要做 5 件事：\n"
        "1) 判断问题复杂度：simple 或 complex；并给 complexity_score=1..5。\n"
        "2) 给 uncertainty_score=1..5（信息缺失/指代不明/目标不清时更高）。\n"
        "3) 判断是否需要反问补充信息（need_clarification）。若需要，给 1~3 个澄清问题。\n"
        "4) 判断是否需要检索知识库（use_rag）。若需要，给 expanded_query（用于向量检索）。\n"
        "5) 判断是否需要 Web Search（use_web）。若需要，给 1~N 条 web_queries（用于搜索引擎摘要检索）。\n"
        "额外：判断是否建议启用“子任务拆分”（suggest_decompose）。\n"
        "\n"
        "复杂度判断：\n"
        "- simple：可以在不做大量假设的情况下直接回答，步骤较少。\n"
        "- complex：需要多步排查/对比/条件分支，或关键信息缺失。\n"
        "\n"
        "use_rag 倾向 true 的情况：\n"
        "- 需要查具体产品知识、配置步骤、已知问题、域名/端点、错误码含义\n"
        "- 用户信息不足但可以通过检索找到通用排查路径\n"
        "use_rag 倾向 false 的情况：\n"
        "- 纯闲聊/寒暄\n"
        "- 纯写作润色且不依赖知识库事实\n"
        "\n"
        "use_web 倾向 true 的情况：\n"
        "- 需要外部公开信息的“摘要/线索”，如：最新公告/变更、公开链接、第三方对比、需要验证某个说法是否存在\n"
        "- 用户明确要求：查一下/找链接/验证/最新/今天/版本/价格/公告\n"
        "use_web 倾向 false 的情况：\n"
        "- 用户提供的信息已足够\n"
        "- 纯内部写作，不需要外部事实\n"
        "\n"
        "当前系统检索策略 policy 为："
        f"{policy}\n"
        "当前系统 Web Search 是否可用 web_search_enabled="
        f"{str(web_enabled).lower()}\n"
        "当前系统 子任务拆分策略 agent_decompose_policy="
        f"{decomp_policy}\n"
        "当前系统 子任务拆分偏好 agent_decompose_bias(0..100)="
        f"{decomp_bias}\n"
        "规则：\n"
        "- policy=force_on 时：use_rag 必须为 true。\n"
        "- policy=force_off 时：use_rag 必须为 false。\n"
        "- web_search_enabled=false 时：use_web 必须为 false。\n"
        "- agent_decompose_policy=force_on 时：suggest_decompose 必须为 true。\n"
        "- agent_decompose_policy=force_off 时：suggest_decompose 必须为 false。\n"
        "- bias 越高：越倾向 suggest_decompose=true（但不要为简单闲聊拆分）。\n"
        "- expanded_query 要尽量包含：产品/模块、关键实体、错误码、21V/世纪互联等限定词（若相关）。\n"
        "- need_clarification=true 时：clarify 必须非空。\n"
        f"- web_queries 最多 {web_q_max} 条，每条尽量短（<=80字）。\n"
        "\n"
        "线程状态（internal，可为空）：\n"
        "- 可能存在未闭合事项 open_issues，以及主题收敛程度 entropy。\n"
        "- 如果用户使用“这个/那个/继续/上面”等指代词，且 open_issues 不止 1 个：优先 need_clarification=true，用 clarify 让用户选定指代对象。\n"
        "- 如果 entropy.level=high 且用户目标不明确：优先 need_clarification=true，给 1~3 个收敛问题。\n"
        f"- 若你建议拆分（suggest_decompose=true）：通常意味着需要多视角并行处理，后续系统会把任务拆成最多 {_max_tasks} 个子任务。\n"
        "\n"
        "输出必须是严格 JSON，不要输出其他文字：\n"
        "{"
        "\"complexity\":\"simple\","
        "\"complexity_score\":2,"
        "\"uncertainty_score\":2,"
        "\"need_clarification\":false,"
        "\"clarify\":[\"...\"],"
        "\"use_rag\":true,"
        "\"expanded_query\":\"...\","
        "\"use_web\":false,"
        "\"web_queries\":[\"...\"],"
        "\"suggest_decompose\":false,"
        "\"reason\":\"...\""
        "}"
    )

    out = ark.chat_generate(
        [
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": prompt + ("\n\n线程状态：\n" + state_text if state_text else "") + "\n\n用户问题：\n" + user_message},
        ]
    )
    obj = parse_json_object(out)
    complexity = str(obj.get("complexity") or "").strip().lower()
    if complexity not in {"simple", "complex"}:
        complexity = "complex"

    try:
        complexity_score = int(obj.get("complexity_score"))
    except Exception:
        complexity_score = 2 if complexity == "simple" else 4
    complexity_score = max(1, min(5, complexity_score))

    try:
        uncertainty_score = int(obj.get("uncertainty_score"))
    except Exception:
        uncertainty_score = 2 if complexity == "simple" else 3
    uncertainty_score = max(1, min(5, uncertainty_score))

    need_clarification = bool(obj.get("need_clarification")) if isinstance(obj.get("need_clarification"), bool) else False

    use_rag = bool(obj.get("use_rag")) if isinstance(obj.get("use_rag"), bool) else False
    if policy == "force_on":
        use_rag = True
    elif policy == "force_off":
        use_rag = False

    use_web = bool(obj.get("use_web")) if isinstance(obj.get("use_web"), bool) else False
    if not web_enabled:
        use_web = False

    expanded = str(obj.get("expanded_query") or user_message).strip() or user_message
    expanded = expanded[:800]
    reason = str(obj.get("reason") or "").strip()
    clarify_raw = obj.get("clarify")
    clarify: list[str] = []
    if isinstance(clarify_raw, list):
        for x in clarify_raw:
            s = str(x or "").strip()
            if s and s not in clarify:
                clarify.append(s)
    clarify = clarify[:3]
    if need_clarification and not clarify:
        need_clarification = False

    web_queries: list[str] = []
    raw_web_q = obj.get("web_queries")
    if isinstance(raw_web_q, list):
        for x in raw_web_q:
            s = str(x or "").strip()
            if s and s not in web_queries:
                web_queries.append(s[:120])
    web_queries = web_queries[:web_q_max]
    if use_web and not web_queries:
        use_web = False

    suggest_decompose = bool(obj.get("suggest_decompose")) if isinstance(obj.get("suggest_decompose"), bool) else False
    if decomp_policy == "force_on":
        suggest_decompose = True
    elif decomp_policy == "force_off":
        suggest_decompose = False
    else:
        # deterministic bias adjustment to make behavior predictable
        if decomp_bias <= 20 and complexity_score <= 3 and uncertainty_score <= 3:
            suggest_decompose = False
        if decomp_bias >= 80 and complexity_score >= 3 and (use_rag or use_web or uncertainty_score >= 3):
            suggest_decompose = True
        if complexity_score <= 2 and uncertainty_score <= 2:
            # never decompose trivial questions
            suggest_decompose = False
    return {
        "complexity": complexity,
        "complexity_score": complexity_score,
        "uncertainty_score": uncertainty_score,
        "need_clarification": need_clarification,
        "clarify": clarify,
        "use_rag": use_rag,
        "expanded_query": expanded,
        "use_web": use_web,
        "web_queries": web_queries,
        "suggest_decompose": suggest_decompose,
        "reason": reason,
    }


def retrieve_knowledge(query: str, top_k: int | None = None) -> list[VectorHit]:
    k = int(top_k) if top_k is not None else rag_top_k()
    k = max(0, min(50, k))
    if k <= 0:
        return []

    llm = DashScopeClient()
    vs = VectorStore()
    vs.ensure_collections()
    try:
        vec = llm.embed_texts([query])[0]
        candidate_k = max(k, min(80, k * 4))
        candidate_k = max(candidate_k, int(rag_teaching_candidates() or 0))
        candidate_k = min(200, candidate_k)
        hits = vs.search_knowledge(vec, limit=candidate_k)
        if not hits:
            return []

        boost = float(rag_teaching_score_boost() or 0.0)
        if boost > 0:
            for h in hits:
                try:
                    if str((h.payload or {}).get("source_kind") or "").strip().lower() == "teaching":
                        h.score = float(h.score) + boost
                except Exception:
                    continue
            hits.sort(key=lambda x: float(x.score), reverse=True)

        threshold = int(getattr(settings, "rag_rerank_min_docs", 0) or 0)
        if threshold <= 0:
            return hits[:k]

        total_docs = vs.count(settings.qdrant_knowledge_collection)
        if total_docs < threshold:
            return hits[:k]

        if not settings.ark_api_key:
            return hits[:k]

        ark: ArkClient | None = None
        try:
            ark = ArkClient()
            order = _rerank_order(ark, query, hits, k)
            if not order:
                return hits[:k]
            return [hits[i] for i in order]
        except Exception:
            return hits[:k]
        finally:
            try:
                if ark is not None:
                    ark.close()
            except Exception:
                pass
    finally:
        llm.close()
