from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from qdrant_client.http.models import PointStruct

from .ark import ArkClient
from .config import decision_profile_enabled, decision_profile_refresh_hours, memory_enabled, memory_top_k
from .db import SessionLocal, User, UserDecisionProfile, UserMemory, UserPersona
from .qwen import DashScopeClient
from .settings import settings
from .vectorstore import VectorStore


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    end = t.rfind("```")
    if end <= 0:
        return t
    inner = t[3:end]
    inner = re.sub(r"^\w+\n", "", inner).strip()
    return inner


def _parse_json_array(text: str) -> list[Any]:
    t = _strip_code_fences(text).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    m = re.search(r"\[.*\]", t, flags=re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    return obj if isinstance(obj, list) else []


def get_user_context(user_id: str, query_text: str) -> dict[str, Any]:
    """
    返回用于提示词拼装的用户上下文（profile + persona + decision_profile + memory snippets）。
    该函数不做 LLM 更新，只做读取与检索，保证快。
    """
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        profile = (user.profile if user else "").strip()

        persona_row = db.get(UserPersona, user_id)
        persona = (persona_row.summary if persona_row else "").strip()

        decision_row = db.get(UserDecisionProfile, user_id)
        decision = (decision_row.summary if decision_row else "").strip()

    finally:
        db.close()

    memory_snippets: list[str] = []
    if memory_enabled() and settings.dashscope_api_key and query_text.strip():
        llm: DashScopeClient | None = None
        try:
            llm = DashScopeClient()
            vs = VectorStore()
            vs.ensure_collections()
            vec = llm.embed_texts([query_text])[0]
            hits = vs.search_memory(vec, limit=int(memory_top_k()), user_id=user_id)
            for h in hits:
                c = str((h.payload or {}).get("content") or "").strip()
                if c:
                    memory_snippets.append(c)
        except Exception:
            memory_snippets = []
        finally:
            try:
                if llm is not None:
                    llm.close()
            except Exception:
                pass

    return {"profile": profile, "persona": persona, "decision": decision, "memory": memory_snippets}


def background_extract_memory(user_id: str, user_message: str) -> None:
    """
    从单条用户消息中抽取“可复用”的记忆点，写入 SQL + Qdrant memory collection。
    失败不抛出（后台任务不能影响主流程）。
    """
    if not memory_enabled():
        return
    text = (user_message or "").strip()
    if len(text) < 6:
        return

    try:
        ark = ArkClient()
        items = _extract_memory_items(ark, text)
    except Exception:
        return
    finally:
        try:
            ark.close()
        except Exception:
            pass

    if not items:
        return

    db = SessionLocal()
    try:
        llm = DashScopeClient()
        vs = VectorStore()
        vs.ensure_collections()

        rows: list[UserMemory] = []
        for it in items[:10]:
            c = str(it or "").strip()
            if not c:
                continue
            rows.append(UserMemory(user_id=user_id, content=c))

        if not rows:
            return

        for r in rows:
            db.add(r)
        db.commit()

        # vectorize + upsert
        vecs = llm.embed_texts([r.content for r in rows])
        points: list[PointStruct] = []
        for r, v in zip(rows, vecs):
            payload = {
                "user_id": user_id,
                "memory_id": r.id,
                "content": r.content,
                "created_at": _utcnow_iso(),
            }
            points.append(PointStruct(id=r.id, vector=v, payload=payload))
        vs.upsert(settings.qdrant_memory_collection, points)
    except Exception:
        return
    finally:
        try:
            llm.close()
        except Exception:
            pass
        db.close()


def _extract_memory_items(ark: ArkClient, user_message: str) -> list[str]:
    prompt = (
        "你在为一个团队协作助手抽取“长期可复用记忆”。\n"
        "只从输入里提取事实/偏好/角色信息，不要编造。\n"
        "输出必须是严格 JSON 数组，每项是一个短句字符串。\n"
        "适合写入记忆的例子：\n"
        "- 我是某公司 M365 管理员，主要负责 Exchange/Entra\n"
        "- 我偏好回答风格：简洁、带操作步骤\n"
        "- 我常用环境：21V（世纪互联）\n"
        "不适合：一次性临时问题、没有复用价值的细节。\n"
        "现在输出 JSON 数组："
    )
    out = ark.chat_generate(
        [
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": prompt + "\n\n输入：\n" + user_message},
        ]
    )
    arr = _parse_json_array(out)
    items: list[str] = []
    for x in arr:
        if isinstance(x, str):
            s = x.strip()
            if s and s not in items:
                items.append(s)
    return items


def background_refresh_persona(user_id: str) -> None:
    """
    用最近对话 + 用户 profile，刷新 persona_summary（可反复覆盖）。
    """
    if not memory_enabled():
        return

    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            return
        profile = (user.profile or "").strip()
        if not profile:
            return

        # Only sample a limited window for cost control
        rows = (
            db.query(UserMemory)
            .filter(UserMemory.user_id == user_id)
            .order_by(UserMemory.created_at.desc())
            .limit(30)
            .all()
        )
        mem_text = "\n".join([f"- {r.content}" for r in rows if (r.content or "").strip()])[:6000]
        if not mem_text:
            return

        ark = ArkClient()
        try:
            prompt = (
                "你在维护一个用户画像 summary，用于后续对话更贴合该用户。\n"
                "只基于输入信息，不要编造。\n"
                "输出必须是纯中文，200~500 字，包含：角色/偏好/常用场景/表达习惯。\n"
            )
            out = ark.chat_generate(
                [
                    {"role": "system", "content": "You write concise user persona summaries."},
                    {"role": "user", "content": prompt + "\n\n用户自述(profile)：\n" + profile + "\n\n记忆点：\n" + mem_text},
                ]
            ).strip()
        finally:
            ark.close()

        row = db.get(UserPersona, user_id)
        if not row:
            row = UserPersona(user_id=user_id, summary="")
        row.summary = out
        db.add(row)
        db.commit()
    except Exception:
        return
    finally:
        db.close()


def background_refresh_decision_profile(user_id: str) -> None:
    """
    用用户自述(profile) + persona + 记忆点，刷新“决策画像”。
    该画像更偏向：风险偏置、证据阈值、排查/决策风格、默认假设与沟通偏好。
    """
    if not decision_profile_enabled():
        return

    db = SessionLocal()
    ark: ArkClient | None = None
    try:
        user = db.get(User, user_id)
        if not user:
            return

        row = db.get(UserDecisionProfile, user_id)
        if row:
            try:
                if (datetime.now(timezone.utc) - row.updated_at) < timedelta(hours=int(decision_profile_refresh_hours())):
                    return
            except Exception:
                pass

        profile = (user.profile or "").strip()
        persona_row = db.get(UserPersona, user_id)
        persona = (persona_row.summary if persona_row else "").strip()

        mem_rows = (
            db.query(UserMemory)
            .filter(UserMemory.user_id == user_id)
            .order_by(UserMemory.created_at.desc())
            .limit(40)
            .all()
        )
        mem_text = "\n".join([f"- {str(r.content or '').strip()}" for r in mem_rows if str(r.content or "").strip()])[:7000]
        if not (profile or persona or mem_text):
            return

        prompt = (
            "你在维护一个用户的“决策画像”(Decision Profile)，用于让后续回答更符合该用户的判断方式。\n"
            "只基于输入信息，不要编造。\n"
            "输出必须是纯中文，150~400 字，尽量具体、可操作。\n"
            "要覆盖但不要生硬列名：\n"
            "- 风险偏置：更保守还是更激进\n"
            "- 证据阈值：更偏好截图/日志/可验证步骤，还是可以接受合理假设\n"
            "- 排查/决策风格：清单式/假设驱动/探索式/先结论后证据等\n"
            "- 默认假设与禁忌：常见前提、不能接受的做法\n"
            "- 沟通偏好：简洁/详细、先问再答等\n"
            "如果信息不足，输出尽量保守，避免强行定性。\n"
        )

        ark = ArkClient()
        out = ark.chat_generate(
            [
                {"role": "system", "content": "You write concise decision profile summaries."},
                {
                    "role": "user",
                    "content": prompt
                    + ("\n\n用户自述(profile)：\n" + profile if profile else "")
                    + ("\n\n用户画像(persona)：\n" + persona if persona else "")
                    + ("\n\n记忆点：\n" + mem_text if mem_text else ""),
                },
            ]
        ).strip()
        out = out[:6000].strip()
        if len(out) < 20:
            return

        if not row:
            row = UserDecisionProfile(user_id=user_id, summary="")
        row.summary = out
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
