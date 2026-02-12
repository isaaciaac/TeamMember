from __future__ import annotations

from typing import Any

from .ark import ArkClient
from .utils import parse_json_array


def semantic_knowledge_points(ark: ArkClient, text: str) -> list[dict[str, Any]]:
    """
    把任意“原始内容”拆成可检索的知识点 seeds。
    输出是 list[{"title": str, "content": str, "tags": [str]}]。
    """
    src = (text or "").strip()
    if not src:
        return []

    prompt = (
        "你是一个企业知识库的“知识点抽取与分块”助手。\n"
        "目标：把输入内容按语义拆成若干条可复用的知识点（seed），每条尽量自洽、可单独检索。\n"
        "硬规则：\n"
        "1) 只基于输入，不允许编造。\n"
        "2) 输出必须是严格 JSON 数组，不要输出任何额外文字。\n"
        "3) 每个元素 schema：{\"title\":\"...\",\"content\":\"...\",\"tags\":[\"...\"]}\n"
        "4) title 要短（<= 40 字），content 要完整（建议 80~400 字）。\n"
        "5) tags 可选，最多 6 个。\n"
    )

    out = ark.chat_generate(
        [
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": prompt + "\n\n输入：\n" + src},
        ]
    )
    arr = parse_json_array(out)
    points: list[dict[str, Any]] = []
    for x in arr:
        if not isinstance(x, dict):
            continue
        title = str(x.get("title") or "").strip()
        content = str(x.get("content") or "").strip()
        tags_raw = x.get("tags")
        tags: list[str] = []
        if isinstance(tags_raw, list):
            for t in tags_raw:
                s = str(t or "").strip()
                if s and s not in tags:
                    tags.append(s)
        if len(title) < 2 or len(content) < 20:
            continue
        points.append({"title": title[:120], "content": content[:4000], "tags": tags[:6]})
    return points

