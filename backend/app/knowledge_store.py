from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from qdrant_client.http.models import PointStruct

from .qwen import DashScopeClient
from .settings import settings
from .vectorstore import VectorStore


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_seed_text(*, title: str, content: str, tags: list[str] | None = None) -> str:
    t = (title or "").strip()
    c = (content or "").strip()
    tags = tags or []
    tags_txt = ", ".join([x for x in (tags or []) if str(x or "").strip()][:6])
    parts = [f"Title: {t}", "", "Content:", c]
    if tags_txt:
        parts.extend(["", f"Tags: {tags_txt}"])
    return "\n".join(parts).strip()


def upsert_knowledge_points(
    points: list[dict[str, Any]],
    *,
    source_kind: str,
    source_name: str,
    item_id: str,
) -> int:
    if not points:
        return 0

    llm = DashScopeClient()
    vs = VectorStore()
    vs.ensure_collections()

    try:
        seed_ids: list[str] = []
        seed_texts: list[str] = []
        payloads: list[dict[str, Any]] = []

        for idx, p in enumerate(points):
            title = str(p.get("title") or "").strip()
            content = str(p.get("content") or "").strip()
            tags_raw = p.get("tags")
            tags: list[str] = []
            if isinstance(tags_raw, list):
                for t in tags_raw:
                    s = str(t or "").strip()
                    if s and s not in tags:
                        tags.append(s)

            if len(title) < 2 or len(content) < 20:
                continue

            seed_id = str(uuid.uuid4())
            seed_text = build_seed_text(title=title, content=content, tags=tags)

            seed_ids.append(seed_id)
            seed_texts.append(seed_text)
            payloads.append(
                {
                    "seed_id": seed_id,
                    "title": title,
                    "content": content,
                    "tags": tags[:6],
                    "source_kind": source_kind,
                    "source_name": source_name,
                    "item_id": item_id,
                    "chunk_index": idx,
                    "created_at": _utcnow_iso(),
                }
            )

        if not seed_texts:
            return 0

        vecs = llm.embed_texts(seed_texts)
        points_upsert: list[PointStruct] = []
        for sid, vec, payload in zip(seed_ids, vecs, payloads):
            points_upsert.append(PointStruct(id=sid, vector=vec, payload=payload))
        vs.upsert(settings.qdrant_knowledge_collection, points_upsert)
        return len(points_upsert)
    finally:
        llm.close()

