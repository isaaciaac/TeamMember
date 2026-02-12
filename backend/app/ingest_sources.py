from __future__ import annotations

import json
from typing import Any, Iterator

import httpx
from sqlalchemy import create_engine, text

from .ark import ArkClient
from .knowledge_chunking import semantic_knowledge_points
from .knowledge_store import upsert_knowledge_points


def ingest_from_sql(*, database_url: str, query: str) -> Iterator[tuple[str, str]]:
    """
    Yield (item_id, item_text).
    要求 query 返回至少三列：id, title, content（title/content 可为空）。
    """
    eng = create_engine(database_url, pool_pre_ping=True)
    with eng.connect() as conn:
        res = conn.execute(text(query))
        for row in res.mappings():
            item_id = str(row.get("id") or "").strip()
            if not item_id:
                continue
            title = str(row.get("title") or "").strip()
            content = str(row.get("content") or "").strip()
            item_text = (title + "\n\n" + content).strip()
            if len(item_text) < 20:
                continue
            yield item_id, item_text


def ingest_from_odata(*, url: str, headers: dict[str, str] | None = None, max_pages: int = 200) -> Iterator[tuple[str, str]]:
    """
    Yield (item_id, item_text). item_id 优先取 item 的 id/guid 字段，否则用 index 生成。
    """
    headers = headers or {}
    client = httpx.Client(timeout=httpx.Timeout(60.0))
    try:
        next_url: str | None = url
        pages = 0
        idx = 0
        while next_url and pages < max_pages:
            pages += 1
            resp = client.get(next_url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("value") if isinstance(payload, dict) else None
            if not isinstance(items, list):
                break
            for it in items:
                idx += 1
                if not isinstance(it, dict):
                    continue
                item_id = str(it.get("id") or it.get("Id") or it.get("guid") or it.get("Guid") or "").strip()
                if not item_id:
                    item_id = f"item_{idx}"
                item_text = json.dumps(it, ensure_ascii=False, indent=2)[:12000]
                if len(item_text) < 20:
                    continue
                yield item_id, item_text
            next_url = str(payload.get("@odata.nextLink") or "").strip() or None
    finally:
        client.close()


def ingest_item_to_knowledge(
    *,
    ark: ArkClient,
    item_id: str,
    item_text: str,
    source_kind: str,
    source_name: str,
) -> dict[str, Any]:
    points = semantic_knowledge_points(ark, item_text)
    upserted = upsert_knowledge_points(points, source_kind=source_kind, source_name=source_name, item_id=item_id)
    return {"item_id": item_id, "points": len(points), "upserted": upserted}

