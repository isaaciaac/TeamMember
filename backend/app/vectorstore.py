from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from .settings import settings


def _distance() -> Distance:
    d = settings.qdrant_distance.lower()
    if d == "cosine":
        return Distance.COSINE
    if d == "dot":
        return Distance.DOT
    if d == "euclid":
        return Distance.EUCLID
    raise ValueError(f"Unsupported QDRANT_DISTANCE: {settings.qdrant_distance}")


@dataclass
class VectorHit:
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self) -> None:
        self._client = QdrantClient(url=settings.qdrant_url)

    def ensure_collections(self) -> None:
        vec_size = int(settings.qdrant_vector_size)
        for name in [settings.qdrant_knowledge_collection, settings.qdrant_memory_collection]:
            if self._client.collection_exists(name):
                info = self._client.get_collection(name)
                existing = int(info.config.params.vectors.size)  # type: ignore[union-attr]
                if existing != vec_size:
                    raise RuntimeError(
                        f"Qdrant collection vector size mismatch for '{name}': existing={existing} want={vec_size}. "
                        f"Fix by using the same embedding model or delete the collection."
                    )
                continue
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vec_size, distance=_distance()),
            )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    def upsert(self, collection: str, points: list[PointStruct]) -> None:
        self._client.upsert(collection_name=collection, points=points)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    def search_knowledge(self, vector: list[float], limit: int) -> list[VectorHit]:
        res = self._client.search(
            collection_name=settings.qdrant_knowledge_collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        return [VectorHit(score=float(r.score), payload=dict(r.payload or {})) for r in res]

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    def search_memory(self, vector: list[float], limit: int, user_id: str) -> list[VectorHit]:
        flt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        res = self._client.search(
            collection_name=settings.qdrant_memory_collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
            query_filter=flt,
        )
        return [VectorHit(score=float(r.score), payload=dict(r.payload or {})) for r in res]

    def count(self, collection: str) -> int:
        res = self._client.count(collection_name=collection, exact=True)
        return int(res.count or 0)

