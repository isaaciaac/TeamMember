from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .db import init_db
from .maintenance import start_maintenance_thread
from .vectorstore import VectorStore


def _create_app() -> FastAPI:
    app = FastAPI(title="TeamMember API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        init_db()
        try:
            VectorStore().ensure_collections()
        except Exception:
            # Qdrant may not be ready yet; endpoints will lazy-init later.
            pass
        try:
            start_maintenance_thread()
        except Exception:
            # Maintenance should never block app start.
            pass

    app.include_router(router, prefix="/api")
    return app


app = _create_app()
