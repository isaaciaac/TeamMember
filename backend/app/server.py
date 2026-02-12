from __future__ import annotations

import uvicorn

from .main import app
from .settings import settings


def main() -> None:
    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port, log_level="info")


if __name__ == "__main__":
    main()

