"""FastAPI application factory and setup."""

from __future__ import annotations

import logging
import sys
import time
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.api.frontend import router as frontend_router
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:\t%(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stocker - OpenEnv",
        version="0.1.0",
        description="RL environment where an AI agent makes stock-trading decisions.",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception: %s", exc)
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        logger.info(
            "%s %s - %s - %.3fs",
            request.method, request.url.path, response.status_code, time.time() - start,
        )
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.include_router(api_router)
    app.include_router(frontend_router)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
