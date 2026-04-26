"""FastAPI application factory and setup."""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.api.frontend import router as frontend_router
from app.config import settings

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
TRAINING_RUNS = Path(__file__).resolve().parents[1] / "training" / "runs"
CACHE_DIR    = Path(__file__).resolve().parents[1] / ".cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:\t%(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def _hydrate_cache_from_hub() -> None:
    """If STOCKER_CACHE_REPO is set, download the cache from that HF dataset
    into .cache/ on startup. Idempotent — skipped when .cache/ already has
    council/ entries. Never fails startup; logs and continues on error."""
    repo = os.getenv("STOCKER_CACHE_REPO", "").strip()
    if not repo:
        return
    if (CACHE_DIR / "council").exists() and any((CACHE_DIR / "council").iterdir()):
        logger.info("Cache already populated; skipping hub download.")
        return
    try:
        from huggingface_hub import snapshot_download
        token = os.getenv("HF_TOKEN") or None
        logger.info("Downloading council cache from %s ...", repo)
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(CACHE_DIR),
            token=token,
            allow_patterns=["council/**/*.json"],
        )
        n = sum(1 for _ in CACHE_DIR.rglob("*.json"))
        logger.info("Cache hydrated: %d entries from %s", n, repo)
    except Exception as e:
        logger.warning("Cache hydration failed (%s); will fall back to live calls / mock.", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _hydrate_cache_from_hub()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stocker - OpenEnv",
        version="0.1.0",
        description="RL environment where an AI agent makes stock-trading decisions.",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
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

    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    if TRAINING_RUNS.is_dir():
        app.mount(
            "/training/runs",
            StaticFiles(directory=str(TRAINING_RUNS)),
            name="training-runs",
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
