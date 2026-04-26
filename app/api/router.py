"""Aggregate API router."""

from fastapi import APIRouter

from app.api import council, env, health, meta, ohlcv, state, training

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(meta.router)
api_router.include_router(env.router)
api_router.include_router(state.router)
api_router.include_router(ohlcv.router)
api_router.include_router(council.router)
api_router.include_router(training.router)
