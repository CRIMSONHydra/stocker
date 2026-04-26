"""Aggregate API router."""

from fastapi import APIRouter

from app.api import corpus, env, health, meta, state

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(meta.router)
api_router.include_router(env.router)
api_router.include_router(state.router)
api_router.include_router(corpus.router)
