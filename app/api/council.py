"""Council preview endpoint: runs the 7 specialists + moderator on the
current env state and returns the resulting CouncilDecision.

Picks a real OpenAI-compatible client (HF Inference Endpoint, vLLM, etc.)
when ``API_BASE_URL`` is set, otherwise falls back to ``MockLLMClient``
so tests and CPU-only laptops still work without API keys / GPU.
"""

import logging
import os

from fastapi import APIRouter, HTTPException

from app.council.llm import MockLLMClient, build_openai_client_from_env
from app.council.runner import Council
from app.models import CouncilDecision

logger = logging.getLogger(__name__)
router = APIRouter(tags=["council"])


def _make_client():
    if os.getenv("API_BASE_URL"):
        client = build_openai_client_from_env()
        logger.info("Council using real client: base_url=%s model=%s",
                    client.base_url, client.model)
        return client
    logger.info("Council using MockLLMClient (no API_BASE_URL set)")
    return MockLLMClient()


_council = Council(client=_make_client(), use_cache=True)


@router.get("/council", response_model=CouncilDecision)
async def get_council() -> CouncilDecision:
    import app.api.env as env_module

    env = env_module.current_env
    if not env.is_ready():
        raise HTTPException(
            status_code=409,
            detail="env not initialized or episode complete — call /reset first",
        )
    return await _council.run_async(env.current_observation())
