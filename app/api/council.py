"""Council preview endpoint: runs the 7 specialists + moderator on the
current env state and returns the resulting CouncilDecision.

Defaults to MockLLMClient so the endpoint works on a laptop with no API
keys / GPU. The mock is deterministic per (ticker, date) and the Council
runner caches results to .cache/council, so repeat hits are instant.
"""

from fastapi import APIRouter, HTTPException

from app.council.llm import MockLLMClient
from app.council.runner import Council
from app.models import CouncilDecision

router = APIRouter(tags=["council"])

_council = Council(client=MockLLMClient(), use_cache=True)


@router.get("/council", response_model=CouncilDecision)
async def get_council() -> CouncilDecision:
    import app.api.env as env_module

    env = env_module.current_env
    if env._done or not env._prices:
        raise HTTPException(
            status_code=409,
            detail="env not initialized or episode complete — call /reset first",
        )
    obs = env._build_observation(env._current_index)
    try:
        return await _council.run_async(obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"council failed: {e}")
