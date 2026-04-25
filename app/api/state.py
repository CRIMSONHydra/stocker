"""State export and restore endpoints."""

from fastapi import APIRouter, HTTPException

from app.models import EnvironmentState

router = APIRouter(tags=["state"])


@router.get("/state", response_model=EnvironmentState)
async def get_state() -> EnvironmentState:
    import app.api.env as env_module
    try:
        return env_module.current_env.state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/state")
async def restore_state(snapshot: EnvironmentState) -> dict:
    import app.api.env as env_module
    from app.core.environment import StockerEnv

    try:
        new_env = StockerEnv(task_id=snapshot.task_id)
        new_env.load_snapshot(snapshot)
        env_module.current_env = new_env
        return {"status": "restored", "task_id": snapshot.task_id}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
