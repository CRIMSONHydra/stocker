"""Core RL environment endpoints: reset and step."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.environment import StockerEnv
from app.models import ResetResult, StepResult, TradeAction

router = APIRouter(tags=["environment"])

current_env = StockerEnv(task_id="task_easy")


class ResetRequest(BaseModel):
    task_id: str = "task_easy"


@router.post("/reset", response_model=ResetResult)
async def reset(body: Optional[ResetRequest] = None) -> ResetResult:
    global current_env
    task_id = body.task_id if body else "task_easy"
    try:
        current_env = StockerEnv(task_id=task_id)
        return current_env.reset()
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/step", response_model=StepResult)
async def step(action: TradeAction) -> StepResult:
    try:
        return current_env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
