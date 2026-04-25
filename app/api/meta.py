"""Environment metadata endpoint."""

from fastapi import APIRouter

from app.core.tasks import list_task_ids

router = APIRouter(tags=["meta"])


@router.get("/meta")
async def meta() -> dict:
    return {
        "name": "stocker",
        "version": "0.1.0",
        "description": "RL environment for stock trading decisions.",
        "tasks": list_task_ids(),
        "action_space": {
            "side": "literal[buy, sell, hold]",
            "quantity": "int (>=0)",
        },
        "observation_space": {
            "ticker": "string",
            "date": "string (ISO date)",
            "price": "float",
            "price_history": "list[float]",
            "fundamentals": "dict",
            "cash": "float",
            "position": "int",
            "portfolio_value": "float",
        },
    }
