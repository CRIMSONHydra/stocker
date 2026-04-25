"""Pydantic data models for the Stocker OpenEnv environment."""

from typing import Literal

from pydantic import BaseModel, Field


class MarketObservation(BaseModel):
    """What the agent sees: market state plus its current portfolio."""
    ticker: str
    date: str
    price: float
    price_history: list[float]
    fundamentals: dict
    cash: float
    position: int
    portfolio_value: float
    task_id: str
    step_number: int
    total_steps: int


class TradeAction(BaseModel):
    """What the agent does: buy / sell / hold a quantity of shares."""
    side: Literal["buy", "sell", "hold"]
    quantity: int = Field(ge=0, default=0)


class RewardResult(BaseModel):
    """Internal reward computation result."""
    score: float
    breakdown: dict[str, float]
    feedback: str


class EnvironmentState(BaseModel):
    """Snapshot of environment state."""
    task_id: str
    current_step: int
    total_steps: int
    done: bool
    cash: float
    position: int
    portfolio_value: float
    action_history: list[dict]
    reward_history: list[float]


class StepResult(BaseModel):
    observation: MarketObservation
    reward: float
    done: bool
    info: dict


class ResetResult(BaseModel):
    observation: MarketObservation
    info: dict
