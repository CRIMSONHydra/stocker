"""Pydantic data models for the Stocker OpenEnv environment."""

from typing import Literal

from pydantic import BaseModel, Field


class MarketObservation(BaseModel):
    """What the agent (council) sees on each step."""
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

    # Council inputs (added in step 2 of the multi-agent rebuild)
    chart_path: str = ""
    headlines: list[dict] = Field(default_factory=list)
    forum_excerpts: list[dict] = Field(default_factory=list)
    indicators: dict = Field(default_factory=dict)
    peers: dict = Field(default_factory=dict)
    macro: list[dict] = Field(default_factory=list)


class TradeAction(BaseModel):
    """What the council emits each step."""
    side: Literal["buy", "sell", "hold"]
    quantity: int = Field(ge=0, default=0)


class SpecialistVote(BaseModel):
    """One specialist's read of the situation."""
    name: str
    signal: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class CouncilDecision(BaseModel):
    """The moderator's merged output for one step."""
    votes: list[SpecialistVote]
    action: TradeAction
    rationale: str


class RewardResult(BaseModel):
    score: float
    breakdown: dict[str, float]
    feedback: str


class EnvironmentState(BaseModel):
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
