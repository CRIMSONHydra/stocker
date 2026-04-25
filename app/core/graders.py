"""Reward and shaping functions for the Stocker environment."""

from app.models import RewardResult, TradeAction


def compute_step_reward(
    action: TradeAction,
    prev_portfolio: float,
    new_portfolio: float,
    starting_cash: float,
    invalid: bool,
) -> RewardResult:
    """Reward = pct change in portfolio value, with a small penalty for invalid actions."""
    breakdown: dict[str, float] = {}

    pnl_pct = (new_portfolio - prev_portfolio) / max(prev_portfolio, 1e-9)
    breakdown["pnl_pct"] = round(pnl_pct, 5)

    penalty = 0.0
    if invalid:
        penalty = 0.01
        breakdown["invalid_action_penalty"] = -penalty

    score = pnl_pct - penalty

    side = action.side
    feedback = f"{side}({action.quantity}) -> portfolio {new_portfolio:.2f}"
    if invalid:
        feedback += " [invalid: insufficient cash/position]"

    return RewardResult(score=round(score, 5), breakdown=breakdown, feedback=feedback)


def compute_trajectory_bonus(
    final_portfolio: float, starting_cash: float, threshold: float = 1.05
) -> float:
    """Bonus if the agent finishes with >threshold of starting capital."""
    ratio = final_portfolio / max(starting_cash, 1e-9)
    if ratio >= threshold:
        return 0.05
    return 0.0
