"""Reward and shaping for the Stocker backtest.

Per-step reward = pct change in portfolio value (with a small invalid-action
penalty). End-of-episode reward adds:
  * alpha bonus:  +0.10 if final portfolio beats buy-and-hold by >= 1%
  * drawdown penalty: linear in the worst drawdown observed during the episode

All rewards are clipped to [-1.0, 1.0] in the caller (StockerEnv.step).
"""
from __future__ import annotations

from app.models import RewardResult, TradeAction


def compute_step_reward(
    action: TradeAction,
    prev_portfolio: float,
    new_portfolio: float,
    starting_cash: float,
    invalid: bool,
) -> RewardResult:
    breakdown: dict[str, float] = {}

    pnl_pct = (new_portfolio - prev_portfolio) / max(prev_portfolio, 1e-9)
    breakdown["pnl_pct"] = round(pnl_pct, 5)

    penalty = 0.0
    if invalid:
        penalty = 0.01
        breakdown["invalid_action_penalty"] = -penalty

    score = pnl_pct - penalty
    feedback = f"{action.side}({action.quantity}) -> portfolio {new_portfolio:.2f}"
    if invalid:
        feedback += " [invalid: insufficient cash/position]"
    return RewardResult(score=round(score, 5), breakdown=breakdown, feedback=feedback)


def compute_trajectory_bonus(
    final_portfolio: float,
    buy_and_hold_value: float,
    starting_cash: float,
    portfolio_curve: list[float] | None = None,
) -> float:
    """End-of-episode shaping: alpha bonus minus drawdown penalty."""
    bonus = 0.0

    # Alpha vs. buy-and-hold (capped at +0.10)
    if buy_and_hold_value > 0:
        alpha = (final_portfolio - buy_and_hold_value) / buy_and_hold_value
        if alpha >= 0.01:
            bonus += min(0.10, alpha)

    # Drawdown penalty (only if the curve is provided)
    if portfolio_curve:
        peak = portfolio_curve[0]
        max_dd = 0.0
        for v in portfolio_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / max(peak, 1e-9)
            if dd > max_dd:
                max_dd = dd
        if max_dd > 0.05:
            bonus -= min(0.10, (max_dd - 0.05))

    return bonus


def compute_max_drawdown(portfolio_curve: list[float]) -> float:
    if not portfolio_curve:
        return 0.0
    peak = portfolio_curve[0]
    max_dd = 0.0
    for v in portfolio_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / max(peak, 1e-9)
        if dd > max_dd:
            max_dd = dd
    return max_dd
