"""Reward and shaping for the Stocker backtest.

Per-step reward is a weighted sum of two components:

  1. Performance factor (W_PERF) — asymmetric piecewise-linear function of the
     gap between the precomputed ideal PnL trajectory and the model's actual
     real (inflation-adjusted) PnL at this step. Small underperformance is
     rewarded; large underperformance is punished; outperformance gets a
     bonus.
  2. Inflation factor (W_INFLATION) — penalty for the share of nominal gain
     eaten by inflation drag at this point in the episode.

A small invalid-action penalty is subtracted on top. Transaction cost is not
a separate term — it is deducted from cash at trade time inside the env, so
nominal_pnl_pct already reflects friction.

End-of-episode shaping (alpha vs. buy-and-hold + drawdown penalty) lives in
compute_trajectory_bonus and is unchanged.

All rewards are clipped to [-1.0, 1.0] in the caller (StockerEnv.step).
"""
from __future__ import annotations

from app.config import Settings
from app.models import RewardResult, TradeAction

TRADING_DAYS_PER_YEAR = 252
INVALID_ACTION_PENALTY = 0.01


def compute_step_reward(
    *,
    action: TradeAction,
    new_portfolio: float,
    starting_cash: float,
    invalid: bool,
    step_index: int,
    total_steps: int,
    ideal_pnl_pct_series: list[float],
    ideal_pnl_pct_total: float,
    settings: Settings,
) -> RewardResult:
    breakdown: dict[str, float] = {}

    # Cumulative nominal PnL fraction (relative to starting cash)
    nominal_pnl_pct = (new_portfolio - starting_cash) / max(starting_cash, 1e-9)

    # Inflation-adjusted real PnL, scaled by elapsed years in the episode
    years_elapsed = max(step_index, 0) / TRADING_DAYS_PER_YEAR
    inflation_growth = (1.0 + settings.annual_inflation_rate) ** years_elapsed - 1.0
    real_pnl_pct = (1.0 + nominal_pnl_pct) / (1.0 + inflation_growth) - 1.0

    breakdown["nominal_pnl_pct"] = round(nominal_pnl_pct, 6)
    breakdown["real_pnl_pct"] = round(real_pnl_pct, 6)
    breakdown["inflation_drag"] = round(inflation_growth, 6)

    # Inflation factor: how much of the nominal gain was eaten by inflation.
    # Always <= 0 (real <= nominal under positive inflation), clipped at -1.
    inflation_factor = max(-1.0, real_pnl_pct - nominal_pnl_pct)
    breakdown["inflation_factor"] = round(inflation_factor, 6)

    # Performance factor: asymmetric piecewise linear in the gap to ideal.
    ideal_at_step = (
        ideal_pnl_pct_series[step_index]
        if 0 <= step_index < len(ideal_pnl_pct_series)
        else ideal_pnl_pct_total
    )
    gap = ideal_at_step - real_pnl_pct
    scale = max(0.05, 0.5 * abs(ideal_pnl_pct_total))

    if gap < 0.0:
        # Model exceeded ideal — bonus up to +2.0 (env clip caps at +1.0).
        performance_factor = 1.0 + min(1.0, abs(gap) / scale)
    elif gap <= scale:
        # Close to ideal — high reward decaying linearly toward 0.
        performance_factor = 1.0 - gap / scale
    else:
        # Far behind ideal — punishment down to -1.0.
        performance_factor = -min(1.0, (gap - scale) / scale)

    breakdown["ideal_pnl_pct_at_step"] = round(ideal_at_step, 6)
    breakdown["gap"] = round(gap, 6)
    breakdown["performance_factor"] = round(performance_factor, 6)

    # Weighted combination
    w_perf = settings.reward_weight_performance
    w_inf = settings.reward_weight_inflation
    weighted_perf = w_perf * performance_factor
    weighted_inf = w_inf * inflation_factor
    breakdown["weighted_performance"] = round(weighted_perf, 6)
    breakdown["weighted_inflation"] = round(weighted_inf, 6)

    score = weighted_perf + weighted_inf

    if invalid:
        score -= INVALID_ACTION_PENALTY
        breakdown["invalid_action_penalty"] = -INVALID_ACTION_PENALTY

    feedback = (
        f"{action.side}({action.quantity}) -> portfolio {new_portfolio:.2f} "
        f"(real {real_pnl_pct:+.3%}, ideal {ideal_at_step:+.3%}, gap {gap:+.3%})"
    )
    if invalid:
        feedback += " [invalid: insufficient cash/position]"

    return RewardResult(score=round(score, 6), breakdown=breakdown, feedback=feedback)


def compute_trajectory_bonus(
    final_portfolio: float,
    buy_and_hold_value: float,
    starting_cash: float,
    portfolio_curve: list[float] | None = None,
) -> float:
    """End-of-episode shaping: alpha bonus minus drawdown penalty."""
    bonus = 0.0

    if buy_and_hold_value > 0:
        alpha = (final_portfolio - buy_and_hold_value) / buy_and_hold_value
        if alpha >= 0.01:
            bonus += min(0.10, alpha)

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
