"""Unit tests for the grader (compute_step_reward).

Covers the asymmetric performance shape, the inflation drag component, and
the env-level transaction cost deduction.
"""
from __future__ import annotations

from app.config import Settings
from app.core.environment import StockerEnv
from app.core.graders import compute_step_reward
from app.models import TradeAction


def _settings(**overrides) -> Settings:
    base = dict(
        transaction_cost_rate=0.001,
        annual_inflation_rate=0.05,
        reward_weight_performance=0.7,
        reward_weight_inflation=0.3,
    )
    base.update(overrides)
    return Settings(**base)


def _call(
    *,
    actual_pnl_pct: float,
    ideal_at_step: float,
    ideal_total: float,
    step_index: int = 5,
    total_steps: int = 50,
    starting_cash: float = 10000.0,
    invalid: bool = False,
    settings: Settings | None = None,
):
    settings = settings or _settings()
    series = [0.0] * total_steps
    series[step_index] = ideal_at_step
    new_portfolio = starting_cash * (1.0 + actual_pnl_pct)
    return compute_step_reward(
        action=TradeAction(side="hold", quantity=0),
        prev_portfolio=starting_cash,
        new_portfolio=new_portfolio,
        starting_cash=starting_cash,
        invalid=invalid,
        step_index=step_index,
        total_steps=total_steps,
        ideal_pnl_pct_series=series,
        ideal_pnl_pct_total=ideal_total,
        settings=settings,
    )


def test_at_ideal_step_yields_high_performance():
    # actual real ~= ideal at this step -> gap ~= 0 -> perf ~= 1
    result = _call(actual_pnl_pct=0.05, ideal_at_step=0.05, ideal_total=0.20)
    assert result.breakdown["performance_factor"] > 0.95
    assert result.score > 0.5


def test_far_behind_ideal_yields_punishment():
    # actual < ideal by much more than scale -> perf = -1
    result = _call(actual_pnl_pct=-0.05, ideal_at_step=0.30, ideal_total=0.30)
    assert result.breakdown["performance_factor"] == -1.0
    assert result.breakdown["weighted_performance"] < 0


def test_outperforming_ideal_yields_bonus():
    # gap < 0 -> perf > 1.0 (env clip will cap, but raw breakdown shows the bonus)
    result = _call(actual_pnl_pct=0.10, ideal_at_step=0.05, ideal_total=0.10)
    assert result.breakdown["performance_factor"] > 1.0
    assert result.breakdown["gap"] < 0


def test_inflation_factor_is_negative_with_positive_pnl():
    # Real < nominal under positive inflation, so inflation_factor < 0
    result = _call(
        actual_pnl_pct=0.05,
        ideal_at_step=0.05,
        ideal_total=0.20,
        step_index=252,  # one year in -> noticeable inflation drag
        total_steps=300,
    )
    assert result.breakdown["inflation_factor"] < 0
    assert result.breakdown["real_pnl_pct"] < result.breakdown["nominal_pnl_pct"]


def test_invalid_action_subtracts_penalty():
    fine = _call(actual_pnl_pct=0.05, ideal_at_step=0.05, ideal_total=0.20)
    bad = _call(actual_pnl_pct=0.05, ideal_at_step=0.05, ideal_total=0.20, invalid=True)
    assert bad.score < fine.score
    assert "invalid_action_penalty" in bad.breakdown


def test_buy_deducts_transaction_cost_from_cash():
    """End-to-end env check: a buy at 0.1% transaction cost shaves cash."""
    env = StockerEnv(task_id="task_easy")
    env.reset()
    cash_before = env.state().cash
    price = env._prices[0]
    qty = 5
    env.step({"side": "buy", "quantity": qty})
    cash_after = env.state().cash

    notional = qty * price
    expected_cost = notional * 1.001  # 0.1% txn cost
    assert abs((cash_before - cash_after) - expected_cost) < 1e-4


def test_sell_deducts_transaction_cost_from_proceeds():
    env = StockerEnv(task_id="task_easy")
    env.reset()
    qty = 5
    env.step({"side": "buy", "quantity": qty})
    cash_after_buy = env.state().cash
    sell_price = env._prices[1]  # next-step price (env advanced after the buy)
    env.step({"side": "sell", "quantity": qty})
    cash_after_sell = env.state().cash

    notional = qty * sell_price
    expected_proceeds = notional * 0.999
    assert abs((cash_after_sell - cash_after_buy) - expected_proceeds) < 1e-4
