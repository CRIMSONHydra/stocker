#!/usr/bin/env python3
"""Precompute per-task ideal profit trajectories used by the grader.

For each task in TASK_META we simulate a perfect-foresight policy on the
episode price series: at every step we go max-long if the next bar is up
and flat otherwise. The resulting per-step cumulative PnL fraction is
written to data/ideal_profits/<task_id>.json. The grader (per-step
performance component) reads these sidecars and compares actual PnL to
the ideal trajectory.

Transaction cost is applied to the ideal trajectory at the same rate the
env uses, so the gap the model sees reflects only foresight quality.

Run:
    uv run python scripts/build_ideal_profit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.core.tasks import TASK_META, list_task_ids  # noqa: E402
from app.data import loader  # noqa: E402

OUTPUT_DIR = ROOT / "data" / "ideal_profits"


def simulate_perfect_foresight(
    prices: list[float],
    starting_cash: float,
    txn_cost_rate: float,
) -> tuple[list[float], list[float]]:
    """Multi-transaction perfect foresight on a single ticker.

    Returns (portfolio_curve, pnl_pct_curve) where each list has len(prices)
    entries and represents the value (and pnl fraction) at the end of each
    step after that step's action has been applied.
    """
    cash = float(starting_cash)
    position = 0
    portfolio_curve: list[float] = []
    pnl_curve: list[float] = []

    n = len(prices)
    for i in range(n):
        price = prices[i]
        going_up = (i + 1 < n) and (prices[i + 1] > price)

        if going_up and position == 0:
            # Affordable shares after paying txn cost: solve qty * price * (1 + r) <= cash.
            qty = int(cash // (price * (1.0 + txn_cost_rate)))
            if qty > 0:
                notional = qty * price
                cash -= notional + notional * txn_cost_rate
                position += qty
        elif (not going_up) and position > 0:
            notional = position * price
            cash += notional - notional * txn_cost_rate
            position = 0

        port_value = cash + position * price
        portfolio_curve.append(port_value)
        pnl_curve.append((port_value - starting_cash) / starting_cash)

    return portfolio_curve, pnl_curve


def build_for_task(task_id: str) -> dict:
    meta = TASK_META[task_id]
    rows = loader.episode_rows(task_id)
    if rows.empty:
        raise RuntimeError(
            f"No episode data for {task_id}. Run scripts/build_dataset.py first."
        )
    prices = rows["close"].astype(float).tolist()
    starting_cash = float(meta["starting_cash"])

    portfolio_curve, pnl_curve = simulate_perfect_foresight(
        prices=prices,
        starting_cash=starting_cash,
        txn_cost_rate=settings.transaction_cost_rate,
    )

    return {
        "task_id": task_id,
        "ticker": meta["ticker"],
        "starting_cash": starting_cash,
        "transaction_cost_rate": settings.transaction_cost_rate,
        "ideal_pnl_pct_series": [round(v, 6) for v in pnl_curve],
        "ideal_pnl_pct_total": round(pnl_curve[-1], 6) if pnl_curve else 0.0,
        "ideal_portfolio_curve": [round(v, 4) for v in portfolio_curve],
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for task_id in list_task_ids():
        payload = build_for_task(task_id)
        out = OUTPUT_DIR / f"{task_id}.json"
        out.write_text(json.dumps(payload, indent=2))
        n = len(payload["ideal_pnl_pct_series"])
        print(
            f"{task_id:<14} steps={n:>3} "
            f"ideal_total={payload['ideal_pnl_pct_total']:+.4f} "
            f"-> {out.relative_to(ROOT)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
