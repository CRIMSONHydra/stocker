"""Task definitions for Stocker.

Tasks are loaded from data/prices.parquet (built by scripts/build_dataset.py).
Each task is one ticker over one episode window. The 3 task IDs are kept
stable so the OpenEnv contract doesn't change.
"""
from __future__ import annotations

import logging

from app.data import loader

logger = logging.getLogger(__name__)


TASK_META = {
    "task_easy": {
        "ticker": "AAPL",
        "description": "AAPL Aug-Sep 2023 — uptrend leading into iPhone 15 launch.",
        "starting_cash": 10000.0,
        "fundamentals": {
            "sector": "tech", "company": "Apple Inc.",
            "pe_ratio_at_episode_start": 32.1, "market_cap_usd": 2.85e12,
        },
    },
    "task_medium": {
        "ticker": "INTC",
        "description": "INTC Jan-Feb 2024 — choppy/sideways post-earnings.",
        "starting_cash": 10000.0,
        "fundamentals": {
            "sector": "semiconductors", "company": "Intel Corp.",
            "pe_ratio_at_episode_start": 105.4, "market_cap_usd": 200e9,
        },
    },
    "task_hard": {
        "ticker": "META",
        "description": "META Sep-Oct 2022 — drawdown then snap-back after Q3 earnings.",
        "starting_cash": 10000.0,
        "fundamentals": {
            "sector": "internet", "company": "Meta Platforms",
            "pe_ratio_at_episode_start": 11.8, "market_cap_usd": 380e9,
        },
    },
}


def list_task_ids() -> list[str]:
    return list(TASK_META.keys())


def get_task_definition(task_id: str) -> dict:
    if task_id not in TASK_META:
        raise KeyError(
            f"Unknown task_id: {task_id}. Available: {list(TASK_META.keys())}"
        )
    meta = TASK_META[task_id]
    rows = loader.episode_rows(task_id)
    if rows.empty:
        raise RuntimeError(
            f"No episode data for {task_id}. Run scripts/build_dataset.py."
        )
    return {
        "task_id": task_id,
        "description": meta["description"],
        "ticker": meta["ticker"],
        "starting_cash": meta["starting_cash"],
        "fundamentals": meta["fundamentals"],
        "dates": rows["date"].tolist(),
        "prices": rows["close"].astype(float).tolist(),
    }
