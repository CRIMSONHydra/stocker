"""Trading scenario dataset for the Stocker environment.

Each task is a sequence of daily market observations for a single ticker, with
a known ground-truth optimal trajectory used only for reward shaping (not
shown to the agent).

Tasks are loaded from:
1. Inline definitions below
2. JSON files in tasks/ directory
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inline task definitions
# ---------------------------------------------------------------------------

TASK_EASY = {
    "task_id": "task_easy",
    "description": "Steady uptrend: a clearly bullish 10-day sequence.",
    "ticker": "ACME",
    "starting_cash": 10000.0,
    "fundamentals": {"sector": "tech", "pe_ratio": 22.0, "market_cap": 5e9},
    "prices": [100.0, 101.5, 103.0, 104.2, 106.0, 107.5, 109.0, 110.5, 112.0, 114.0],
}

TASK_MEDIUM = {
    "task_id": "task_medium",
    "description": "Volatile sideways market: noisy mean-reverting prices.",
    "ticker": "VOLT",
    "starting_cash": 10000.0,
    "fundamentals": {"sector": "energy", "pe_ratio": 14.5, "market_cap": 1.2e9},
    "prices": [50.0, 52.0, 49.5, 51.0, 48.5, 50.5, 52.5, 49.0, 51.5, 50.0],
}

TASK_HARD = {
    "task_id": "task_hard",
    "description": "Bull-then-bear reversal: agent must time the exit.",
    "ticker": "FLIP",
    "starting_cash": 10000.0,
    "fundamentals": {"sector": "biotech", "pe_ratio": 35.0, "market_cap": 800e6},
    "prices": [40.0, 42.5, 45.0, 47.0, 49.5, 50.0, 47.0, 43.5, 39.0, 35.0],
}

_INLINE_TASKS = {
    "task_easy": TASK_EASY,
    "task_medium": TASK_MEDIUM,
    "task_hard": TASK_HARD,
}

_TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "tasks"


def _load_tasks_from_dir(tasks_dir: Path) -> dict[str, dict]:
    loaded: dict[str, dict] = {}
    if not tasks_dir.is_dir():
        return loaded
    for json_file in sorted(tasks_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                task = json.load(f)
            task_id = task.get("task_id", json_file.stem)
            loaded[task_id] = task
            logger.debug("Loaded task '%s' from %s", task_id, json_file.name)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load task from %s: %s", json_file, e)
    return loaded


TASKS_BY_ID: dict[str, dict] = {}
TASKS_BY_ID.update(_INLINE_TASKS)
TASKS_BY_ID.update(_load_tasks_from_dir(_TASKS_DIR))


def get_task_definition(task_id: str) -> dict:
    if task_id not in TASKS_BY_ID:
        raise KeyError(
            f"Unknown task_id: {task_id}. Available: {list(TASKS_BY_ID.keys())}"
        )
    return TASKS_BY_ID[task_id]


def list_task_ids() -> list[str]:
    return list(TASKS_BY_ID.keys())
