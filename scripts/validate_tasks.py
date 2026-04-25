#!/usr/bin/env python3
"""Validate every Stocker task definition."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.tasks import TASKS_BY_ID  # noqa: E402


def validate() -> int:
    errors: list[str] = []

    for task_id, task in sorted(TASKS_BY_ID.items()):
        for field in ("ticker", "starting_cash", "prices"):
            if field not in task:
                errors.append(f"{task_id}: missing '{field}'")

        prices = task.get("prices", [])
        if len(prices) < 3:
            errors.append(f"{task_id}: needs >=3 prices, got {len(prices)}")
        if any(p <= 0 for p in prices):
            errors.append(f"{task_id}: non-positive price found")

        cash = task.get("starting_cash", 0)
        if cash <= 0:
            errors.append(f"{task_id}: starting_cash must be > 0")

        print(f"{task_id:<18} ticker={task.get('ticker'):<6} "
              f"steps={len(prices)} cash={cash}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(" -", e)
        return 1
    print(f"\nAll {len(TASKS_BY_ID)} tasks valid.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
