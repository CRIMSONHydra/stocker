#!/usr/bin/env python3
"""Validate that the bundled dataset is intact and the env can step every task."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.environment import StockerEnv  # noqa: E402
from app.core.tasks import TASK_META, list_task_ids  # noqa: E402
from app.data import loader  # noqa: E402


def validate() -> int:
    errors: list[str] = []

    # Parquet sanity
    for name in ("prices", "indicators", "news", "forums", "peers", "macro"):
        try:
            df = getattr(loader, name)()
        except Exception as e:
            errors.append(f"{name}.parquet not loadable: {e}")
            continue
        if df.empty:
            errors.append(f"{name}.parquet is empty")

    for task_id in list_task_ids():
        meta = TASK_META[task_id]
        rows = loader.episode_rows(task_id)
        if rows.empty:
            errors.append(f"{task_id}: no in-episode rows")
            continue
        if len(rows) < 10:
            errors.append(f"{task_id}: only {len(rows)} steps (< 10)")
        env = StockerEnv(task_id=task_id)
        env.reset()
        result = env.step({"side": "hold", "quantity": 0})
        if not (-1.0 <= result.reward <= 1.0):
            errors.append(f"{task_id}: reward {result.reward} out of [-1,1]")
        chart = loader.chart_path(meta["ticker"], rows.iloc[0]["date"])
        if not chart:
            errors.append(f"{task_id}: missing chart for first step")
        print(
            f"{task_id:<14} ticker={meta['ticker']:<6} steps={len(rows):>3} "
            f"chart_ok={'yes' if chart else 'NO '}"
        )

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(" -", e)
        return 1
    print(f"\nAll {len(list_task_ids())} tasks OK.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
