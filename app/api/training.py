"""Training metrics endpoint: surfaces what is actually on disk in
``training/runs/<run>/`` — currently a per-task ``summary.csv`` with
total_reward / final_portfolio / buy_and_hold / alpha_pct columns plus PNG
curve plots saved by ``training/eval_rollout.py``.
"""

import csv
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(tags=["training"])

RUNS_DIR = Path(__file__).resolve().parents[2] / "training" / "runs"


def _latest_run() -> Path | None:
    if not RUNS_DIR.is_dir():
        return None
    candidates = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@router.get("/training/metrics")
async def get_metrics() -> dict:
    run = _latest_run()
    if run is None:
        return {"status": "no_runs", "summary": [], "mean_alpha_pct": 0.0}

    summary_path = run / "summary.csv"
    summary: list[dict] = []
    if summary_path.is_file():
        with summary_path.open() as f:
            for row in csv.DictReader(f):
                summary.append({
                    "task_id": row["task_id"],
                    "total_reward": float(row["total_reward"]),
                    "final_portfolio": float(row["final_portfolio"]),
                    "buy_and_hold": float(row["buy_and_hold"]),
                    "alpha_pct": float(row["alpha_pct"]),
                })

    mean_alpha = (
        sum(r["alpha_pct"] for r in summary) / len(summary) if summary else 0.0
    )

    def _png(name: str) -> str | None:
        p = run / name
        return f"/training/runs/{run.name}/{name}" if p.is_file() else None

    return {
        "status": "completed" if summary else "no_runs",
        "run_name": run.name,
        "summary": summary,
        "mean_alpha_pct": round(mean_alpha, 2),
        "reward_curve_png": _png("reward_curve.png"),
        "portfolio_curve_png": _png("portfolio_curve.png"),
    }
