#!/usr/bin/env python3
"""Offline evaluation rollout for the Stocker council.

Runs the council on every task in `app.core.tasks.list_task_ids` and dumps:
  - <out>/results.json    full per-step trace
  - <out>/reward_curve.png cumulative reward per step, one line per task
  - <out>/portfolio_curve.png portfolio value per step, one line per task
  - <out>/summary.csv     per-task aggregates

Use --mock for a deterministic, no-network sanity run (CI-safe).
Use --moderator-lora <name> to evaluate a trained LoRA adapter from
training/runs/<run_id>/moderator-lora/.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.council.llm import MockLLMClient, build_openai_client_from_env
from app.council.runner import Council
from app.core.environment import StockerEnv
from app.core.tasks import list_task_ids

logger = logging.getLogger(__name__)


def run_one_task(council: Council, task_id: str) -> dict:
    env = StockerEnv(task_id=task_id)
    obs = env.reset().observation
    starting_cash = env._task["starting_cash"]

    rewards: list[float] = []
    cum_rewards: list[float] = []
    portfolio_curve: list[float] = [starting_cash]
    actions: list[dict] = []
    cum = 0.0

    while True:
        decision = council.run(obs)
        result = env.step(decision.action)
        rewards.append(result.reward)
        cum += result.reward
        cum_rewards.append(cum)
        portfolio_curve.append(result.info.get("portfolio_value", portfolio_curve[-1]))
        actions.append({
            "side": decision.action.side,
            "quantity": decision.action.quantity,
            "rationale": decision.rationale[:200],
        })
        if result.done:
            break
        obs = result.observation

    final_pv = portfolio_curve[-1]
    bnh_pv = env._buy_and_hold_value()
    return {
        "task_id": task_id,
        "rewards": rewards,
        "cum_rewards": cum_rewards,
        "portfolio_curve": portfolio_curve,
        "actions": actions,
        "final_portfolio": final_pv,
        "buy_and_hold_portfolio": bnh_pv,
        "alpha_pct": (final_pv / max(bnh_pv, 1e-9) - 1.0) * 100.0,
        "starting_cash": starting_cash,
    }


def plot_curves(results: list[dict], out_dir: Path) -> None:
    # Reward curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.plot(range(1, len(r["cum_rewards"]) + 1), r["cum_rewards"], label=r["task_id"])
    ax.set_title("Cumulative reward per step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png", dpi=120)
    plt.close(fig)

    # Portfolio curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.plot(range(len(r["portfolio_curve"])), r["portfolio_curve"], label=r["task_id"])
    ax.set_title("Portfolio value per step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio value (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "portfolio_curve.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use MockLLMClient")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--moderator-lora", default=None)
    parser.add_argument(
        "--out",
        default=None,
        help="Output dir (default: training/runs/eval_<timestamp>)",
    )
    parser.add_argument("--tasks", default="all")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path("training/runs") / (
        f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    client = MockLLMClient() if args.mock else build_openai_client_from_env()
    council = Council(
        client=client, moderator_lora=args.moderator_lora, use_cache=not args.no_cache
    )

    task_ids = list_task_ids() if args.tasks == "all" else args.tasks.split(",")
    results = [run_one_task(council, t) for t in task_ids]

    # Save artifacts
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    plot_curves(results, out_dir)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "total_reward", "final_portfolio", "buy_and_hold", "alpha_pct"])
        for r in results:
            w.writerow([
                r["task_id"],
                round(sum(r["rewards"]), 4),
                round(r["final_portfolio"], 2),
                round(r["buy_and_hold_portfolio"], 2),
                round(r["alpha_pct"], 2),
            ])

    print(f"Eval rollout done. Artifacts in: {out_dir}")
    for r in results:
        print(
            f"  {r['task_id']:<14} total_reward={sum(r['rewards']):+.4f}  "
            f"final={r['final_portfolio']:.2f}  bnh={r['buy_and_hold_portfolio']:.2f}  "
            f"alpha={r['alpha_pct']:+.2f}%"
        )


if __name__ == "__main__":
    main()
