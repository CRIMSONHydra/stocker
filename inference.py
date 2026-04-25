"""
Inference Script — Stocker OpenEnv (multi-agent council)
========================================================
Runs the 7-specialist + moderator council on every step of every task.

MANDATORY environment variables (only required without --mock):
    API_BASE_URL   OpenAI-compatible endpoint    (default: http://localhost:8000/v1)
    MODEL_NAME     Model id served at that endpoint  (default: google/gemma-4-E4B-it)
    HF_TOKEN       API key (any value works for local vLLM)

STDOUT FORMAT
    [START]   task=<name> env=stocker model=<model>
    [STEP]    step=<n> action=<side(qty)> reward=<x.xxxx> done=<true|false> error=<msg|null>
    [COUNCIL] step=<n> votes=<json-array-of-7-votes> rationale=<moderator-rationale>
    [END]     success=<true|false> steps=<n> score=<x.xxxx> rewards=<r1,...,rn>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from app.council.llm import MockLLMClient, build_openai_client_from_env
from app.council.runner import Council
from app.core.environment import StockerEnv
from app.core.tasks import list_task_ids

BENCHMARK = "stocker"


# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_council(step: int, votes: list, rationale: str) -> None:
    payload = {
        "votes": [v.model_dump() for v in votes],
        "rationale": rationale,
    }
    print(f"[COUNCIL] step={step} payload={json.dumps(payload, separators=(',', ':'))}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
def run_episode(council: Council, task_id: str, model_name: str) -> dict:
    env = StockerEnv(task_id=task_id)
    reset = env.reset()
    obs = reset.observation

    rewards: list[float] = []
    step_details: list[dict] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=model_name)
    try:
        for step in range(1, env.state().total_steps + 1):
            error: Optional[str] = None
            try:
                decision = council.run(obs)
                action = decision.action
            except Exception as e:
                error = str(e)
                from app.models import TradeAction
                action = TradeAction(side="hold", quantity=0)
                decision = None

            result = env.step(action)
            reward = result.reward
            rewards.append(reward)
            steps_taken = step

            action_str = f"{action.side}({action.quantity})"
            log_step(step, action_str, reward, result.done, error)
            if decision is not None:
                log_council(step, decision.votes, decision.rationale)

            step_details.append({
                "step": step,
                "side": action.side,
                "quantity": action.quantity,
                "reward": reward,
                "info": result.info,
                "done": result.done,
                "votes": [v.model_dump() for v in decision.votes] if decision else [],
                "moderator_rationale": decision.rationale if decision else "",
            })

            if result.done:
                break
            obs = result.observation

        score = sum(rewards)
        success = score >= 0.0  # any non-negative net P&L counts as success
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": round(score, 4),
        "total_reward": round(sum(rewards), 4),
        "steps": steps_taken,
        "success": success,
        "step_details": step_details,
    }


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the Stocker council inference")
    parser.add_argument("--task", default="all", help="Task name or 'all' (default: all)")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockLLMClient — deterministic, no network. For CI/dev.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable .cache/council on-disk caching",
    )
    parser.add_argument(
        "--moderator-lora", default=None,
        help="vLLM LoRA adapter name to apply to moderator calls "
             "(see scripts/serve_vllm.sh --lora). Ignored in --mock.",
    )
    parser.add_argument("--output", default=None, help="Path to write JSON results")
    args = parser.parse_args()

    if args.mock:
        client = MockLLMClient()
        model_name = "mock"
    else:
        client = build_openai_client_from_env()
        model_name = client.model

    council = Council(
        client=client,
        moderator_lora=args.moderator_lora,
        use_cache=not args.no_cache,
    )

    tasks = list_task_ids() if args.task == "all" else [args.task]
    results = []
    for tid in tasks:
        results.append(run_episode(council, tid, model_name))

    # Summary table to stderr
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"{'Task':<14} {'Score':>10} {'Reward':>10} {'Steps':>6} {'Pass':>6}", file=sys.stderr)
    for r in results:
        print(
            f"{r['task_id']:<14} {r['score']:>10.4f} {r['total_reward']:>10.4f} "
            f"{r['steps']:>6} {'yes' if r['success'] else 'no':>6}",
            file=sys.stderr,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model": model_name,
                "benchmark": BENCHMARK,
                "tasks": results,
                "total_score": round(
                    sum(r["score"] for r in results) / max(len(results), 1), 4
                ),
            }, f, indent=2)
        print(f"\nResults written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
