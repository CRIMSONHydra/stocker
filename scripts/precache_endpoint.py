#!/usr/bin/env python3
"""Pre-cache all specialist votes for task_easy using the configured LLM endpoint.

Run this once while the HF endpoint is warm so that training and eval can
read from cache (instant) without re-calling the endpoint.

Usage:
    export API_BASE_URL=https://<endpoint>.endpoints.huggingface.cloud/v1
    export MODEL_NAME=ggml-org/gemma-4-26B-A4B-it-GGUF
    export HF_TOKEN=hf_xxx
    python scripts/precache_endpoint.py [--tasks task_easy,task_medium,task_hard]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="task_easy",
                   help="Comma-separated task IDs (default: task_easy)")
    p.add_argument("--mock", action="store_true",
                   help="Use MockLLMClient instead of endpoint (offline test)")
    args = p.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",")]

    if args.mock:
        from app.council.llm import MockLLMClient
        client = MockLLMClient()
        log("Using MockLLMClient (offline mode)")
    else:
        from app.council.llm import build_openai_client_from_env
        client = build_openai_client_from_env()
        log(f"Endpoint → {client.base_url}")
        log(f"Model    → {client.model}")

    from app.council.runner import Council
    from app.core.environment import StockerEnv

    council = Council(client=client, use_cache=True)

    # Build the full plan first so we know the total
    plan: list[tuple] = []
    for task_id in task_ids:
        env = StockerEnv(task_id=task_id)
        obs = env.reset().observation
        while True:
            for sp in council.specialists:
                cache_path = council._vote_cache_path(sp, obs)
                plan.append((sp, obs, cache_path.exists()))
            result = env.step({"side": "hold", "quantity": 0})
            if result.done:
                break
            obs = result.observation

    todo  = [(sp, obs) for sp, obs, hit in plan if not hit]
    hits  = sum(1 for _, _, hit in plan if hit)
    log(f"Plan: {len(plan)} votes — {hits} cached, {len(todo)} to fetch")

    if not todo:
        log("All votes already cached. Nothing to do.")
        return

    t0 = time.time()
    HEARTBEAT = 25
    for i, (sp, obs) in enumerate(todo, 1):
        council._cached_vote(sp, obs)
        if i % HEARTBEAT == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-9)
            eta = (len(todo) - i) / max(rate, 1e-9)
            log(f"  {i}/{len(todo)}  {sp.name}@{obs.ticker}/{obs.date}  "
                f"rate={rate:.2f}/s  ETA={eta/60:.1f} min")

    elapsed = time.time() - t0
    log(f"Done in {elapsed/60:.2f} min. {len(plan)} entries in cache.")


if __name__ == "__main__":
    main()
