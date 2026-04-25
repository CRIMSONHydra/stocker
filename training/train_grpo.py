#!/usr/bin/env python3
"""GRPO fine-tuning of the moderator on the Stocker council env.

Approach
--------
1) Walk through every (task_id, step) deterministically. For each, run the 7
   frozen specialists (cached after first build) and capture their votes.
2) Build a training dataset of moderator prompts. Each row contains the
   system + user text and the env-state snapshot needed to compute reward.
3) GRPO loop (TRL): for each prompt, sample G candidate moderator outputs,
   parse each into a TradeAction, simulate ONE env step to get the reward,
   then update the LoRA adapter on Gemma 4 E4B IT against those G rewards.

This is a per-step GRPO. Multi-step rollouts (sequence-level reward) are an
obvious extension and are left as a follow-up.

Outputs (under training/runs/<run_id>/):
  - moderator-lora/        the trained PEFT adapter
  - loss.png, reward.png   matplotlib snapshots
  - tensorboard/           TB event files
  - eval_report.json       pre/post comparison

Compute
-------
- Default targets a single 4070M with bitsandbytes 4-bit base + LoRA rank 16.
- Set TRAIN_DEVICE=cuda or run on cloud (HF Endpoints / Spaces with GPU).
- Specialist calls go through the same vLLM endpoint as inference (so vLLM
  must be running). For deterministic dry-runs use --mock-specialists.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def build_prompt_dataset(use_mock_specialists: bool, cache: bool):
    """Pre-compute specialist votes for every (task, step) and return a list
    of training rows: {task_id, step_index, prompt, env_snapshot}.
    """
    from app.council.llm import MockLLMClient, build_openai_client_from_env
    from app.council.runner import Council
    from app.council.moderator import Moderator
    from app.core.environment import StockerEnv
    from app.core.tasks import list_task_ids

    client = MockLLMClient() if use_mock_specialists else build_openai_client_from_env()
    council = Council(client=client, use_cache=cache)
    moderator = Moderator(client)  # used only for prompt-building

    rows = []
    for task_id in list_task_ids():
        env = StockerEnv(task_id=task_id)
        obs = env.reset().observation
        # Snapshot state for replay during reward computation:
        snapshots = []
        while True:
            # 7 frozen specialists vote (cached)
            votes = []
            for sp in council.specialists:
                votes.append(council._cached_vote(sp, obs))

            messages = moderator._build_messages(obs, votes)
            snapshots.append({
                "task_id": task_id,
                "step_index": obs.step_number - 1,
                "messages": messages,
                "votes": [v.model_dump() for v in votes],
                "env_state": env.state().model_dump(),
            })

            # Advance env with a placeholder hold so we get the *next* observation
            result = env.step({"side": "hold", "quantity": 0})
            if result.done:
                break
            obs = result.observation
        rows.extend(snapshots)
    return rows


def reward_for_completion(completion_text: str, snapshot: dict) -> float:
    """Replay env to the given step, apply the completion's action, and
    return the env reward."""
    from app.council.llm import parse_json_object
    from app.core.environment import StockerEnv
    from app.models import EnvironmentState, TradeAction

    parsed = parse_json_object(completion_text)
    side = str(parsed.get("side", "hold")).lower()
    if side not in ("buy", "sell", "hold"):
        side = "hold"
    try:
        qty = max(0, int(parsed.get("quantity", 0)))
    except (ValueError, TypeError):
        qty = 0

    env = StockerEnv(task_id=snapshot["task_id"])
    env.reset()
    env.load_snapshot(EnvironmentState(**snapshot["env_state"]))
    result = env.step(TradeAction(side=side, quantity=qty))
    return float(result.reward)


# ---------------------------------------------------------------------------
def run_grpo(args, run_dir: Path, dataset):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Flatten chat messages -> single prompt string the way TRL expects
    def _row_to_prompt(row):
        msgs = row["messages"]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    train_rows = [{"prompt": _row_to_prompt(r), "snapshot": r} for r in dataset]
    hf_ds = Dataset.from_list(train_rows)

    def reward_fn(completions, **kwargs):
        # `completions` is a list of decoded strings; `kwargs` contains the
        # original row fields including "snapshot".
        snapshots = kwargs["snapshot"]
        rewards = []
        for comp, snap in zip(completions, snapshots):
            try:
                rewards.append(reward_for_completion(comp, snap))
            except Exception as e:
                logger.warning("reward_fn error: %s", e)
                rewards.append(0.0)
        return rewards

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_cfg = GRPOConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        max_prompt_length=2048,
        max_completion_length=192,
        logging_dir=str(run_dir / "tensorboard"),
        report_to=["tensorboard"],
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=hf_ds,
        peft_config=lora_cfg,
    )

    trainer.train()
    trainer.save_model(str(run_dir / "moderator-lora"))
    return trainer


# ---------------------------------------------------------------------------
def plot_history(run_dir: Path, history: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        logger.warning("Empty trainer history; skipping plots.")
        return
    losses = [(h.get("step") or h.get("global_step"), h["loss"]) for h in history if "loss" in h]
    rewards = [(h.get("step") or h.get("global_step"), h["reward"]) for h in history if "reward" in h]

    if losses:
        xs, ys = zip(*losses)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys)
        ax.set_title("GRPO loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "loss.png", dpi=120)
        plt.close(fig)

    if rewards:
        xs, ys = zip(*rewards)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys)
        ax.set_title("GRPO reward")
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "reward.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.getenv("MODEL_NAME", "google/gemma-4-E4B-it"))
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mock-specialists", action="store_true",
                   help="Use MockLLMClient for the 7 specialists (testing only)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_id = datetime.utcnow().strftime("grpo_%Y%m%d_%H%M%S")
    run_dir = Path(args.out) if args.out else Path("training/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    print(f"[1/3] Building moderator-prompt dataset ...")
    dataset = build_prompt_dataset(use_mock_specialists=args.mock_specialists, cache=True)
    (run_dir / "dataset.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "messages"} for r in dataset], indent=2
    ))
    print(f"  {len(dataset)} (task,step) prompts ready.")

    print(f"[2/3] Running GRPO ...")
    trainer = run_grpo(args, run_dir, dataset)

    print(f"[3/3] Plotting curves ...")
    plot_history(run_dir, trainer.state.log_history)

    print(f"Done. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
