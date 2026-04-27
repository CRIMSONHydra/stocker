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
def build_prompt_dataset(
    use_mock_specialists: bool,
    cache: bool,
    task_ids: list[str] | None = None,
):
    """Pre-compute specialist votes for every (task, step) and return a list
    of training rows: {task_id, step_index, prompt, env_snapshot}.

    task_ids: subset of list_task_ids() to train on; None = all.
    """
    from app.council.llm import MockLLMClient, build_openai_client_from_env
    from app.council.runner import Council
    from app.council.moderator import Moderator
    from app.core.environment import StockerEnv
    from app.core.tasks import list_task_ids

    client = MockLLMClient() if use_mock_specialists else build_openai_client_from_env()
    council = Council(client=client, use_cache=cache)
    moderator = Moderator(client)  # used only for prompt-building

    active_tasks = task_ids if task_ids is not None else list_task_ids()
    rows = []
    for task_id in active_tasks:
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


PARSE_FAILURE_PENALTY = 0.1  # explicit penalty for malformed JSON output


def _derive_ideal_action(snapshot: dict) -> dict:
    """Greedy-hindsight ideal action for a snapshot. Looks at the price
    `lookahead_steps` ahead and returns buy when price will rise, sell
    when price will fall, hold otherwise. Used for imitation pretrain only.
    """
    from app.config import settings as global_settings
    from app.core.tasks import get_task_definition

    task = get_task_definition(snapshot["task_id"])
    prices = task["prices"]
    step = snapshot["step_index"]
    K = max(1, int(getattr(global_settings, "lookahead_steps", 5)))
    if step + K >= len(prices):
        return {"side": "hold", "quantity": 0,
                "rationale": "End of episode — hold."}

    cur, fut = prices[step], prices[step + K]
    pct = (fut - cur) / max(cur, 1e-9)
    state = snapshot["env_state"]
    cash = state.get("cash", 10000)
    pos = state.get("position", 0)
    max_buy = int(cash // max(cur, 1e-9))

    if pct > 0.005 and max_buy > 0:
        qty = max(1, max_buy // 2)
        return {"side": "buy", "quantity": qty,
                "rationale": f"+{pct:.2%} forecast over next {K} bars — buy."}
    if pct < -0.005 and pos > 0:
        return {"side": "sell", "quantity": pos,
                "rationale": f"{pct:.2%} forecast over next {K} bars — exit."}
    return {"side": "hold", "quantity": 0,
            "rationale": "Flat forecast — hold."}


def _imitation_pretrain(base_model, tokenizer, dataset, run_dir, args) -> None:
    """One-pass SFT on (prompt → ideal-action JSON) pairs. Cheap warmstart
    so GRPO starts with at least valid JSON output and a sensible prior."""
    import json as _json
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    # IMPORTANT: must attach a LoRA adapter BEFORE SFT, or the 4-bit base
    # weights are frozen and there's nothing to train. Same lora_cfg shape
    # GRPO will use later; trainer.save_model preserves it.
    sft_lora = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )

    rows = []
    for r in dataset:
        prompt = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True
        )
        ideal = _derive_ideal_action(r)
        completion = _json.dumps(ideal)
        rows.append({"text": prompt + completion})
    sft_ds = Dataset.from_list(rows)
    print(f"[grpo] Imitation pretrain on {len(rows)} (prompt → ideal-action) pairs ...")

    sft_cfg = SFTConfig(
        output_dir=str(run_dir / "sft_warmstart"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr * 4,            # warmstart can use a higher LR
        num_train_epochs=1,
        logging_dir=str(run_dir / "tensorboard_sft"),
        report_to=["tensorboard"],
        save_strategy="no",
        bf16=base_model.dtype == __import__("torch").bfloat16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )
    sft_trainer = SFTTrainer(
        model=base_model,
        processing_class=tokenizer,
        args=sft_cfg,
        train_dataset=sft_ds,
        peft_config=sft_lora,
    )
    sft_trainer.train()
    print("[grpo] Imitation pretrain done. Continuing to GRPO ...")


def reward_for_completion(completion_text: str, snapshot: dict) -> float:
    """Replay env to the snapshot, apply the completion's action, then roll
    forward `rollout_horizon` steps with `hold` so the consequences of the
    action (price moves while holding the new position) accumulate into the
    reward. Without this multi-step rollout, single-step reward is nearly
    invariant to action and GRPO has no advantage signal.

    Also applies an explicit parse-failure penalty so the model learns to
    output valid JSON instead of getting silently defaulted to hold(0)."""
    from app.config import settings as global_settings
    from app.council.llm import parse_json_object
    from app.core.environment import StockerEnv
    from app.models import EnvironmentState, TradeAction

    parsed = parse_json_object(completion_text)
    parse_failed = not parsed  # empty dict from parse_json_object means malformed JSON
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
    cum_reward = float(result.reward)

    # Roll forward with `hold` so the position taken is actually marked-to-
    # market over future bars. This is what gives the agent a real signal
    # that buying at low prices / selling at high prices is good.
    horizon = int(getattr(global_settings, "rollout_horizon", 5))
    steps_done = 0
    while steps_done < horizon and not result.done:
        result = env.step(TradeAction(side="hold", quantity=0))
        cum_reward += float(result.reward)
        steps_done += 1

    if parse_failed:
        cum_reward -= PARSE_FAILURE_PENALTY

    return cum_reward


# ---------------------------------------------------------------------------
def run_grpo(args, run_dir: Path, dataset):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # GPU path: load in 4-bit so base + LoRA + reference + activations all
    # fit in 24 GB on the L4. Without this the bf16 model alone is ~8 GB and
    # GRPO's reference copy + KV cache + activations OOMs.
    # CPU path: skip BnB (CUDA-only), load in fp32. Used only for end-to-end
    # code-path tests with a tiny model — actual training needs GPU.
    if torch.cuda.is_available():
        bnb_compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"[grpo] Loading {model_id} in 4-bit (compute dtype={bnb_compute_dtype}) ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=bnb_compute_dtype,
        )
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True
        )
    else:
        print(f"[grpo] CUDA unavailable — loading {model_id} in fp32 on CPU "
              "(end-to-end code-path test only; not a real training run).")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        )

    # Flatten chat messages -> single prompt string the way TRL expects
    def _row_to_prompt(row):
        msgs = row["messages"]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    train_rows = [{"prompt": _row_to_prompt(r), "snapshot": r} for r in dataset]
    hf_ds = Dataset.from_list(train_rows)

    # ── Imitation pretrain (option C) ─────────────────────────────────────
    # Brief SFT pass on derived ideal actions to warmstart the LoRA before
    # GRPO. Without it, GRPO starts from random LoRA noise and the agent
    # may need many steps to discover even basic JSON formatting.
    if getattr(args, "imitation_warmstart", False):
        _imitation_pretrain(base_model, tokenizer, dataset, run_dir, args)

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

    # TRL ≥0.13 enforces:
    #   generation_batch_size (= per_device_train_batch_size * grad_accum * num_proc)
    #   must be divisible by num_generations
    # We require the user to pass batch_size that satisfies that.
    if (args.batch_size * args.grad_accum) % args.num_generations != 0:
        raise SystemExit(
            f"per_device_train_batch_size ({args.batch_size}) * grad_accum "
            f"({args.grad_accum}) = {args.batch_size * args.grad_accum} must be "
            f"divisible by num_generations ({args.num_generations}). "
            f"Try --batch-size {args.num_generations} or --num-generations "
            f"{args.batch_size * args.grad_accum}."
        )

    grpo_cfg = GRPOConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        logging_dir=str(run_dir / "tensorboard"),
        report_to=["tensorboard"],
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        # Gradient checkpointing trades a little compute for ~30% activation
        # memory savings. Critical on the L4 24 GB.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=base_model,
        processing_class=tokenizer,
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
    p.add_argument("--batch-size", type=int, default=8,
                   help="per_device batch (must satisfy "
                        "(batch * grad_accum) %% num_generations == 0)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mock-specialists", action="store_true",
                   help="Use MockLLMClient for the 7 specialists (testing only)")
    p.add_argument("--tasks", default="all",
                   help="Comma-separated task IDs to train on, or 'all' (default)")
    p.add_argument("--imitation-warmstart", action="store_true",
                   help="One-pass SFT on derived ideal actions before GRPO. "
                        "Gives the model a sensible prior so GRPO doesn't "
                        "start from random LoRA noise.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    task_ids = None if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]

    run_id = datetime.utcnow().strftime("grpo_%Y%m%d_%H%M%S")
    run_dir = Path(args.out) if args.out else Path("training/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    print(f"[1/3] Building moderator-prompt dataset (tasks={args.tasks}) ...")
    dataset = build_prompt_dataset(
        use_mock_specialists=args.mock_specialists, cache=True, task_ids=task_ids
    )
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
