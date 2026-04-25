"""
Inference Script — Stocker OpenEnv
==================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM (default: HuggingFace Router)
    MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-7B-Instruct)
    HF_TOKEN       Your HuggingFace / API key
    API_KEY        Alternative to HF_TOKEN

- Place inference.py in the project root. Uses OpenAI Client for LLM calls.

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import argparse
import json
import os
import re
import sys
import textwrap
from typing import Optional

from openai import OpenAI

from app.core.environment import StockerEnv
from app.core.tasks import list_task_ids

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
BENCHMARK = "stocker"

MAX_STEPS = 50
TEMPERATURE = 0.2
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.0  # any non-negative net P&L counts as success

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a disciplined stock trader. At each step you observe the current
    price, recent price history, your cash, and your share position. You must
    decide: buy, sell, or hold — and if buying or selling, how many shares.

    Rules:
    - You cannot buy more than your cash allows.
    - You cannot sell more shares than you own.
    - Aim to maximize the final portfolio value.

    Respond with ONLY a valid JSON object (no markdown, no extra text):
    {"side": "buy|sell|hold", "quantity": <int>}""")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(observation: dict) -> str:
    return f"""Trade decision for {observation['ticker']} on {observation['date']}.

Current price: {observation['price']:.2f}
Recent prices: {observation['price_history']}
Fundamentals: {observation['fundamentals']}
Cash: {observation['cash']:.2f}
Position: {observation['position']} shares
Portfolio value: {observation['portfolio_value']:.2f}
Step {observation['step_number']} of {observation['total_steps']}"""


def parse_response(text: str) -> dict:
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        start = cleaned.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                if depth == 0:
                    parsed = json.loads(cleaned[start : i + 1])
                    side = str(parsed.get("side", "hold")).lower()
                    if side not in ("buy", "sell", "hold"):
                        side = "hold"
                    qty = int(parsed.get("quantity", 0))
                    return {"side": side, "quantity": max(0, qty)}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    side_match = re.search(r"side[\"']?\s*[:=]\s*[\"']?(buy|sell|hold)", text, re.I)
    qty_match = re.search(r"quantity[\"']?\s*[:=]\s*[\"']?(-?\d+)", text, re.I)
    if side_match:
        return {
            "side": side_match.group(1).lower(),
            "quantity": max(0, int(qty_match.group(1))) if qty_match else 0,
        }

    return {"side": "hold", "quantity": 0}


def run_episode(
    client: OpenAI, model: str, task_id: str, use_json_mode: bool = True
) -> dict:
    env = StockerEnv(task_id=task_id)
    reset_result = env.reset()
    obs = reset_result.observation.model_dump()

    rewards: list[float] = []
    step_details: list[dict] = []
    steps_taken = 0

    log_start(task=task_id, env=BENCHMARK, model=model)

    try:
        for step in range(1, MAX_STEPS + 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
            ]
            error: Optional[str] = None

            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                }
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(**kwargs)
                llm_text = (response.choices[0].message.content or "").strip()
            except Exception as e:
                err = str(e)
                if use_json_mode and ("response_format" in err or "json" in err.lower()):
                    use_json_mode = False
                    try:
                        response = client.chat.completions.create(
                            model=model, messages=messages,
                            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                        )
                        llm_text = (response.choices[0].message.content or "").strip()
                    except Exception as e2:
                        error = str(e2)
                        llm_text = '{"side": "hold", "quantity": 0}'
                else:
                    error = err
                    llm_text = '{"side": "hold", "quantity": 0}'

            action_dict = parse_response(llm_text)
            step_result = env.step(action_dict)

            reward = step_result.reward
            done = step_result.done
            rewards.append(reward)
            steps_taken = step

            action_str = f"{action_dict['side']}({action_dict['quantity']})"
            log_step(step, action_str, reward, done, error)

            step_details.append({
                "step": step,
                "side": action_dict["side"],
                "quantity": action_dict["quantity"],
                "reward": reward,
                "info": step_result.info,
                "done": done,
            })

            if done:
                break
            obs = step_result.observation.model_dump()

        score = sum(rewards)
        success = score >= SUCCESS_SCORE_THRESHOLD
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


def main():
    parser = argparse.ArgumentParser(description="Run LLM inference on Stocker")
    parser.add_argument("--task", default="all", help="Task name or 'all'")
    parser.add_argument("--model", default=None, help=f"Model (default: {MODEL_NAME})")
    parser.add_argument("--output", default=None, help="Path to write JSON results")
    parser.add_argument(
        "--no-json-mode", action="store_true",
        help="Disable JSON response mode for models that don't support it",
    )
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: No API key. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.")
        raise SystemExit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    model = args.model or MODEL_NAME

    tasks = list_task_ids() if args.task == "all" else [args.task]

    results = []
    for task_id in tasks:
        results.append(run_episode(client, model, task_id, use_json_mode=not args.no_json_mode))

    print(f"\n{'='*56}", file=sys.stderr)
    print(f"{'Task':<18} {'Score':>10} {'Reward':>10} {'Steps':>6} {'Pass':>6}", file=sys.stderr)
    for r in results:
        print(
            f"{r['task_id']:<18} {r['score']:>10.4f} {r['total_reward']:>10.4f} "
            f"{r['steps']:>6} {'yes' if r['success'] else 'no':>6}",
            file=sys.stderr,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model": model,
                "api_base_url": API_BASE_URL,
                "benchmark": BENCHMARK,
                "tasks": results,
                "total_score": round(sum(r["score"] for r in results) / max(len(results), 1), 4),
            }, f, indent=2)
        print(f"\nResults written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
