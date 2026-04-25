# CLAUDE.md

## Project Overview
**Stocker** — multi-agent council RL environment for stock trading on
OpenEnv. Seven specialist LLM analysts vote each step; a moderator LLM
merges votes into a `(side, quantity)` trade. The moderator is GRPO-trained
via TRL on top of `google/gemma-4-E4B-it`.

## Stack
- **Language:** Python 3.10+ (Docker image runs 3.13)
- **Framework:** FastAPI + Uvicorn
- **Models:** Pydantic v2 + pydantic-settings
- **OpenEnv:** `openenv-core>=0.2.0`
- **Package manager:** `uv` (deps grouped into optional extras: `data`, `eval`, `serve`, `train`)
- **Tests:** pytest + FastAPI TestClient (`MockLLMClient` for council tests — no GPU/API needed)
- **Serving:** vLLM (OpenAI-compatible at :8000) — see [scripts/serve_vllm.sh](scripts/serve_vllm.sh)
- **Training:** TRL `GRPOTrainer` + PEFT LoRA — see [training/train_grpo.py](training/train_grpo.py)

## Layout
```
.
├── app/
│   ├── api/                # HTTP routers (health, meta, env, state, frontend)
│   ├── council/            # 7 specialists + moderator + asyncio runner
│   │   ├── llm.py          # OpenAILLMClient, MockLLMClient, parse_json_object
│   │   ├── specialists.py  # ChartPattern / Seasonal / Indicator / News / Forum / Peer / Geo
│   │   ├── moderator.py    # merges votes → TradeAction (extra_body for LoRA)
│   │   └── runner.py       # Council.run / run_async + .cache/council/* on-disk cache
│   ├── core/               # environment.py, graders.py, tasks.py
│   ├── data/loader.py      # parquet lookups + chart_path
│   ├── config.py
│   ├── main.py
│   └── models.py           # Pydantic schemas (the public OpenEnv contract)
├── data/                   # bundled by scripts/build_dataset.py
│   ├── *.parquet
│   ├── charts/             # 768x768 candlestick PNGs
│   └── sources/            # curated news / forums / macro JSON
├── server/app.py           # OpenEnv entry point (`server.app:main`)
├── inference.py            # council-driven OpenEnv inference loop (root)
├── client.py
├── scripts/                # build_dataset, render_charts, serve_vllm, validate_tasks
├── training/               # eval_rollout, train_grpo, runs/
├── tests/
├── Dockerfile
├── openenv.yaml
└── pyproject.toml
```

## Conventions

- All Pydantic schemas live in [app/models.py](app/models.py). Don't move
  them. Names: `MarketObservation`, `TradeAction`, `SpecialistVote`,
  `CouncilDecision`, `RewardResult`, `StepResult`, `ResetResult`,
  `EnvironmentState`.
- `StockerEnv` exposes `reset() / step() / state() / load_snapshot()`.
- The OpenEnv contract is **single-agent**. Multi-agent council lives in
  `app/council/` and `inference.py` — never inside `step()`.
- `inference.py` MUST keep the `[START] / [STEP] / [END]` log format and
  also emit `[COUNCIL]` per step — graders parse all four.
- Reward is clipped to `[-1.0, 1.0]` at the env boundary
  ([app/core/graders.py](app/core/graders.py)).
- Specialists return JSON `{"signal": float, "confidence": float, "rationale": str}`
  — the parser in [app/council/llm.py](app/council/llm.py) tolerates fenced
  code blocks and trailing prose.
- The moderator response goes with `extra_body={"lora_request": {"name": ...}}`
  when a LoRA is configured, otherwise plain base.
- Cache layout: `.cache/council/<role>/base/<ticker>__<date>.json` for
  specialists, `.cache/council/moderator/<lora>/<ticker>__<date>__<hash>.json`
  for the moderator (votes hash differentiates).
- Tests use `MockLLMClient` — never `HF_TOKEN` in CI. The mock routes by
  system-prompt keyword, so changing role keywords requires updating
  `MockLLMClient.SIGNAL_BIAS`.

## Don't

- Don't introduce a frontend build step — the HTML lives inline in
  [app/api/frontend.py](app/api/frontend.py).
- Don't bake live API calls into specialists. Council inputs come from the
  bundled parquet dataset (`app/data/loader.py`). Live sources are the data
  builder's job.
- Don't pull large model weights into the Docker image; serve via vLLM
  externally.
- Don't break the per-step `[COUNCIL]` log line format — the writeup +
  trainer's reward replay both depend on it.
- Don't make specialists rely on the moderator's LoRA. Specialists must
  remain frozen so their cached votes are reusable across GRPO steps.
