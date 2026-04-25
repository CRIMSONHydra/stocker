---
title: Stocker - OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - multi-agent
  - finance
---

# Stocker — Multi-Agent Council RL Environment

A long-term stock-trading RL environment built on **OpenEnv**. Every step,
seven specialist analyst agents look at the same situation through different
lenses and vote; a moderator merges their votes into a single trade
`(side, quantity)`. The moderator is fine-tuned via **GRPO** (TRL) using the
env's own reward as the training signal. Base model: **`google/gemma-4-E4B-it`**
(multimodal — the chart specialist consumes candlestick PNGs).

## Council architecture

```
                       OpenEnv interface (single-agent contract)
        ┌───────────────────────────────────────────────────┐
        │  POST /reset → MarketObservation(ticker, date, …) │
        │  POST /step  → TradeAction(side, quantity)        │
        └────────────────────────▲──────────────────────────┘
                                 │  inference.py orchestrates:
   ┌─────────────────────────────┴─────────────────────────────┐
   │                          per step                          │
   │  ┌──────────────────────────────────────────────────────┐ │
   │  │           7 specialists run in PARALLEL              │ │
   │  │  ChartPattern   (vision: 60-day candlestick PNG)     │ │
   │  │  SeasonalTrend  (long-term + cycle context)          │ │
   │  │  Indicator      (RSI / MACD / SMA / BB / ATR)        │ │
   │  │  News           (curated headlines, ~7-day window)   │ │
   │  │  ForumSentiment (Reddit excerpts)                    │ │
   │  │  PeerCommodity  (peer stocks + gold/oil correlation) │ │
   │  │  Geopolitics    (CPI / FOMC / sanctions / fiscal)    │ │
   │  │  → SpecialistVote(signal in [-1,1], confidence, why) │ │
   │  └──────────────────────────────────────────────────────┘ │
   │                       │                                    │
   │                       ▼                                    │
   │           Moderator (Gemma 4 E4B IT + LoRA)                │
   │       sees 7 votes → outputs TradeAction + rationale       │
   └────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                       env.step(action)  →  reward
                  (alpha vs buy-and-hold − drawdown)
```

Key invariant: **the OpenEnv contract stays single-agent.** Council
orchestration lives in `inference.py` + `app/council/`, not inside `step()`.
This means `openenv validate` passes and graders can re-run inference exactly.

## Tasks

| Task | Ticker | Window | Difficulty |
|------|--------|--------|------------|
| `task_easy`   | AAPL | 2023-08 → 2023-09 | Steady regime, modest moves |
| `task_medium` | INTC | 2024-01 → 2024-02 | Choppy / sideways post-earnings |
| `task_hard`   | META | 2022-09 → 2022-10 | Drawdown then snap-back after Q3 print |

Episodes are 41–43 trading days each.

## Action space

| Field | Type | Description |
|-------|------|-------------|
| `side` | `buy \| sell \| hold` | |
| `quantity` | `int (≥ 0)` | Ignored when `side=hold`. Constrained by cash/position. |

## Observation space

The agent sees portfolio state plus *all council inputs* in one struct
(specialists slice their own view from it):

| Field | Type | Used by |
|-------|------|---------|
| `ticker`, `date`, `price`, `price_history` | core | env, every specialist |
| `cash`, `position`, `portfolio_value` | core | moderator, reward |
| `fundamentals` | dict | seasonal, peer |
| `chart_path` | path to PNG | ChartPattern (vision) |
| `headlines` | list[dict] | News |
| `forum_excerpts` | list[dict] | ForumSentiment |
| `indicators` | dict | Indicator |
| `peers` | dict | PeerCommodity |
| `macro` | list[dict] | Geopolitics, SeasonalTrend |
| `task_id`, `step_number`, `total_steps` | bookkeeping | all |

## Reward

Per step:

```
reward_step = ΔPortfolio% − invalid_action_penalty
```

End-of-episode shaping (added to the last step):

```
+ alpha_bonus   = min(0.10, max(0, alpha_vs_buy_and_hold))
- dd_penalty    = min(0.10, max(0, max_drawdown − 0.05))
```

Reward is clipped to `[-1, 1]` at the env boundary. See
[app/core/graders.py](app/core/graders.py).

## Quick start

```bash
# 1. Install (CPU-only deps)
pip install uv
uv pip install -e ".[dev,data,eval]"

# 2. Build the bundled dataset (yfinance + indicators + chart PNGs)
python scripts/build_dataset.py
python scripts/validate_tasks.py

# 3. Smoke-test with the deterministic mock client (no GPU, no API key)
python inference.py --task all --mock --no-cache

# 4. Local server + interactive frontend
./run.sh
# http://localhost:7860/web
# http://localhost:7860/docs   (Swagger)

# 5. Real run with Gemma 4 E4B IT served locally via vLLM
pip install vllm                                      # or uv pip install -e ".[serve]"
./scripts/serve_vllm.sh                               # boots http://localhost:8000/v1
export API_BASE_URL=http://localhost:8000/v1
export MODEL_NAME=google/gemma-4-E4B-it
export HF_TOKEN=any
python inference.py --task all
```

### Remote serving (when the local 4070M is too tight)

**Option A — HF Inference Endpoints** (no code changes, just env vars):

```bash
# After deploying google/gemma-4-E4B-it as a dedicated endpoint on HF:
export API_BASE_URL=https://<endpoint-id>.endpoints.huggingface.cloud/v1
export MODEL_NAME=google/gemma-4-E4B-it
export HF_TOKEN=hf_xxxxxxxxxxxx
python inference.py --task all
```

**Option B — Colab T4 end-to-end** (recommended when downloads are a problem):

Open [training/train_grpo.ipynb](training/train_grpo.ipynb) on Colab. The
notebook clones the repo, downloads Gemma on Colab's network (gigabit, no
WSL overhead), pre-caches all 7 specialists' votes, runs GRPO, and saves
the LoRA + plots — all in one runtime. Uses `TransformersLLMClient` to call
the loaded model in-process, so no separate vLLM server is needed.

**Option C — Edit in VSCode, execute on Colab GPU.** Run
[training/colab_launcher.ipynb](training/colab_launcher.ipynb) on Colab —
it clones the repo, installs deps, builds the dataset, then exposes a
Jupyter server through a free `cloudflared` tunnel and prints a single URL.
In VSCode locally:

1. Install the **Jupyter extension** (`ms-toolsai.jupyter`).
2. Command Palette → *Jupyter: Specify Jupyter Server for Connections* →
   paste the URL.
3. Open `training/train_grpo.ipynb` locally and pick the remote kernel.

You edit `.ipynb` cells in VSCode; execution happens on Colab's T4. When
the runtime evicts (idle / 12-hour limit), re-run the launcher's `launch`
cell for a fresh URL.

## Inference output format

Standard OpenEnv `[START] / [STEP] / [END]` lines, plus a `[COUNCIL]` line per
step that dumps all 7 votes (used by the writeup to show council disagreement):

```
[START]   task=task_easy env=stocker model=google/gemma-4-E4B-it
[STEP]    step=1 action=buy(10) reward=0.0042 done=false error=null
[COUNCIL] step=1 payload={"votes":[{"name":"chart_pattern","signal":0.42,...}, ...],"rationale":"..."}
...
[END]     success=true steps=43 score=0.0871 rewards=0.0042,...
```

## Training (GRPO)

The 7 specialists are **frozen** (base inference). Only the **moderator** is
fine-tuned via `trl.GRPOTrainer` with a LoRA adapter on the Gemma 4 E4B IT
base. Reward = env's own per-step reward.

```bash
uv pip install -e ".[train]"
# In a separate terminal: ./scripts/serve_vllm.sh
python -m training.train_grpo --epochs 2 --num-generations 8 --batch-size 2 --grad-accum 4
```

Outputs land in `training/runs/grpo_<timestamp>/`:

- `moderator-lora/` — the trained PEFT adapter
- `loss.png`, `reward.png` — TB-derived plots (ref.md item 3)
- `tensorboard/` — full TB logs
- `args.json`, `dataset.json` — exact training config

For the Colab path: open [training/train_grpo.ipynb](training/train_grpo.ipynb).

### Compute notes

- Base + LoRA + KV cache fits a single **RTX 4070M (8GB)** at 4-bit (bitsandbytes).
- Specialist outputs are cached by `(role, ticker, date)`, so they are
  computed once and re-used across every GRPO step. Only the moderator is
  re-rolled per training update.
- Cloud fallback: spin up an HF Inference Endpoint with the Gemma weights and
  point `API_BASE_URL` at it.

## Evaluation rollout

```bash
# Pre-training baseline (no LoRA)
python -m training.eval_rollout --out training/runs/eval_pre

# After GRPO
python -m training.eval_rollout --moderator-lora moderator --out training/runs/eval_post
```

Each run produces `reward_curve.png`, `portfolio_curve.png`, `summary.csv`,
and a full per-step trace in `results.json`.

## Layout

```
.
├── app/
│   ├── api/                 # FastAPI routers (health, meta, env, state, frontend)
│   ├── council/             # 7 specialists + moderator + parallel runner
│   │   ├── llm.py           # OpenAILLMClient + MockLLMClient + JSON parsers
│   │   ├── specialists.py   # 7 role-specific prompted agents
│   │   ├── moderator.py     # merges votes → TradeAction (+ optional LoRA)
│   │   └── runner.py        # asyncio.gather + on-disk cache
│   ├── core/                # environment.py, graders.py, tasks.py
│   ├── data/loader.py       # parquet + chart lookups
│   └── models.py            # all Pydantic schemas
├── data/                    # bundled at build time by scripts/build_dataset.py
│   ├── *.parquet
│   ├── charts/              # 768x768 candlestick PNGs
│   └── sources/             # curated news / forums / macro JSON
├── server/app.py            # OpenEnv ASGI entrypoint
├── inference.py             # council-driven OpenEnv inference loop
├── client.py                # Python HTTP client
├── scripts/
│   ├── build_dataset.py     # one-time data generator (yfinance + indicators + charts)
│   ├── render_charts.py
│   ├── serve_vllm.sh        # local OpenAI-compatible LLM endpoint
│   └── validate_tasks.py
├── training/
│   ├── eval_rollout.py      # offline backtest, reward + portfolio curves
│   ├── train_grpo.py        # TRL GRPO on the moderator LoRA
│   └── train_grpo.ipynb     # Colab/Kaggle wrapper
├── tests/
├── Dockerfile
├── openenv.yaml
└── pyproject.toml
```

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/web` | GET | Interactive frontend |
| `/health` | GET | Health check |
| `/meta` | GET | Environment metadata |
| `/reset` | POST | Reset (body: `{"task_id": "task_easy"}`) |
| `/step` | POST | Submit `{"side": "...", "quantity": N}` |
| `/state` | GET / POST | Export / restore environment state |
| `/docs` | GET | Swagger UI |

## Credits & references

- Built on OpenEnv (`openenv-core>=0.2.0`) following the
  [submission spec](ref.md).
- Real OHLCV: yfinance (Yahoo Finance public endpoints).
- Curated headlines + forum excerpts ship in [data/sources/](data/sources/);
  drop-in replace with your own sources.
- Model: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it).

## TODO / hand-off

- Swap curated news for a live HF dataset of dated headlines (e.g. GDELT slice).
- Add more tasks under `data/sources/news.json` etc. + extend `TASK_META`.
- Multi-step rollouts in GRPO (currently 1-step reward per generation).
- Training-run links: replace the Colab placeholder with the user's run.
