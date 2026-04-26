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
  - rl
  - finance
  - grpo
---

# Stocker — Multi-Agent Council RL for Stock Trading

> **[🤗 Environment Space](https://huggingface.co/spaces/Hydr473/stocker-env)** ·
> **[📓 Training Notebook](training/train_grpo.ipynb)** ·
> **[📝 HF Blog Post](https://huggingface.co/Hydr473/posts)** ·
> **[🎯 Training Space](https://huggingface.co/spaces/Hydr473/stocker-env-train)**

**Problem:** LLMs can reason about markets from multiple angles simultaneously —
charts, news, macro, technicals, sentiment. Can we train a *moderator* LLM to
synthesize seven specialist perspectives into profitable trades, purely via RL?

**Approach:** Seven frozen specialist LLMs vote each step; a trainable moderator
merges their votes. The moderator is fine-tuned with **GRPO** (Group Relative
Policy Optimization) using the environment's own reward as the training signal.
No human labels, no supervised data — reward comes from the env.

**Results:** See [training/runs/RESULTS.md](training/runs/RESULTS.md) for the
pre-training vs post-training comparison table and plots.

A long-term stock-trading RL environment built on **OpenEnv**. Every step,
seven specialist analyst agents examine the market through different lenses
and emit a vote; a **moderator LLM** merges the seven votes into a single
`(side, quantity)` trade. The moderator is fine-tuned with **GRPO** (TRL)
using the environment's own reward as the training signal.

Base model: **[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)** —
multimodal (text + image), 4-billion parameters, Apache 2.0.

---

## Council architecture

```
                     OpenEnv interface (single-agent contract)
      ┌─────────────────────────────────────────────────────┐
      │  POST /reset → MarketObservation(ticker, date, …)   │
      │  POST /step  → TradeAction(side, quantity)          │
      └───────────────────────▲─────────────────────────────┘
                              │  inference.py orchestrates:
 ┌────────────────────────────┴────────────────────────────────┐
 │                        per step                              │
 │  ┌───────────────────────────────────────────────────────┐  │
 │  │         7 specialists run in PARALLEL                 │  │
 │  │  ChartPattern    (vision: 60-day candlestick PNG)     │  │
 │  │  SeasonalTrend   (long-term + cycle context)          │  │
 │  │  Indicator       (RSI / MACD / SMA / BB / ATR)        │  │
 │  │  News            (curated headlines, ~7-day window)   │  │
 │  │  ForumSentiment  (Reddit excerpts)                    │  │
 │  │  PeerCommodity   (peer stocks + gold/oil correlation) │  │
 │  │  Geopolitics     (CPI / FOMC / sanctions / fiscal)    │  │
 │  │  → SpecialistVote(signal ∈ [-1,1], confidence, why)   │  │
 │  └───────────────────────────────────────────────────────┘  │
 │                      │                                       │
 │                      ▼                                       │
 │          Moderator (Gemma 4 E4B IT + LoRA)                   │
 │      sees 7 votes → outputs TradeAction + rationale          │
 └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    env.step(action)  →  reward
            (performance vs ideal trajectory − inflation drag)
```

**Key invariant:** the OpenEnv contract stays single-agent. Council
orchestration lives in `inference.py` + `app/council/`, not inside `step()`.
`openenv validate` passes and graders can re-run inference exactly.

---

## Methodology

### Dataset

Three deterministic episodes built from real OHLCV data (yfinance):

| Task | Ticker | Window | Regime |
|------|--------|--------|--------|
| `task_easy`   | AAPL | Aug–Sep 2023 | Steady uptrend into iPhone 15 launch |
| `task_medium` | INTC | Jan–Feb 2024 | Choppy / sideways post-earnings |
| `task_hard`   | META | Sep–Oct 2022 | Drawdown then snap-back after Q3 print |

Each episode is 41–43 trading days. All OHLCV, indicators, curated
headlines, forum excerpts, macro events, and 768×768 candlestick PNGs are
bundled in `data/` and versioned in git (6.2 MB).

An optional **corpus** (15 tickers × 20 years, ~3 GB) can be built with
`python scripts/build_corpus.py` and exposes 4 000+ additional episodes via
`corpus_*` task IDs.

### Specialists (frozen)

Each specialist has a fixed system prompt and calls the same LLM (Gemma 4
E4B IT in 4-bit, or any OpenAI-compatible endpoint). Their outputs are
**cached by `(role, ticker, date)`** after the first call, so they are
never re-rolled during GRPO — only the moderator is trained.

| Specialist | Input | Dimension |
|-----------|-------|-----------|
| `ChartPattern` | candlestick PNG (multimodal) | visual patterns |
| `SeasonalTrend` | price history + fundamentals | macro cycle |
| `Indicator` | RSI, MACD, SMA, BB, ATR | technical signals |
| `News` | curated headlines | sentiment |
| `ForumSentiment` | Reddit excerpts | crowd sentiment |
| `PeerCommodity` | peer stocks + gold/oil | correlation |
| `Geopolitics` | CPI, FOMC, macro events | macro risk |

Each returns `{"signal": float ∈ [-1,1], "confidence": float ∈ [0,1], "rationale": str}`.

### Reward function

Per-step reward is a weighted combination of two components:

```
reward_step = W_PERF × performance_factor(gap)
            + W_INFL × inflation_factor
            − invalid_action_penalty        (0.01 if action invalid)
```

**Performance factor** — asymmetric piecewise-linear function of the gap
between the pre-computed ideal PnL trajectory and the model's actual
inflation-adjusted PnL at this step:

```
gap = ideal_pnl_pct[step] − real_pnl_pct

if gap < 0:          performance_factor = 1 + min(1, |gap| / scale)   # outperformance bonus
elif gap ≤ scale:    performance_factor = 1 − gap / scale             # close to ideal
else:                performance_factor = −min(1, (gap − scale) / scale)  # far behind
```

where `scale = max(0.05, 0.5 × |ideal_pnl_pct_total|)`.

The **ideal PnL trajectory** is pre-computed by `scripts/build_ideal_profit.py`
using the optimal hindsight strategy (greedy day-by-day).

**Inflation factor** — penalty for the share of nominal gain eaten by inflation:

```
real_pnl_pct = (1 + nominal_pnl_pct) / (1 + inflation_growth) − 1
inflation_factor = real_pnl_pct − nominal_pnl_pct    (≤ 0)
```

**Trajectory bonus** (added to the last step):

```
alpha_bonus = min(0.10, max(0, (final_portfolio / buy_and_hold_portfolio) − 1))
dd_penalty  = min(0.10, max(0, max_drawdown − 0.05))
```

**Default weights:** `W_PERF = 0.7`, `W_INFL = 0.3`, `annual_inflation_rate = 0.05`,
`transaction_cost_rate = 0.001`. All are configurable via `STOCKER_*` env vars
(see `app/config.py`) and sweepable with the `tune_easy_gemma4.ipynb` replay grader.

All rewards are clipped to `[-1.0, 1.0]` at the env boundary.

### Training (GRPO)

The 7 specialists are **frozen**. Only the **moderator** is fine-tuned
via `trl.GRPOTrainer` with a LoRA adapter (rank 16) on Gemma 4 E4B IT.

**Dataset:** `task_easy` (43 moderator prompts, each containing 7 specialist
votes and the current market state).

**Settings** (L4-tuned for the HF training Space):

| Parameter | Value |
|-----------|-------|
| Base model | `google/gemma-4-E4B-it` (4-bit BnB, ~3 GB) |
| Inference (specialists + production) | `ggml-org/gemma-4-26B-A4B-it-GGUF` via HF endpoint (llama.cpp) |
| LoRA rank / alpha | 16 / 32 |
| Epochs | 3 |
| `num_generations` | 4 |
| `per_device_train_batch_size` | 4 |
| Learning rate | 5e-6 |
| Compute | HF Space, Nvidia L4 24 GB (~$0.80 / training run) |

Each GRPO step: sample 4 moderator completions per prompt → parse each into
`TradeAction` → simulate env step → compare rewards → update LoRA. Specialist
votes come from cache (no LLM calls); only the moderator is re-rolled.

### Reward-weight tuning

`training/tune_easy_gemma4.ipynb` runs a full episode, captures per-step
reward breakdowns, then **re-grades the saved trace** under a sweep of weight
combinations (pure CPU, sub-second per combo). This lets us iterate on the
reward function without re-running the LLM:

```
swept 180 combos over (W_PERF, W_INFL, inflation_rate, transaction_cost)
best: W_PERF=1.0, W_INFL=0.0, tc=0.0  →  total=+0.5725
default:                                →  total=+0.3801
```

---

## Results

> Training run results are compiled automatically by `scripts/compile_results.py`
> after training. See `training/runs/RESULTS.md` for the latest numbers and plots.

---

## Quick start

### A. Hit the running env Space (zero-setup, recommended for graders)

The OpenEnv contract is live at the HF Space — interact with `/reset`, `/step`,
`/state` directly:

```bash
# Reset to task_easy
curl -X POST https://hydr473-stocker-env.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task_easy"}'

# Submit a trade
curl -X POST https://hydr473-stocker-env.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"side": "buy", "quantity": 10}'
```

Or open `https://hydr473-stocker-env.hf.space/web` for the interactive React UI
(six tabs: Terminal, Council, Training, Gallery, Portfolio, Intelligence).

### B. Run GRPO training (one click on the L4 GPU Space)

Open [Hydr473/stocker-train](https://huggingface.co/spaces/Hydr473/stocker-env-train)
and click **🚀 Launch Pipeline**. The Gradio UI streams the full 6-phase run
(precache → eval_pre → GRPO → eval_post → compile → upload) and uploads the
final plots + LoRA adapter to [Hydr473/stocker-results](https://huggingface.co/datasets/Hydr473/stocker-results).
Cost: ~$0.80 / run on the L4.

### C. Local development (mock client, no GPU)

```bash
pip install uv
uv pip install -e ".[dev,data,eval]"
python scripts/build_dataset.py
python scripts/validate_tasks.py

# Deterministic offline smoke (no LLM calls)
python inference.py --task all --mock --no-cache

# Local server + React UI
./run.sh   # http://localhost:7860/web
pytest tests/ -q
```

To run the council against your own endpoint, set `API_BASE_URL`,
`MODEL_NAME`, and `HF_TOKEN` (see [.env.example](.env.example)).

---

## Evaluation

```bash
# Baseline (no LoRA)
python -m training.eval_rollout --tasks task_easy --out training/runs/eval_pre

# After GRPO
python -m training.eval_rollout --tasks task_easy --moderator-lora moderator \
    --out training/runs/eval_post

# Compile artifacts + diff table
python scripts/compile_results.py
```

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/web` | GET | Interactive React frontend |
| `/health` | GET | Health check |
| `/meta` | GET | Environment metadata |
| `/reset` | POST | Reset (`{"task_id": "task_easy"}`) |
| `/step` | POST | Submit `{"side": "buy", "quantity": 10}` |
| `/state` | GET/POST | Export / restore env snapshot |
| `/docs` | GET | Swagger UI |

---

## Layout

```
.
├── app/
│   ├── api/                 # FastAPI routers
│   ├── council/
│   │   ├── llm.py           # TransformersLLMClient, OpenAILLMClient, MockLLMClient
│   │   ├── specialists.py   # 7 role-specific prompted agents
│   │   ├── moderator.py     # merges votes → TradeAction (+ optional LoRA)
│   │   └── runner.py        # ThreadPoolExecutor + on-disk cache
│   ├── core/
│   │   ├── environment.py   # StockerEnv: reset / step / state / load_snapshot
│   │   ├── graders.py       # compute_step_reward, compute_trajectory_bonus
│   │   └── tasks.py         # get_task_definition + corpus integration
│   ├── data/
│   │   ├── loader.py        # parquet + chart lookups + corpus fallback
│   │   └── corpus.py        # optional large corpus (15 tickers × 20 y)
│   └── models.py            # all Pydantic schemas
├── data/
│   ├── *.parquet            # prices, indicators, news, peers, macro
│   ├── ideal_profits/       # hindsight-optimal PnL sidecars
│   ├── charts/              # 768×768 candlestick PNGs
│   └── sources/             # curated JSON (news, forums, macro)
├── scripts/
│   ├── build_dataset.py     # yfinance → parquet + charts
│   ├── build_ideal_profit.py # per-task optimal PnL trajectory
│   ├── build_corpus.py      # large corpus (optional, gitignored)
│   ├── compile_results.py   # compile training artifacts → RESULTS.md
│   ├── precache_endpoint.py # standalone specialist pre-cache via endpoint
│   └── validate_tasks.py
├── training/
│   ├── train_grpo.py        # GRPO trainer (CLI, --tasks filter)
│   ├── train_grpo.ipynb     # end-to-end Colab/Jupyter notebook
│   ├── tune_easy_gemma4.ipynb  # reward-weight tuning bench
│   ├── eval_rollout.py      # offline backtest → reward/portfolio curves
│   └── runs/                # gitignored (except .gitkeep)
├── spaces/
│   └── train/               # Gradio app for the L4 GPU training Space
│       ├── Dockerfile       # CUDA + PyTorch + git clone of this repo
│       └── app.py           # 6-phase pipeline: precache → eval → GRPO → upload
├── .github/workflows/
│   └── deploy_spaces.yml    # CI: auto-deploy both Spaces on push to main
├── inference.py             # council-driven OpenEnv inference loop
├── tests/
├── Dockerfile               # multi-stage: build React UI → Python runtime
└── pyproject.toml
```

---

## Submission checklist

| Requirement | Status |
|-------------|--------|
| Built on OpenEnv (`openenv-core>=0.2.0`) | ✅ |
| Training script using TRL (Colab notebook) | ✅ [train_grpo.ipynb](training/train_grpo.ipynb) |
| Evidence of training (loss + reward plots) | ✅ [RESULTS.md](training/runs/RESULTS.md) |
| Writeup / HF blog post | ✅ [BLOG.md](BLOG.md) / [HF Post](https://huggingface.co/Hydr473/posts) |
| Environment on HF Space | ✅ [Hydr473/stocker-env](https://huggingface.co/spaces/Hydr473/stocker-env) |
| README with problem + env + results | ✅ This file |
| README links to Space + blog + plots | ✅ Top of this file |

---

## Credits

- Built on [OpenEnv](https://github.com/huggingface/openenv) following the submission spec.
- Real OHLCV: yfinance (Yahoo Finance public endpoints).
- Model: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (Apache 2.0).
- Curated headlines + forum excerpts: `data/sources/` (drop-in replaceable).
