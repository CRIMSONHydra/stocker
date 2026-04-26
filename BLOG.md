---
title: "Teaching a Multi-Agent Council to Trade Stocks with GRPO"
thumbnail: /blog/assets/stocker/council_diagram.png
authors:
  - user: naverdo
---

# Teaching a Multi-Agent Council to Trade Stocks with GRPO

> **TL;DR** — We built a seven-specialist multi-agent council where seven
> frozen LLM analysts vote each trading step and a trainable moderator merges
> their votes into a trade. The moderator is fine-tuned with GRPO using the
> RL environment's own reward as the signal. This post covers the full
> pipeline: environment design, reward function, specialist architecture,
> training setup, and results on the AAPL Aug–Sep 2023 episode.

---

## Motivation

Large language models can read a candlestick chart, reason about macroeconomics,
and parse earnings headlines. Can they trade? And if we give one LLM seven
specialized perspectives — each looking at the same market step through a
different lens — can we then *train* the moderator that merges those perspectives
using reinforcement learning?

That's Stocker. It's an [OpenEnv](https://github.com/huggingface/openenv)-compatible
RL environment built around a multi-agent council. The seven specialists are
**frozen** (their cached votes are free to reuse across every training step).
Only the **moderator** is fine-tuned, via GRPO on top of
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it).

---

## Environment design

### The single-agent contract

OpenEnv expects a single-agent loop: `POST /reset` → observe → `POST /step` →
reward. The council is entirely outside `step()` — in `inference.py` and
`app/council/`. This preserves OpenEnv compatibility: `openenv validate` passes,
graders can replay the exact sequence of actions.

### Episodes

Three deterministic episodes from real OHLCV data (yfinance):

| Task | Ticker | Window | Regime |
|------|--------|--------|--------|
| `task_easy` | AAPL | Aug–Sep 2023 | Uptrend into iPhone 15 launch |
| `task_medium` | INTC | Jan–Feb 2024 | Choppy post-earnings |
| `task_hard` | META | Sep–Oct 2022 | Drawdown then snap-back after Q3 |

Each is 41–43 trading days. All OHLCV, indicators, headlines, forum excerpts,
macro events, and 768×768 candlestick PNGs are versioned in the repo (~6.2 MB).

### Action space

```json
{"side": "buy" | "sell" | "hold", "quantity": int}
```

`quantity` is silently clamped to what cash or position allows. Invalid actions
(e.g., sell when flat) are detected at step time and carry a small penalty.

---

## The seven specialists

Every step, seven agents run in **parallel threads** (no asyncio — Jupyter runs
a live event loop) and each returns:

```json
{"signal": float,     // [-1.0 = strong sell, +1.0 = strong buy]
 "confidence": float, // [0.0, 1.0]
 "rationale": str}    // free-form explanation
```

| Agent | Input modality | What it looks at |
|-------|----------------|-----------------|
| `ChartPattern` | Image (candlestick PNG) | Visual patterns, breakouts, support/resistance |
| `SeasonalTrend` | Text | Long-term cycles, seasonality, fundamentals |
| `Indicator` | Text | RSI, MACD, SMA-50/200, Bollinger Bands, ATR |
| `News` | Text | Curated headlines in a ±7-day window |
| `ForumSentiment` | Text | Reddit excerpts, crowd sentiment |
| `PeerCommodity` | Text | Peer stocks, gold/oil correlation |
| `Geopolitics` | Text | CPI prints, FOMC decisions, macro events |

`ChartPattern` uses Gemma's multimodal path — the PNG is encoded as a data URL
and passed as an `image_url` content part. The other six are text-only.

### Caching

Specialist votes are cached to disk at
`.cache/council/<role>/base/<ticker>__<date>.json`. Since the specialists are
frozen, cache hits are instant. For the AAPL Aug–Sep episode (43 steps × 7
specialists = 301 entries), the first fill takes ~5–10 minutes on a T4; every
subsequent GRPO step reads from cache in milliseconds.

---

## Reward function

The reward function evolved through several iterations. The final design uses
**two components** and an end-of-episode bonus:

### Performance factor

We pre-compute the *ideal PnL trajectory* for each task using a hindsight
greedy strategy (`scripts/build_ideal_profit.py`). At each step, the moderator's
inflation-adjusted PnL is compared to this ideal via an asymmetric piecewise-
linear function:

```python
gap   = ideal_pnl_pct[step] - real_pnl_pct
scale = max(0.05, 0.5 * abs(ideal_pnl_pct_total))

if gap < 0:           # outperformed ideal
    perf = 1.0 + min(1.0, abs(gap) / scale)   # up to +2.0 (clipped by env)
elif gap <= scale:    # close to ideal
    perf = 1.0 - gap / scale                   # [0, 1]
else:                 # far behind
    perf = -min(1.0, (gap - scale) / scale)    # [-1, 0]
```

The asymmetry rewards exceeding the ideal, not just matching it.

### Inflation factor

Nominal PnL isn't real return. We deflate by elapsed years in the episode:

```python
real_pnl_pct    = (1 + nominal_pnl_pct) / (1 + inflation_growth) - 1
inflation_factor = real_pnl_pct - nominal_pnl_pct   # ≤ 0
```

This creates a small but persistent pressure to act quickly rather than
holding cash and watching inflation erode it.

### Combined reward

```
reward_step = W_PERF × perf_factor + W_INFL × inflation_factor − invalid_penalty
```

Default weights: `W_PERF=0.7`, `W_INFL=0.3`. All four hyperparameters
(`W_PERF`, `W_INFL`, `annual_inflation_rate`, `transaction_cost_rate`) are
sweepable without re-running the LLM — the `tune_easy_gemma4.ipynb` replay
grader re-grades the saved trajectory under all weight combinations in under
one second.

### Trajectory bonus (final step)

```
alpha_bonus = min(0.10, max(0, (final_portfolio / buy_and_hold) − 1))
dd_penalty  = min(0.10, max(0, max_drawdown − 0.05))
```

All rewards are clipped to `[-1.0, 1.0]` at the env boundary.

---

## Reward-weight tuning

Before training we ran the `tune_easy_gemma4.ipynb` tuning bench to understand
the reward landscape:

- Loaded Gemma 4 E4B IT in 4-bit on a Colab T4.
- Ran one full episode of `task_easy` (43 steps), capturing the complete
  reward breakdown per step (nominal PnL, real PnL, inflation drag, gap to
  ideal, performance factor, weighted sums).
- Saved the trace to `training/runs/easy_gemma4_<ts>/trace.jsonl`.
- **Re-graded the trace** under a 180-combo sweep of `(W_PERF, W_INFL,
  annual_inflation_rate, transaction_cost_rate)` — pure CPU, ~0.3 seconds
  total.

Key finding: the replay-vs-env reward matched to within 5×10⁻⁶ (assertion:
< 1×10⁻⁵), confirming the grader faithfully mirrors the env.

Sweep results (selected):

| W_PERF | W_INFL | infl_rate | tc_rate | Total reward |
|--------|--------|-----------|---------|-------------|
| 1.0 | 0.0 | 0.05 | 0.000 | +0.5725 |
| 0.7 | 0.3 | 0.05 | 0.001 | +0.3801 *(default)* |
| 0.5 | 0.3 | 0.05 | 0.001 | +0.2680 |

We trained with the defaults (`W_PERF=0.7`, `W_INFL=0.3`) to measure the
effect of the inflation signal. The sweep findings are stored in
`training/runs/easy_gemma4_<ts>/chosen_settings.json` for the next run.

---

## GRPO training

### Setup

The GRPOTrainer from TRL wraps our reward function. For each training step:
1. Load `per_device_train_batch_size=4` moderator prompts from the dataset.
2. Sample `num_generations=4` moderator completions per prompt (16 total).
3. Parse each completion into a `TradeAction`.
4. For each action, reload the env to that snapshot and take one step → reward.
5. Update the LoRA adapter using the GRPO policy gradient estimate.

```python
grpo_cfg = GRPOConfig(
    per_device_train_batch_size=4,  # (4 * 1) % 4 == 0 ✓
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    num_train_epochs=3,
    num_generations=4,
    bf16=False, fp16=True,          # T4 has no bf16
)
lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
```

### Key implementation details

- **ThreadPoolExecutor, not `asyncio.run`** — Jupyter runs a live event loop,
  so `asyncio.run()` raises `RuntimeError`. We use `ThreadPoolExecutor.map()`
  for parallel specialist calls.
- **`device_map={"":0}` + `bnb_4bit_use_double_quant=True`** — forces the
  full 4-bit model onto GPU 0; `device_map="auto"` silently splits layers to
  CPU and breaks BnB validation.
- **fp16 on T4** — Colab T4 doesn't support bf16; we auto-detect with
  `torch.cuda.is_bf16_supported()`.
- **`torchao>=0.16`** — PEFT ≥0.13 probes torchao at LoRA injection; Colab
  ships 0.10 which fails with a version mismatch. Explicit upgrade in the
  install cell.
- **Divisibility constraint** — TRL ≥0.13 requires
  `(per_device_train_batch_size × grad_accum) % num_generations == 0`.
  Our 4 × 1 ÷ 4 == 0 satisfies this.

---

## Results

See `training/runs/RESULTS.md` (generated by `scripts/compile_results.py`
after training) for the full pre vs. post comparison table, training curves,
and portfolio plots.

The trained adapter ships in `training/runs/grpo_<timestamp>/moderator-lora/`
and can be loaded via:

```python
from peft import PeftModel
client.model = PeftModel.from_pretrained(
    client.model, "training/runs/grpo_.../moderator-lora", adapter_name="moderator"
)
client.moderator_lora = "moderator"
```

---

## Reproducing

```bash
git clone https://github.com/CRIMSONHydra/stocker.git && cd stocker

# CPU smoke test — no GPU, no API key needed
pip install uv && uv pip install -e ".[dev,data,eval]"
python scripts/build_dataset.py
python scripts/build_ideal_profit.py
pytest tests/ -q

# Full Colab run (T4)
# Open training/train_grpo.ipynb → Runtime → T4 GPU → Run All
```

---

## What's next

- **Multi-step rollouts** — current GRPO trains on 1-step rewards; sequence-
  level shaping would let the moderator learn to hold across multiple steps.
- **Live corpus** — swap curated JSON for a streaming HF dataset of dated
  headlines (e.g., a GDELT slice).
- **More tasks** — the corpus path (`data/corpus/`) exposes 4 000+ episodes;
  training on a curriculum of regimes should improve generalization.
- **Specialist fine-tuning** — once the moderator is stable, unfreeze
  individual specialists with a shared reward signal.

---

## Resources

- **GitHub:** [CRIMSONHydra/stocker](https://github.com/CRIMSONHydra/stocker)
- **Model:** [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- **Framework:** [OpenEnv](https://github.com/huggingface/openenv) · [TRL](https://github.com/huggingface/trl) · [PEFT](https://github.com/huggingface/peft)
- **Training notebook:** [training/train_grpo.ipynb](training/train_grpo.ipynb)
- **Tuning notebook:** [training/tune_easy_gemma4.ipynb](training/tune_easy_gemma4.ipynb)
