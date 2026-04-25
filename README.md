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
---

# Stocker - OpenEnv Environment

An RL environment where an AI agent makes stock-trading decisions (buy / sell / hold)
over a sequence of daily market observations and is rewarded for portfolio P&L.

## Tasks

| Task | Difficulty | Steps | Description |
|------|-----------|-------|-------------|
| `task_easy`   | Easy   | 10 | Steady uptrend |
| `task_medium` | Medium | 10 | Volatile sideways market |
| `task_hard`   | Hard   | 10 | Bull-then-bear reversal |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `side` | `buy \| sell \| hold` | Trade direction |
| `quantity` | `int (>= 0)` | Number of shares (ignored if `hold`) |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | `string` | Stock ticker symbol |
| `date` | `string` | Day label (e.g. `day_3`) |
| `price` | `float` | Current price |
| `price_history` | `list[float]` | Prices observed so far |
| `fundamentals` | `dict` | Static facts about the company |
| `cash` | `float` | Current cash balance |
| `position` | `int` | Current number of shares held |
| `portfolio_value` | `float` | `cash + position * price` |
| `task_id` | `string` | Active task |
| `step_number` | `int` | 1-indexed step |
| `total_steps` | `int` | Total steps in the episode |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | inference only | LLM endpoint (default: HuggingFace Router) |
| `MODEL_NAME` | inference only | Model identifier |
| `HF_TOKEN` | inference only | HuggingFace API key |

The server itself requires no API keys.

## Quick Start

```bash
# Install dependencies
pip install uv
uv sync

# Run the server
./run.sh
# Server: http://localhost:7860
# Frontend: http://localhost:7860/web
# Swagger: http://localhost:7860/docs

# Smoke test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:7860/step  -H "Content-Type: application/json" \
  -d '{"side": "buy", "quantity": 10}'

# Run inference
export HF_TOKEN=hf_xxx
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
python inference.py --task all
```

## Inference Script

`inference.py` lives in the project root, uses the OpenAI client, and emits
`[START]`, `[STEP]`, `[END]` log lines required by OpenEnv.

```
[START] task=<task_name> env=stocker model=<model_name>
[STEP]  step=<n> action=<side(qty)> reward=<x.xx> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
```

## Reward

For each step:

```
reward = (new_portfolio - prev_portfolio) / prev_portfolio - invalid_action_penalty
```

A small bonus (`+0.05`) is added on the final step if the agent ends the episode
with at least 1.05× starting capital. Reward is clipped to `[-1.0, 1.0]`.

## Docker

```bash
docker build -t stocker .
docker run -p 7860:7860 stocker
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/web` | GET | Interactive frontend |
| `/health` | GET | Health check |
| `/meta` | GET | Environment metadata |
| `/reset` | POST | Reset environment |
| `/step` | POST | Submit trade action |
| `/state` | GET / POST | Export / restore state |
| `/docs` | GET | Swagger UI |

## Validation

```bash
./validate-submission.sh https://your-space.hf.space .
python scripts/validate_tasks.py
```

## TODO (scaffolding hand-off)

- Replace inline `prices` with realistic OHLCV data (CSV or HF datasets).
- Add more tasks under `tasks/*.json`.
- Tune the reward shaping (transaction costs, risk penalty, Sharpe ratio).
- Wire HF Spaces deployment in `.env` + `push-hf.sh`.
- Add training script (Unsloth / TRL) per OpenEnv submission requirements.
