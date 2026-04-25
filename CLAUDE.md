# CLAUDE.md

## Project Overview
**Stocker** — a stock-trading RL environment built on top of OpenEnv. The agent
sees daily market observations and must produce a `(side, quantity)` trade.

## Stack
- **Language:** Python 3.10+ (Docker image runs 3.13)
- **Framework:** FastAPI + Uvicorn
- **Models:** Pydantic v2 + pydantic-settings
- **OpenEnv:** `openenv-core>=0.2.0`
- **Package manager:** `uv`
- **Tests:** pytest + FastAPI TestClient

## Layout
```
.
├── app/                # FastAPI application
│   ├── api/            # HTTP routers (health, meta, env, state, frontend)
│   ├── core/           # environment.py, graders.py, tasks.py
│   ├── config.py
│   ├── main.py
│   └── models.py       # Pydantic schemas
├── server/app.py       # OpenEnv entry point (server.app:main)
├── tasks/              # JSON task definitions (loaded at import)
├── tests/              # pytest suite
├── scripts/            # helper CLIs
├── inference.py        # LLM rollout script (root, OpenEnv requirement)
├── client.py           # Python HTTP client
├── Dockerfile          # builds via uv
├── openenv.yaml        # OpenEnv spec
└── run.sh              # local dev server
```

## Conventions
- All Pydantic models live in [app/models.py](app/models.py) — keep them
  named: `MarketObservation`, `TradeAction`, `RewardResult`, `StepResult`,
  `ResetResult`, `EnvironmentState`.
- `StockerEnv` exposes `reset() / step() / state() / load_snapshot()`.
- `inference.py` MUST keep the `[START] / [STEP] / [END]` stdout format —
  graders parse it.
- Tasks are inline in [app/core/tasks.py](app/core/tasks.py) plus optional
  JSON files in [tasks/](tasks/) (auto-loaded at import).
- Reward is always clipped to `[-1.0, 1.0]` at the boundary.
- The server runs on port `7860` (HF Spaces convention).

## Don't
- Don't introduce a frontend build step — the HTML lives inline in
  [app/api/frontend.py](app/api/frontend.py).
- Don't bundle large data files in the Docker image; pull at runtime if
  needed.
- Don't break the OpenAI-client pattern in `inference.py` — judges rerun it.
