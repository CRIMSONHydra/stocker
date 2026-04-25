"""Council layer tests using the deterministic MockLLMClient."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.council.llm import MockLLMClient
from app.council.runner import CACHE_ROOT, Council
from app.council.specialists import SPECIALISTS
from app.core.environment import StockerEnv


@pytest.fixture(autouse=True)
def _clean_cache(tmp_path, monkeypatch):
    # Use a per-test cache dir so previous runs don't pollute.
    new_cache = tmp_path / "council"
    monkeypatch.setattr("app.council.runner.CACHE_ROOT", new_cache)
    yield
    if new_cache.exists():
        shutil.rmtree(new_cache, ignore_errors=True)


def test_seven_specialists_registered():
    assert len(SPECIALISTS) == 7
    names = [c.name for c in [cls(MockLLMClient()) for cls in SPECIALISTS]]
    assert sorted(names) == sorted([
        "chart_pattern", "seasonal_trend", "indicator", "news",
        "forum_sentiment", "peer_commodity", "geopolitics",
    ])


def test_council_produces_valid_decision():
    env = StockerEnv("task_easy")
    obs = env.reset().observation

    council = Council(client=MockLLMClient(), use_cache=False)
    decision = council.run(obs)

    assert len(decision.votes) == 7
    for v in decision.votes:
        assert -1.0 <= v.signal <= 1.0
        assert 0.0 <= v.confidence <= 1.0
        assert v.rationale
    assert decision.action.side in ("buy", "sell", "hold")
    assert decision.action.quantity >= 0


def test_council_cache_round_trips(tmp_path):
    env = StockerEnv("task_easy")
    obs = env.reset().observation

    council = Council(client=MockLLMClient(), use_cache=True)
    d1 = council.run(obs)
    d2 = council.run(obs)
    # Same inputs -> identical decision
    assert d1.action.side == d2.action.side
    assert d1.action.quantity == d2.action.quantity
    assert [v.signal for v in d1.votes] == [v.signal for v in d2.votes]


def test_council_drives_env_to_completion():
    env = StockerEnv("task_easy")
    obs = env.reset().observation
    council = Council(client=MockLLMClient(), use_cache=False)

    done = False
    steps = 0
    while not done and steps < 100:
        decision = council.run(obs)
        result = env.step(decision.action)
        obs = result.observation
        done = result.done
        steps += 1

    assert done
    assert steps == env.state().total_steps
