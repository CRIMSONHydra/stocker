"""Smoke tests for StockerEnv."""

from app.core.environment import StockerEnv
from app.core.tasks import list_task_ids


def test_env_reset_and_step_each_task():
    for task_id in list_task_ids():
        env = StockerEnv(task_id=task_id)
        reset = env.reset()
        assert reset.observation.task_id == task_id
        assert reset.observation.step_number == 1

        result = env.step({"side": "hold", "quantity": 0})
        assert -1.0 <= result.reward <= 1.0
        assert result.observation.step_number >= 2 or result.done


def test_env_buy_then_hold_runs_to_completion():
    env = StockerEnv(task_id="task_easy")
    env.reset()
    env.step({"side": "buy", "quantity": 10})
    done = False
    steps = 0
    while not done and steps < 50:
        r = env.step({"side": "hold", "quantity": 0})
        done = r.done
        steps += 1
    assert done


def test_invalid_buy_does_not_change_position():
    env = StockerEnv(task_id="task_easy")
    env.reset()
    r = env.step({"side": "buy", "quantity": 10**9})
    assert r.info["position"] == 0
