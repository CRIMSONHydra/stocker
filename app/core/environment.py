"""StockerEnv — the Stocker OpenEnv environment."""

import uuid

from app.core.graders import compute_step_reward, compute_trajectory_bonus
from app.core.tasks import get_task_definition
from app.models import (
    EnvironmentState,
    MarketObservation,
    ResetResult,
    StepResult,
    TradeAction,
)


class StockerEnv:
    """Stock-trading RL environment."""

    def __init__(self, task_id: str = "task_easy"):
        self.task_id = task_id
        self._task: dict = {}
        self._prices: list[float] = []
        self._current_index: int = 0
        self._done: bool = True
        self._cash: float = 0.0
        self._position: int = 0
        self._action_history: list[dict] = []
        self._reward_history: list[float] = []
        self._episode_id: str = ""

    # ------------------------------------------------------------------ reset
    def reset(self, task_id: str | None = None) -> ResetResult:
        if task_id is not None:
            self.task_id = task_id

        self._task = get_task_definition(self.task_id)
        self._prices = list(self._task["prices"])
        self._current_index = 0
        self._done = False
        self._cash = float(self._task.get("starting_cash", 10000.0))
        self._position = 0
        self._action_history = []
        self._reward_history = []
        self._episode_id = str(uuid.uuid4())[:8]

        return ResetResult(
            observation=self._build_observation(0),
            info={
                "task_id": self.task_id,
                "episode_id": self._episode_id,
                "total_steps": len(self._prices),
                "description": self._task.get("description", ""),
                "starting_cash": self._cash,
            },
        )

    # ------------------------------------------------------------------- step
    def step(self, action: TradeAction | dict) -> StepResult:
        if isinstance(action, dict):
            try:
                action = TradeAction.model_validate(action)
            except Exception as e:
                return StepResult(
                    observation=self._build_observation(self._current_index),
                    reward=0.0,
                    done=self._done,
                    info={"error": f"Invalid action: {e}"},
                )

        if self._done:
            return StepResult(
                observation=self._terminal_observation(),
                reward=0.0,
                done=True,
                info={"message": "Episode already finished. Call reset()."},
            )

        price = self._prices[self._current_index]
        prev_portfolio = self._cash + self._position * price
        invalid = self._apply_action(action, price)
        new_portfolio = self._cash + self._position * price

        result = compute_step_reward(
            action=action,
            prev_portfolio=prev_portfolio,
            new_portfolio=new_portfolio,
            starting_cash=float(self._task.get("starting_cash", 10000.0)),
            invalid=invalid,
        )
        reward = result.score

        self._action_history.append(action.model_dump())
        self._reward_history.append(reward)

        self._current_index += 1
        if self._current_index >= len(self._prices):
            self._done = True
            final_price = self._prices[-1]
            final_portfolio = self._cash + self._position * final_price
            reward += compute_trajectory_bonus(
                final_portfolio, float(self._task.get("starting_cash", 10000.0))
            )

        reward = max(-1.0, min(1.0, reward))

        next_obs = (
            self._terminal_observation() if self._done
            else self._build_observation(self._current_index)
        )

        return StepResult(
            observation=next_obs,
            reward=round(reward, 5),
            done=self._done,
            info={
                "trade_feedback": result.feedback,
                "reward_breakdown": result.breakdown,
                "portfolio_value": round(new_portfolio, 4),
                "cash": round(self._cash, 4),
                "position": self._position,
            },
        )

    # ------------------------------------------------------------------ state
    def state(self) -> EnvironmentState:
        price = (
            self._prices[self._current_index]
            if self._current_index < len(self._prices)
            else (self._prices[-1] if self._prices else 0.0)
        )
        return EnvironmentState(
            task_id=self.task_id,
            current_step=self._current_index,
            total_steps=len(self._prices),
            done=self._done,
            cash=round(self._cash, 4),
            position=self._position,
            portfolio_value=round(self._cash + self._position * price, 4),
            action_history=self._action_history,
            reward_history=self._reward_history,
        )

    def load_snapshot(self, snapshot: EnvironmentState) -> None:
        self._task = get_task_definition(snapshot.task_id)
        self._prices = list(self._task["prices"])
        self._current_index = snapshot.current_step
        self._done = snapshot.done
        self._cash = snapshot.cash
        self._position = snapshot.position
        self._action_history = snapshot.action_history
        self._reward_history = snapshot.reward_history

    # ---------------------------------------------------------------- helpers
    def _apply_action(self, action: TradeAction, price: float) -> bool:
        """Apply trade. Returns True if the action was invalid (insufficient
        cash/position) — the action is then treated as a hold."""
        if action.side == "hold" or action.quantity <= 0:
            return False

        if action.side == "buy":
            cost = action.quantity * price
            if cost > self._cash:
                return True
            self._cash -= cost
            self._position += action.quantity
            return False

        if action.side == "sell":
            if action.quantity > self._position:
                return True
            self._cash += action.quantity * price
            self._position -= action.quantity
            return False

        return True

    def _build_observation(self, index: int) -> MarketObservation:
        price = self._prices[index]
        return MarketObservation(
            ticker=self._task["ticker"],
            date=f"day_{index + 1}",
            price=price,
            price_history=self._prices[: index + 1],
            fundamentals=self._task.get("fundamentals", {}),
            cash=round(self._cash, 4),
            position=self._position,
            portfolio_value=round(self._cash + self._position * price, 4),
            task_id=self.task_id,
            step_number=index + 1,
            total_steps=len(self._prices),
        )

    def _terminal_observation(self) -> MarketObservation:
        final_price = self._prices[-1] if self._prices else 0.0
        portfolio = self._cash + self._position * final_price
        return MarketObservation(
            ticker=self._task.get("ticker", ""),
            date="[EPISODE COMPLETE]",
            price=final_price,
            price_history=self._prices,
            fundamentals=self._task.get("fundamentals", {}),
            cash=round(self._cash, 4),
            position=self._position,
            portfolio_value=round(portfolio, 4),
            task_id=self.task_id,
            step_number=len(self._prices),
            total_steps=len(self._prices),
        )
