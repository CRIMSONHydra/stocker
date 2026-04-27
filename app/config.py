"""Application settings using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "stocker"
    allow_origins: list[str] = ["*"]
    port: int = 7860

    # Reward shaping. Bumped weights + inflation rate to push the moderator
    # toward actually trading instead of converging to "always hold" — see
    # docstring in app/core/graders.py for the full design.
    transaction_cost_rate: float = 0.001
    annual_inflation_rate: float = 0.10        # was 0.05 — sharper "cash decays" signal
    reward_weight_performance: float = 0.8     # was 0.7
    reward_weight_inflation: float = 0.5       # was 0.3 — aggressively penalize sitting in cash

    # Lookahead bonus: at training step N, reward proportional to the price
    # change between step N and N+lookahead_steps, times the agent's net
    # exposure delta. Gives the single-step reward a directional signal.
    lookahead_steps: int = 5
    lookahead_weight: float = 0.5              # 0 disables the term entirely

    # Multi-step rollout horizon used by training/train_grpo.reward_for_completion.
    # After applying the model's action, the env is rolled forward this many
    # steps with `hold` and the cumulative reward is what the agent sees.
    rollout_horizon: int = 5

    model_config = {"env_prefix": "STOCKER_", "case_sensitive": False}


settings = Settings()
