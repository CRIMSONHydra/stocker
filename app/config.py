"""Application settings using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "stocker"
    allow_origins: list[str] = ["*"]
    port: int = 7860

    transaction_cost_rate: float = 0.001
    annual_inflation_rate: float = 0.05
    reward_weight_performance: float = 0.7
    reward_weight_inflation: float = 0.3

    model_config = {"env_prefix": "STOCKER_", "case_sensitive": False}


settings = Settings()
