"""Application settings using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "stocker"
    allow_origins: list[str] = ["*"]
    port: int = 7860

    model_config = {"env_prefix": "STOCKER_", "case_sensitive": False}


settings = Settings()
