"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_model: Optional[str] = "mistral"
    app_env: str = "development"
    log_level: str = "INFO"
    chroma_path: str = "./data/chroma_db"
    mock_mode: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get or create cached settings instance."""
    return Settings()