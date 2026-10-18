"""Local-first environment settings for PostgreSQL and optional Redis."""
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    postgres_dsn: SecretStr = SecretStr("")
    pg_pool_min_size: int = Field(default=1, ge=1, le=50)
    pg_pool_max_size: int = Field(default=10, ge=1, le=100)
    redis_url: SecretStr = SecretStr("")  # empty disables Redis completely
    redis_cache_ttl_seconds: int = Field(default=60, ge=1, le=3600)
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.5-flash-lite"
    embedding_model_path: Path = Path("models/all-MiniLM-L6-v2")
    embedding_dimensions: int = 384
    entity_similarity_floor: float = Field(default=0.78, ge=0.0, le=1.0)
    fact_similarity_floor: float = Field(default=0.68, ge=0.0, le=1.0)
    max_facts_per_message: int = Field(default=15, ge=1, le=40)
    auth_token: SecretStr = SecretStr("")  # optional local shared token, not per-user auth

    @field_validator("embedding_dimensions")
    @classmethod
    def dim384(cls, value: int) -> int:
        if value != 384:
            raise ValueError("This project requires the 384-dimensional MiniLM model.")
        return value

    @model_validator(mode="after")
    def pool_limits(self):
        if self.pg_pool_min_size > self.pg_pool_max_size:
            raise ValueError("PG_POOL_MIN_SIZE cannot exceed PG_POOL_MAX_SIZE")
        return self
