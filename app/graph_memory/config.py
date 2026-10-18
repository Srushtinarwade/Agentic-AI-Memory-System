"""Local-first, intentionally small settings surface."""
from pathlib import Path
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    neo4j_database: str = "neo4j"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.5-flash-lite"
    embedding_model_path: Path = Path("models/all-MiniLM-L6-v2")
    embedding_dimensions: int = 384
    entity_similarity_floor: float = Field(default=0.78, ge=0.0, le=1.0)
    fact_similarity_floor: float = Field(default=0.68, ge=0.0, le=1.0)
    max_facts_per_message: int = Field(default=15, ge=1, le=40)
    auth_token: SecretStr = SecretStr("")  # optional local shared token (NOT per-user auth)
    startup_index_wait_seconds: int = Field(default=90, ge=1, le=600)

    @field_validator("embedding_dimensions")
    @classmethod
    def dim384(cls, value: int) -> int:
        if value != 384:
            raise ValueError("This project requires the 384-dimensional MiniLM model.")
        return value
