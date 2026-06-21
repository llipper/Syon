"""Configurações principais do Syon via variáveis de ambiente."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent


class TrainingSettings(BaseSettings):
    """Configurações de treinamento carregadas de variáveis de ambiente."""

    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / "secrets" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_path: Path = Field(default=PROJECT_ROOT / "data", validation_alias="DATA_PATH")
    raw_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "raw",
        validation_alias="RAW_DATA_PATH",
    )
    processed_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed",
        validation_alias="PROCESSED_DATA_PATH",
    )
    model_name: str = Field(default="syon-7b", validation_alias="MODEL_NAME")
    model_path: Path = Field(default=PROJECT_ROOT / "models", validation_alias="MODEL_PATH")
    checkpoint_path: Path = Field(
        default=PROJECT_ROOT / "training" / "checkpoints",
        validation_alias="CHECKPOINT_PATH",
    )
    batch_size: int = Field(default=32, validation_alias="BATCH_SIZE")
    learning_rate: float = Field(default=2e-4, validation_alias="LEARNING_RATE")
    num_epochs: int = Field(default=3, validation_alias="NUM_EPOCHS")
    warmup_steps: int = Field(default=2000, validation_alias="WARMUP_STEPS")
    world_size: int = Field(default=1, validation_alias="WORLD_SIZE")
    rank: int = Field(default=0, validation_alias="RANK")
    master_addr: str = Field(default="localhost", validation_alias="MASTER_ADDR")
    master_port: int = Field(default=29500, validation_alias="MASTER_PORT")


class InferenceSettings(BaseSettings):
    """Configurações de inferência."""

    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / "secrets" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    inference_batch_size: int = Field(default=8, validation_alias="INFERENCE_BATCH_SIZE")
    max_token_length: int = Field(default=4096, validation_alias="MAX_TOKEN_LENGTH")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        validation_alias="INFERENCE_DEVICE",
    )
    default_temperature: float = Field(default=0.7, validation_alias="INFERENCE_TEMPERATURE")
    default_top_p: float = Field(default=0.95, validation_alias="INFERENCE_TOP_P")


class APISettings(BaseSettings):
    """Configurações da API REST."""

    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / "secrets" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    api_workers: int = Field(default=4, validation_alias="API_WORKERS")
    cors_origins: str = Field(default="*", validation_alias="API_CORS_ORIGINS")
    max_body_size_mb: int = Field(default=16, validation_alias="API_MAX_BODY_SIZE_MB")
    request_timeout_seconds: int = Field(
        default=120,
        validation_alias="API_REQUEST_TIMEOUT",
    )


class SecuritySettings(BaseSettings):
    """Configurações de segurança e autenticação."""

    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / "secrets" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    secret_key: str = Field(default="change-me-in-production", validation_alias="SECRET_KEY")
    algorithm: str = Field(default="HS256", validation_alias="ALGORITHM")
    api_key_header: str = Field(default="X-API-Key", validation_alias="API_KEY_HEADER")
    rate_limit_per_minute: int = Field(default=60, validation_alias="RATE_LIMIT_PER_MINUTE")
    enable_input_sanitization: bool = Field(
        default=True,
        validation_alias="ENABLE_INPUT_SANITIZATION",
    )


class LoggingSettings(BaseSettings):
    """Configurações de logging."""

    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / "secrets" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_path: Path = Field(default=PROJECT_ROOT / "logs", validation_alias="LOG_PATH")
    environment: str = Field(default="development", validation_alias="SYON_ENV")


class Settings(BaseSettings):
    """Configuração agregada do Syon (merged)."""

    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / "secrets" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    training: TrainingSettings = Field(default_factory=TrainingSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    api: APISettings = Field(default_factory=APISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @property
    def is_production(self) -> bool:
        return self.logging.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.logging.environment.lower() == "development"

    def environment_config_path(self) -> Path:
        env_name = self.logging.environment.lower()
        return CONFIG_DIR / "environments" / f"{env_name}.yaml"


@lru_cache
def get_settings() -> Settings:
    """Retorna instância singleton das configurações."""
    return Settings()