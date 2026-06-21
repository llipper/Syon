"""Utilitários de logging."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

from utils.file_utils import load_yaml

DEFAULT_LOG_CONFIG = Path(__file__).resolve().parents[1] / "config" / "logging" / "logging_config.yaml"


def setup_logger(
    name: str = "syon",
    *,
    level: str | int | None = None,
    config_path: str | Path | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configura e retorna um logger do Syon."""
    config_file = Path(config_path) if config_path else DEFAULT_LOG_CONFIG

    if config_file.exists():
        config = load_yaml(config_file)
        if log_file is not None:
            _override_log_file(config, Path(log_file))
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def _override_log_file(config: dict[str, Any], log_file: Path) -> None:
    """Substitui caminho do handler de arquivo na configuração."""
    handlers = config.get("handlers", {})
    for handler in handlers.values():
        if handler.get("class", "").endswith("FileHandler") or handler.get("class", "").endswith(
            "RotatingFileHandler"
        ):
            handler["filename"] = str(log_file)