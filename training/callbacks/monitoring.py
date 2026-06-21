"""Monitoring callback."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from training.callbacks.base import Callback
from training.core.loss_functions import perplexity_calculation

if TYPE_CHECKING:
    from training.core.trainer import Trainer

logger = logging.getLogger("syon.training")


class MetricsCallback(Callback):
    """Loga métricas de treinamento."""

    def __init__(self, log_dir: Path | None = None):
        self.log_dir = log_dir or Path("training/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._batch_start: float = 0.0
        self._tokens_processed = 0

    def log_loss(self, metrics: dict[str, float]) -> None:
        logger.info("loss=%.4f lr=%.2e", metrics.get("total", 0), metrics.get("lr", 0))

    def log_perplexity(self, loss_value: float) -> None:
        ppl = perplexity_calculation(__import__("torch").tensor(loss_value))
        logger.info("perplexity=%.2f", ppl)

    def log_throughput(self, tokens: int, elapsed: float) -> None:
        if elapsed > 0:
            logger.info("throughput=%.1f tokens/s", tokens / elapsed)

    def log_resource_usage(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info("gpu_memory=%.2f GB", allocated)
        except Exception:
            pass

    def on_batch_begin(self, trainer: Trainer, batch: Any) -> None:
        self._batch_start = time.perf_counter()

    def on_batch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        elapsed = time.perf_counter() - self._batch_start
        self.log_loss(metrics)
        if "perplexity" in metrics:
            logger.info("perplexity=%.2f", metrics["perplexity"])
        self.log_throughput(batch_size := 1, elapsed)
        self.log_resource_usage()

        log_file = self.log_dir / f"training_log_step_{trainer.global_step}.json"
        with log_file.open("w", encoding="utf-8") as handle:
            json.dump({"step": trainer.global_step, "metrics": metrics, "batch_size": batch_size}, handle)