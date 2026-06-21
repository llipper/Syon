"""Validation callback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from training.callbacks.base import Callback

if TYPE_CHECKING:
    from training.core.trainer import Trainer


class ValidationCallback(Callback):
    """Executa validação periódica e early stopping."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.eval_every = int(config.get("training", {}).get("eval_every_steps", 500))
        self.early_stopping_patience = int(config.get("training", {}).get("early_stopping_patience", 5))
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.log_dir = Path(config.get("log_dir", "training/logs"))

    def evaluate_on_validation(self, trainer: Trainer, val_dataloader) -> dict[str, float]:
        """Avalia modelo no conjunto de validação."""
        metrics: dict[str, list[float]] = {"val_loss": [], "val_perplexity": []}
        for batch in val_dataloader:
            step_metrics = trainer.validation_step(batch)
            for key, value in step_metrics.items():
                metrics.setdefault(key, []).append(value)
        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def early_stopping_check(self, val_loss: float) -> bool:
        """Retorna True se treinamento deve parar."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.early_stopping_patience

    def log_val_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Persiste métricas de validação."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "validation_metrics.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **metrics}) + "\n")

    def on_batch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if trainer.global_step > 0 and trainer.global_step % self.eval_every == 0:
            self.log_val_metrics(trainer.global_step, {"train_loss": metrics.get("total", 0)})