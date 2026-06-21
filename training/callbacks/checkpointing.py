"""Checkpointing callback."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from training.callbacks.base import Callback

if TYPE_CHECKING:
    from training.core.trainer import Trainer


class CheckpointCallback(Callback):
    """Salva checkpoints periódicos durante treinamento."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.save_every = int(config.get("training", {}).get("save_every_steps", 1000))
        self.max_checkpoints = int(config.get("training", {}).get("max_checkpoints", 5))
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "training/checkpoints"))

    def save_checkpoint(self, trainer: Trainer) -> Path:
        """Salva checkpoint completo."""
        return trainer.checkpoint(self.checkpoint_dir)

    def save_optimizer_state(self, trainer: Trainer, path: Path) -> None:
        """Salva estado do optimizer separadamente."""
        import torch

        if trainer.optimizer_scheduler:
            torch_state = trainer.optimizer_scheduler.optimizer.state_dict()
            torch.save(torch_state, path / "optimizer.bin")

    def save_training_state(self, trainer: Trainer, path: Path) -> None:
        """Salva metadados de treinamento."""
        import json

        state = {
            "global_step": trainer.global_step,
            "current_epoch": trainer.current_epoch,
        }
        with (path / "training_state.json").open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def cleanup_old_checkpoints(self) -> None:
        """Remove checkpoints antigos além do limite."""
        if not self.checkpoint_dir.exists():
            return
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_ckpt in checkpoints[self.max_checkpoints :]:
            shutil.rmtree(old_ckpt.parent, ignore_errors=True) if old_ckpt.parent.is_dir() else old_ckpt.unlink(missing_ok=True)

    def on_batch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if trainer.global_step > 0 and trainer.global_step % self.save_every == 0:
            self.save_checkpoint(trainer)
            self.cleanup_old_checkpoints()