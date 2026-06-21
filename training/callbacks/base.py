"""Classe base para callbacks de treinamento."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from training.core.trainer import Trainer


class Callback:
    """Interface base para callbacks do loop de treinamento."""

    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        pass

    def on_batch_begin(self, trainer: Trainer, batch: Any) -> None:
        pass

    def on_batch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        pass