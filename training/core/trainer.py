"""Trainer principal — migrado de training/parallel_trainer.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from syon.config import CONFIGS_DIR
from training.callbacks.base import Callback
from training.callbacks.checkpointing import CheckpointCallback
from training.callbacks.monitoring import MetricsCallback
from training.callbacks.validation import ValidationCallback
from training.core.distributed import DistributedTrainer
from training.core.gradient_accumulation import accumulate_gradients, effective_batch_size
from training.core.loss_functions import ContrastiveLoss, SecurityAwareLoss, custom_loss_weighted, perplexity_calculation
from training.core.mixed_precision import MixedPrecisionTrainer
from training.core.optimization import AdamWScheduler
from training.data_pipeline.dataloader import CodeDataset, SecurityDataset, create_dataloader
from training.dataset.composition import load_composition
from training.dataset.curator import DatasetCurator
from training.distillation import BidirectionalDistillation
from training.utils import load_checkpoint, save_checkpoint


def load_training_config(path: Path | None = None) -> dict[str, Any]:
    """Carrega configuração de treinamento."""
    if path and path.exists():
        config_path = path
    else:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "base_config.yaml"
        if not config_path.exists():
            config_path = CONFIGS_DIR / "training" / "parallel_sft.yaml"
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class Trainer:
    """
    Orchestrator principal de treinamento Syon.
    Suporta train_epoch, validation_step, checkpoint e resume.
    """

    def __init__(self, config: dict[str, Any], data_dir: Path):
        self.config = config
        self.data_dir = Path(data_dir)
        self.composition = load_composition()
        self.curator = DatasetCurator(self.data_dir, self.composition)

        loss_cfg = config.get("loss", {})
        self.security_loss = SecurityAwareLoss(
            weight=float(loss_cfg.get("security_aware_weight", 0.15))
        )
        self.contrastive_loss = ContrastiveLoss(
            weight=float(loss_cfg.get("contrastive_weight", 0.1))
        )
        self.distillation = BidirectionalDistillation()

        self._student_model = None
        self._teacher_model = None
        self.optimizer_scheduler: AdamWScheduler | None = None
        self.mixed_precision = MixedPrecisionTrainer(
            enabled=config.get("training", {}).get("mixed_precision", True)
        )
        self.distributed: DistributedTrainer | None = None

        self.global_step = 0
        self.current_epoch = 0
        self.callbacks: list[Callback] = [
            MetricsCallback(),
            CheckpointCallback(config),
            ValidationCallback(config),
        ]

    def prepare_data(self) -> list:
        """Prepara e cura dataset de treinamento."""
        samples = self.curator.curate_all()
        if not samples:
            raise FileNotFoundError(
                f"Nenhum dado encontrado em {self.data_dir}. "
                "Popule data/raw/<domain>/<source>.jsonl conforme composition.yaml"
            )
        return samples

    def setup_models(self, student_model, teacher_model) -> None:
        """Configura modelos student/teacher e optimizer."""
        self._student_model = student_model
        self._teacher_model = teacher_model
        training_cfg = self.config.get("training", {})
        total_steps = int(training_cfg.get("max_steps", 100000))
        self.optimizer_scheduler = AdamWScheduler(student_model, self.config, total_steps)

        if training_cfg.get("distributed", False):
            self.distributed = DistributedTrainer(student_model, self.config)

    def train_step(self, batch) -> dict[str, float]:
        """Executa um step de treinamento."""
        if self._student_model is None or self._teacher_model is None:
            raise RuntimeError("Modelos não configurados. Chame setup_models() primeiro.")

        student_out = self.mixed_precision.forward_pass(self._student_model, batch)
        with torch.no_grad():
            teacher_out = self._teacher_model(**batch)

        distill_loss = self.distillation(student_out.logits, teacher_out.logits, batch["labels"])
        security_loss = self.security_loss(student_out.logits, batch["labels"])
        contrastive_loss = self.contrastive_loss(
            student_out.hidden_states[-1].mean(dim=1),
            teacher_out.hidden_states[-1].mean(dim=1),
        )
        total = custom_loss_weighted(
            {"distillation": distill_loss, "security": security_loss, "contrastive": contrastive_loss},
            {"distillation": 1.0, "security": 1.0, "contrastive": 1.0},
        )

        assert self.optimizer_scheduler is not None
        accumulation_steps = int(self.config.get("training", {}).get("gradient_accumulation_steps", 1))

        scaled_loss = total / accumulation_steps
        self.mixed_precision.backward_pass(scaled_loss)
        if self.distributed:
            self.distributed.sync_gradients()

        if accumulate_gradients(self.global_step, accumulation_steps):
            self.mixed_precision.step_optimizer(self.optimizer_scheduler.optimizer)
            self.optimizer_scheduler.optimizer.zero_grad()
            self.optimizer_scheduler.step()

        metrics = {
            "total": float(total.item()),
            "distillation": float(distill_loss.item()),
            "security_aware": float(security_loss.item()),
            "contrastive": float(contrastive_loss.item()),
            "perplexity": perplexity_calculation(total),
            "lr": self.optimizer_scheduler.get_lr(),
        }
        self.global_step += 1
        return metrics

    def train_epoch(self, dataloader) -> dict[str, float]:
        """Loop de treinamento por epoch."""
        if self._student_model is None:
            raise RuntimeError("Modelo não configurado.")

        self._student_model.train()
        epoch_metrics: dict[str, list[float]] = {}

        for callback in self.callbacks:
            callback.on_epoch_begin(self, self.current_epoch)

        for batch in dataloader:
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch)

            step_metrics = self.train_step(batch)

            for key, value in step_metrics.items():
                epoch_metrics.setdefault(key, []).append(value)

            for callback in self.callbacks:
                callback.on_batch_end(self, step_metrics)

        aggregated = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        self.current_epoch += 1

        for callback in self.callbacks:
            callback.on_epoch_end(self, aggregated)

        return aggregated

    def validation_step(self, batch) -> dict[str, float]:
        """Executa validação em um batch."""
        if self._student_model is None:
            raise RuntimeError("Modelo não configurado.")

        self._student_model.eval()
        with torch.no_grad():
            student_out = self._student_model(**batch)
            loss = self.security_loss(student_out.logits, batch["labels"])
            return {
                "val_loss": float(loss.item()),
                "val_perplexity": perplexity_calculation(loss),
            }

    def checkpoint(self, path: Path | None = None) -> Path:
        """Salva checkpoint do treinamento."""
        if self._student_model is None or self.optimizer_scheduler is None:
            raise RuntimeError("Modelo/optimizer não configurados.")

        checkpoint_dir = path or Path(self.config.get("checkpoint_dir", "training/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"

        save_checkpoint(
            path=ckpt_path,
            model=self._student_model,
            optimizer=self.optimizer_scheduler.optimizer,
            scheduler=self.optimizer_scheduler.scheduler,
            training_state={
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "config": self.config,
            },
        )
        return ckpt_path

    def resume(self, checkpoint_path: Path) -> None:
        """Retoma treinamento a partir de checkpoint."""
        if self._student_model is None or self.optimizer_scheduler is None:
            raise RuntimeError("Modelo/optimizer não configurados.")

        state = load_checkpoint(
            path=checkpoint_path,
            model=self._student_model,
            optimizer=self.optimizer_scheduler.optimizer,
            scheduler=self.optimizer_scheduler.scheduler,
        )
        self.global_step = state.get("global_step", 0)
        self.current_epoch = state.get("current_epoch", 0)

    def run(self, max_steps: int | None = None) -> None:
        """Inicializa pipeline de treinamento."""
        training_cfg = self.config.get("training", {})
        steps = max_steps or int(training_cfg.get("max_steps", 100000))
        _ = self.prepare_data()

        batch_size = int(training_cfg.get("batch_size", 8))
        accum = int(training_cfg.get("gradient_accumulation_steps", 1))
        world_size = int(training_cfg.get("world_size", 1))

        print(f"[Syon] Treinamento iniciado — {steps} steps configurados")
        print(f"[Syon] Batch efetivo: {effective_batch_size(batch_size, accum, world_size)}")
        print(f"[Syon] Composição validada: {self.composition.validate_weights()}")
        print("[Syon] Aguardando modelos student/teacher via setup_models()")


# Backward compatibility alias
ParallelTrainer = Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Syon Trainer")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    config = load_training_config(args.config)
    trainer = Trainer(config, args.data_dir)
    if args.resume:
        trainer.setup_models(None, None)  # noqa: user must provide models
    trainer.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()