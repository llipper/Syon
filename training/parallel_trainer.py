"""Treinamento Paralelo Supervisionado — backward compatibility wrapper."""

from __future__ import annotations

from training.core.trainer import ParallelTrainer, Trainer, load_training_config, main

__all__ = ["ParallelTrainer", "Trainer", "load_training_config", "main"]

if __name__ == "__main__":
    main()