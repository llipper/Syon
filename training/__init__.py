"""Pipeline de treinamento Syon."""

__all__ = ["Trainer", "ParallelTrainer", "load_training_config", "train_main"]


def __getattr__(name: str):
    """Lazy imports para evitar carregar torch no import do pacote."""
    if name in ("Trainer", "ParallelTrainer", "load_training_config"):
        from training.core.trainer import ParallelTrainer, Trainer, load_training_config

        return {"Trainer": Trainer, "ParallelTrainer": ParallelTrainer, "load_training_config": load_training_config}[name]
    if name == "train_main":
        from training.parallel_trainer import main as train_main

        return train_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")