"""Treino Syon 3 via Hugging Face — somente infraestrutura."""

__all__ = ["main"]


def main() -> None:
    from training.hf.syon3_hf_trainer import main as _run

    _run()