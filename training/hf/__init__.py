"""
Treino Syon 3 via Hugging Face — SOMENTE infraestrutura.

- Trainer: loop de otimização
- datasets API: download de texto (diálogos PT)

Modelo: Syon 3 proprietário, pesos do zero.
"""

from training.hf.syon3_hf_trainer import main

__all__ = ["main"]