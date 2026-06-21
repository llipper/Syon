"""Collate functions para batches de treinamento."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def pad_sequences(sequences: list[list[int]], pad_value: int = 0) -> Tensor:
    """Pad sequências para mesmo comprimento."""
    max_len = max(len(s) for s in sequences) if sequences else 0
    padded = [s + [pad_value] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)


def create_attention_masks(input_ids: Tensor, pad_token_id: int = 0) -> Tensor:
    """Cria máscaras de atenção (1 para tokens reais, 0 para padding)."""
    return (input_ids != pad_token_id).long()


def default_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate padrão para batches de texto."""
    texts = [item.get("text", "") for item in batch]
    return {
        "texts": texts,
        "batch_size": len(batch),
        "domains": [item.get("domain", "unknown") for item in batch],
    }


def prepare_batch(batch: list[dict[str, Any]], tokenizer=None, max_length: int = 512) -> dict[str, Tensor]:
    """Prepara batch tokenizado para o modelo."""
    texts = [item.get("text", "") for item in batch]

    if tokenizer is not None:
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
    else:
        # Fallback: hash-based pseudo tokens para desenvolvimento
        input_ids = torch.tensor(
            [[hash(t) % 32000 for t in text.split()[:max_length]] or [0] for text in texts],
            dtype=torch.long,
        )
        input_ids = pad_sequences(input_ids.tolist())
        labels = input_ids.clone()

    attention_mask = create_attention_masks(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }