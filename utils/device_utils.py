"""Utilitários de detecção de dispositivo (CPU/GPU)."""

from __future__ import annotations

from typing import Literal

import torch

DeviceType = Literal["cpu", "cuda", "mps"]


def get_device(prefer: str = "auto") -> torch.device:
    """Retorna o dispositivo PyTorch mais adequado."""
    preference = prefer.lower().strip()

    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference in {"cpu", "cuda", "mps"}:
        if preference == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if preference == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            return torch.device("cpu")
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_available_gpus() -> list[dict[str, int | str]]:
    """Lista GPUs CUDA disponíveis com metadados básicos."""
    if not torch.cuda.is_available():
        return []

    gpus: list[dict[str, int | str]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_mb": props.total_memory // (1024 * 1024),
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return gpus