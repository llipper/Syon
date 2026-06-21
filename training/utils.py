"""Utilitários de treinamento — checkpoints e métricas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> dict[str, Any]:
    """Carrega checkpoint de treinamento."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("training_state", {
        "global_step": checkpoint.get("global_step", 0),
        "current_epoch": checkpoint.get("current_epoch", 0),
    })


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Salva checkpoint completo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "training_state": training_state or {},
        "global_step": (training_state or {}).get("global_step", 0),
        "current_epoch": (training_state or {}).get("current_epoch", 0),
    }
    if optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)

    if training_state:
        meta_path = path.parent / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(training_state, handle, indent=2)


def get_model_size(model: torch.nn.Module) -> dict[str, float]:
    """Estima tamanho do modelo em parâmetros e MB."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return {
        "total_params": total_params,
        "trainable_params": trainable,
        "size_mb": round(size_mb, 2),
    }


def compute_flops(model: torch.nn.Module, input_shape: tuple[int, ...] = (1, 512)) -> int:
    """Estimativa simplificada de FLOPs."""
    try:
        from torchprofile import profile_macs  # type: ignore[import-untyped]

        dummy = torch.randint(0, 32000, input_shape)
        macs = profile_macs(model, dummy)
        return int(macs * 2)
    except Exception:
        params = sum(p.numel() for p in model.parameters())
        return params * input_shape[0] * input_shape[1] * 2