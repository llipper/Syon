"""Utilitários de monitoramento de memória."""

from __future__ import annotations

import os
from typing import Any

import torch


def get_memory_usage() -> dict[str, Any]:
    """Retorna uso de memória do processo e GPU (se disponível)."""
    usage: dict[str, Any] = {
        "process_rss_mb": _get_process_rss_mb(),
        "process_vms_mb": _get_process_vms_mb(),
    }

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        props = torch.cuda.get_device_properties(device)
        usage["gpu"] = {
            "device_index": device,
            "device_name": props.name,
            "allocated_mb": allocated // (1024 * 1024),
            "reserved_mb": reserved // (1024 * 1024),
            "total_mb": props.total_memory // (1024 * 1024),
        }

    return usage


def _get_process_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _get_process_vms_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().vms / (1024 * 1024)
    except ImportError:
        return 0.0