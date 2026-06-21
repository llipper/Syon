"""Detecção e otimização para GPU NVIDIA."""

from __future__ import annotations

from typing import Any


def detect_gpu() -> dict[str, Any]:
    """Detecta GPUs CUDA disponíveis."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "devices": []}

        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        return {"available": True, "count": len(devices), "devices": devices}
    except ImportError:
        return {"available": False, "count": 0, "devices": [], "error": "torch não instalado"}


def allocate_gpu_memory(fraction: float = 0.9) -> dict[str, Any]:
    """Reserva fração da memória GPU para inferência."""
    info = detect_gpu()
    if not info["available"]:
        return {"allocated": False, "message": "GPU não disponível"}

    try:
        import torch

        torch.cuda.set_per_process_memory_fraction(fraction)
        return {"allocated": True, "fraction": fraction, "devices": info["devices"]}
    except Exception as exc:
        return {"allocated": False, "error": str(exc)}


def cuda_optimizations() -> dict[str, bool]:
    """Ativa otimizações CUDA comuns."""
    opts = {
        "cudnn_benchmark": False,
        "tf32": False,
        "flash_attention": False,
    }
    try:
        import torch

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            opts["cudnn_benchmark"] = True
            if hasattr(torch.backends.cuda, "matmul") and hasattr(
                torch.backends.cuda.matmul, "allow_tf32"
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
                opts["tf32"] = True
    except ImportError:
        pass
    return opts


def multi_gpu_inference(device_ids: list[int] | None = None) -> dict[str, Any]:
    """Configura inferência multi-GPU via device_map."""
    info = detect_gpu()
    if not info["available"]:
        return {"enabled": False, "device_map": "cpu"}
    ids = device_ids or list(range(info["count"]))
    return {"enabled": True, "device_ids": ids, "device_map": "auto"}