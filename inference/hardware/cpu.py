"""Otimização de inferência em CPU."""

from __future__ import annotations

import os
from typing import Any


def optimize_for_cpu(num_threads: int | None = None) -> dict[str, Any]:
    """Configura threads e flags para inferência em CPU."""
    threads = num_threads or os.cpu_count() or 4
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    opts: dict[str, Any] = {
        "num_threads": threads,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }

    try:
        import torch

        torch.set_num_threads(threads)
        opts["torch_num_threads"] = threads
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, threads // 2))
    except ImportError:
        pass

    return opts


def use_optimizations(enable_mkldnn: bool = True) -> dict[str, bool]:
    """Ativa otimizações MKLDNN/oneDNN quando disponíveis."""
    result = {"mkldnn": False, "ipex": False}
    try:
        import torch

        if enable_mkldnn and hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
            result["mkldnn"] = True
    except ImportError:
        pass
    return result


def estimate_cpu_time(num_tokens: int, tokens_per_sec: float = 25.0) -> dict[str, float]:
    """Estima tempo de geração em CPU."""
    seconds = num_tokens / max(tokens_per_sec, 1e-6)
    return {
        "num_tokens": num_tokens,
        "tokens_per_sec": tokens_per_sec,
        "estimated_seconds": round(seconds, 2),
    }