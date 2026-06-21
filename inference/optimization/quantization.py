"""Utilitários de quantização para inferência."""

from __future__ import annotations

from typing import Any


def quantize_int8(model: Any) -> dict[str, Any]:
    """Estima quantização INT8 — requer integração com torch.quantization em produção."""
    return {
        "status": "stub",
        "bits": 8,
        "message": "Quantização INT8 requer pipeline de calibração dedicado",
        "model_type": type(model).__name__,
    }


def quantize_int4(model: Any) -> dict[str, Any]:
    """Estima quantização INT4 (GGUF/GPTQ)."""
    return {
        "status": "stub",
        "bits": 4,
        "message": "Use ModelLoader.load_quantized_model() para variantes GGUF",
        "model_type": type(model).__name__,
    }


def estimate_memory(
    num_parameters: int,
    bits: int = 16,
    context_length: int = 8192,
    batch_size: int = 1,
) -> dict[str, float]:
    """Estima uso de memória em GB para inferência."""
    weight_gb = (num_parameters * bits) / (8 * 1024**3)
    kv_cache_gb = (context_length * batch_size * 2 * 4096 * 2) / (1024**3)
    total = weight_gb + kv_cache_gb
    return {
        "weights_gb": round(weight_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "total_gb": round(total, 2),
    }


def benchmark_quantized(baseline_ms: float, quantized_ms: float) -> dict[str, float]:
    """Compara latência entre modelo base e quantizado."""
    speedup = baseline_ms / max(quantized_ms, 1e-6)
    return {
        "baseline_ms": baseline_ms,
        "quantized_ms": quantized_ms,
        "speedup": round(speedup, 2),
    }