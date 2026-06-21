"""Otimizações de atenção para inferência."""

from __future__ import annotations

from typing import Any


def flash_attention_v2(enabled: bool = True) -> dict[str, Any]:
    """Configura Flash Attention v2 quando disponível."""
    try:
        import torch

        has_flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        return {
            "enabled": enabled and has_flash,
            "backend": "sdpa" if has_flash else "eager",
            "message": "Flash Attention via scaled_dot_product_attention"
            if has_flash
            else "Fallback para atenção padrão",
        }
    except ImportError:
        return {"enabled": False, "backend": "none", "message": "PyTorch não disponível"}


def paged_attention(block_size: int = 16) -> dict[str, Any]:
    """Configuração de paged attention (compatível com vLLM-style serving)."""
    return {
        "block_size": block_size,
        "enabled": True,
        "message": f"Paged attention com blocos de {block_size} tokens",
    }


def memory_efficient_attention(use_checkpoint: bool = False) -> dict[str, Any]:
    """Opções de atenção com menor footprint de memória."""
    return {
        "gradient_checkpointing": use_checkpoint,
        "scaled_dot_product_attention": True,
        "message": "Atenção memory-efficient configurada",
    }