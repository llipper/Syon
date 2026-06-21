"""Estratégias de decodificação para geração de tokens."""

from __future__ import annotations

from typing import Any

import numpy as np


def greedy_decoding(logits: np.ndarray) -> int:
    """Seleciona o token com maior probabilidade."""
    return int(np.argmax(logits))


def temperature_sampling(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Amostragem com temperatura; temperature=0 equivale a greedy."""
    if temperature <= 0:
        return greedy_decoding(logits)
    scaled = logits / max(temperature, 1e-8)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


def nucleus_sampling(logits: np.ndarray, top_p: float = 0.95, temperature: float = 1.0) -> int:
    """Top-p (nucleus) sampling com temperatura opcional."""
    if temperature > 0:
        scaled = logits / max(temperature, 1e-8)
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
    else:
        probs = np.exp(logits - np.max(logits))

    probs = probs / np.sum(probs)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    cutoff = int(np.searchsorted(cumulative, top_p, side="right"))
    cutoff = max(cutoff, 1)
    selected_indices = sorted_indices[:cutoff]
    selected_probs = probs[selected_indices]
    selected_probs = selected_probs / np.sum(selected_probs)
    chosen = int(np.random.choice(selected_indices, p=selected_probs))
    return chosen


def apply_generation_strategy(
    logits: np.ndarray,
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> int:
    """Aplica estratégia de geração conforme parâmetros."""
    if not do_sample or temperature <= 0:
        return greedy_decoding(logits)
    if top_p < 1.0:
        return nucleus_sampling(logits, top_p=top_p, temperature=temperature)
    return temperature_sampling(logits, temperature=temperature)