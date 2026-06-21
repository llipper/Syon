"""KV-cache para otimização de inferência autoregressiva."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KVCache:
    """Cache de chaves/valores para atenção incremental."""

    num_layers: int
    max_seq_len: int
    hidden_size: int
    dtype: str = "float16"
    _keys: list[Any] = field(default_factory=list, repr=False)
    _values: list[Any] = field(default_factory=list, repr=False)
    position: int = 0

    def allocate(self) -> None:
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers
        self.position = 0

    def update(self, layer_idx: int, key: Any, value: Any) -> None:
        if layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} >= num_layers {self.num_layers}")
        self._keys[layer_idx] = key
        self._values[layer_idx] = value
        self.position += 1

    def get(self, layer_idx: int) -> tuple[Any, Any]:
        return self._keys[layer_idx], self._values[layer_idx]

    def clear(self) -> None:
        self.allocate()

    @property
    def memory_bytes_estimate(self) -> int:
        elem_size = 2 if self.dtype == "float16" else 4
        return self.num_layers * self.max_seq_len * self.hidden_size * 2 * elem_size


def allocate_cache(num_layers: int, max_seq_len: int, hidden_size: int) -> KVCache:
    cache = KVCache(num_layers=num_layers, max_seq_len=max_seq_len, hidden_size=hidden_size)
    cache.allocate()
    return cache


def update_cache(cache: KVCache, layer_idx: int, key: Any, value: Any) -> KVCache:
    cache.update(layer_idx, key, value)
    return cache


def memory_efficient_cache(num_layers: int, max_seq_len: int, hidden_size: int) -> KVCache:
    """Aloca cache com dtype reduzido para economia de memória."""
    return KVCache(
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        dtype="float16",
    )