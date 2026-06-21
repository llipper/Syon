"""Syon - LLM especializado em programação e cybersegurança."""

from __future__ import annotations

from typing import TYPE_CHECKING

from syon.config import SyonConfig

__version__ = "1.0.0b0"
__all__ = ["SyonModel", "SyonConfig", "__version__"]

if TYPE_CHECKING:
    from syon.model import SyonModel


def __getattr__(name: str):
    if name == "SyonModel":
        from syon.model import SyonModel

        return SyonModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")