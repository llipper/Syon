"""Utilitários de processamento de texto."""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINES = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Normaliza espaços em branco preservando quebras de linha."""
    lines = []
    for line in text.splitlines():
        lines.append(_MULTI_WHITESPACE.sub(" ", line.strip()))
    return "\n".join(lines)


def clean_text(
    text: str,
    *,
    normalize_unicode: bool = True,
    strip_control_chars: bool = True,
    collapse_newlines: bool = True,
) -> str:
    """Limpa texto removendo caracteres indesejados e normalizando espaços."""
    cleaned = text

    if normalize_unicode:
        cleaned = unicodedata.normalize("NFKC", cleaned)

    if strip_control_chars:
        cleaned = _CONTROL_CHARS.sub("", cleaned)

    cleaned = normalize_whitespace(cleaned)

    if collapse_newlines:
        cleaned = _MULTI_NEWLINES.sub("\n\n", cleaned)

    return cleaned.strip()