"""Sanitização de entradas de código e texto."""

from __future__ import annotations

import re
import unicodedata


PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]+['\"]"),
]


def sanitize_code(code: str, max_length: int = 500_000) -> str:
    """Remove caracteres de controle e limita tamanho."""
    normalized = unicodedata.normalize("NFKC", code)
    cleaned = "".join(ch for ch in normalized if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return cleaned[:max_length]


def remove_pii(text: str, replacement: str = "[REDACTED]") -> str:
    """Remove padrões comuns de PII."""
    result = text
    for pattern in PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def escape_special_chars(text: str) -> str:
    """Escapa caracteres especiais para contexto seguro."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\x00", "")
    )


def validate_encoding(text: str) -> dict[str, bool | str]:
    """Valida encoding UTF-8 e ausência de null bytes."""
    try:
        text.encode("utf-8")
        has_null = "\x00" in text
        return {"valid": not has_null, "encoding": "utf-8", "has_null_bytes": has_null}
    except UnicodeEncodeError as exc:
        return {"valid": False, "encoding": "invalid", "error": str(exc)}