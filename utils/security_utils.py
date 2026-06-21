"""Utilitários de segurança para entrada e hashing."""

from __future__ import annotations

import hashlib
import html
import re
from typing import Any

_INJECTION_PATTERNS = [
    re.compile(r"(?i)(ignore\s+(all\s+)?(previous|prior)\s+instructions)"),
    re.compile(r"(?i)(system\s*:\s*)"),
    re.compile(r"(?i)(<\s*/?\s*script\s*>)"),
    re.compile(r"(?i)(;\s*(drop|delete|truncate)\s+table)"),
    re.compile(r"(?i)(\$\(.*\)|`.*`)"),
]

_NULL_BYTES = re.compile(r"\x00")


def sanitize_input(
    text: str,
    *,
    max_length: int = 65536,
    escape_html: bool = True,
    strip_null_bytes: bool = True,
    block_injection_patterns: bool = True,
) -> str:
    """Sanitiza entrada de usuário removendo conteúdo potencialmente perigoso."""
    sanitized = text.strip()

    if strip_null_bytes:
        sanitized = _NULL_BYTES.sub("", sanitized)

    if escape_html:
        sanitized = html.escape(sanitized, quote=True)

    if block_injection_patterns:
        for pattern in _INJECTION_PATTERNS:
            sanitized = pattern.sub("[FILTERED]", sanitized)

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def generate_hash(
    value: str | bytes,
    *,
    algorithm: str = "sha256",
    encoding: str = "utf-8",
) -> str:
    """Gera hash hexadecimal de uma string ou bytes."""
    hasher = hashlib.new(algorithm)

    if isinstance(value, str):
        hasher.update(value.encode(encoding))
    else:
        hasher.update(value)

    return hasher.hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    """Compara strings em tempo constante para evitar timing attacks."""
    return hashlib.sha256(a.encode()).digest() == hashlib.sha256(b.encode()).digest()