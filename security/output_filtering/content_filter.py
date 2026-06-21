"""Filtragem de conteúdo malicioso na saída do modelo."""

from __future__ import annotations

import re

from security.input_validation.sanitizer import remove_pii
from syon.exceptions import SecurityPolicyError

MALICIOUS_OUTPUT_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)rm\s+-rf\s+/"),
    re.compile(r"(?i)drop\s+table\s+"),
]

EXPLOIT_PATTERNS = [
    re.compile(r"(?i)(metasploit|shellcode|buffer\s*overflow\s*exploit)"),
    re.compile(r"(?i)(privilege\s*escalation\s*exploit)"),
]

HARMFUL_PATTERNS = [
    re.compile(r"(?i)(como\s+criar\s+malware)"),
    re.compile(r"(?i)(ransomware\s+source\s+code)"),
]


def filter_malicious_code(text: str) -> str:
    for pattern in MALICIOUS_OUTPUT_PATTERNS:
        if pattern.search(text):
            raise SecurityPolicyError("Output bloqueado: conteúdo potencialmente malicioso")
    return text


def filter_exploits(text: str) -> str:
    for pattern in EXPLOIT_PATTERNS:
        if pattern.search(text):
            raise SecurityPolicyError("Output bloqueado: conteúdo de exploit detectado")
    return text


def filter_harmful_content(text: str) -> str:
    for pattern in HARMFUL_PATTERNS:
        if pattern.search(text):
            raise SecurityPolicyError("Output bloqueado: conteúdo prejudicial detectado")
    return text


def filter_pii(text: str) -> str:
    return remove_pii(text)


def check_output(text: str) -> str:
    """Filtra outputs potencialmente maliciosos e redige PII."""
    filter_malicious_code(text)
    filter_exploits(text)
    filter_harmful_content(text)
    return filter_pii(text)