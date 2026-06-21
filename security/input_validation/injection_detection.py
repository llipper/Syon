"""Detecção de injection e jailbreak em prompts."""

from __future__ import annotations

import re

from syon.exceptions import SecurityPolicyError

INJECTION_SIGNALS = [
    "ignore previous instructions",
    "ignore all instructions",
    "you are now dan",
    "jailbreak",
    "system prompt override",
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?(previous\s+)?instructions"),
    re.compile(r"(?i)you\s+are\s+now\s+(dan|evil|unrestricted)"),
    re.compile(r"(?i)system\s*prompt\s*override"),
    re.compile(r"(?i)reveal\s+(the\s+)?(system\s+)?prompt"),
]

SQL_INJECTION_PATTERNS = [
    re.compile(r"(?i)('\s*or\s*'1'\s*=\s*'1)"),
    re.compile(r"(?i)(union\s+select)"),
    re.compile(r"(?i)(;\s*drop\s+table)"),
]

COMMAND_INJECTION_PATTERNS = [
    re.compile(r"(?i)(;\s*rm\s+-rf)"),
    re.compile(r"(?i)(\|\s*bash)"),
    re.compile(r"(?i)(&&\s*curl\s+)"),
]

CODE_INJECTION_PATTERNS = [
    re.compile(r"(?i)(eval\s*\()"),
    re.compile(r"(?i)(exec\s*\()"),
    re.compile(r"(?i)(__import__\s*\()"),
]

REFUSAL_CHECKS = {
    "malware": "Não gero malware executável.",
    "zero-day exploit": "Recuso exploits zero-day comprovados.",
    "bypass authentication": "Não auxilio bypass de autenticação moderna.",
    "drm bypass": "Não contorno proteções de DRM.",
}


def detect_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    if any(signal in lowered for signal in INJECTION_SIGNALS):
        return True
    return any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS)


def detect_sql_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in SQL_INJECTION_PATTERNS)


def detect_command_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in COMMAND_INJECTION_PATTERNS)


def detect_code_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in CODE_INJECTION_PATTERNS)


def check_input(prompt: str) -> None:
    """Bloqueia tentativas óbvias de jailbreak/prompt injection."""
    if detect_prompt_injection(prompt):
        raise SecurityPolicyError("Prompt bloqueado: possível prompt injection detectado")


def should_refuse_topic(prompt: str) -> str | None:
    """Retorna motivo de recusa para tópicos proibidos."""
    lowered = prompt.lower()
    for keyword, message in REFUSAL_CHECKS.items():
        if keyword in lowered:
            return message
    return None