"""Verificações de conformidade HIPAA (stub funcional)."""

from __future__ import annotations

import re
from typing import Any

from security.input_validation.sanitizer import remove_pii

PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"(?i)\b(mrn|medical\s*record|diagnosis|patient\s*id)\b"),
    re.compile(r"(?i)\b(hip|knee|surgery|prescription)\b"),
]


def protect_pii(text: str) -> str:
    """Redige informações de saúde protegidas (PHI)."""
    result = remove_pii(text)
    for pattern in PHI_PATTERNS:
        result = pattern.sub("[PHI_REDACTED]", result)
    return result


def audit_logging(event: str, actor: str, resource: str) -> dict[str, Any]:
    """Gera entrada de auditoria HIPAA."""
    import time

    return {
        "timestamp": time.time(),
        "event": event,
        "actor": actor,
        "resource": resource,
        "logged": True,
    }


def access_controls(role: str, action: str) -> bool:
    """Verifica se papel tem permissão para ação em dados PHI."""
    allowed = {
        "admin": {"read", "write", "delete", "export"},
        "clinician": {"read", "write"},
        "auditor": {"read", "export"},
        "guest": set(),
    }
    return action in allowed.get(role, set())


def encryption_requirements(data_classification: str) -> dict[str, bool]:
    """Requisitos de criptografia por classificação."""
    requires_encryption = data_classification in {"phi", "pii", "restricted"}
    return {
        "encryption_at_rest": requires_encryption,
        "encryption_in_transit": requires_encryption,
        "classification": data_classification,
    }


def check_hipaa_compliance(
    text: str,
    *,
    role: str = "guest",
    action: str = "read",
) -> dict[str, Any]:
    """Verifica conformidade HIPAA básica."""
    has_phi = any(p.search(text) for p in PHI_PATTERNS)
    access_ok = access_controls(role, action) if has_phi else True
    issues: list[str] = []
    if has_phi and not access_ok:
        issues.append(f"Papel '{role}' não autorizado para '{action}' em PHI")
    if has_phi and "encryption_at_rest" not in encryption_requirements("phi"):
        issues.append("Criptografia em repouso requerida para PHI")
    return {
        "compliant": len(issues) == 0,
        "has_phi": has_phi,
        "access_authorized": access_ok,
        "issues": issues,
        "protected_text": protect_pii(text) if has_phi else text,
    }