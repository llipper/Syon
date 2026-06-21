"""Verificações de conformidade GDPR (stub funcional)."""

from __future__ import annotations

import re
from typing import Any

from security.input_validation.sanitizer import remove_pii

PII_DETECTED_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)


def right_to_be_forgotten(user_id: str, stored_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Remove registros associados a um usuário (direito ao esquecimento)."""
    remaining = [r for r in stored_records if r.get("user_id") != user_id]
    deleted = len(stored_records) - len(remaining)
    return {"user_id": user_id, "records_deleted": deleted, "remaining": len(remaining)}


def data_portability(user_id: str, stored_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exporta dados do usuário em formato portável."""
    return [r for r in stored_records if r.get("user_id") == user_id]


def consent_management(
    user_id: str,
    consent_given: bool,
    purpose: str,
) -> dict[str, Any]:
    """Registra consentimento do usuário."""
    return {
        "user_id": user_id,
        "consent_given": consent_given,
        "purpose": purpose,
        "valid": consent_given and bool(purpose),
    }


def privacy_impact_assessment(data_types: list[str]) -> dict[str, Any]:
    """Avaliação simplificada de impacto à privacidade."""
    sensitive = {"health", "biometric", "financial", "location", "children"}
    flagged = [d for d in data_types if d.lower() in sensitive]
    risk = "high" if flagged else "low" if not data_types else "medium"
    return {"risk_level": risk, "sensitive_data_types": flagged}


def check_gdpr_compliance(text: str, *, has_consent: bool = True) -> dict[str, Any]:
    """Verifica requisitos básicos GDPR em texto/dados processados."""
    has_pii = bool(PII_DETECTED_PATTERN.search(text))
    redacted = remove_pii(text) if has_pii else text
    issues: list[str] = []
    if has_pii and not has_consent:
        issues.append("PII detectado sem consentimento documentado")
    if has_pii and text == redacted:
        issues.append("PII presente e não redigido")
    return {
        "compliant": len(issues) == 0,
        "has_pii": has_pii,
        "has_consent": has_consent,
        "issues": issues,
        "redacted_preview": redacted[:200] if has_pii else None,
    }