"""Módulos de conformidade regulatória."""

from security.compliance.gdpr import check_gdpr_compliance, consent_management, right_to_be_forgotten
from security.compliance.hipaa import check_hipaa_compliance, protect_pii

__all__ = [
    "check_gdpr_compliance",
    "right_to_be_forgotten",
    "consent_management",
    "check_hipaa_compliance",
    "protect_pii",
]