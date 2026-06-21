"""Controle de acesso e auditoria."""

from security.access_control.audit_log import (
    generate_audit_report,
    log_access,
    log_authorization_failure,
)
from security.access_control.rbac import RBACManager

__all__ = [
    "RBACManager",
    "log_access",
    "log_authorization_failure",
    "generate_audit_report",
]