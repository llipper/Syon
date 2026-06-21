"""Logging de auditoria de acesso."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditEntry:
    timestamp: float
    event_type: str
    actor: str
    resource: str
    success: bool
    details: dict[str, Any] = field(default_factory=dict)


_AUDIT_LOG: list[AuditEntry] = []


def log_access(actor: str, resource: str, success: bool = True, **details: Any) -> AuditEntry:
    entry = AuditEntry(
        timestamp=time.time(),
        event_type="access",
        actor=actor,
        resource=resource,
        success=success,
        details=details,
    )
    _AUDIT_LOG.append(entry)
    return entry


def log_authorization_failure(actor: str, resource: str, reason: str) -> AuditEntry:
    return log_access(actor, resource, success=False, reason=reason, event="auth_failure")


def generate_audit_report(
    *,
    actor: str | None = None,
    since: float | None = None,
) -> dict[str, Any]:
    entries = _AUDIT_LOG
    if actor:
        entries = [e for e in entries if e.actor == actor]
    if since is not None:
        entries = [e for e in entries if e.timestamp >= since]

    failures = [e for e in entries if not e.success]
    return {
        "total_events": len(entries),
        "failures": len(failures),
        "success_rate": round((len(entries) - len(failures)) / max(len(entries), 1), 2),
        "events": [
            {
                "timestamp": e.timestamp,
                "type": e.event_type,
                "actor": e.actor,
                "resource": e.resource,
                "success": e.success,
                "details": e.details,
            }
            for e in entries[-100:]
        ],
    }


def clear_audit_log() -> None:
    """Limpa log em memória (apenas para testes)."""
    _AUDIT_LOG.clear()