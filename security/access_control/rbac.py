"""Role-Based Access Control para APIs Syon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_ROLES: dict[str, set[str]] = {
    "admin": {"inference", "security_analysis", "training", "admin"},
    "developer": {"inference", "security_analysis", "code_generation"},
    "analyst": {"security_analysis", "inference"},
    "viewer": {"inference"},
    "guest": set(),
}


@dataclass
class RBACManager:
    """Gerencia papéis e permissões de usuários."""

    roles: dict[str, set[str]] = field(default_factory=lambda: dict(DEFAULT_ROLES))
    user_roles: dict[str, str] = field(default_factory=dict)

    def assign_role(self, user_id: str, role: str) -> None:
        if role not in self.roles:
            raise ValueError(f"Papel desconhecido: {role}")
        self.user_roles[user_id] = role

    def check_permission(self, user_id: str, permission: str) -> bool:
        role = self.user_roles.get(user_id, "guest")
        return permission in self.roles.get(role, set())

    def get_user_permissions(self, user_id: str) -> set[str]:
        role = self.user_roles.get(user_id, "guest")
        return set(self.roles.get(role, set()))

    def add_role(self, role: str, permissions: set[str]) -> None:
        self.roles[role] = permissions

    def list_roles(self) -> dict[str, list[str]]:
        return {role: sorted(perms) for role, perms in self.roles.items()}