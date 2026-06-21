"""Utilitários de validação de schemas e dados."""

from __future__ import annotations

from typing import Any


def validate_schema(
    data: dict[str, Any],
    schema: dict[str, Any],
    *,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """
    Valida um dicionário contra um schema simples.

    Schema suporta:
    - required: list[str]
    - properties: dict[str, dict] com keys type, enum, minLength, maxLength
    """
    errors: list[str] = []

    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field in required:
        if field not in data:
            errors.append(f"Campo obrigatório ausente: {field}")

    if strict:
        allowed = set(properties.keys())
        for key in data:
            if key not in allowed:
                errors.append(f"Campo não permitido: {key}")

    for field, rules in properties.items():
        if field not in data:
            continue
        value = data[field]
        field_errors = _validate_field(field, value, rules)
        errors.extend(field_errors)

    return len(errors) == 0, errors


def _validate_field(field: str, value: Any, rules: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_type = rules.get("type")

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    if expected_type and expected_type in type_map:
        if not isinstance(value, type_map[expected_type]):
            errors.append(f"{field}: tipo esperado {expected_type}, recebido {type(value).__name__}")

    if "enum" in rules and value not in rules["enum"]:
        errors.append(f"{field}: valor '{value}' não está em {rules['enum']}")

    if isinstance(value, str):
        min_length = rules.get("minLength")
        max_length = rules.get("maxLength")
        if min_length is not None and len(value) < min_length:
            errors.append(f"{field}: comprimento mínimo {min_length}")
        if max_length is not None and len(value) > max_length:
            errors.append(f"{field}: comprimento máximo {max_length}")

    if isinstance(value, (int, float)):
        minimum = rules.get("minimum")
        maximum = rules.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"{field}: valor mínimo {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{field}: valor máximo {maximum}")

    return errors