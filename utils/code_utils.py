"""Utilitários de análise e formatação de código."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from syon.code.languages import normalize_language


@dataclass
class SyntaxValidationResult:
    language: str
    valid: bool
    error: str | None = None


def format_code(code: str, language: str = "python") -> str:
    """Formata código (stub: normaliza espaços e remove linhas vazias extras)."""
    lang = normalize_language(language)
    lines = [line.rstrip() for line in code.splitlines()]

    if lang == "python":
        return _format_python_stub(lines)

    return "\n".join(line for line in lines if line.strip() or line == "")


def validate_syntax(code: str, language: str = "python") -> SyntaxValidationResult:
    """Valida sintaxe básica do código para linguagens suportadas."""
    lang = normalize_language(language)

    if lang == "python":
        try:
            ast.parse(code)
            return SyntaxValidationResult(language=lang, valid=True)
        except SyntaxError as exc:
            return SyntaxValidationResult(
                language=lang,
                valid=False,
                error=f"{exc.msg} (linha {exc.lineno})",
            )

    if lang in {"javascript", "typescript"}:
        return _validate_braces(code, lang)

    if lang in {"json"}:
        import json

        try:
            json.loads(code)
            return SyntaxValidationResult(language=lang, valid=True)
        except json.JSONDecodeError as exc:
            return SyntaxValidationResult(language=lang, valid=False, error=str(exc))

    # Stub para demais linguagens: verificação estrutural mínima
    if not code.strip():
        return SyntaxValidationResult(language=lang, valid=False, error="Código vazio")
    return SyntaxValidationResult(language=lang, valid=True)


def _format_python_stub(lines: list[str]) -> str:
    formatted: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        formatted.append(line)
        previous_blank = is_blank
    return "\n".join(formatted).strip() + "\n"


def _validate_braces(code: str, language: str) -> SyntaxValidationResult:
    pairs = {"(": ")", "{": "}", "[": "]"}
    stack: list[str] = []
    in_string: str | None = None

    for char in code:
        if in_string:
            if char == in_string:
                in_string = None
            continue
        if char in {"'", '"', "`"}:
            in_string = char
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                return SyntaxValidationResult(
                    language=language,
                    valid=False,
                    error="Delimitadores desbalanceados",
                )

    if stack:
        return SyntaxValidationResult(
            language=language,
            valid=False,
            error="Delimitadores não fechados",
        )
    return SyntaxValidationResult(language=language, valid=True)