"""Pós-processamento de saídas de inferência."""

from __future__ import annotations

import re
from typing import Any


def clean_output(text: str) -> str:
    """Remove artefatos comuns de geração."""
    cleaned = text.strip()
    cleaned = re.sub(r"<\|(?:system|user|assistant)\|>", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_code(text: str, language: str = "python") -> str:
    """Extrai e formata blocos de código da resposta."""
    fence = f"```{language}"
    if fence in text:
        match = re.search(rf"```{re.escape(language)}\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    generic = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return clean_output(text)


def validate_output(text: str, min_length: int = 1, max_length: int = 100_000) -> dict[str, Any]:
    """Valida saída gerada quanto a tamanho e conteúdo vazio."""
    cleaned = clean_output(text)
    issues: list[str] = []
    if len(cleaned) < min_length:
        issues.append(f"Saída muito curta (< {min_length} caracteres)")
    if len(cleaned) > max_length:
        issues.append(f"Saída excede limite ({max_length} caracteres)")
    if not cleaned:
        issues.append("Saída vazia após limpeza")
    return {
        "valid": len(issues) == 0,
        "text": cleaned,
        "issues": issues,
        "length": len(cleaned),
    }


def extract_key_info(text: str) -> dict[str, list[str]]:
    """Extrai blocos de código e menções CWE da saída."""
    code_blocks = re.findall(r"```[\w]*\s*\n(.*?)```", text, re.DOTALL)
    cwes = re.findall(r"CWE-\d+", text)
    return {"code_blocks": code_blocks, "cwe_mentions": sorted(set(cwes))}