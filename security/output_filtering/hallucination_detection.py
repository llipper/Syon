"""Detecção heurística de alucinações em saídas."""

from __future__ import annotations

import re
from typing import Any


CITATION_PATTERN = re.compile(
    r"(?i)(?:according to|fonte:|source:)\s+([A-Z][A-Za-z\s]+(?:\(\d{4}\))?)"
)
CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,}")
UNCERTAINTY_MARKERS = [
    "pode ser",
    "provavelmente",
    "acredito que",
    "não tenho certeza",
    "might be",
    "possibly",
]


def detect_factual_inconsistency(text: str, context: str = "") -> dict[str, Any]:
    """Detecta contradições simples entre contexto e resposta."""
    issues: list[str] = []
    if context:
        context_lower = context.lower()
        for sentence in text.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            negations = ["não", "never", "impossível", "cannot"]
            if any(n in sentence.lower() for n in negations):
                key_terms = [w for w in sentence.lower().split() if len(w) > 5]
                contradictions = [t for t in key_terms if t in context_lower]
                if contradictions:
                    issues.append(f"Possível contradição: {sentence[:80]}")
    return {"inconsistent": len(issues) > 0, "issues": issues}


def detect_fake_citations(text: str) -> dict[str, Any]:
    """Identifica citações suspeitas sem referência verificável."""
    citations = CITATION_PATTERN.findall(text)
    cves = CVE_PATTERN.findall(text)
    suspicious = [c for c in citations if len(c.split()) < 2]
    return {
        "citations_found": citations,
        "cves_found": cves,
        "suspicious_citations": suspicious,
        "likely_hallucinated": len(suspicious) > 0 and len(cves) == 0,
    }


def confidence_scoring(text: str) -> dict[str, Any]:
    """Pontua confiança heurística da resposta."""
    lowered = text.lower()
    uncertainty_count = sum(1 for m in UNCERTAINTY_MARKERS if m in lowered)
    has_cve = bool(CVE_PATTERN.search(text))
    has_code = "```" in text
    score = 0.7
    if uncertainty_count:
        score -= min(0.3, uncertainty_count * 0.1)
    if has_cve:
        score += 0.1
    if has_code:
        score += 0.05
    score = max(0.0, min(1.0, score))
    return {
        "confidence": round(score, 2),
        "uncertainty_markers": uncertainty_count,
        "has_structured_evidence": has_cve or has_code,
    }