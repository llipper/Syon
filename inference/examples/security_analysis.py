#!/usr/bin/env python3
"""Exemplo de análise de segurança com SyonModel e SAST."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from security.vulnerability_scanning.sast import SecurityAnalyzer
from syon.model import SyonModel


SAMPLE_CODE = '''
import pickle
import os

def load_data(user_input):
    return pickle.loads(user_input)

def run_cmd(cmd):
    os.system(cmd)
'''


def main() -> None:
    analyzer = SecurityAnalyzer()
    result = analyzer.analyze(SAMPLE_CODE, language="python")

    print(f"Linguagem: {result.language}")
    print(f"Nível de risco: {result.risk_level}")
    print(f"CVSS estimado: {result.cvss_estimate}")
    print(f"OWASP: {', '.join(result.owasp_categories) or 'nenhum'}")
    print("\nFindings:")
    for finding in result.findings:
        print(f"  [{finding.severity}] {finding.cwe} (linha {finding.line}): {finding.description}")
    print("\nRecomendações:")
    for rec in result.recommendations:
        print(f"  - {rec}")

    model_path = ROOT / "models" / "syon-7b.gguf"
    if model_path.exists():
        print("\n--- Análise complementar via LLM ---")
        model = SyonModel.load(model_path)
        llm_result = model.analyze_security(SAMPLE_CODE, language="python")
        print(f"Risco (SAST): {llm_result.risk_level}")


if __name__ == "__main__":
    main()