#!/usr/bin/env python3
"""Exemplo básico de inferência com SyonModel."""

from __future__ import annotations

import sys
from pathlib import Path

# Permite execução direta: python inference/examples/basic_inference.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syon.model import SyonModel


def main() -> None:
    model_path = ROOT / "models" / "syon-7b.gguf"
    if not model_path.exists():
        print(f"Modelo não encontrado em {model_path}")
        print("Use SyonModel com caminho válido ou coloque o GGUF em models/")
        return

    model = SyonModel.load(model_path)
    prompt = (
        "<|system|>Você é Syon, especialista em programação.\n"
        "<|user|>Explique o que é SQL injection em uma frase.\n"
        "<|assistant|>"
    )
    response = model.complete(prompt, max_tokens=128, temperature=0.3)
    print("Resposta:", response)


if __name__ == "__main__":
    main()