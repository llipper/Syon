"""Carrega Syon 3 para inferência no Kaggle — descobre paths de código e pesos."""

from __future__ import annotations

import sys
from pathlib import Path


def find_code_root(input_dir: Path = Path("/kaggle/input")) -> Path:
    for marker in input_dir.rglob("syon3.py"):
        root = marker.parent.parent.parent
        if (root / "training").is_dir():
            return root
    raise FileNotFoundError("Código Syon 3 não encontrado em /kaggle/input")


def find_weights_dir(input_dir: Path = Path("/kaggle/input")) -> Path:
    candidates: list[Path] = []
    for config in input_dir.rglob("config.json"):
        parent = config.parent
        if (parent / "pytorch_model.bin").exists() and (parent / "tokenizer.json").exists():
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError("Pesos Syon 3 não encontrados (config.json + pytorch_model.bin)")
    # Preferir versão numérica mais alta (ex: .../default/4/)
    def version_key(p: Path) -> int:
        for part in reversed(p.parts):
            if part.isdigit():
                return int(part)
        return 0
    return max(candidates, key=version_key)


def bootstrap() -> tuple[Path, Path]:
    inp = Path("/kaggle/input")
    code = find_code_root(inp)
    weights = find_weights_dir(inp)
    for p in (str(code), str(code / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return code, weights


def load_model(device: str = "cuda"):
    from models.architecture.syon3 import Syon3
    from models.tokenizer.syon_bpe import SyonBPETokenizer

    _, weights = bootstrap()
    model = Syon3.from_pretrained(weights, device=device)
    tokenizer = SyonBPETokenizer.from_pretrained(weights)
    model.eval()
    return model, tokenizer, weights