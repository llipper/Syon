"""
Empacota dados para upload como Dataset Kaggle.

Uso (no PC, apos import + build SFT):
    python scripts/kaggle/package_kaggle_data.py --sft data/gemma_sft --conversation data/raw/conversation

Gera:
    dist/kaggle-datasets/syon-gemma-sft/       -> upload como dataset
    dist/kaggle-datasets/syon-conversation-pt/
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def copy_tree(src: Path, dst: Path) -> int:
    if not src.exists():
        print(f"  SKIP (nao existe): {src}")
        return 0
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n = sum(1 for f in dst.rglob("*") if f.is_file())
    print(f"  {n} arquivos -> {dst}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Empacotar dados para Kaggle")
    parser.add_argument("--sft", type=Path, default=ROOT / "data/gemma_sft")
    parser.add_argument("--conversation", type=Path, default=ROOT / "data/raw/conversation")
    parser.add_argument("--out", type=Path, default=ROOT / "dist/kaggle-datasets")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("[Syon/Kaggle] Empacotando datasets...")
    copy_tree(args.sft, out / "syon-gemma-sft")
    copy_tree(args.conversation, out / "syon-conversation-pt")

    print(f"\n[Syon/Kaggle] Pronto em {out}")
    print("Upload no Kaggle:")
    print("  1. kaggle.com -> Datasets -> New Dataset")
    print("  2. Upload pasta syon-gemma-sft -> nome: syon-gemma-sft")
    print("  3. Upload pasta syon-conversation-pt -> nome: syon-conversation-pt")


if __name__ == "__main__":
    main()