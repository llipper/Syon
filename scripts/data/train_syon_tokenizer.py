"""Treina tokenizer BPE Syon do zero no corpus local."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from models.tokenizer.syon_bpe import SyonBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina Syon BPE do zero")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "models/tokenizer/syon3-bpe")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    print(f"[Syon] Treinando tokenizer BPE do zero em {args.data_dir}")
    tok = SyonBPETokenizer.train_from_corpus_dir(
        args.data_dir, vocab_size=args.vocab_size, max_length=args.max_length
    )
    args.output.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(args.output)
    print(f"[Syon] ✓ Tokenizer: {args.output} | vocab={tok.vocab_size}")


if __name__ == "__main__":
    main()