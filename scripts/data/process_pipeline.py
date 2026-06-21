"""
Processa data/raw → data/processed (train/val/test/merged).

Uso:
    python scripts/data/process_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from training.dataset.curator import DatasetCurator
from training.dataset.composition import load_composition


def process(raw_dir: Path, processed_dir: Path, seed: int = 42) -> dict:
    curator = DatasetCurator(raw_dir, load_composition())
    samples = curator.curate_all()
    if not samples:
        raise FileNotFoundError(f"Sem dados em {raw_dir}. Rode build_master_curriculum.py primeiro.")

    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    splits = {
        "train": samples[:train_end],
        "validation": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    stats = {}
    for split_name, split_samples in splits.items():
        out_dir = processed_dir / "merged"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"full_{split_name}.jsonl" if split_name != "validation" else out_dir / "full_val.jsonl"
        if split_name == "validation":
            path = processed_dir / "merged" / "full_val.jsonl"
        elif split_name == "train":
            path = processed_dir / "merged" / "full_train.jsonl"
        else:
            path = processed_dir / "merged" / "full_test.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for s in split_samples:
                f.write(json.dumps({"text": s.text, "metadata": {**s.metadata, "domain": s.domain, "source": s.source}}, ensure_ascii=False) + "\n")
        stats[split_name] = len(split_samples)
        print(f"  {path.name}: {len(split_samples)}")

    stats_path = processed_dir / "merged" / "statistics.json"
    stats_path.write_text(json.dumps({"total": n, "splits": stats}, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    print(f"[Syon] Processando {args.raw} → {args.processed}")
    stats = process(args.raw, args.processed)
    print(f"[Syon] ✓ Splits: {stats}")


if __name__ == "__main__":
    main()