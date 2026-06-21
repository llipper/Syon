"""
Geração de dataset para treino Syon 3.

Gera:
  1. Entradas manuais premium (curriculum_entries.py)
  2. Amostras procedurais (curriculum_generator.py)
  3. Manifest + estatísticas

Uso:
    python scripts/data/build_syon3_curriculum.py
    python scripts/data/build_syon3_curriculum.py --min-samples 15000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data.curriculum_entries import ENTRIES
from scripts.data.curriculum_generator import generate_all, generate_until_unique

AUGMENT_SUFFIXES = [
    "\n\n[SENIOR] Considere observabilidade, SLOs e failure modes.",
    "\n\n[MASTER] Documente trade-offs em ADR.",
    "\n\n[SECURITY] Defense in depth + least privilege.",
    "\n\n[ARCH] Evolução incremental, evite big-bang.",
]


def write_jsonl(path: Path, records: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def augment_record(record: dict, seed: int) -> dict:
    rng = random.Random(seed)
    text = record["text"]
    if rng.random() > 0.4:
        text += rng.choice(AUGMENT_SUFFIXES)
    meta = dict(record.get("metadata", {}))
    meta["augmented"] = True
    meta["seed"] = seed
    return {"text": text, "metadata": meta}


def merge_entries_and_generated(min_unique: int) -> dict[str, dict[str, list[dict]]]:
    merged: dict[str, dict[str, list[dict]]] = {}
    seen: set[str] = set()

    def add_record(domain: str, source: str, rec: dict) -> None:
        h = hashlib.sha256(rec["text"].encode()).hexdigest()
        if h in seen:
            return
        seen.add(h)
        merged.setdefault(domain, {}).setdefault(source, []).append(rec)

    for domain, sources in ENTRIES.items():
        for source, records in sources.items():
            for rec in records:
                add_record(domain, source, rec)

    generated = generate_until_unique(max(min_unique - len(seen), 0))
    for domain, sources in generated.items():
        for source, records in sources.items():
            for rec in records:
                add_record(domain, source, rec)

    return merged


def build(output_dir: Path, min_samples: int = 10000, augment: int = 3) -> dict[str, int]:
    data = merge_entries_and_generated(min_unique=min_samples)

    stats: dict[str, int] = {}
    total = 0
    for domain, sources in data.items():
        for source, records in sources.items():
            path = output_dir / domain / f"{source}.jsonl"
            n = write_jsonl(path, records)
            stats[f"{domain}/{source}"] = n
            total += n
            print(f"  {domain}/{source}: {n}")

    stats["_total"] = total
    manifest_dir = output_dir.parent / "curriculum"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output": str(output_dir),
        "total_samples": total,
        "min_samples_target": min_samples,
        "stats": stats,
        "domains": list(data.keys()),
    }
    (manifest_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Syon 3 curriculum")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--min-samples", type=int, default=10000)
    parser.add_argument("--augment", type=int, default=3)
    args = parser.parse_args()

    print(f"[Syon 3] Gerando curriculum → {args.output} (mínimo {args.min_samples} amostras)")
    stats = build(args.output, min_samples=args.min_samples, augment=args.augment)
    print(f"[Syon 3] Total: {stats['_total']} amostras geradas")


if __name__ == "__main__":
    main()