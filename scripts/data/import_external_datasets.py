"""
Importa datasets externos para data/raw/ (escala master).

Exemplos:
    python scripts/data/import_external_datasets.py --source huggingface --dataset bigcode/the-stack-smol --domain programming --source-name github_repositories --limit 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def import_huggingface(dataset_id: str, output: Path, domain: str, source_name: str, limit: int) -> int:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("content") or row.get("text") or row.get("code", "")
            if not text or len(text) < 50:
                continue
            f.write(json.dumps({"text": text[:8000], "metadata": {"source": dataset_id}}) + "\n")
            count += 1
            if count >= limit:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["huggingface"], default="huggingface")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--domain", default="programming")
    parser.add_argument("--source-name", default="github_repositories")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    out = args.output_dir / args.domain / f"{args.source_name}.jsonl"
    if args.source == "huggingface":
        n = import_huggingface(args.dataset, out, args.domain, args.source_name, args.limit)
    print(f"Importados {n} registros → {out}")


if __name__ == "__main__":
    main()