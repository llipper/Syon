"""
Importa datasets REAIS de conversação em português → data/raw/conversation/.

Fontes (HuggingFace ou pasta local):
  - instruct-aira-dataset-v3 portuguese (~50k diálogos)
  - UltrachatBR (~1.45M diálogos PT)

Uso local (sem download):
    python scripts/data/import_conversation_datasets.py ^
      --aira-parquet "UltrachatBR/../dataset-v3-bucket/portuguese-00000-of-00001.parquet" ^
      --ultrachat-dir UltrachatBR ^
      --ultrachat-limit 100000
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

# UltrachatBR usa literais Python com escapes inválidos (ex.: "\ ") — Python 3.12+ avisa.
_INVALID_ESCAPE_RE = re.compile(
    r"\\(?![\\'\"abfnrtv]|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|[0-7]{1,3})"
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROLE_MAP = {
    "user": "user",
    "human": "user",
    "humano": "user",
    "assistant": "model",
    "assistente": "model",
    "model": "model",
    "gpt": "model",
    "bot": "model",
}


def normalize_role(role: str) -> str | None:
    return ROLE_MAP.get(str(role).strip().lower())


def normalize_turns(raw_turns: list[dict[str, str]]) -> list[dict[str, str]] | None:
    turns: list[dict[str, str]] = []
    for t in raw_turns:
        role = normalize_role(t.get("role", t.get("from", "")))
        content = str(t.get("content", t.get("value", ""))).strip()
        if not role or len(content) < 2:
            continue
        turns.append({"role": role, "content": content})
    if len(turns) < 2:
        return None
    # alternância user/model — descarta se só um lado
    if not any(t["role"] == "user" for t in turns) or not any(t["role"] == "model" for t in turns):
        return None
    return turns


def _sanitize_conversa_literal(conversa: str) -> str:
    """Remove backslashes inválidos antes do literal_eval (ex.: '\\ ' → ' ')."""
    return _INVALID_ESCAPE_RE.sub(lambda m: m.group(0)[1:], conversa)


def parse_ultrachat_conversa(conversa: str) -> list[dict[str, str]] | None:
    cleaned = _sanitize_conversa_literal(conversa)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            data = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    raw: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "humano" in item:
            raw.append({"role": "user", "content": str(item["humano"])})
        if "assistente" in item:
            raw.append({"role": "model", "content": str(item["assistente"])})
        if "human" in item:
            raw.append({"role": "user", "content": str(item["human"])})
        if "assistant" in item:
            raw.append({"role": "model", "content": str(item["assistant"])})
    return normalize_turns(raw)


def write_jsonl(path: Path, records: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def import_aira_local(parquet_path: Path, output: Path, limit: int) -> int:
    import pyarrow.parquet as pq

    print(f"[Syon/Conv] Lendo instruct-aira local: {parquet_path}")
    table = pq.read_table(parquet_path)
    records: list[dict] = []
    for batch in table.to_batches(max_chunksize=256):
        for row in batch.to_pylist():
            turns = normalize_turns(row.get("conversations") or [])
            if not turns:
                continue
            records.append({
                "turns": turns,
                "metadata": {
                    "source": "nicholasKluge/instruct-aira-dataset-v3",
                    "conversation_id": row.get("conversation_id", ""),
                    "split": "portuguese",
                    "local_path": str(parquet_path),
                },
            })
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break
    out = output / "instruct_aira_pt.jsonl"
    n = write_jsonl(out, records)
    print(f"  → {n} diálogos em {out}")
    return n


def import_aira(output: Path, limit: int, *, local_parquet: Path | None = None) -> int:
    if local_parquet and local_parquet.exists():
        return import_aira_local(local_parquet, output, limit)

    from datasets import load_dataset

    print(f"[Syon/Conv] Baixando instruct-aira PT (limite {limit})...")
    ds = load_dataset("nicholasKluge/instruct-aira-dataset-v3", split="portuguese", streaming=True)
    records: list[dict] = []
    for row in ds:
        convs = row.get("conversations") or []
        turns = normalize_turns(convs)
        if not turns:
            continue
        records.append({
            "turns": turns,
            "metadata": {
                "source": "nicholasKluge/instruct-aira-dataset-v3",
                "conversation_id": row.get("conversation_id", ""),
                "split": "portuguese",
            },
        })
        if len(records) >= limit:
            break
    out = output / "instruct_aira_pt.jsonl"
    n = write_jsonl(out, records)
    print(f"  → {n} diálogos em {out}")
    return n


def import_ultrachat_local(ultrachat_dir: Path, output: Path, limit: int) -> int:
    shards = sorted(ultrachat_dir.glob("train_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"Nenhum train_*.jsonl em {ultrachat_dir}")

    print(f"[Syon/Conv] Lendo UltrachatBR local: {ultrachat_dir} ({len(shards)} shards)")
    records: list[dict] = []
    for shard in shards:
        with shard.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turns = parse_ultrachat_conversa(str(row.get("conversa", "")))
                if not turns:
                    continue
                records.append({
                    "turns": turns,
                    "metadata": {
                        "source": "recogna-nlp/UltrachatBR",
                        "conversation_id": row.get("conversation_id", ""),
                        "local_shard": shard.name,
                    },
                })
                if len(records) >= limit:
                    break
        if len(records) >= limit:
            break

    out = output / "ultrachat_br.jsonl"
    n = write_jsonl(out, records)
    print(f"  → {n} diálogos em {out}")
    return n


def import_ultrachat_br(output: Path, limit: int, *, local_dir: Path | None = None) -> int:
    if local_dir and local_dir.exists():
        return import_ultrachat_local(local_dir, output, limit)

    from datasets import load_dataset

    print(f"[Syon/Conv] Baixando UltrachatBR (limite {limit})...")
    ds = load_dataset("recogna-nlp/UltrachatBR", split="train", streaming=True)
    records: list[dict] = []
    for row in ds:
        turns = parse_ultrachat_conversa(str(row.get("conversa", "")))
        if not turns:
            continue
        records.append({
            "turns": turns,
            "metadata": {
                "source": "recogna-nlp/UltrachatBR",
                "conversation_id": row.get("conversation_id", ""),
            },
        })
        if len(records) >= limit:
            break
    out = output / "ultrachat_br.jsonl"
    n = write_jsonl(out, records)
    print(f"  → {n} diálogos em {out}")
    return n


def import_all(
    output_dir: Path,
    *,
    aira_limit: int = 50_000,
    ultrachat_limit: int = 30_000,
    force: bool = False,
    aira_parquet: Path | None = None,
    ultrachat_dir: Path | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "import_manifest.json"
    if not force and manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(prev.get("total_dialogues", 0)) >= (aira_limit + ultrachat_limit) * 0.9:
                print(f"[Syon/Conv] Já importado: {prev['total_dialogues']} diálogos")
                return prev
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    aira_n = import_aira(output_dir, aira_limit, local_parquet=aira_parquet)
    ultra_n = import_ultrachat_br(output_dir, ultrachat_limit, local_dir=ultrachat_dir)
    total = aira_n + ultra_n

    manifest = {
        "output": str(output_dir),
        "total_dialogues": total,
        "sources": {
            "instruct_aira_pt": aira_n,
            "ultrachat_br": ultra_n,
        },
        "aira_limit": aira_limit,
        "ultrachat_limit": ultrachat_limit,
        "local_paths": {
            "aira_parquet": str(aira_parquet) if aira_parquet else None,
            "ultrachat_dir": str(ultrachat_dir) if ultrachat_dir else None,
        },
        "real_datasets": True,
        "hardcoded": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Syon/Conv] ✓ {total} diálogos reais importados")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Importar conversação PT real")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/conversation")
    parser.add_argument("--aira-limit", type=int, default=50_000)
    parser.add_argument("--ultrachat-limit", type=int, default=30_000)
    parser.add_argument("--aira-parquet", type=Path, default=ROOT / "dataset-v3-bucket/portuguese-00000-of-00001.parquet")
    parser.add_argument("--ultrachat-dir", type=Path, default=ROOT / "UltrachatBR")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    aira_local = args.aira_parquet if args.aira_parquet.exists() else None
    ultra_local = args.ultrachat_dir if args.ultrachat_dir.exists() else None
    if aira_local:
        print(f"[Syon/Conv] Aira local: {aira_local}")
    if ultra_local:
        print(f"[Syon/Conv] UltrachatBR local: {ultra_local}")

    import_all(
        args.output,
        aira_limit=args.aira_limit,
        ultrachat_limit=args.ultrachat_limit,
        force=args.force,
        aira_parquet=aira_local,
        ultrachat_dir=ultra_local,
    )


if __name__ == "__main__":
    main()