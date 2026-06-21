"""
Empacota Syon 3 para upload no Kaggle Models.

Uso:
    python scripts/kaggle/package_kaggle_model.py --source /kaggle/working/syon-output/syon-3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

REQUIRED = ("config.json", "pytorch_model.bin", "tokenizer.json")


def package(source: Path, output: Path, variation: str = "default") -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"Fonte não encontrada: {source}")

    missing = [f for f in REQUIRED if not (source / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos obrigatórios ausentes em {source}: {missing}\n"
            "Treine primeiro com run_scratch_kaggle.py"
        )

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for name in REQUIRED:
        shutil.copy2(source / name, output / name)
    if (source / "syon_model.json").exists():
        shutil.copy2(source / "syon_model.json", output / "syon_model.json")

    card = {
        "model_name": "Syon 3",
        "variation": variation,
        "from_scratch": True,
        "architecture": "syon_3",
        "description": "Syon 3 — LLM proprietário para programação e cybersegurança",
        "files": sorted(p.name for p in output.iterdir() if p.is_file()),
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
    }
    (output / "kaggle_model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"[Syon 3] Pacote Kaggle Model → {output}")
    for f in sorted(output.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.2f} MB)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Empacota Syon 3 para Kaggle Models")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/kaggle-model/syon-3")
    parser.add_argument("--variation", default="default")
    args = parser.parse_args()
    package(args.source, args.output, variation=args.variation)


if __name__ == "__main__":
    main()