"""Exporta modelo treinado para models/pretrained/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone


def export(source: Path, dest_name: str, models_dir: Path) -> Path:
    dest = models_dir / "pretrained" / dest_name
    dest.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)

    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "name": dest_name,
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Syon] Exportado → {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", default="syon-master-lora")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    export(args.source, args.name, args.models_dir)


if __name__ == "__main__":
    main()