"""Fine-tune Gemma 2 com dataset Syon (QLoRA) no Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def find_gemma_path(input_dir: Path = Path("/kaggle/input")) -> Path:
    candidates: list[tuple[int, Path]] = []
    for cfg in input_dir.rglob("config.json"):
        text = cfg.read_text(encoding="utf-8", errors="ignore").lower()
        if "gemma" not in text and "gemma" not in str(cfg).lower():
            continue
        root = cfg.parent
        if not any((root / f).exists() for f in ("model.safetensors", "pytorch_model.bin", "model-00001-of-00002.safetensors")):
            # safetensors shards
            if not list(root.glob("*.safetensors")):
                continue
        ver = 0
        for part in root.parts:
            if part.isdigit():
                ver = max(ver, int(part))
        candidates.append((ver, root))
    if not candidates:
        raise FileNotFoundError("Gemma 2 não encontrado em /kaggle/input — Add Input: google/gemma-2")
    candidates.sort(key=lambda x: x[0])
    # Preferir 2b sobre 9b no Kaggle T4
    for _, p in candidates:
        if "2b" in str(p).lower():
            return p
    return candidates[-1][1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/syon_gemma_kaggle.yaml")
    parser.add_argument("--gemma-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/working/syon/data/raw"))
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    import yaml

    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    gemma = args.gemma_path or find_gemma_path()
    config.setdefault("model_params", {})["base_model"] = str(gemma)
    print(f"[Syon/Gemma] Base: {gemma}")

    # Gera dataset Syon se necessário
    if not args.data_dir.exists() or not list(args.data_dir.rglob("*.jsonl")):
        from scripts.data.build_master_curriculum import build

        build(args.data_dir, min_samples=int(config.get("data_params", {}).get("min_samples", 5000)))

    # Patch config temporário
    tmp_cfg = Path("/kaggle/working/syon_gemma_config.yaml")
    tmp_cfg.write_text(yaml.dump(config), encoding="utf-8")

    from scripts.kaggle.kaggle_train import train_on_kaggle

    train_on_kaggle(tmp_cfg, args.data_dir, args.max_steps)


if __name__ == "__main__":
    main()