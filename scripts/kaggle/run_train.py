"""Entry point Kaggle — configura PYTHONPATH e roda treino Syon 3."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

if __name__ == "__main__":
    from training.hf.syon3_hf_trainer import main

    main()