"""Atalho: treino master no Kaggle."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
subprocess.check_call([
    sys.executable, "-m", "training.master_trainer",
    "--config", str(ROOT / "training/configs/master_kaggle.yaml"),
    "--data-dir", "/kaggle/working/syon/data/raw",
    "--augment", "30",
] + sys.argv[1:])