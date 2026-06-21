"""Pipeline COMPLETO no Kaggle — Syon treinado DO ZERO."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
scratch = ROOT / "scripts/kaggle/run_scratch_kaggle.py"
subprocess.check_call([sys.executable, str(scratch)], cwd=str(ROOT))