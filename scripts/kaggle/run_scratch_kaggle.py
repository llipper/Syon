"""Pipeline Syon DO ZERO no Kaggle — tokenizer + arquitetura + pré-treino + 3 fases."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/kaggle/working/syon") if Path("/kaggle/working/syon").exists() else Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw"
CONFIG = ROOT / "training/configs/syon_scratch_kaggle.yaml"
PRETRAIN_BEST = ROOT / "training/checkpoints/pretrain/best"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(ROOT), str(ROOT / "src")]
    if extra := env.get("PYTHONPATH"):
        paths.append(extra)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd, cwd=str(ROOT), env=_subprocess_env())


steps: list[list[str]] = [
    [sys.executable, "scripts/data/build_master_curriculum.py", "--output", str(DATA), "--min-samples", "5000"],
    [sys.executable, "scripts/data/process_pipeline.py", "--raw", str(DATA)],
    [sys.executable, "scripts/data/train_syon_tokenizer.py", "--data-dir", str(DATA),
     "--output", str(ROOT / "models/tokenizer/syon3-bpe")],
    [sys.executable, "-m", "training.pretrain", "--config", str(CONFIG), "--data-dir", str(DATA)],
    [sys.executable, "-m", "training.master_trainer", "--config", str(CONFIG), "--phase", "1",
     "--data-dir", str(DATA), "--resume", str(PRETRAIN_BEST)],
    [sys.executable, "-m", "training.master_trainer", "--config", str(CONFIG), "--phase", "2",
     "--data-dir", str(DATA), "--resume", str(ROOT / "training/checkpoints/phase1/best")],
    [sys.executable, "-m", "training.master_trainer", "--config", str(CONFIG), "--phase", "3",
     "--data-dir", str(DATA), "--resume", str(ROOT / "training/checkpoints/phase2/best")],
]

for i, cmd in enumerate(steps, 1):
    print(f"\n{'='*50}\n[Syon Scratch] Passo {i}/{len(steps)}\n{'='*50}")
    run(cmd)

print("\n✓ SYON 3 COMPLETO → /kaggle/working/syon-output/syon-3")