"""Roda treino de conversação HF no Kaggle."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/kaggle/working/syon")
CONFIG = ROOT / "training/configs/syon3_hf_kaggle.yaml"


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError("Rode kaggle_bootstrap.py primeiro")

    env = os.environ.copy()
    paths = [str(ROOT), str(ROOT / "src")]
    if extra := env.get("PYTHONPATH"):
        paths.append(extra)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cmd = [
        sys.executable,
        "-m",
        "training.hf.syon3_hf_trainer",
        "--config",
        str(CONFIG),
    ]
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    print("\nOK → /kaggle/working/syon-output/syon-3")


if __name__ == "__main__":
    main()