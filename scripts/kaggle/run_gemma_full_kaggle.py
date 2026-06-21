"""Pipeline FULL Kaggle: dados -> SFT Gemma -> 5 fases treino."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/kaggle/working/syon") if Path("/kaggle/working/syon").exists() else Path(__file__).resolve().parents[2]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def env():
    e = os.environ.copy()
    paths = [str(ROOT), str(ROOT / "src")]
    if extra := e.get("PYTHONPATH"):
        paths.append(extra)
    e["PYTHONPATH"] = os.pathsep.join(paths)
    return e


def has_prebuilt_sft() -> bool:
    candidates = [
        ROOT / "data/gemma_sft/gemma_sft_train.jsonl",
        Path("/kaggle/input/datasets/regyfelipe/syon-gemma-sft/gemma_sft_train.jsonl"),
    ]
    for p in Path("/kaggle/input").rglob("gemma_sft_train.jsonl") if Path("/kaggle/input").exists() else []:
        candidates.append(p)
    return any(p.exists() and p.stat().st_size > 1_000_000 for p in candidates)


steps = []

if not has_prebuilt_sft():
    steps.append([
        sys.executable, "scripts/data/build_master_curriculum.py",
        "--output", str(ROOT / "data/raw"), "--min-samples", "10000", "--augment", "5",
    ])
    conv = ROOT / "data/raw/conversation/instruct_aira_pt.jsonl"
    if not conv.exists():
        steps.append([
            sys.executable, "scripts/data/import_conversation_datasets.py",
            "--output", str(ROOT / "data/raw/conversation"),
            "--aira-limit", "50000", "--ultrachat-limit", "100000",
        ])
    steps.append([
        sys.executable, "scripts/data/build_gemma_sft_dataset.py",
        "--raw", str(ROOT / "data/raw"), "--output", str(ROOT / "data/gemma_sft"),
        "--min-curriculum", "10000", "--no-conversation-import",
    ])
else:
    print("[Gemma FULL] SFT pre-built detectado - pulando preparacao")

steps.append([
    sys.executable, "scripts/kaggle/kaggle_gemma_full_train.py",
    "--config", str(ROOT / "training/configs/syon_gemma_full_kaggle.yaml"),
])

for i, cmd in enumerate(steps, 1):
    print(f"\n{'='*50}\n[Gemma FULL] {i}/{len(steps)}\n{'='*50}")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env())

print("\nOK -> /kaggle/working/syon-output/syon-gemma-full/syon-gemma-full")