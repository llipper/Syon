"""
Pipeline Syon 3 local — dados + tokenizer + pretrain + 4 fases.

Uso:
    python -m training.pipeline
    python -m training.pipeline --phase 2 --resume training/checkpoints/phase1/best
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "training/configs/syon3.yaml"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(ROOT), str(ROOT / "src")]
    if extra := env.get("PYTHONPATH"):
        paths.append(extra)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd, cwd=str(ROOT), env=_subprocess_env())


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_steps(
    config: dict[str, Any],
    config_path: Path,
    *,
    min_samples: int,
    augment: int,
    skip_conversation: bool,
    skip_pretrain: bool,
    phase: int,
    resume: Path | None,
) -> list[list[str]]:
    data_cfg = config.get("data", {})
    conv_cfg = data_cfg.get("conversation", {})
    data_dir = ROOT / data_cfg.get("raw_dir", "data/raw")
    config_str = str(config_path)
    tok_dir = ROOT / config["model"]["tokenizer_dir"]
    pretrain_best = ROOT / "training/checkpoints/pretrain/best"

    conv_cmd = [
        sys.executable,
        "scripts/data/import_conversation_datasets.py",
        "--output",
        str(data_dir / "conversation"),
        "--aira-limit",
        str(conv_cfg.get("aira_limit", 50_000)),
        "--ultrachat-limit",
        str(conv_cfg.get("ultrachat_limit", 100_000)),
    ]
    aira = conv_cfg.get("aira_parquet")
    ultra = conv_cfg.get("ultrachat_dir")
    if aira:
        conv_cmd.extend(["--aira-parquet", str(ROOT / aira)])
    if ultra:
        conv_cmd.extend(["--ultrachat-dir", str(ROOT / ultra)])

    all_steps: list[list[str]] = [
        [
            sys.executable,
            "scripts/data/build_syon3_curriculum.py",
            "--output",
            str(data_dir),
            "--min-samples",
            str(min_samples),
            "--augment",
            str(augment),
        ],
    ]
    if not skip_conversation:
        all_steps.append(conv_cmd)
    all_steps.extend(
        [
            [sys.executable, "scripts/data/process_pipeline.py", "--raw", str(data_dir)],
            [
                sys.executable,
                "scripts/data/train_syon_tokenizer.py",
                "--data-dir",
                str(data_dir),
                "--output",
                str(tok_dir),
            ],
        ]
    )
    if not skip_pretrain:
        all_steps.append(
            [
                sys.executable,
                "-m",
                "training.pretrain",
                "--config",
                config_str,
                "--data-dir",
                str(data_dir),
            ]
        )

    phase_resumes = {
        1: resume or pretrain_best,
        2: ROOT / "training/checkpoints/phase1/best",
        3: ROOT / "training/checkpoints/phase2/best",
        4: ROOT / "training/checkpoints/phase3/best",
    }
    phases = [phase] if phase > 0 else [1, 2, 3, 4]
    for p in phases:
        cmd = [
            sys.executable,
            "-m",
            "training.syon3_trainer",
            "--config",
            config_str,
            "--phase",
            str(p),
            "--data-dir",
            str(data_dir),
        ]
        r = phase_resumes.get(p)
        if r:
            cmd.extend(["--resume", str(r)])
        all_steps.append(cmd)

    return all_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Syon 3 — pipeline local completo")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--augment", type=int, default=5)
    parser.add_argument("--skip-conversation", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--phase", type=int, default=0, help="0=todas as fases")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    min_samples = args.min_samples or int(config.get("data", {}).get("min_samples", 10_000))

    steps = build_steps(
        config,
        config_path,
        min_samples=min_samples,
        augment=args.augment,
        skip_conversation=args.skip_conversation,
        skip_pretrain=args.skip_pretrain,
        phase=args.phase,
        resume=args.resume,
    )

    out_dir = config.get("output", {}).get("pretrained_dir", "models/pretrained/syon-3")
    print(f"[Syon 3] Pipeline local | {len(steps)} passos | saída: {out_dir}")

    for i, cmd in enumerate(steps, 1):
        print(f"\n{'='*50}\n[Syon 3] Passo {i}/{len(steps)}\n{'='*50}")
        run(cmd)

    print(f"\nOK → {ROOT / out_dir}")


if __name__ == "__main__":
    main()