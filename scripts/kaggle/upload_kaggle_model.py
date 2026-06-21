"""Upload do Syon 3 para Kaggle Models via kagglehub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Syon 3 → Kaggle Models")
    parser.add_argument("--package-dir", type=Path, default=ROOT / "dist/kaggle-model/syon-3")
    parser.add_argument("--handle", required=True, help="ex: regyfelipe/syon-3/pytorch/default")
    parser.add_argument("--notes", default="Syon 3 — treinado do zero")
    parser.add_argument("--license", default="Apache 2.0")
    args = parser.parse_args()

    if not args.package_dir.is_dir():
        raise FileNotFoundError(f"Rode package_kaggle_model.py antes: {args.package_dir}")

    try:
        import kagglehub
    except ImportError:
        print("Instale: pip install kagglehub")
        sys.exit(1)

    print(f"[Syon 3] Upload → {args.handle}")
    kagglehub.model_upload(
        args.handle,
        str(args.package_dir),
        version_notes=args.notes,
        license_name=args.license,
    )
    print("[Syon 3] ✓ Model publicado no Kaggle")


if __name__ == "__main__":
    main()