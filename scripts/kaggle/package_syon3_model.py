"""
Empacota kl/ local para publicar no Kaggle Model (regyfelipe/syon-3-v1).

Uso:
    python scripts/kaggle/package_syon3_model.py
    python scripts/kaggle/package_syon3_model.py --kl kl --out dist/kaggle-models/syon-3-v1

Upload no Kaggle:
    1. https://www.kaggle.com/models/regyfelipe/syon-3-v1 → New version
    2. Framework: PyTorch | Upload a pasta com subpasta kl/
    3. Ou zip: dist/kaggle-models/syon-3-v1.zip (extrai para kl/ na raiz do upload)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = ("pytorch_model.bin", "config.json", "tokenizer.json")
OPTIONAL = ("syon_model.json", "model.safetensors")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def validate_kl(kl_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED if not (kl_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"kl/ incompleto em {kl_dir}\n"
            f"Faltando: {', '.join(missing)}\n"
            "Treine localmente ou copie kl/ com pytorch_model.bin antes de empacotar."
        )
    return list(REQUIRED) + [n for n in OPTIONAL if (kl_dir / n).exists()]


def package(kl_dir: Path, out_dir: Path, *, make_zip: bool) -> dict:
    files = validate_kl(kl_dir)
    dest_kl = out_dir / "kl"
    if dest_kl.exists():
        shutil.rmtree(dest_kl)
    dest_kl.mkdir(parents=True)

    manifest_files: list[dict] = []
    for name in files:
        src = kl_dir / name
        dst = dest_kl / name
        shutil.copy2(src, dst)
        size = src.stat().st_size
        manifest_files.append(
            {"name": name, "bytes": size, "size": human_size(size), "sha256": sha256_file(src)}
        )
        print(f"[package] {name} ({human_size(size)})")

    manifest = {
        "model": "syon-3-v1",
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "source_kl": str(kl_dir.resolve()),
        "kaggle_mount_hint": "/kaggle/input/models/regyfelipe/syon-3-v1/pytorch/default/<version>/kl",
        "files": manifest_files,
        "has_weights": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path: Path | None = None
    if make_zip:
        zip_path = out_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file() and path != zip_path:
                    zf.write(path, path.relative_to(out_dir.parent))
        print(f"[package] ZIP → {zip_path} ({human_size(zip_path.stat().st_size)})")

    return {"out_dir": str(out_dir), "zip": str(zip_path) if zip_path else None, "manifest": manifest}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Empacota kl/ para Kaggle Model syon-3-v1")
    parser.add_argument("--kl", type=Path, default=root / "kl")
    parser.add_argument("--out", type=Path, default=root / "dist" / "kaggle-models" / "syon-3-v1")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    report = package(args.kl.resolve(), args.out.resolve(), make_zip=not args.no_zip)
    print("\n[package] Pronto para upload no Kaggle Model.")
    print(f"  Pasta: {report['out_dir']}")
    if report["zip"]:
        print(f"  ZIP:   {report['zip']}")
    print("  Após Add Data no notebook, bootstrap deve mostrar has_weights: true")


if __name__ == "__main__":
    main()