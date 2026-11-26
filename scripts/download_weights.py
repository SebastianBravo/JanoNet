"""Descarga los pesos de los modelos y verifica su SHA256.

    python scripts/download_weights.py                 # u100, rs_ca y rs_ca_tr
    python scripts/download_weights.py --all           # también e6 (opcional)
    python scripts/download_weights.py --verify-only   # solo comprueba los ficheros presentes
    python scripts/download_weights.py --base-url <URL>  # aloja los ficheros en otra parte

Lee ``weights/registry.json`` (fichero, tamaño, SHA256 y URL de cada modelo). Un fichero
ya presente con el SHA256 correcto no se vuelve a descargar; uno con SHA256 distinto se
descarta. Los pesos se distribuyen bajo CC BY-NC 4.0 y están sujetos a los términos de
uso de los datasets de entrenamiento (ver MODEL_CARD.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from janonet import paths


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:  # noqa: S310 - URL del registro
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description="Descarga y verifica los pesos de JanoNet")
    ap.add_argument("models", nargs="*", help="modelos a descargar (por defecto, los no opcionales)")
    ap.add_argument("--all", action="store_true", help="incluir los opcionales (e6)")
    ap.add_argument("--base-url", default=None, help="URL base: se descarga <base-url>/<fichero>")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    wdir = paths.weights_dir()
    registry = json.loads((paths.project_root() / "weights" / "registry.json").read_text(encoding="utf-8"))["models"]
    names = args.models or [n for n, m in registry.items() if args.all or not m.get("optional")]
    unknown = [n for n in names if n not in registry]
    if unknown:
        print(f"modelos desconocidos: {unknown}; disponibles: {sorted(registry)}", file=sys.stderr)
        return 2
    wdir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name in names:
        entry = registry[name]
        dest = wdir / entry["file"]
        if dest.exists():
            if sha256(dest) == entry["sha256"]:
                print(f"{name}: OK ({dest})")
                continue
            print(f"{name}: SHA256 distinto, se descarta {dest}")
            dest.unlink()
        if args.verify_only:
            print(f"{name}: FALTA {dest}")
            failed += 1
            continue
        url = f"{args.base_url.rstrip('/')}/{entry['file']}" if args.base_url else entry.get("url")
        if not url:
            print(f"{name}: los pesos aún no están publicados (registry.json sin URL); usa --base-url", file=sys.stderr)
            failed += 1
            continue
        download(url, dest)
        if sha256(dest) != entry["sha256"]:
            print(f"{name}: SHA256 no coincide tras descargar; fichero eliminado", file=sys.stderr)
            dest.unlink()
            failed += 1
        else:
            print(f"{name}: descargado y verificado ({entry['bytes'] / 1e6:.1f} MB)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
