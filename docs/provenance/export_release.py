"""Prepara los pesos publicables y las fichas de cada modelo (no publica nada).

1. De cada checkpoint de entrenamiento del estudio extrae **solo el ``state_dict``**
   (float32) y lo guarda en ``weights/<modelo>.pt``; comprueba que carga con
   ``strict=True`` y que sus tensores son idénticos a los del checkpoint.
2. Escribe ``configs/release/<modelo>.json`` (y ``ensemble.json``) con config, umbrales
   A y B, escalera de operación por prevalencia, métricas de prueba, ``srm_stats``,
   versión y SHA256 de los pesos (``janonet.release``).
3. Escribe ``weights/registry.json``: fichero, tamaño, SHA256, licencia y URL de
   descarga de cada modelo, en el GitHub Release ``v0.1.0`` del repositorio.

Necesita las predicciones de referencia completas (con las filas locales de
FakeAVCeleb) para que umbrales y métricas sean los del documento.

Uso::

    python docs/provenance/export_release.py --checkpoint u100=<best.ckpt> --checkpoint rs_ca=<...> \\
        --checkpoint rs_ca_tr=<...> --checkpoint e6=<...>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from janonet import __version__, paths
from janonet.eval import reference as R
from janonet.eval import study as S
from janonet.models import build_model, load_state_dict_from_checkpoint
from janonet.models.loading import load_config
from janonet.models.srm import load_srm_stats
from janonet.release import model_card

WEIGHTS_LICENSE = (
    "CC BY-NC 4.0 (sujeto a los términos de uso de DFDC, FaceForensics++, DeeperForensics-1.0 y FakeAVCeleb)"
)
OPTIONAL = {"e6"}
#: Dónde se publican los pesos (GitHub Release del repositorio).
RELEASE_REPO = "SebastianBravo/JanoNet"
RELEASE_TAG = "v0.1.0"
RELEASE_BASE_URL = f"https://github.com/{RELEASE_REPO}/releases/download/{RELEASE_TAG}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", action="append", required=True, help="modelo=ruta (repetible)")
    args = ap.parse_args()
    ckpts = dict(c.split("=", 1) for c in args.checkpoint)
    if not R.complete():
        print("faltan las filas locales de FakeAVCeleb: los umbrales no serían los del documento", file=sys.stderr)
        return 1

    st = S.build_study(R.load_predictions(), R.samples_with_study_method())
    st.add_ensemble()
    wdir, rdir = paths.weights_dir(), paths.configs_dir() / "release"
    wdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    stats = load_srm_stats(paths.srm_stats_path())
    registry: dict[str, dict] = {}

    for name, ck in ckpts.items():
        cfg = load_config(name)
        sd = load_state_dict_from_checkpoint(ck)
        # float32 los tensores de coma flotante; los enteros (num_batches_tracked) se quedan como están.
        sd = {
            k: (v.detach().float() if v.is_floating_point() else v.detach()).contiguous().clone() for k, v in sd.items()
        }
        out = wdir / f"{name}.pt"
        torch.save(sd, out)
        back = torch.load(out, map_location="cpu", weights_only=True)
        model = build_model(cfg["model"], pretrained=False)
        model.load_state_dict(back, strict=True)
        orig = load_state_dict_from_checkpoint(ck)
        assert all(
            back[k].dtype == (torch.float32 if orig[k].is_floating_point() else orig[k].dtype)
            and torch.equal(back[k], orig[k].to(back[k].dtype))
            for k in orig
        ), f"{name}: tensores distintos"
        weights = {
            "file": out.name,
            "bytes": out.stat().st_size,
            "sha256": sha256(out),
            "format": "state_dict float32 (torch.save)",
            "license": WEIGHTS_LICENSE,
            "source_checkpoint_sha256": sha256(Path(ck)),
            "optional": name in OPTIONAL,
        }
        srm = stats if "srm" in cfg["model"]["branches"] else None
        card = model_card(st, cfg["meta"]["study_config"], name, {**cfg["model"], "meta": cfg["meta"]}, srm, weights)
        (rdir / f"{name}.json").write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
        registry[name] = {k: weights[k] for k in ("file", "bytes", "sha256", "license", "optional")}
        registry[name]["url"] = f"{RELEASE_BASE_URL}/{out.name}"
        print(
            f"{name}: {out.stat().st_size / 1e6:.1f} MB, sha256 {weights['sha256'][:16]}…, "
            f"A {card['thresholds']['A']:.6f}, B {card['thresholds']['B']:.6f}"
        )

    ens = load_config("ensemble")
    card = model_card(st, "ENS", "ensemble", ens, None, None)
    card["ensemble"] = ens["ensemble"]
    (rdir / "ensemble.json").write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"ensemble: A {card['thresholds']['A']:.6f}, B {card['thresholds']['B']:.6f}")
    (wdir / "registry.json").write_text(
        json.dumps({"janonet_version": __version__, "models": registry}, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
