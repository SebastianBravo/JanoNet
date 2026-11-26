"""Predicción de uno o varios videos con los modelos publicados.

    python scripts/predict.py examples/deeperforensics_end_to_end_100_M121.mp4
    python scripts/predict.py video.mp4 --objective friction --value 4 --pi 0.01 --json outputs/pred.json

Cada video pasa por el mismo preprocesamiento del entrenamiento (RetinaFace, reglas de
limpieza, recorte a 224, clip central de 16 cuadros); si no lo cumple, se informa «no
procesable» con el motivo. Con el puntaje de cada modelo se decide en el punto de
operación de su ficha (``configs/release/<modelo>.json``) que corresponde al objetivo del
negocio y a la prevalencia ``--pi``:

- ``max_detection``: recall ≥ 0,95 en validación;
- ``friction``: fricción ≤ ``--value`` (3, 4, 5, 6, 8 o 10) en validación a ``--pi``;
- ``recall``: recall ≥ ``--value`` (0,80, 0,85, 0,90 o 0,95) en validación;
- ``compromise`` (por defecto): fricción ≤ 5 a ``--pi`` (el umbral B si ``--pi 0.02``).

Necesita los pesos (``python scripts/download_weights.py``) y el extra ``[preprocess]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from janonet import paths
from janonet.inference.video import predict_video
from janonet.models.loading import load_config, load_model
from janonet.release import OBJECTIVES, load_card, pick_operating_point

ENSEMBLE_MEMBERS = ("u100", "rs_ca_tr")


def main() -> int:
    ap = argparse.ArgumentParser(description="Predice si uno o varios videos son deepfakes")
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["u100", "rs_ca_tr", "rs_ca", "ensemble"],
        help="modelos a usar; 'ensemble' necesita u100 y rs_ca_tr (se cargan solos)",
    )
    ap.add_argument("--objective", default="compromise", choices=OBJECTIVES)
    ap.add_argument("--value", type=float, default=None, help="fricción o recall objetivo (con friction/recall)")
    ap.add_argument("--pi", type=float, default=0.02, choices=[0.005, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--json", type=Path, default=None, help="guarda el resultado completo en JSON")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.objective in ("friction", "recall") and args.value is None:
        ap.error(f"--objective {args.objective} necesita --value")
    if args.threads:
        torch.set_num_threads(args.threads)

    networks = [m for m in args.models if m != "ensemble"]
    if "ensemble" in args.models:
        networks += [m for m in ENSEMBLE_MEMBERS if m not in networks]
    models = {}
    for m in networks:
        w = paths.weights_dir() / f"{m}.pt"
        if not w.exists():
            print(f"falta {w}: ejecuta  python scripts/download_weights.py", file=sys.stderr)
            return 2
        models[m] = load_model(m, w, device="cpu")
    points = {m: pick_operating_point(load_card(m), args.pi, args.objective, args.value) for m in args.models}

    from janonet.preprocess.detect import FaceDetector

    detector = FaceDetector(0.9)
    results = []
    for video in args.videos:
        r = predict_video(video, models, detector=detector, ensemble=ENSEMBLE_MEMBERS)
        out = {"video": str(video), "processable": r.processable, "reasons": r.reasons, "models": {}}
        print(f"\n{video}")
        if not r.processable:
            print("  NO PROCESABLE, solicitar recaptura: " + "; ".join(r.reasons))
        for m in args.models:
            if not r.processable:
                break
            p, s = points[m], r.scores[m]
            fake = s >= p["umbral"]
            out["models"][m] = {
                "score": s,
                "threshold": p["umbral"],
                "decision": "FAKE" if fake else "REAL",
                "operating_point": {"type": p["tipo"], "target": p["objetivo"], "pi": args.pi},
                "test_at_threshold": p["test"],
            }
            name = load_config(m)["meta"]["display_name"]
            print(
                f"  {name:<32} puntaje {s:.4f}  umbral {p['umbral']:.4f}  -> {'FALSO' if fake else 'real'}"
                f"   (en prueba: recall {p['test']['tpr']:.3f}, fricción {p['test']['fric']:.2f})"
            )
        out["timings_s"] = r.timings
        print(f"  tiempo: {r.timings.get('total', r.timings.get('preprocesamiento', 0)):.1f} s")
        results.append(out)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
