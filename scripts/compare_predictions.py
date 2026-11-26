"""Compara predicciones propias con las predicciones guardadas del estudio.

    python scripts/compare_predictions.py outputs/predictions_test.csv [--saturated-only ids.txt]

Para cada configuración: diferencia absoluta máxima, media y percentil 99 del puntaje,
y cuántas decisiones cambian en el umbral B del documento. Los videos cuyo puntaje
guardado es exactamente 1,0 (saturación de la sigmoide en float16) se reportan aparte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from janonet.eval import reference as R
from janonet.eval import study as S
from janonet.models.ensemble import ensemble_score


def main() -> int:
    ap = argparse.ArgumentParser(description="Diferencia frente a las predicciones guardadas")
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    new = pd.read_csv(args.predictions, float_precision="round_trip")
    ref = R.load_predictions()
    ref = ref[ref["split"].isin(new["split"].unique())]
    st = S.build_study(R.load_predictions(), R.samples_with_study_method())
    thresholds = {c: float(v["B"]) for c, v in st.thresholds.items()}
    if "ENS" in set(new["config"]):
        st.add_ensemble()
        thresholds["ENS"] = float(st.thresholds["ENS"]["B"])
        piv = ref[ref["config"].isin(S.ENSEMBLE)].pivot(index=["split", "sample_id"], columns="config", values="score")
        ens = ensemble_score([piv[m].to_numpy(np.float32) for m in S.ENSEMBLE])
        ref = pd.concat(
            [
                ref,
                pd.DataFrame(
                    {
                        "config": "ENS",
                        "split": piv.index.get_level_values(0),
                        "sample_id": piv.index.get_level_values(1),
                        "score": ens,
                    }
                ),
            ]
        )
    u100_sat = set(ref.loc[(ref["config"] == "U100") & (ref["score"] >= 1.0), "sample_id"])

    m = new.merge(ref, on=["config", "split", "sample_id"], suffixes=("_new", "_ref"))
    m["d"] = (m["score_new"] - m["score_ref"]).abs()
    report = {}
    for c, g in m.groupby("config"):
        t = thresholds[c]
        flips = (g["score_new"] >= t) != (g["score_ref"] >= t)
        sat = g["sample_id"].isin(u100_sat)
        report[c] = {
            "n": int(len(g)),
            "max_abs": float(g.loc[~sat, "d"].max()) if (~sat).any() else None,
            "mean_abs": float(g.loc[~sat, "d"].mean()) if (~sat).any() else None,
            "p99_abs": float(g.loc[~sat, "d"].quantile(0.99)) if (~sat).any() else None,
            "flips_B": int(flips[~sat].sum()),
            "n_saturados_U100": int(sat.sum()),
            "saturados_max_abs": float(g.loc[sat, "d"].max()) if sat.any() else None,
            "saturados_flips_B": int(flips[sat].sum()),
        }
    print(pd.DataFrame(report).T.to_string())
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
