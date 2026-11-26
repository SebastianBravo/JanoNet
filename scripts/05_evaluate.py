"""Tablas de resultados del documento desde predicciones por video.

    # con las predicciones de referencia del estudio (reference/), y comparación con
    # las tablas del documento (reference/document/)
    python scripts/05_evaluate.py --out outputs/evaluation --compare [--no-bootstrap]

    # con predicciones propias (salida de 04_predict_split.py), mismas columnas:
    # config, split, sample_id, score
    python scripts/05_evaluate.py --predictions outputs/predictions.csv --configs U100 RS-ca RS-ca-tr

Escribe r2_global, r2_escalera (con fricciones 15 y 20 además de 3–10), r2_prevalencia,
r2_umbrales.json, r3_* y r4_*, r5_segmentos. Con ``--compare`` imprime la diferencia
máxima frente a cada tabla del documento.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from janonet.eval import reference as R
from janonet.eval import study as S


def main() -> int:
    ap = argparse.ArgumentParser(description="Tablas de resultados del documento")
    ap.add_argument("--predictions", type=Path, default=None)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/evaluation"))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument(
        "--allow-subset",
        action="store_true",
        help="evaluar solo los videos con predicción (pruebas cortas; no son las cifras del documento)",
    )
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions, float_precision="round_trip") if args.predictions else R.load_predictions()
    configs = args.configs or S.STUDY_CONFIGS
    st = S.build_study(preds, R.samples_with_study_method(), configs, allow_subset=args.allow_subset)
    full = set(S.STUDY_CONFIGS) <= set(configs)
    args.out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    tables["r2_global.csv"] = S.table_global(st, configs)
    tables["r2_escalera.csv"] = S.table_ladder(st, configs, frictions=(3, 4, 5, 6, 8, 10, 15, 20))
    prev_cfgs = [c for c in ["U100", "R-tr", "RS-cat", "RS-ca", "RS-ca-tr", "E6"] if c in configs]
    tables["r2_prevalencia.csv"] = S.table_prevalence(st, prev_cfgs)
    (args.out / "r2_umbrales.json").write_text(
        json.dumps({c: {k: float(v) for k, v in st.thresholds[c].items()} for c in configs}, indent=1)
    )
    if full:
        samples = R.samples_with_study_method()
        train_donors = set(
            samples.loc[(samples["split"] == "train") & samples["df_donor_actor"].notna(), "df_donor_actor"]
        )
        tables["r3_subsets.csv"], macro = S.table_subsets(st, train_donors)
        tables["r3_macro.csv"] = macro.reset_index()
        tables["r3_demografia.csv"] = S.table_demography(st)
        tables["r3_fps.csv"] = S.table_fps(st)
        tables["r3_errores.csv"] = S.table_errors(st)
        tables["r5_segmentos.csv"] = S.table_segments(st)
        if not args.no_bootstrap:
            t0 = time.perf_counter()
            tables["r3_boot.csv"] = S.bootstrap_pairs(st)
            point, ci = S.ensemble_and_pi(st)
            tables["r4_point.csv"] = point
            ci.to_csv(args.out / "r4_ci.csv")
            print(f"bootstrap: {time.perf_counter() - t0:.0f} s")
    for name, df in tables.items():
        df.to_csv(args.out / name, index=name in ("r2_global.csv",))
    print(f"{len(tables)} tablas -> {args.out}")

    if args.compare:
        if not R.complete():
            print("aviso: faltan las filas locales de FakeAVCeleb; la comparación no puede ser exacta")
        worst = 0.0
        for name, df in tables.items():
            d = R.compare_table(df, name)
            num = {k: v for k, v in d.items() if not k.startswith("_")}
            m = max(num.values()) if num else 0.0
            worst = max(worst, m)
            print(f"  {name:<22} filas {int(d['_rows']):>4}  faltan {int(d['_missing'])}  máx |dif| {m:.3g}")
        if "r4_point.csv" in tables:
            d = R.compare_r4_ci(pd.read_csv(args.out / "r4_ci.csv", header=[0, 1], index_col=[0, 1, 2, 3]))
            print(f"  {'r4_ci.csv':<22} máx |dif| {d:.3g}")
            worst = max(worst, d)
        print(f"diferencia máxima global: {worst:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
