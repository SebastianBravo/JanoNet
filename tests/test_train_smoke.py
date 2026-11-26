"""Corrida mínima de punta a punta: entrenar U100 una época sobre 16 videos y evaluar su checkpoint.

Necesita la caché (``JANONET_CACHE``) y descarga los pesos de ImageNet la primera vez.
Las corridas de verificación de la Fase 4 (128 videos, los cuatro modelos, pérdida antes
y después) están documentadas en PROVENANCE.md §4.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from janonet import paths

pytestmark = pytest.mark.slow


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=paths.project_root())


def test_entrenar_y_evaluar_el_checkpoint(tmp_path):
    cache = os.environ.get("JANONET_CACHE")
    if not cache or not Path(cache).is_dir():
        pytest.skip("JANONET_CACHE no apunta a la caché")
    run = tmp_path / "u100"
    _run(
        "scripts/03_train.py",
        "--model",
        "u100",
        "--run-dir",
        str(run),
        "--subset",
        "16",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--accum-steps",
        "1",
        "--num-workers",
        "0",
        "--train-eval-subset",
        "0",
        "--device",
        "cpu",
        "--resume",
        "never",
    )
    assert (run / "best.ckpt").exists() and (run / "metrics_epoch.csv").exists()
    val = pd.read_csv(run / "val_scores_epoch00.csv")
    ids = tmp_path / "ids.txt"
    ids.write_text("\n".join(map(str, val["sample_id"])))
    out = tmp_path / "pred.csv"
    _run(
        "scripts/04_predict_split.py",
        "--models",
        "u100",
        "--split",
        "val",
        "--sample-ids",
        str(ids),
        "--checkpoint",
        f"u100={run / 'best.ckpt'}",
        "--device",
        "cpu",
        "--out",
        str(out),
    )
    pred = pd.read_csv(out).set_index("sample_id")["score"]
    # El checkpoint evaluado fuera del bucle da exactamente los puntajes del bucle.
    np.testing.assert_array_equal(pred.loc[val["sample_id"]].to_numpy(), val["score"].to_numpy())
