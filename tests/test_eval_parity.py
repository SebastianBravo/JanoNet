"""Paridad de la evaluación con las tablas del documento (reference/document/).

Con las predicciones guardadas del estudio, el código de ``janonet.eval`` debe
reproducir:

- estimaciones puntuales, umbrales, escalera, fricción por prevalencia y conteos de
  errores: **exactas** (salvo el redondeo de coma flotante, ≤ 1e-12);
- intervalos bootstrap por video y por grupo: también exactos, porque se respeta la
  semilla (7) y el orden de llamadas al generador de los scripts del documento.

Necesita las filas locales de FakeAVCeleb (``manifests/local/`` y ``reference/local/``):
sin ellas el conjunto de prueba está incompleto y la comparación no tiene sentido.
"""

from __future__ import annotations

import numpy as np
import pytest

from janonet.eval import reference as R
from janonet.eval import study as S

pytestmark = pytest.mark.parity

TOL = 1e-12


@pytest.fixture(scope="module")
def st() -> S.Study:
    if not R.complete():
        pytest.skip("faltan las filas locales de FakeAVCeleb (manifests/local/, reference/local/)")
    return S.build_study(R.load_predictions(), R.samples_with_study_method())


@pytest.fixture(scope="module")
def train_donors() -> set[str]:
    s = R.samples_with_study_method()
    return set(s.loc[(s["split"] == "train") & s["df_donor_actor"].notna(), "df_donor_actor"])


def _check(table, name):
    d = R.compare_table(table, name)
    assert d.pop("_missing") == 0, f"{name}: faltan filas"
    d.pop("_rows")
    worst = max(d.values()) if d else 0.0
    assert worst <= TOL, f"{name}: diferencia máxima {worst:.3g} ({ {k: v for k, v in d.items() if v > TOL} })"


def test_umbrales_A_y_B_identicos(st):
    ref = R.document_thresholds()
    for c in S.STUDY_CONFIGS:
        assert float(st.thresholds[c]["A"]) == ref[c]["A"], c
        assert float(st.thresholds[c]["B"]) == ref[c]["B"], c


def test_tabla_global(st):
    _check(S.table_global(st, S.STUDY_CONFIGS), "r2_global.csv")


def test_escalera(st):
    _check(S.table_ladder(st, S.STUDY_CONFIGS), "r2_escalera.csv")


def test_sensibilidad_a_la_prevalencia(st):
    _check(S.table_prevalence(st, ["U100", "R-tr", "RS-cat", "RS-ca", "RS-ca-tr", "E6"]), "r2_prevalencia.csv")


def test_subconjuntos_y_macro(st, train_donors):
    sub, macro = S.table_subsets(st, train_donors)
    _check(sub, "r3_subsets.csv")
    _check(macro.reset_index(), "r3_macro.csv")


def test_demografia_fps_y_errores(st):
    _check(S.table_demography(st), "r3_demografia.csv")
    _check(S.table_fps(st), "r3_fps.csv")
    _check(S.table_errors(st), "r3_errores.csv")


def test_segmentos(st):
    _check(S.table_segments(st), "r5_segmentos.csv")


def test_ensamble_reproduce_ap_real_0897(st):
    from sklearn.metrics import average_precision_score

    st.add_ensemble()
    ap = average_precision_score(1 - st.test_y, 1 - st.test_scores["ENS"])
    assert round(ap, 3) == 0.897


@pytest.mark.slow
def test_bootstrap_por_video_y_por_grupo(st):
    _check(S.bootstrap_pairs(st), "r3_boot.csv")


@pytest.mark.slow
def test_bootstrap_ensamble_y_prevalencia(st):
    point, ci = S.ensemble_and_pi(st)
    _check(point, "r4_point.csv")
    assert R.compare_r4_ci(ci) <= TOL
    assert np.isfinite(ci.to_numpy()).all()
