"""Fuentes, reglas de limpieza, grafo de identidades, partición, tabla maestra y caché.

Corren con los manifiestos (sin videos). Los que necesitan las filas de FakeAVCeleb
(``manifests/local/``, fuera de git) se saltan si no están.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from janonet.data import cache, sources
from janonet.data import manifest as M
from janonet.preprocess.geometry import frontal_score, make_square_crop_from_box
from janonet.preprocess.video import evaluate_cleaning_rules, load_preprocess_config


@pytest.fixture(scope="module")
def pre_public() -> pd.DataFrame:
    return M.read_preprocessing(include_local=False)


@pytest.fixture(scope="module")
def samples_public() -> pd.DataFrame:
    return M.read_samples(include_local=False)


@pytest.fixture(scope="module")
def pre_any() -> pd.DataFrame:
    """Lo que haya: parte pública más la local si existe."""
    return M.read_preprocessing()


@pytest.fixture(scope="module")
def samples_any() -> pd.DataFrame:
    return M.read_samples()


@pytest.fixture(scope="module")
def pre() -> pd.DataFrame:
    df = M.read_preprocessing()
    if not M.has_fakeavceleb(df, "preprocessing"):
        pytest.skip("faltan las filas locales de FakeAVCeleb (manifests/local/)")
    return df


@pytest.fixture(scope="module")
def samples() -> pd.DataFrame:
    df = M.read_samples()
    if not M.has_fakeavceleb(df, "samples"):
        pytest.skip("faltan las filas locales de FakeAVCeleb (manifests/local/)")
    return df


# ── Manifiestos ─────────────────────────────────────────────────────────────


def test_parte_publica_sin_fakeavceleb(pre_public, samples_public):
    assert (pre_public["dataset"] != M.LOCAL_ONLY_DATASET).all()
    assert (samples_public["dataset"] != M.LOCAL_ONLY_DATASET).all()
    assert len(pre_public) == 140_423 - 10_209
    assert len(samples_public) == 94_234 - 5_787
    # Los sample_id publicados son los del estudio: FakeAVCeleb ocupaba 0..5786.
    assert samples_public["sample_id"].min() == 5_787 and samples_public["sample_id"].max() == 94_233


def test_sin_fakeavceleb_la_particion_no_se_reconstruye(pre_public):
    with pytest.raises(RuntimeError, match="FakeAVCeleb"):
        M.build_split(pre_public)


def test_manifiestos_tienen_los_tamanos_del_estudio(pre, samples):
    assert len(pre) == 140_423
    assert int(pre["meets_the_clean_rules"].sum()) == 94_234
    assert len(samples) == 94_234
    comp = samples.groupby(["split", "label"]).size().unstack()
    assert comp.loc["train"].to_dict() == {"FAKE": 64_249, "REAL": 11_218}
    assert comp.loc["val"].to_dict() == {"FAKE": 7_764, "REAL": 1_405}
    assert comp.loc["test"].to_dict() == {"FAKE": 8_134, "REAL": 1_464}
    assert samples["group_id"].nunique() == 946


def test_manifiestos_sin_rutas_absolutas(pre_any, samples_any):
    for df in (pre_any, samples_any):
        ids = df["video_id"]
        assert not ids.str.contains(r"^[A-Za-z]:|\\|^/", regex=True).any()
        assert ids.map(sources.prefix_of).notna().all()


def test_sample_id_sigue_el_orden_heredado(samples_any):
    keys = samples_any.sort_values("sample_id")["video_id"].map(sources.legacy_sort_key)
    assert keys.is_monotonic_increasing


def test_orden_de_filas_heredado(pre_any):
    keys = pre_any.sort_values("manifest_order")["video_id"].map(sources.legacy_manifest_order_key).tolist()
    assert keys == sorted(keys)


def test_reconstruir_particion_y_tabla_maestra_da_lo_mismo(pre, samples):
    rebuilt = M.build_samples(pre, M.build_split(pre))
    for col in M.SAMPLES_COLUMNS:
        a, b = rebuilt[col], samples[col]
        if a.dtype.kind == "f":
            assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True), col
        else:
            assert (a.astype("string").fillna("<NA>") == b.astype("string").fillna("<NA>")).all(), col


def test_sin_fuga_de_identidades_controladas(pre):
    from janonet.data.identity import leakage_report

    rep = leakage_report(M.build_split(pre), M.dfdc_meta_from_preprocessing(pre))
    controlled = rep.drop(index=["df_donor_actor"])
    assert (controlled["cross_splits"] == 0).all()
    # Limitación documentada: 81 de 100 actores donantes de DeeperForensics cruzan particiones.
    assert rep.loc["df_donor_actor", "cross_splits"] == 81


# ── Reglas de limpieza: reproducen el veredicto y el mensaje de las 140 423 filas ──


def test_reglas_de_limpieza_reproducen_el_manifiesto(pre_any):
    rules = load_preprocess_config()["cleaning"]
    nonfrontal = np.rint(pre_any["nonfrontal_fraction"] * pre_any["valid_face_frames"]).astype(int)
    for row, nf in zip(pre_any.itertuples(index=False), nonfrontal, strict=True):
        counts = dict(
            checked_frames=row.checked_frames,
            valid_face_frames=row.valid_face_frames,
            nonfrontal_frames=int(nf),
            too_many_faces_frames=row.too_many_faces_frames,
            missing_face_frames=row.missing_face_frames,
            processed_frames=row.processed_frames,
        )
        ok, msg, _ = evaluate_cleaning_rules(counts, rules)
        expected = "" if pd.isna(row.clean_rules_message) else row.clean_rules_message
        assert ok == bool(row.meets_the_clean_rules), row.video_id
        assert msg == expected, row.video_id


# ── Fuentes ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("video_id", "dataset", "label", "method"),
    [
        ("dfdc/aaagqkcdis.mp4", "dfdc", None, "unknown"),
        ("ffpp/manipulated_sequences/Deepfakes/c23/videos/000_003.mp4", "FaceForensics++", "FAKE", "Deepfakes"),
        ("ffpp/original_sequences/youtube/c23/videos/000.mp4", "deeper-forensics", "REAL", "original"),
        ("deeperforensics/manipulated_videos/end_to_end/000_M101.mp4", "deeper-forensics", "FAKE", "end_to_end"),
        ("fakeavceleb/RealVideo-RealAudio/African/men/id00076/00109.mp4", "FakeAVCeleb", "REAL", "original"),
    ],
)
def test_dataset_etiqueta_y_metodo_desde_video_id(video_id, dataset, label, method):
    assert sources.dataset_of(video_id) == dataset
    assert sources.label_from_path(video_id) == label
    assert sources.method_of(video_id) == method


def test_layouts_alternativos(monkeypatch, tmp_path):
    monkeypatch.setenv("JANONET_DF_ROOT", str(tmp_path / "df"))
    monkeypatch.setenv("JANONET_FAVC_ROOT", str(tmp_path / "favc"))
    df = sources.candidate_paths("deeperforensics/manipulated_videos/end_to_end/000_M101.mp4")
    assert tmp_path / "df" / "FAKE" / "end_to_end" / "000_M101.mp4" in df
    fv = sources.candidate_paths("fakeavceleb/FakeVideo-RealAudio/African/men/id00076/00109_1.mp4")
    assert tmp_path / "favc" / "FAKE" / "African" / "men" / "id00076" / "00109_1.mp4" in fv


# ── Geometría ───────────────────────────────────────────────────────────────


def test_recorte_cuadrado_margen_cero():
    # Caja del cuadro 0 del original FF++ 002 (RetinaFace): 190x292 -> lado 292.
    x1, y1, x2, y2 = make_square_crop_from_box(np.array([548, 159, 738, 451], np.float32), 1280, 720)
    assert (x2 - x1, y2 - y1) == (292, 292)
    assert y1 == round(305 + 0.05 * 292 - 146)


def test_puntaje_frontal_cero_para_cara_simetrica():
    lmk = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], np.float32)
    assert frontal_score(lmk) == pytest.approx(0.0)


# ── Caché y ventana ─────────────────────────────────────────────────────────


def test_bin_ida_y_vuelta(tmp_path):
    frames = np.random.default_rng(0).integers(0, 256, (32, 224, 224, 3), dtype=np.uint8)
    p = cache.write_bin(cache.bin_path(tmp_path, 12345), frames, 12345, 77)
    assert p == tmp_path / "12" / "12345.bin" and p.stat().st_size == 4_816_928
    header, back = cache.read_bin(p, verify_crc=True)
    assert (header.sample_id, header.window_start) == (12345, 77)
    assert np.array_equal(back, frames)
    assert np.array_equal(cache.read_window(p, 8, 16), frames[8:24])


def test_ventanas_iguales_al_codigo_original():
    assert cache.center_window_offset() == 8
    assert cache.max_window_offset(stride=2) == 1
    # Valores calculados con dataset_analysis/src/tfm/data/dataset.py.
    assert [cache.derive_seed(20260910, e, s) for e, s in [(0, 0), (3, 12345), (9, 94233)]] == [
        14279606124926765295,
        16141018570476252002,
        4647067427731745803,
    ]
    assert [cache.train_window_offset(20260910, e, s, 16) for e, s in [(0, 0), (3, 12345), (9, 94233)]] == [6, 13, 1]
