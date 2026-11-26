"""Reproducción de las tablas de resultados del documento a partir de las predicciones por video.

Cada función reproduce un bloque de los scripts que generaron las cifras del documento
(``analisis_documento_JanoNet/scripts``), con la misma aritmética y, en el bootstrap,
**la misma semilla y el mismo orden de llamadas al generador**, de modo que los
intervalos coinciden exactamente:

=========================  ===========================================  ==================
función                    salida                                       origen
=========================  ===========================================  ==================
``table_global``           r2_global.csv (tablas VIII, X)                analisis2.py §1
``table_ladder``           r2_escalera.csv (fig. 9)                      analisis2.py §2
``table_prevalence``       r2_prevalencia.csv (tabla XII, fig. 14)       analisis2.py §3
``bootstrap_pairs``        r3_boot.csv                                   analisis3.py (BOOT)
``table_subsets``          r3_subsets.csv, r3_macro.csv                  analisis3.py
``table_demography``       r3_demografia.csv                             analisis3.py
``table_fps``              r3_fps.csv                                    analisis3.py
``table_errors``           r3_errores.csv (fig. 11a)                     analisis3.py
``ensemble_and_pi``        r4_point.csv, r4_ci.csv (fig. 10)             analisis4.py
``table_segments``         r5_segmentos.csv (figs. 12 y 13)              analisis5.py
=========================  ===========================================  ==================

Entrada: un ``Study`` con los puntajes oficiales de validación y prueba por
configuración, alineados por ``sample_id`` (orden ascendente), y los atributos de los
videos de prueba.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from janonet.eval.operating import (
    PI,
    PREVALENCES,
    clopper_pearson,
    production_metrics,
    standard_metrics,
    threshold_friction,
    threshold_max_f1_real,
    threshold_recall,
)
from janonet.models.ensemble import ensemble_score

#: Configuraciones del documento (las 13 de analisis2.py:8).
STUDY_CONFIGS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "U79", "U100", "R-tr", "RS-cat", "RS-ca", "RS-ca-tr"]
FAMILY = ["U100", "R-tr", "RS-cat", "RS-ca", "RS-ca-tr"]
ENSEMBLE = ("U100", "RS-ca-tr")
SEED_TABLES = 20261123
SEED_BOOT = 7
N_BOOT = 1000


@dataclass
class Study:
    """Puntajes oficiales por configuración y atributos de los videos de prueba."""

    val_scores: dict[str, np.ndarray]
    val_y: np.ndarray
    test_scores: dict[str, np.ndarray]
    test_y: np.ndarray
    base: pd.DataFrame  # atributos de prueba, en el mismo orden que test_scores
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for c, sv in self.val_scores.items():
            self.thresholds[c] = dict(
                A=threshold_max_f1_real(sv, self.val_y), B=threshold_friction(sv, self.val_y, 5.0)
            )

    def add_ensemble(self, name: str = "ENS", members: tuple[str, str] = ENSEMBLE) -> None:
        self.val_scores[name] = ensemble_score([self.val_scores[m] for m in members])
        self.test_scores[name] = ensemble_score([self.test_scores[m] for m in members])
        sv = self.val_scores[name]
        self.thresholds[name] = dict(A=threshold_max_f1_real(sv, self.val_y), B=threshold_friction(sv, self.val_y, 5.0))


def build_study(predictions: pd.DataFrame, samples: pd.DataFrame, configs: list[str] | None = None) -> Study:
    """``predictions``: ``sample_id, split, config, score``; ``samples``: tabla maestra."""
    configs = configs or STUDY_CONFIGS
    s = samples.set_index("sample_id")
    out = {}
    for split in ("val", "test"):
        ids = np.sort(samples.loc[samples["split"] == split, "sample_id"].to_numpy())
        scores = {}
        for c in configs:
            p = predictions[(predictions["config"] == c) & (predictions["split"] == split)].set_index("sample_id")
            if len(p) != len(ids) or not p.index.isin(ids).all():
                raise ValueError(f"predicciones incompletas de {c} en {split}: {len(p)} de {len(ids)}")
            # float32: la precisión en que se guardaron y con la que se calcularon las cifras
            # del documento (afecta a 1 - s en el AP con empates y al ensamble).
            scores[c] = p.loc[ids, "score"].to_numpy(dtype=np.float32)
        out[split] = (ids, scores, s.loc[ids, "y"].to_numpy())
    ids = out["test"][0]
    base = study_attributes(samples.set_index("sample_id").loc[ids].reset_index())
    return Study(out["val"][1], out["val"][2], out["test"][1], out["test"][2], base)


def study_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Atributos de segmento con los nombres de los scripts del documento."""
    parts = df["video_id"].str.split("/")
    is_favc = df["dataset"] == "FakeAVCeleb"
    favc_method = df["method_study"] if "method_study" in df else df["method"]
    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].to_numpy(),
            "label": df["y"].to_numpy(),
            "dataset": df["dataset"].to_numpy(),
            "metodo_manipulacion": df["method"].to_numpy(),
            "favc_metodo": np.where(is_favc, np.where(df["label"] == "REAL", "real", favc_method), None),
            "favc_etnia": np.where(is_favc, parts.str[2], None),
            "favc_genero": np.where(is_favc, parts.str[3], None),
            "group_id": df["group_id"].to_numpy(),
            "resize_action": df["resize_action"].to_numpy(),
            "fps": df["fps"].to_numpy(),
            "donor": df["df_donor_actor"].to_numpy(),
        }
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# analisis2.py
# ─────────────────────────────────────────────────────────────────────────────


def table_global(st: Study, configs: list[str]) -> pd.DataFrame:
    rows = []
    for c in configs:
        s, y = st.test_scores[c], st.test_y
        th = st.thresholds[c]
        a = standard_metrics(s, y, th["A"])
        b = production_metrics(s, y, th["B"])
        bv = production_metrics(st.val_scores[c], st.val_y, th["B"])
        lo, hi = clopper_pearson(b["fp"], b["fp"] + b["tn"])
        tlo, thi = clopper_pearson(b["tp"], b["tp"] + b["fn"])
        rows.append(
            dict(
                cfg=c,
                **a,
                **{"B_" + k: v for k, v in b.items()},
                B_fric_val=bv["fric"],
                B_fpr_lo=lo,
                B_fpr_hi=hi,
                B_tpr_lo=tlo,
                B_tpr_hi=thi,
                thA=th["A"],
                thB=th["B"],
            )
        )
    return pd.DataFrame(rows).set_index("cfg")


def table_ladder(
    st: Study, configs: list[str], frictions=(3, 4, 5, 6, 8, 10), recalls=(0.80, 0.85, 0.90, 0.95), pi: float = PI
) -> pd.DataFrame:
    rows = []
    for c in configs:
        sv, s = st.val_scores[c], st.test_scores[c]
        for tgt in frictions:
            t = threshold_friction(sv, st.val_y, tgt, pi)
            rows.append(dict(cfg=c, tipo="fric", objetivo=tgt, umbral=t, **production_metrics(s, st.test_y, t, pi)))
        for rt in recalls:
            t = threshold_recall(sv, st.val_y, rt)
            rows.append(dict(cfg=c, tipo="recall", objetivo=rt, umbral=t, **production_metrics(s, st.test_y, t, pi)))
    return pd.DataFrame(rows)


def table_prevalence(st: Study, configs: list[str], prevalences=PREVALENCES) -> pd.DataFrame:
    rows = []
    for c in configs:
        for pi in prevalences:
            t = threshold_friction(st.val_scores[c], st.val_y, 5.0, pi)
            rows.append(dict(cfg=c, pi=pi, umbral=t, **production_metrics(st.test_scores[c], st.test_y, t, pi)))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# analisis3.py
# ─────────────────────────────────────────────────────────────────────────────

BOOT_PAIRS = [
    ("RS-cat", "U100"),
    ("RS-ca", "RS-cat"),
    ("RS-ca-tr", "RS-ca"),
    ("R-tr", "U100"),
    ("RS-ca-tr", "U100"),
    ("E2", "E0"),
    ("E3", "E2"),
    ("E6", "E3"),
    ("E6", "E0"),
]
BOOT_METRICS = ["ap_real", "recall_B", "prec_fake_B", "f1_fake_B", "fric_B"]


def _groups(st: Study):
    grp = st.base["group_id"].to_numpy()
    ug = np.unique(grp)
    return ug, {g: np.where(grp == g)[0] for g in ug}


def bootstrap_indices(
    st: Study, rng: np.random.Generator, kind: str, n: int = N_BOOT, min_real_group: int | None = 50
) -> list[np.ndarray]:
    """Réplicas por video (estratificado por clase) o por grupo de identidad.

    En el bootstrap por grupo de analisis3 se descartan (consumiendo el generador) las
    réplicas con menos de ``min_real_group`` REAL; en el de analisis4 no
    (``min_real_group=None``).
    """
    y = st.test_y
    i_real, i_fake = np.where(y == 0)[0], np.where(y == 1)[0]
    ug, gidx = _groups(st)
    out = []
    for _ in range(n):
        if kind == "video":
            idx = np.concatenate([rng.choice(i_real, len(i_real)), rng.choice(i_fake, len(i_fake))])
        else:
            gs = rng.choice(ug, len(ug))
            idx = np.concatenate([gidx[g] for g in gs])
            if min_real_group is not None and (y[idx] == 0).sum() < min_real_group:
                continue
        out.append(idx)
    return out


def _stat_pair(st: Study, c: str, idx: np.ndarray) -> np.ndarray:
    s, y = st.test_scores[c][idx], st.test_y[idx]
    m = production_metrics(s, y, st.thresholds[c]["B"])
    return np.array([average_precision_score(1 - y, 1 - s), m["tpr"], m["prec_fake"], m["f1_fake"], m["fric"]])


def bootstrap_pairs(st: Study, pairs=BOOT_PAIRS, singles=None, seed: int = SEED_BOOT, n: int = N_BOOT) -> pd.DataFrame:
    """IC 95 % de diferencias emparejadas y de cada configuración, por video y por grupo."""
    singles = singles or FAMILY + ["E0", "E6"]
    rng = np.random.default_rng(seed)
    full = np.arange(len(st.test_y))
    rows = []
    for kind in ("video", "grupo"):
        idxs = bootstrap_indices(st, rng, kind, n)
        for a, b in pairs:
            diffs = np.array([_stat_pair(st, a, i) - _stat_pair(st, b, i) for i in idxs])
            pa, pb = _stat_pair(st, a, full), _stat_pair(st, b, full)
            lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
            rows += [
                dict(boot=kind, a=a, b=b, metrica=m, diff=pa[j] - pb[j], lo=lo[j], hi=hi[j])
                for j, m in enumerate(BOOT_METRICS)
            ]
        for c in singles:
            vals = np.array([_stat_pair(st, c, i) for i in idxs])
            pc = _stat_pair(st, c, full)
            lo, hi = np.percentile(vals, [2.5, 97.5], axis=0)
            rows += [
                dict(boot=kind, a=c, b="", metrica=m, diff=pc[j], lo=lo[j], hi=hi[j])
                for j, m in enumerate(BOOT_METRICS)
            ]
    return pd.DataFrame(rows)


SWAP_FF = ["Deepfakes", "FaceSwap", "FaceShifter", "DeepFakeDetection"]


def _subsets(base: pd.DataFrame, train_donors: set[str]):
    ff_real = (base.dataset == "deeper-forensics") & (base.label == 0)
    yield "DFDC", base.dataset == "dfdc"
    yield "FF++ (c23)", (base.dataset == "FaceForensics++") | ff_real
    yield "DeeperForensics", base.dataset == "deeper-forensics"
    yield "FakeAVCeleb", base.dataset == "FakeAVCeleb"
    swap = (
        (base.label == 0)
        | (base.dataset == "dfdc")
        | (base.dataset == "deeper-forensics")
        | ((base.dataset == "FaceForensics++") & base.metodo_manipulacion.isin(SWAP_FF))
        | ((base.dataset == "FakeAVCeleb") & base.favc_metodo.isin(["faceswap", "fsgan"]))
    )
    yield "Solo face swap", swap
    yield "Global", base.dataset.notna()
    shared = (base.dataset == "deeper-forensics") & (base.label == 1) & base.donor.isin(train_donors)
    yield "Global sin donantes compartidos", ~shared


def table_subsets(st: Study, train_donors: set[str], configs=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas por fuente, solo face swap y sin donantes compartidos; y su media macro."""
    configs = configs or FAMILY + ["E0", "E2", "E6"]
    rows = []
    for name, mask in _subsets(st.base, train_donors):
        mk = mask.to_numpy()
        for c in configs:
            s, yy = st.test_scores[c][mk], st.test_y[mk]
            if (yy == 0).sum() == 0 or (yy == 1).sum() == 0:
                continue
            th = st.thresholds[c]
            a = standard_metrics(s, yy, th["A"])
            pm = production_metrics(s, yy, th["B"])
            rows.append(
                dict(
                    sub=name,
                    cfg=c,
                    nR=(yy == 0).sum(),
                    nF=(yy == 1).sum(),
                    auc=a["auc"],
                    ap_real=a["ap_real"],
                    acc05=((s >= 0.5) == yy).mean(),
                    f1_fake05=standard_metrics(s, yy, 0.5)["f1_fake"],
                    acc_A=a["acc"],
                    f1_real_A=a["f1_real"],
                    f1_fake_A=a["f1_fake"],
                    p_real_A=a["p_real"],
                    r_real_A=a["r_real"],
                    tprB=pm["tpr"],
                    fprB=pm["fpr"],
                    fricB=pm["fric"],
                )
            )
    sub = pd.DataFrame(rows)
    sources = ["DFDC", "FF++ (c23)", "DeeperForensics", "FakeAVCeleb"]
    macro = sub[sub["sub"].isin(sources)].groupby("cfg")[["auc", "ap_real", "f1_real_A", "tprB", "fprB"]].mean()
    return sub, macro


def table_demography(st: Study, configs=("U100", "RS-ca-tr")) -> pd.DataFrame:
    fav = st.base.dataset == "FakeAVCeleb"
    rows = []
    for c in configs:
        for col in ("favc_etnia", "favc_genero"):
            for g, sub in st.base[fav].groupby(col):
                mk = sub.index.to_numpy()
                s, yy = st.test_scores[c][mk], st.test_y[mk]
                p = s >= st.thresholds[c]["B"]
                rows.append(
                    dict(
                        cfg=c,
                        var=col,
                        grupo=g,
                        nF=(yy == 1).sum(),
                        nR=(yy == 0).sum(),
                        tpr=(p & (yy == 1)).sum() / max((yy == 1).sum(), 1),
                        fp=(p & (yy == 0)).sum(),
                    )
                )
    return pd.DataFrame(rows)


def table_fps(st: Study, configs=("U100", "RS-ca-tr")) -> pd.DataFrame:
    fps_b = pd.cut(st.base.fps, [0, 25.5, 30.5, 100], labels=["<=25", "25-30", ">30"])
    rows = []
    for b, sub in st.base.groupby(fps_b, observed=False):
        mk = sub.index.to_numpy()
        yy = st.test_y[mk]
        for c in configs:
            s = st.test_scores[c][mk]
            rows.append(
                dict(
                    fps=b,
                    cfg=c,
                    nR=(yy == 0).sum(),
                    nF=(yy == 1).sum(),
                    auc=roc_auc_score(yy, s) if 0 < yy.mean() < 1 else np.nan,
                )
            )
    return pd.DataFrame(rows)


ERROR_PAIRS = [("U100", "RS-ca-tr"), ("U100", "RS-cat"), ("U100", "RS-ca"), ("U100", "R-tr"), ("E0", "E1")]


def table_errors(st: Study, pairs=ERROR_PAIRS) -> pd.DataFrame:
    """Falsos no detectados y reales señalados en el umbral B, y cuántos comparten dos modelos."""
    y = st.test_y
    rows = []
    for a, b in pairs:
        pa, pb = st.test_scores[a] >= st.thresholds[a]["B"], st.test_scores[b] >= st.thresholds[b]["B"]
        fn_a, fn_b = (~pa) & (y == 1), (~pb) & (y == 1)
        fp_a, fp_b = pa & (y == 0), pb & (y == 0)
        rows.append(
            dict(
                a=a,
                b=b,
                fnA=fn_a.sum(),
                fnB=fn_b.sum(),
                fn_ambos=(fn_a & fn_b).sum(),
                fpA=fp_a.sum(),
                fpB=fp_b.sum(),
                fp_ambos=(fp_a & fp_b).sum(),
                jac_fn=(fn_a & fn_b).sum() / max((fn_a | fn_b).sum(), 1),
                jac_fp=(fp_a & fp_b).sum() / max((fp_a | fp_b).sum(), 1),
            )
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# analisis4.py
# ─────────────────────────────────────────────────────────────────────────────

ENS_PAIRS = [
    ("ENS", "RS-ca-tr"),
    ("ENS", "U100"),
    ("RS-ca-tr", "U100"),
    ("RS-ca", "U100"),
    ("R-tr", "U100"),
    ("RS-cat", "U100"),
]


def ensemble_and_pi(st: Study, pis=(0.005, 0.02), seed: int = SEED_BOOT, n: int = N_BOOT):
    """Diferencias productivas emparejadas (con el ensamble) a varias prevalencias; puntos e IC 95 %."""
    if "ENS" not in st.test_scores:
        st.add_ensemble()
    th_pi = {
        c: {pi: threshold_friction(st.val_scores[c], st.val_y, 5.0, pi) for pi in PREVALENCES}
        for c in ["U100", "RS-ca-tr", "ENS", "R-tr", "RS-ca", "RS-cat"]
    }
    y = st.test_y

    def stat(c, idx, pi):
        s, yy = st.test_scores[c][idx], y[idx]
        m = production_metrics(s, yy, th_pi[c][pi], pi)
        return np.array(
            [average_precision_score(1 - yy, 1 - s), roc_auc_score(yy, s), m["tpr"], m["prec_fake"], m["fric"]]
        )

    rng = np.random.default_rng(seed)
    out = []
    for kind in ("video", "grupo"):
        for idx in bootstrap_indices(st, rng, kind, n, min_real_group=None):
            for pi in pis:
                for a, b in ENS_PAIRS:
                    out.append([kind, pi, a, b, *(stat(a, idx, pi) - stat(b, idx, pi))])
    cols = ["ap", "auc", "tpr", "prec", "fric"]
    D = pd.DataFrame(out, columns=["boot", "pi", "a", "b", *cols])
    full = np.arange(len(y))
    F = pd.DataFrame(
        [[pi, a, b, *(stat(a, full, pi) - stat(b, full, pi))] for pi in pis for a, b in ENS_PAIRS],
        columns=["pi", "a", "b", *cols],
    )
    Q = D.groupby(["boot", "pi", "a", "b"]).quantile([0.025, 0.975]).unstack()
    return F, Q


# ─────────────────────────────────────────────────────────────────────────────
# analisis5.py
# ─────────────────────────────────────────────────────────────────────────────

SEGMENT_CONFIGS = ["U100", "R-tr", "RS-cat", "RS-ca", "RS-ca-tr", "ENS"]
_FF_SEGMENTS = [
    ("Deepfakes", "FF++ Deepfakes"),
    ("FaceSwap", "FF++ FaceSwap"),
    ("FaceShifter", "FF++ FaceShifter"),
    ("DeepFakeDetection", "FF++ DFD"),
    ("Face2Face", "FF++ Face2Face"),
    ("NeuralTextures", "FF++ NeuralTextures"),
]


def table_segments(st: Study, configs=SEGMENT_CONFIGS) -> pd.DataFrame:
    """Recall por fuente y método y FPR por fuente, más recall y FPR por acción de redimensionado, en B."""
    if "ENS" not in st.test_scores:
        st.add_ensemble()
    base = st.base

    def seg_fake():
        f = base.label == 1
        yield "DFDC", f & (base.dataset == "dfdc")
        for mm, nm in _FF_SEGMENTS:
            yield nm, f & (base.dataset == "FaceForensics++") & (base.metodo_manipulacion == mm)
        yield "DeeperForensics", f & (base.dataset == "deeper-forensics")
        for mm in sorted(base.favc_metodo.dropna().unique()):
            yield "FAVC " + mm, f & (base.dataset == "FakeAVCeleb") & (base.favc_metodo == mm)

    def seg_real():
        r = base.label == 0
        yield "DFDC", r & (base.dataset == "dfdc")
        yield "FF++/DeeperForensics (originales)", r & (base.dataset == "deeper-forensics")
        yield "FakeAVCeleb", r & (base.dataset == "FakeAVCeleb")

    rows = []

    def add(kind, name, mask):
        mk = np.asarray(mask.fillna(False), dtype=bool)
        n = mk.sum()
        for c in configs:
            k = (st.test_scores[c][mk] >= st.thresholds[c]["B"]).sum()
            lo, hi = clopper_pearson(k, n)
            with np.errstate(invalid="ignore", divide="ignore"):
                val = k / n
            rows.append(dict(tipo=kind, seg=name, n=n, cfg=c, k=k, val=val, lo=lo, hi=hi))

    for kind, gen in (("recall", seg_fake), ("fpr", seg_real)):
        for name, mask in gen():
            add(kind, name, mask)
    for ra in ("upsample", "downsample"):
        for cl, kind in ((1, "recall"), (0, "fpr")):
            add(kind, "resize " + ra, (base.resize_action == ra) & (base.label == cl))
    return pd.DataFrame(rows)
