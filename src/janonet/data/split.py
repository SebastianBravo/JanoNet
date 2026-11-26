"""Partición train/val/test por grupos de identidad, estratificada por (dataset, etiqueta).

El grupo es indivisible. Se minimiza

    coste = sum_j sum_c ( n_jc / N_c - f_j )^2

(proporción observada de cada celda ``(dataset, label)`` frente a la fracción objetivo
del conjunto), primero con un greedy de mayor a menor grupo y luego con refinamiento
local hasta que ningún traslado baje el coste. La normalización es por el total de la
celda ``N_c``, no por el objetivo del conjunto.

Los bloques de FF++/DeepFakeDetection (100 % FAKE) se pre-asignan: el mayor a train y
los otros a val y test.

Origen: ``dataset_analysis/code/notebook_analisis_identidades.ipynb``, celda 12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dfd_forced_assignment(frame: pd.DataFrame) -> dict[str, str]:
    dfd = frame[frame["identity_kind"] == "ffpp_dfd_actor"]
    if dfd.empty:
        return {}
    blocks = dfd.groupby("group_id").size().sort_values(ascending=False)
    targets = ["train", "val", "test"]
    return {gid: targets[i % len(targets)] for i, gid in enumerate(blocks.index)}


def group_split(
    frame: pd.DataFrame,
    fractions: dict[str, float],
    seed: int = 42,
    forced: dict[str, str] | None = None,
    refine: bool = True,
    max_passes: int = 40,
) -> pd.Series:
    """Asigna cada ``group_id`` a un conjunto. Devuelve una Series indexada por ``group_id``."""
    forced = forced or {}
    splits = list(fractions)
    cell = frame["dataset"] + " | " + frame["label"]
    piv = pd.crosstab(frame["group_id"], cell)
    gids = piv.index.to_numpy()
    M = piv.to_numpy(dtype=float)

    totals = M.sum(axis=0)
    targets = np.vstack([totals * fractions[s] for s in splits])
    inv2 = 1.0 / np.maximum(np.vstack([totals] * len(splits)), 1.0) ** 2

    rng = np.random.default_rng(seed)
    sizes = M.sum(axis=1)
    order = np.lexsort((rng.random(len(sizes)), -sizes))

    cur = np.zeros_like(targets)
    assign = np.full(len(gids), -1, dtype=int)
    fixed = {i for i, g in enumerate(gids) if g in forced}

    for i in order:
        v = M[i]
        if gids[i] in forced:
            k = splits.index(forced[gids[i]])
        else:
            k = int(np.argmin(((2 * (cur - targets) * v + v**2) * inv2).sum(axis=1)))
        cur[k] += v
        assign[i] = k

    if refine:
        movable = np.array([i for i in range(len(gids)) if i not in fixed])
        for _ in range(max_passes):
            rng.shuffle(movable)
            moved = 0
            for i in movable:
                v, a = M[i], assign[i]
                d = cur - targets
                remove = ((-2 * d[a] * v + v**2) * inv2[a]).sum()
                best, dest = 0.0, a
                for b in range(len(splits)):
                    if b == a:
                        continue
                    gain = remove + ((2 * d[b] * v + v**2) * inv2[b]).sum()
                    if gain < best - 1e-12:
                        best, dest = gain, b
                if dest != a:
                    cur[a] -= v
                    cur[dest] += v
                    assign[i] = dest
                    moved += 1
            if not moved:
                break

    return pd.Series([splits[k] for k in assign], index=gids, name="split")


def assign_splits(
    grouped: pd.DataFrame,
    fractions: dict[str, float],
    seed: int,
    force_dfd_blocks_apart: bool = True,
    max_passes: int = 40,
) -> pd.DataFrame:
    forced = dfd_forced_assignment(grouped) if force_dfd_blocks_apart else {}
    mapping = group_split(grouped, fractions, seed=seed, forced=forced, max_passes=max_passes)
    out = grouped.copy()
    out["split"] = out["group_id"].map(mapping)
    if out["split"].isna().any():
        raise RuntimeError("quedaron videos sin conjunto asignado")
    return out
