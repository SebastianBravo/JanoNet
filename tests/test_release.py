"""Fichas de publicación y pesos: coherentes con el documento y con el registro."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from janonet import paths
from janonet.eval import reference as R
from janonet.models import build_model
from janonet.models.loading import load_config, load_model

MODELS = ("u100", "rs_ca", "rs_ca_tr", "e6")


def _card(name):
    return json.loads((paths.configs_dir() / "release" / f"{name}.json").read_text(encoding="utf-8"))


def _registry():
    return json.loads((paths.project_root() / "weights" / "registry.json").read_text(encoding="utf-8"))["models"]


@pytest.mark.parametrize("name", MODELS)
def test_umbrales_de_la_ficha_son_los_del_documento(name):
    card = _card(name)
    doc = R.document_thresholds()[card["study_config"]]
    assert card["thresholds"]["A"] == doc["A"] and card["thresholds"]["B"] == doc["B"]


@pytest.mark.parametrize("name", [*MODELS, "ensemble"])
def test_escalera_completa_y_json_valido(name):
    text = (paths.configs_dir() / "release" / f"{name}.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    card = json.loads(text)
    assert set(card["operating_points"]) == {"0.005", "0.01", "0.02", "0.05", "0.1"}
    for pts in card["operating_points"].values():
        assert [p["objetivo"] for p in pts if p["tipo"] == "fric"] == [3, 4, 5, 6, 8, 10, 15, 20]
        assert [p["objetivo"] for p in pts if p["tipo"] == "recall"] == [0.80, 0.85, 0.90, 0.95]
    # El punto de fricción 5 a pi = 0,02 es el umbral B.
    b = next(p for p in card["operating_points"]["0.02"] if p["tipo"] == "fric" and p["objetivo"] == 5)
    assert b["umbral"] == card["thresholds"]["B"]


def test_ensamble_reproduce_el_resumen_del_documento():
    t = _card("ensemble")["test_metrics"]
    assert round(t["standard_at_A"]["ap_real"], 3) == 0.897
    assert round(t["production_at_B_pi_0.02"]["tpr"], 3) == 0.941


@pytest.mark.parametrize("name", MODELS)
def test_registro_y_ficha_coinciden(name):
    reg, card = _registry()[name], _card(name)
    assert reg["sha256"] == card["weights"]["sha256"] and reg["file"] == f"{name}.pt"
    assert card["config"]["branches"] == load_config(name)["model"]["branches"]


@pytest.mark.parametrize("name", MODELS)
def test_pesos_presentes_verifican_y_cargan(name):
    reg = _registry()[name]
    p = paths.weights_dir() / reg["file"]
    if not p.exists():
        pytest.skip(f"{p} no descargado")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == reg["sha256"] and p.stat().st_size == reg["bytes"]
    sd = torch.load(p, map_location="cpu", weights_only=True)
    assert all(v.dtype == torch.float32 for v in sd.values() if v.is_floating_point())
    build_model(load_config(name)["model"], pretrained=False).load_state_dict(sd, strict=True)
    assert load_model(name, p) is not None
