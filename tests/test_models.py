"""Formas de salida, recuentos de parámetros y validación de configs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from janonet.models import ConfigError, build_model
from janonet.models.ensemble import ensemble_logit, ensemble_score
from janonet.models.scoring import LOGIT_EPS, logit, official_score

from .conftest import MODELS, model_cfg

#: Parámetros totales y entrenables medidos con el código original
#: (dataset_analysis/src/tfm/models/assemble.py::build_model) sobre las configs de
#: los runs finales. state_dict fp32: 16,2 / 64,9 / 71,8 / 71,8 MB.
EXPECTED_PARAMS = {
    "u100": (4_008_829, 4_008_829),
    "rs_ca": (16_171_197, 16_171_197),
    "rs_ca_tr": (17_898_173, 17_898_173),
    "e6": (17_898_173, 14_302_785),
}
EXPECTED_KEYS = {"u100": 360, "rs_ca": 496, "rs_ca_tr": 523, "e6": 523}


@pytest.fixture(scope="module", params=MODELS)
def built(request):
    cfg = model_cfg(request.param)
    return request.param, build_model(cfg["model"], pretrained=False).eval()


def test_formas_de_salida(built):
    name, model = built
    clip = torch.rand(2, 16, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(clip)
    assert tuple(out["frame_logits"].shape) == (2, 16)
    if model_cfg(name)["model"]["temporal"] == "transformer":
        assert out["mode"] == "transformer"
        assert tuple(out["video_logits"].shape) == (2,)
    else:
        assert out["mode"] == "mean" and out["video_logits"] is None
    score = official_score(out)
    assert tuple(score.shape) == (2,)
    assert torch.all((score >= 0) & (score <= 1))


def test_rejilla_fusionada_7x7(built):
    name, model = built
    with torch.no_grad():
        grid = model.fused_grid(torch.rand(1, 2, 3, 224, 224))
    expected_c = {"u100": 1280}.get(name, 512)
    assert tuple(grid.shape) == (2, expected_c, 7, 7)


def test_recuento_de_parametros_igual_al_original(built):
    name, model = built
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert (total, trainable) == EXPECTED_PARAMS[name]
    assert len(model.state_dict()) == EXPECTED_KEYS[name]


def test_e6_entrena_el_10_3_por_ciento_de_la_rgb():
    model = build_model(model_cfg("e6")["model"], pretrained=False)
    frac = model.rgb_backbone.trainable_fraction()
    assert frac == pytest.approx(412_160 / 4_007_548)


def test_bn_congeladas_quedan_en_eval_tras_train():
    model = build_model(model_cfg("e6")["model"], pretrained=False).train()
    assert not model.rgb_backbone.features[0].training
    assert model.rgb_backbone.features[8].training


@pytest.mark.parametrize(
    "cfg",
    [
        {"branches": ["rgb"], "fusion": "cross_attention", "temporal": "mean"},
        {"branches": ["rgb", "srm"], "fusion": "none", "temporal": "mean"},
        {"branches": ["rgb"], "fusion": "none", "temporal": "mean", "transformer": {"d_model": 256}},
        {"branches": ["rgb", "rgb"], "fusion": "none", "temporal": "mean"},
        {"branches": ["rgb"], "fusion": "none", "temporal": "residual_diff"},
    ],
)
def test_configs_imposibles_abortan(cfg):
    with pytest.raises(ConfigError):
        build_model(cfg, pretrained=False)


def test_logit_acotado_con_eps():
    assert logit(1.0) == pytest.approx(np.log((1 - LOGIT_EPS) / LOGIT_EPS))
    assert logit(1.0) == pytest.approx(13.815509557935018, abs=1e-9)


def test_ensamble_es_media_de_logits():
    a = np.array([0.9, 0.2, 1.0])
    b = np.array([0.6, 0.1, 0.5])
    np.testing.assert_allclose(ensemble_logit([a, b]), (logit(a) + logit(b)) / 2)
    s = ensemble_score([a, b])
    assert np.all((s > 0) & (s < 1))
    with pytest.raises(ValueError):
        ensemble_logit([a])
