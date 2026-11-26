"""Los checkpoints de los modelos finales cargan en las clases nuevas con strict=True."""

from __future__ import annotations

import pytest
import torch

from janonet.models import build_model, load_state_dict_from_checkpoint

from .conftest import MODELS, model_cfg

pytestmark = pytest.mark.checkpoints


@pytest.mark.parametrize("name", MODELS)
def test_carga_estricta_sin_claves_sobrantes_ni_faltantes(name, checkpoint_for):
    ckpt = checkpoint_for(name)
    sd = load_state_dict_from_checkpoint(ckpt)
    model = build_model(model_cfg(name)["model"], pretrained=False)
    expected = set(model.state_dict())
    assert set(sd) - expected == set(), "claves sobrantes"
    assert expected - set(sd) == set(), "claves faltantes"
    result = model.load_state_dict(sd, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for k, v in model.state_dict().items():
        assert v.shape == sd[k].shape and v.dtype == sd[k].dtype, k


@pytest.mark.parametrize("name", MODELS)
def test_inferencia_con_pesos_reales_es_finita(name, checkpoint_for):
    model = build_model(model_cfg(name)["model"], pretrained=False)
    model.load_state_dict(load_state_dict_from_checkpoint(checkpoint_for(name)), strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 16, 3, 224, 224, generator=torch.Generator().manual_seed(1)))
    assert torch.isfinite(out["frame_logits"]).all()
