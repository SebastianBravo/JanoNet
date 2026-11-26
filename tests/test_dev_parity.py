"""Paridad numérica contra el código original (opcional).

Con los mismos pesos y la misma entrada, el modelo portado debe dar exactamente las
mismas salidas que ``tfm.models.assemble.build_model`` del repositorio de
desarrollo. Se salta si no están ``JANONET_DEV_REPO`` y los checkpoints.
"""

from __future__ import annotations

import pytest
import torch

from janonet import paths
from janonet.models import build_model, load_state_dict_from_checkpoint

from .conftest import MODELS, model_cfg

pytestmark = [pytest.mark.dev_parity, pytest.mark.checkpoints]


@pytest.mark.parametrize("name", MODELS)
def test_salidas_identicas_al_modelo_original(name, dev_repo, checkpoint_for):
    from tfm.models.assemble import build_model as dev_build_model

    cfg = model_cfg(name)["model"]
    sd = load_state_dict_from_checkpoint(checkpoint_for(name))

    dev_cfg = {k: v for k, v in cfg.items()}
    for branch in ("rgb", "srm"):
        if branch in dev_cfg:
            dev_cfg[branch] = {**dev_cfg[branch], "pretrained": False}
    dev = dev_build_model(dev_cfg, srm_stats_path=paths.srm_stats_path())
    dev.load_state_dict(sd, strict=True)
    new = build_model(cfg, pretrained=False)
    new.load_state_dict(sd, strict=True)
    dev.eval()
    new.eval()

    clip = torch.rand(2, 16, 3, 224, 224, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        a, b = dev(clip), new(clip)
    torch.testing.assert_close(b["frame_logits"], a["frame_logits"], rtol=0, atol=0)
    if a["video_logits"] is not None:
        torch.testing.assert_close(b["video_logits"], a["video_logits"], rtol=0, atol=0)
