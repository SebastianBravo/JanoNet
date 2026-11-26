"""Piezas del entrenamiento: orden de datos, muestreo, pérdida, checkpoints y reanudación."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from janonet.data import manifest as M
from janonet.data.clip_dataset import ResumableSampler, stratified_sample_ids
from janonet.train.checkpoint import CheckpointError, CheckpointManager, config_hash
from janonet.train.loop import TrainConfig, Trainer, compute_loss, pos_weight_from_counts, rewind_to_cycle


def _h(a) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def test_permutacion_por_epoca_igual_al_codigo_original():
    # Valores de dataset_analysis/src/tfm/data/sampler.py con la semilla de los modelos finales.
    p = ResumableSampler(75_467, base_seed=20260910, epoch=0).permutation()
    assert p[:8].tolist() == [17190, 70118, 35394, 26695, 21107, 17455, 50738, 56979]
    assert _h(p) == "b9759b3db1f11068"
    assert _h(ResumableSampler(75_467, base_seed=20260910, epoch=3).permutation()) == "8e34eae3f753c97a"


def test_muestreo_estratificado_igual_al_codigo_original():
    s = M.read_samples()
    if not M.has_fakeavceleb(s, "samples"):
        pytest.skip("faltan las filas locales de FakeAVCeleb")
    train = s[s["split"] == "train"]
    assert _h(stratified_sample_ids(train, 128, 20260910)) == "af0311ce93566987"
    assert _h(stratified_sample_ids(train, 9000, 20260910)) == "b592ff3d7f9d5718"


def test_pos_weight_del_estudio():
    s = M.read_samples()
    if not M.has_fakeavceleb(s, "samples"):
        pytest.skip("faltan las filas locales de FakeAVCeleb")
    y = s.loc[s["split"] == "train", "y"].to_numpy()
    assert pos_weight_from_counts(y) == pytest.approx(0.174601939329795, abs=1e-15)


def test_sampler_reanuda_desde_indice():
    a = ResumableSampler(100, base_seed=1, epoch=2)
    full = list(a)
    b = ResumableSampler(100, base_seed=1, epoch=2)
    b.set_start_index(37)
    assert list(b) == full[37:]


def test_rebobinado_al_ciclo_de_acumulacion():
    assert rewind_to_cycle(48, 6, 8, 4) == (32, 4)
    assert rewind_to_cycle(32, 4, 8, 4) == (32, 4)
    with pytest.raises(ValueError):
        rewind_to_cycle(30, 4, 8, 4)


def test_perdida_segun_modulo_temporal():
    crit = nn.BCEWithLogitsLoss()
    y = torch.tensor([1.0, 0.0])
    fl = torch.tensor([[2.0, 0.0], [-1.0, 1.0]])
    mean = compute_loss({"mode": "mean", "frame_logits": fl}, y, crit)
    assert mean == pytest.approx(float(crit(fl, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))))
    vl = torch.tensor([0.5, -0.5])
    tr = compute_loss({"mode": "transformer", "frame_logits": fl, "video_logits": vl}, y, crit)
    assert tr == pytest.approx(float(crit(vl, y)))


def test_checkpoint_de_otra_receta_no_se_reanuda(tmp_path):
    m = CheckpointManager(tmp_path, config_hash({"a": 1}))
    m.save({"x": torch.zeros(1)})
    assert m.load_latest() is not None
    with pytest.raises(CheckpointError):
        CheckpointManager(tmp_path, config_hash({"a": 2})).load_latest()


# ── Reanudación bit a bit con piezas mínimas ────────────────────────────────


class _Data(torch.utils.data.Dataset):
    def __init__(self, n=24):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randint(0, 255, (n, 2, 3, 4, 4), generator=g, dtype=torch.uint8)
        self.y = (torch.arange(n) % 3 == 0).long()
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], i, int(self.y[i])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(48, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        b, t = x.shape[:2]
        return {"mode": "mean", "frame_logits": self.lin(self.drop(x.reshape(b, t, -1))).squeeze(-1)}


def _trainer(run_dir, stop_after=None):
    torch.manual_seed(5)
    ds = _Data()
    sampler = ResumableSampler(len(ds), base_seed=11)
    cfg = TrainConfig(
        epochs=2, batch_size=4, accum_steps=2, amp=False, checkpoint_every_n_steps=0, log_every_n_steps=0, num_workers=0
    )
    # Generador propio: así crear el iterador del cargador no consume el RNG global
    # (el que usa el dropout), y la reanudación no desplaza las máscaras.
    g = torch.Generator().manual_seed(0)
    dl = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=4, drop_last=True, generator=g)
    dv = torch.utils.data.DataLoader(ds, batch_size=8, generator=torch.Generator().manual_seed(0))
    model = _Model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    t = Trainer(
        cfg=cfg,
        model=model,
        optimizer=opt,
        criterion=nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        train_loader=dl,
        val_loader=dv,
        dataset=ds,
        sampler=sampler,
        ckpt=CheckpointManager(run_dir, "h"),
        run_dir=run_dir,
    )
    if stop_after is not None:
        orig = t.train_one_epoch
        calls = {"n": 0}

        def patched():
            # Se pide la parada tras `stop_after` micro-lotes de la primera época.
            it = t.train_loader

            class _Cut:
                def __iter__(self_inner):
                    for k, batch in enumerate(it):
                        if calls["n"] == 0 and k == stop_after:
                            t.stop.request_stop()
                        yield batch

            t.train_loader = _Cut()
            try:
                return orig()
            finally:
                t.train_loader = it
                calls["n"] += 1

        t.train_one_epoch = patched
    return t


def test_reanudar_da_los_mismos_pesos(tmp_path):
    seguido = _trainer(tmp_path / "a")
    seguido.fit()
    cortado = _trainer(tmp_path / "b", stop_after=2)  # 3 micro-lotes con accum_steps=2: a mitad de ciclo
    assert cortado.fit()["interrupted"]
    reanudado = _trainer(tmp_path / "b")
    assert reanudado.resume()
    reanudado.fit()
    for k, v in seguido.model.state_dict().items():
        torch.testing.assert_close(reanudado.model.state_dict()[k], v, rtol=0, atol=0)
