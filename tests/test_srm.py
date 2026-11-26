"""Rama SRM: kernels, banco de filtros, normalización y estadísticas."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from janonet import paths
from janonet.models import srm as S

#: SHA256 de ``srm_kernels().tobytes()`` (float64, (3, 5, 5)) y de
#: ``srm_filter_bank().numpy().tobytes()`` (float32, (3, 3, 5, 5)) calculados con el
#: código original (dataset_analysis/src/tfm/data/srm.py, commit edce201).
SHA_KERNELS_ORIGINAL = "169ad62a72a05be3f2570110af6c1bcd1790c2386c5676ab4783e03b18f5e967"
SHA_BANK_ORIGINAL = "bafd5e9cfdc334bdd2c4294ea21976401512eb29654c8243b9f2ef330bf72092"
#: SHA256 de dataset_analysis/data/training/srm_stats.json.
SHA_STATS_ORIGINAL = "f5dd98bd64995d1b2b395dfeaf356f866dd1c049e6d384a427a66d55751bfd37"


def test_kernels_identicos_al_original_por_hash():
    k = S.srm_kernels()
    assert k.dtype == np.float64 and k.shape == (3, 5, 5)
    assert hashlib.sha256(k.tobytes()).hexdigest() == SHA_KERNELS_ORIGINAL


def test_banco_identico_al_original_por_hash():
    b = S.srm_filter_bank()
    assert tuple(b.shape) == (3, 3, 5, 5)
    assert hashlib.sha256(b.numpy().tobytes()).hexdigest() == SHA_BANK_ORIGINAL


def test_kernels_valores_explicitos():
    k = S.srm_kernels()
    np.testing.assert_array_equal(k[0, 1:4, 1:4], np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4)
    np.testing.assert_array_equal(k[2, 2, 1:4], np.array([1, -2, 1]) / 2)
    assert k[1, 2, 2] == -12 / 12
    # Pasa-altos: cada kernel suma cero (K2/12 deja un redondeo de ~1e-16).
    np.testing.assert_allclose(k.sum(axis=(1, 2)), 0.0, atol=1e-15)


def test_cada_canal_suma_los_tres_canales_rgb():
    b = S.srm_filter_bank()
    for i in range(3):
        for c in range(3):
            torch.testing.assert_close(b[i, c], b[i, 0], rtol=0, atol=0)


def test_residuo_de_imagen_plana_es_cero_con_replicate():
    x = torch.full((1, 3, 32, 32), 0.37)
    r = S.apply_srm(x, S.srm_filter_bank())
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)


def test_estadisticas_identicas_al_original():
    p = paths.srm_stats_path()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == SHA_STATS_ORIGINAL
    st = S.load_srm_stats(p)
    assert st["padding_mode"] == "replicate"


def test_pesos_srm_no_son_parametros_ni_van_al_state_dict():
    capa = S.SRMResidual.from_stats_file(paths.srm_stats_path())
    assert list(capa.parameters()) == []
    assert capa.state_dict() == {}


def test_normalizacion_es_resta_y_division():
    st = S.load_srm_stats(paths.srm_stats_path())
    capa = S.SRMResidual.from_stats_file(paths.srm_stats_path())
    x = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    raw = capa.raw(x)
    m = torch.tensor(st["mean"]).reshape(1, 3, 1, 1)
    s = torch.tensor(st["std"]).reshape(1, 3, 1, 1)
    torch.testing.assert_close(capa(x), (raw - m) / s)


def test_padding_incoherente_aborta():
    with pytest.raises(ValueError):
        S.check_padding_mode("zeros", {"padding_mode": "replicate"})


@pytest.mark.dev_parity
def test_kernels_iguales_al_modulo_original(dev_repo):
    from tfm.data import srm as dev_srm

    np.testing.assert_array_equal(S.srm_kernels(), dev_srm.srm_kernels())
    torch.testing.assert_close(S.srm_filter_bank(), dev_srm.srm_filter_bank(), rtol=0, atol=0)
