# Procedencia de los valores

Cada valor numérico o de diseño de este repositorio se extrajo del código y de los
artefactos que **realmente corrieron** en el desarrollo. No se reinventó ninguno.
Este documento dice de dónde sale cada uno.

**Abreviaturas de origen**

- `DEV`: repositorio de desarrollo `dataset_analysis` (commit `edce201`, 2026-10-30).
  Los runs no registraron `git_sha` (`run_manifest.json -> git_sha: null`), así que la
  procedencia se fija por **fichero y hash**, no por commit.
- `PRE`: repositorio de preprocesamiento `deepfake_detection` (commit `cb5fe94`),
  script `code/preprocess_v2.py`.
- `RUN:<nombre>`: carpeta del run, p. ej. `DEV/runs/rgb/U100-s20260910/`.
- `EXP`: `DEV/export_resultados/` (extracción de resultados que alimentó el documento).
- `DOC`: `analisis_documento_JanoNet.zip` (scripts y CSV del documento).

> Se completa por fases: la Fase 1 cubre modelos, SRM y checkpoints; las Fases 2–5
> añaden preprocesamiento, partición, umbrales, escalera y métricas.

---

## 1. Modelos (Fase 1)

### 1.1 Configs de los modelos finales

Las configs de `configs/models/` copian la **config resuelta** que guardó cada run
(`config.yaml`), no las plantillas de `DEV/configs/`.

| Modelo | Config que corrió | Inicio (UTC) | Checkpoint |
|---|---|---|---|
| `u100` | `RUN:U100-s20260910/config.yaml` | 2026-10-22 09:26 | `best.ckpt` (época con máx. AP REAL en val) |
| `rs_ca` | `RUN:RS-ca-s20260910/config.yaml` | 2026-10-24 00:14 | `best.ckpt` |
| `rs_ca_tr` | `RUN:RS-ca-tr-s20260910/config.yaml` | 2026-10-23 18:06 | `best.ckpt` |
| `e6` | `DEV/runs/E6-seed20260910/config.yaml` | 2026-09-12 09:21 | `best.ckpt` |

| Clave | u100 | rs_ca | rs_ca_tr | e6 | Línea en `config.yaml` (u100 / rs_ca / rs_ca_tr / e6) |
|---|---|---|---|---|---|
| `branches` | `[rgb]` | `[rgb, srm]` | `[rgb, srm]` | `[rgb, srm]` | 25 / 35 / 44 / 42 |
| `fusion` | `none` | `cross_attention` | `cross_attention` | `cross_attention` | 27 / 38 / 47 / 45 |
| `temporal` | `mean` | `mean` | `transformer` | `transformer` | 28 / 39 / 48 / 46 |
| `rgb.freeze_until` | `null` | `null` | `null` | `features.7` | 24 / 24 / 24 / 22 |
| `srm.padding_mode` | — | `replicate` | `replicate` | `replicate` | — / 30 / 30 / 28 |
| `fusion_params` (d_model, n_heads, dropout) | — | 256, 8, 0.0 | 256, 8, 0.0 | 256, 8, 0.0 | — / 32-34 / 32-34 / 30-32 |
| `transformer` (d_model, n_layers, n_heads, dropout, pos, pool) | — | — | 256, 2, 8, 0.1, learned, mean | ídem | — / — / 36-42 / 34-40 |
| `epochs` | 10 | 10 | 10 | 20 | 2 |
| `batch_size` × `accum_steps` | 16 × 2 | 8 × 4 | 8 × 4 | 32 × 1 | 3-4 |
| `lr`, `weight_decay` | 1e-4, 1e-4 | ídem | ídem | ídem | 5-6 |
| `amp` | true | true | true | true | 11 |
| `seed` | 20260910 | 20260910 | 20260910 | 20260910 | 16 |
| `pos_weight_alpha` | 1.0 | 1.0 | 1.0 | (no existía; equivale a 1.0) | 19 |

**`pos_weight` efectivo** = `n_REAL / n_FAKE` de train = 11 218 / 64 249 =
**0,174601939329795** en los cuatro (`run_manifest.json:44` de cada run).

**Operativos.** `num_workers` era 16 en el HPC (nodo POWER9); aquí vale 4 por defecto
porque no altera el resultado. El resto de claves operativas se copian tal cual.

**Entorno de entrenamiento** (`run_manifest.json:15,41`): torch 2.1.2, torchvision
0.16.2, CUDA 12.2, Tesla V100-SXM2-16GB, HPC ZINE (Pontificia Universidad Javeriana).

### 1.2 Arquitectura

| Pieza | Valor | Origen |
|---|---|---|
| Rama RGB | EfficientNet-B0 `features` (ImageNet `IMAGENET1K_V1`), salida 1280×7×7 | `DEV/src/tfm/models/rgb_backbone.py:32,94-111` |
| Normalización RGB | media (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | `DEV/src/tfm/data/dataset.py:236-237` |
| BN congeladas en `eval()` | `train()` sobrescrito | `DEV/src/tfm/models/rgb_backbone.py:153-163` |
| Rama SRM | ResNet-18 sin `avgpool/fc`, `stem_stride=2`, `maxpool`, 512×7×7 | `DEV/src/tfm/models/srm_backbone.py:53-90` |
| Atención cruzada | proyección 1×1 a 256, dos `MultiheadAttention` (8 cabezas) RGB←SRM y SRM←RGB, residual + LayerNorm, concatenación (512 canales) | `DEV/src/tfm/models/fusion.py:76-143` |
| Pooling | `AdaptiveAvgPool2d(1)` tras la fusión | `DEV/src/tfm/models/assemble.py:213,255-260` |
| Temporal `mean` | lineal por cuadro; puntaje = media de las 16 probabilidades | `DEV/src/tfm/models/temporal.py:50-65` |
| Temporal `transformer` | encoder pre-norm, PE aprendida (trunc. normal 0.02), FF = 4·d_model, pooling medio, un logit por video | `DEV/src/tfm/models/temporal.py:90-197` |
| Puntaje oficial | `mean -> prob_mean`, `transformer -> prob_video` | `DEV/src/tfm/eval/aggregate.py:40-44` |

Se **descartó** el modo temporal `residual_diff` (experimento T6-H5) y las cabezas del
cribado (`DEV/src/tfm/models/heads.py`).

### 1.3 Rama SRM: kernels y normalización

| Valor | Origen |
|---|---|
| K1 = [[-1,2,-1],[2,-4,2],[-1,2,-1]] / 4 (3×3 en 5×5) | `DEV/src/tfm/data/srm.py:40-46,66` |
| K2 = 5×5 de segundo orden / 12 | `DEV/src/tfm/data/srm.py:49-55,66` |
| K3 = [1,-2,1] / 2 (1×3 horizontal en 5×5) | `DEV/src/tfm/data/srm.py:58-64,66` |
| Banco `(3,3,5,5)`: cada kernel sobre R, G y B, sumado | `DEV/src/tfm/data/srm.py:75-84` |
| Relleno `replicate` de 2 px | `DEV/src/tfm/data/srm.py:87-131` |
| Entrada [0, 1] (uint8/255), sin ImageNet | `DEV/src/tfm/data/srm.py:13-17` |
| Normalización `(r - mean) / std` | `DEV/src/tfm/data/srm.py:189-192` |
| `configs/srm_stats.json` | copia byte a byte de `DEV/data/training/srm_stats.json` (SHA256 `f5dd98bd…51bfd37`): 2000 videos de train, 8000 cuadros, semilla 20260812 |

Verificación: SHA256 de los kernels (`169ad62a…`) y del banco (`bafd5e9c…`) idénticos
a los del código original (`tests/test_srm.py`).

### 1.4 Puntaje, logit y ensamble

| Valor | Origen |
|---|---|
| `logit(clip(p, 1e-6, 1 - 1e-6))` en float64 | `EXP/scripts/lib_export.py:43,370-372` (`LOGIT_EPS` = el de `DEV/src/tfm/eval/aggregate.py:78`) |
| Ensamble = media de `logit_oficial` de U100 y RS-ca-tr | decisión del usuario (Fase 0). Reproduce AP REAL de prueba 0,8968 → 0,897 (fig. 11) |

### 1.5 Checkpoints de origen (SHA256)

| Modelo | Fichero | Tamaño | SHA256 |
|---|---|---|---|
| `u100` | `RUN:U100-s20260910/best.ckpt` (Pyky) | 48 605 984 B | `9e8ea4bb6669328baceaffe5792cb34bae8edf9a6b96c581a342b72081f9752d` |
| `rs_ca` | `RUN:RS-ca-s20260910/best.ckpt` (Pyky) | 194 705 460 B | `193b8eda326a59e1a263e63aaede16029b58ee0dcdd35994b41a976bbfa47c1d` |
| `rs_ca_tr` | `RUN:RS-ca-tr-s20260910/best.ckpt` (Pyky) | 215 462 216 B | `05393284ae98ffe52fbad910cb69cc2ee2f0a1818292893b85c587642d33493a` |
| `e6` | `DEV/runs/E6-seed20260910/best.ckpt` | 186 523 918 B | `b7a08f6fca4d0eab9c67dab2a8935b5ddb658e7ea5ac3b58b9c5a3f4b8b2ea9b` |

Son checkpoints de entrenamiento (modelo + optimizador + RNG). El `state_dict` va en la
clave `model`. Los pesos publicados (Fase 5) llevarán solo el `state_dict`.
