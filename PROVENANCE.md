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

---

## 2. Datos y preprocesamiento (Fase 2)

### 2.1 Parámetros del preprocesamiento (`configs/preprocess.yaml`)

Los parámetros configurables salen de las columnas `param_*` de los cuatro manifiestos
de preprocesamiento (`DEV/data/raw/manifest_*_no_align_margin_0_0_resize_224.csv`),
**constantes en las 140 423 filas**; las constantes fijas, de `PRE/code/preprocess_v2.py`.

| Valor | Origen |
|---|---|
| `resize.size = 224`, `crop.margin = 0.0`, `detector.detect_every = 5`, `detector.min_face_size = 60`, `frontal.threshold = 0.18`, `cleaning.min_valid_fraction = 0.95`, `cleaning.max_nonfrontal_fraction = 0.20`, `crop.align_faces = false`, `video.codec = mp4v` | columnas `param_*` del manifiesto (ver también `PRE/commands/preprocess-command.example.txt`) |
| `detector.score_threshold = 0.9` | `PRE/code/preprocess_v2.py:354` |
| detección cada `detect_every` cuadros o mientras no haya detección válida | `PRE/code/preprocess_v2.py:519-552` |
| `frontal.landmark_smoothing_alpha = 0.7` (solo para la frontalidad) | `PRE/code/preprocess_v2.py:541-550` |
| puntaje de frontalidad (pesos 0.45 / 0.35 / 0.10 / 0.10) | `PRE/code/preprocess_v2.py:298-346` |
| `crop.center_shift_y = 0.05` | `PRE/code/preprocess_v2.py:265` |
| recorte con la caja sin suavizar de la última detección válida | `PRE/code/preprocess_v2.py:559-601` |
| `resize.interpolation = cubic` | `PRE/code/preprocess_v2.py:637-643` |
| `cleaning.max_too_many_faces_fraction = 0.05` | `PRE/code/preprocess_v2.py:697` |
| `cleaning.max_missing_face_frames = 0` | `PRE/code/preprocess_v2.py:704` |
| caché: 32 cuadros, `window_start = (processed_frames - 32) // 2` | `DEV/src/tfm/data/samples.py:52,422-435` |
| formato `.bin` (cabecera de 32 B `TFMF`, payload RGB uint8) | `DEV/src/tfm/data/binstore.py:1-65` |
| decodificación BGR→RGB, sin `CAP_PROP_POS_FRAMES` | `DEV/src/tfm/data/binstore.py:373-418` |
| clip de 16 cuadros, ventana central offset 8, offset de train `derive_seed(seed, época, sample_id)` (splitmix64) | `DEV/src/tfm/data/dataset.py:32-106` |

Verificación: las reglas portadas reproducen **el veredicto y el mensaje** de limpieza
de las 140 423 filas del manifiesto (`tests/test_data.py`).

### 2.2 Fuentes, etiquetas y métodos

| Valor | Origen |
|---|---|
| Etiquetas: DFDC desde su `metadata.json`; FF++ manipulado = FAKE; DeeperForensics por carpeta `FAKE/REAL`; FakeAVCeleb por tipo | `DEV/code/add_labels.py:134-176` |
| Subconjuntos: DeeperForensics `end_to_end` y `reenact_postprocess`; FakeAVCeleb `FakeVideo-RealAudio` y `RealVideo-RealAudio` | `dataset_unzip/code/unzip_deeperforensics.py` (`TARGET_CATEGORIES`) y `unzip_fakeavceleb.py` (`FOLDER_MAP`) |
| Método de FakeAVCeleb (faceswap, fsgan, wav2lip) | columna `method` de `meta_data.csv` de FakeAVCeleb v1.2 |
| `video_id` canónico y equivalencia con las rutas del estudio | `docs/provenance/export_dev_manifests.py` (`video_id_from_dev_output`) |

### 2.3 Grafo de identidades y partición (`configs/data.yaml`)

| Valor | Origen |
|---|---|
| `split.seed = 42`, `fractions = 0.80/0.10/0.10`, `pool_column = meets_the_clean_rules`, `force_dfd_blocks_apart = true` | `DEV/code/notebook_analisis_identidades.ipynb`, celda 2 |
| `dfdc_granularity = part` | `DEV/src/tfm/data/samples.py:37` (`identity_split_dfdc_part.csv`, la que entrenó; el notebook marca `original` como primaria pero no es la que se usó) |
| union-find, nodos por dataset | notebook, celda 8 |
| greedy + refinamiento local, `max_passes = 40` | notebook, celda 12 |
| `resize_bucket_edges`, `duration_bucket_edges` | `DEV/src/tfm/data/samples.py:58-79` |
| `sample_id` por orden de la ruta del video procesado | `DEV/src/tfm/data/samples.py:317-326` → `sources.legacy_sort_key` |

### 2.4 Manifiestos (`manifests/`)

Generados con `docs/provenance/export_dev_manifests.py` desde
`DEV/data/with_labels/manifest_all_datasets_with_labels.csv`,
`DEV/analysis_outputs/identity_split/identity_split_dfdc_part.csv`,
`DEV/data/training/samples.parquet` (fingerprint `d57973b4…6b12c748`) y los
`metadata.json` de DFDC. El script verifica, antes de escribir, que el código de
`janonet` reconstruye desde `preprocessing.csv.gz` **la misma partición, los mismos
`group_id` y los mismos `sample_id`** en las 94 234 filas (29 columnas idénticas).

Las filas de FakeAVCeleb (10 209 de preprocesamiento, 5 787 de la tabla maestra) se
escriben en `manifests/local/`, fuera de git, porque los términos de uso de FakeAVCeleb
prohíben distribuir datos derivados. Verificado que `01_build_manifest.py sources
--datasets fakeavceleb` reproduce desde la descarga sus 10 209 `video_id`, etiquetas,
métodos y posiciones de fila, y que `split --check --write-local` reconstruye la tabla
completa idéntica.

### 2.5 Paridad del preprocesamiento

`tests/test_preprocess_parity.py` reprocesa 20 videos desde el original (estratificados
por dataset y etiqueta, semilla 20261125; 11 downsample y 9 upsample) y compara los 32
cuadros de la ventana con la caché del estudio:

- **19 de 20 idénticos bit a bit**, incluido el clip central de 16 cuadros.
- `sample_id` 8079 (original FF++ 718): máximo 6 niveles en 3 564 de 4 816 896 valores
  (0,07 %), cuadros 20–26 de la ventana. Causa aislada: OpenCV 4.10/4.11 decodifica el
  video de origen distinto a la 4.12 con la que se generó la caché → RetinaFace da una
  caja desplazada 1 px en un cuadro → el recorte se propaga por el GOP de mp4v. Con
  OpenCV 4.12 sale idéntico (con TensorFlow en CPU y en GPU). OpenCV 4.12 exige
  numpy >= 2, incompatible con TensorFlow 2.10.
- Las magnitudes del manifiesto (cuadros procesados, tamaños de caja y de recorte,
  acción de redimensionado, contadores y veredicto de limpieza) coinciden en los 20.
- Detalle por video: `outputs/parity_preprocess.json` (fuera de git).
