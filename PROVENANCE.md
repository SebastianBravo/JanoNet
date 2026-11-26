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

---

## 3. Evaluación (Fase 3)

### 3.1 Definiciones (`src/janonet/eval/operating.py`)

Las del documento v12, tomadas de `DOC/scripts/analisis2.py` (sha256 del zip
`d0cc7f6e…2d2e2`):

| Valor | Origen |
|---|---|
| π = 0,02; fricción = 1 + (1 − π)/π · FPR/TPR | `analisis2.py:10-11` |
| Umbral B = menor umbral de validación con fricción ≤ 5 a π = 0,02, **sin** el factor 1,173 | `analisis2.py:12-24` (el desarrollo usaba `t_fric5_comp`, con factor: no se porta) |
| Umbral A = máximo F1 REAL sobre 2001 cuantiles del puntaje de validación | `analisis2.py:25-32` (distinto del `threshold.json` del desarrollo, que barría todos los puntajes) |
| Escalera: fricción 3, 4, 5, 6, 8, 10 (+15 y 20) y recall 0,80, 0,85, 0,90, 0,95 | `analisis2.py:68-72`; 15 y 20 de `EXP/escalera_operacion.csv` |
| Métricas productivas (precisión, F1 y alertas reponderadas a π) | `analisis2.py:36-42` |
| IC de Clopper–Pearson | `analisis2.py:33-35` |
| Sensibilidad a π ∈ {0,005, 0,01, 0,02, 0,05, 0,10} | `analisis2.py:76-80` |
| Bootstrap por video (estratificado) y por grupo (`group_id`), 1000 réplicas, semilla 7 | `analisis3.py:4,16-27`; `analisis4.py:14-27` |
| Ensamble: media de logits con ε = 1e-6, en float32 | `analisis3.py:109-113`; `analisis4.py:3-7` |

### 3.2 Predicciones de referencia (`reference/`)

`reference/predictions.csv.gz` sale de `EXP/predicciones_por_video.parquet` (la
`pv.parquet` de los scripts) con `docs/provenance/export_dev_predictions.py`: semilla
20260910, 13 configuraciones, validación y prueba, columna `score_oficial` (en E4–E6 es
la salida por video reinferida, `prob_video_reinferido`). Las filas de FakeAVCeleb van a
`reference/local/` (fuera de git). `reference/document/` son los CSV agregados del zip,
copiados sin cambios.

### 3.3 Detalles que hacen falta para la paridad exacta

- **Los puntajes son float32.** La tabla del estudio los guardaba en float32 y el
  documento calculó `1 − s` (AP REAL) y el ensamble en esa precisión. Con float64 el AP
  REAL de U100 cambia en 1,5e-7 por los empates en 1,0 (videos saturados).
- **scikit-learn 1.7.2.** Con la 1.5.2 el AP con empates difiere (1,5e-7 en U100); la 1.7
  reproduce el documento. Funciona con numpy 1.26.
- **`thA`/`thB` de `r2_global.csv` están redondeados a float32** en el CSV del documento;
  los valores completos son los de `r2_umbrales.json`, que coinciden bit a bit.
- **Método de FakeAVCeleb.** El documento asignó el método cruzando `meta_data.csv` sin
  el tipo de video, y en 15 videos limpios (3 en prueba) un nombre repetido en
  `FakeVideo-FakeAudio` le dio `faceswap-wav2lip` a videos que en
  `FakeVideo-RealAudio` son `wav2lip`. El manifiesto guarda el método correcto; la regla
  del documento se reproduce solo para la paridad (`sources.fakeavceleb_method_study`,
  `manifests/local/fakeavceleb_method_study.csv.gz`). Afecta a dos filas de
  `r5_segmentos.csv` (`FAVC faceswap-wav2lip`, n = 3, y `FAVC wav2lip`).

### 3.4 Resultado de la paridad

Con `scripts/05_evaluate.py --compare` (y `tests/test_eval_parity.py`):

| Tabla | Filas | Diferencia máxima |
|---|---|---|
| r2_global, r2_escalera, r2_prevalencia, r2_umbrales | 13 / 130 / 30 / 13 | ≤ 1,0e-15 (umbrales idénticos) |
| r3_subsets, r3_macro, r3_demografia, r3_fps, r3_errores | 56 / 8 / 14 / 6 / 5 | ≤ 1,1e-16 |
| r5_segmentos | 120 | 3,3e-16 |
| r3_boot (bootstrap por video y por grupo) | 160 | 2,2e-16 |
| r4_point, r4_ci (ensamble, π = 0,005 y 0,02) | 12 / 24 | 1,8e-15 |

Las diferencias son de redondeo de coma flotante (orden de suma y versión de scipy en
los IC binomiales): los intervalos bootstrap son exactos porque se respetan la semilla
y el orden de llamadas al generador.

### 3.5 Reinferencia con el pipeline nuevo (CPU, float32)

`scripts/04_predict_split.py` + `scripts/compare_predictions.py` sobre 353 videos de
prueba: 200 al azar (semilla 20261126) y los 156 en que el `prob_mean` guardado de U100
valía exactamente 1,0 (3 en común). Checkpoints de §1.5, clip central de la caché.

**Hallazgo: la rama SRM depende de la precisión con que se calculó el residuo.** Los
modelos se entrenaron y evaluaron con AMP (`run_manifest.json`, `amp: true`), así que la
convolución SRM corría en float16. En float32 puro, los modelos con SRM se alejan mucho
de sus puntajes publicados; redondeando a float16 solo la entrada, los pesos y la
respuesta del banco SRM (`srm.residual_precision: float16`, ver `models/srm.py`), la
diferencia cae al orden de la que deja el resto de AMP. En GPU con `autocast` fp16 la
diferencia es ≤ 0,001 en los cuatro modelos (60 videos, GTX 1080).

| Modelo | float32 puro: máx / media | residuo SRM en float16: máx / media / p99 | Decisiones distintas en B |
|---|---|---|---|
| U100 | 0,012 / 0,0005 | 0,012 / 0,0005 / 0,0072 | 0 |
| RS-ca | 0,189 / 0,019 | 0,0036 / 0,0004 / 0,0028 | 0 (4 en float32) |
| RS-ca-tr | 0,257 / 0,024 | 0,018 / 0,0015 / 0,016 | 0 (6 en float32) |
| E6 | 0,761 / 0,087 | 0,020 / 0,0008 / 0,0080 | 0 (18 en float32) |
| Ensamble | 0,253 / 0,013 | 0,008 / 0,0007 / 0,0066 | 0 (3 en float32) |

**Videos saturados de U100.** Los 156 videos con `prob_mean = 1,0` guardado también dan
1,0 en float32: sus 16 logits por cuadro superan 16,6, donde la sigmoide float32 también
satura. Diferencia 0, sin cambios de decisión. El ensamble sobre ellos difiere ≤ 8e-6.

Tiempo en CPU (Windows, i7-6700, 4 núcleos): 0,54 s/video U100 y ~1,0 s/video los
modelos con SRM (clip de 16 cuadros, lote de 4).

---

## 4. Entrenamiento (Fase 4)

| Valor | Origen |
|---|---|
| AdamW sobre los parámetros entrenables, sin programador de tasa | `DEV/scripts/04_train.py:449-452` (`scheduler=None`, `:519`) |
| `BCEWithLogitsLoss(pos_weight = alpha · n_REAL/n_FAKE)` del train efectivo | `DEV/scripts/04_train.py:431-446`; `DEV/src/tfm/models/assemble.py:149-181` |
| Pérdida: media de 16 BCE (`mean`) o una BCE del logit de video (`transformer`) | `DEV/src/tfm/train/loop.py:177-186` |
| AMP float16 + `torch.cuda.amp.GradScaler`; acumulación con pérdida / `accum_steps` | `DEV/src/tfm/train/loop.py:65-82,732-760` |
| `torch.manual_seed(seed)` antes de construir el modelo | `DEV/scripts/04_train.py:423` |
| Ventana de train `derive_seed(seed, época, sample_id) % 17`; permutación por época `default_rng(derive_seed(seed, época))` | `DEV/src/tfm/data/dataset.py:95-106`; `DEV/src/tfm/data/sampler.py:75-88` |
| `drop_last=True`; val en orden natural, clip central | `DEV/scripts/04_train.py:275-283` |
| `best.ckpt` = máximo AP REAL de validación con el puntaje oficial | `DEV/src/tfm/train/loop.py:835-856` |
| Subconjunto de train para la curva de sobreajuste (`train_eval_subset: 9000`) | `DEV/scripts/04_train.py:263-271` |
| Estadísticas SRM: 2000 × 4 cuadros, semillas 20260812 y 1337 | `DEV/scripts/03b_compute_srm_stats.py` |
| Tiempos y memoria por modelo (README) | `metrics_epoch.csv` de cada run (`epoch_seconds`, `vram_peak_mb`); reloj total con evaluación: `DEV/CLAUDE.md`, «Semillas de la RGB ajustada §0» |

**Verificado:**

- Orden de datos idéntico al original: la permutación de la época 0 y la 3 con la semilla
  20260910 y los subconjuntos estratificados de 128 y 9000 videos coinciden por hash
  (`tests/test_train.py`).
- `pos_weight` del train completo = 0,174601939329795, el de los `run_manifest.json`.
- `02_preprocess.py srm-stats` recalcula `configs/srm_stats.json` desde la caché con la
  misma muestra (mismo solape de 58 videos entre semillas y mismos 401 408 000 píxeles
  por canal): diferencia relativa en la desviación ≤ 2,5e-8 (orden de suma CPU/GPU).
- Reanudación bit a bit, también cortando **a mitad de un ciclo de acumulación** con
  dropout activo (`tests/test_train.py`). Dos cambios frente al original para lograrlo
  sin depender de que el modelo no tenga dropout: cada `DataLoader` lleva su propio
  generador (crear el iterador consumía el RNG global) y, al guardar a mitad de ciclo, se
  guarda el estado del RNG del inicio del ciclo, que es adonde se rebobina.

Fuera del port: MLflow (las métricas van a `metrics_epoch.csv`), las predicciones por
cuadro en parquet y el panel de métricas de monitoreo por época (queda el AP REAL de
validación y, opcional, el de un subconjunto fijo de train).

---

## 5. Pesos y fichas (Fase 5)

`docs/provenance/export_release.py` extrae de cada checkpoint de §1.5 **solo el
`state_dict`** en float32 y verifica que carga con `strict=True` y que sus tensores son
idénticos a los del checkpoint:

| Modelo | Fichero | Bytes | SHA256 |
|---|---|---|---|
| `u100` | `u100.pt` | 16 298 206 | `0f55483a0dceb040444339dc17a91292286ce732b6d615aa56272c083dce49c3` |
| `rs_ca` | `rs_ca.pt` | 65 020 906 | `6db94dff39bf9c821c124a63b8ba314c38b2b9b178bd4de416110b239e5feb88` |
| `rs_ca_tr` | `rs_ca_tr.pt` | 71 964 331 | `df0f55a6cfa31edae2d5a1ffd84037cc5af875191ed391d4f7ca5ab717fd7659` |
| `e6` | `e6.pt` | 71 933 905 | `5de40c267b90383137d1169c877ad0777305413aa1cbe42bf1119914d8763297` |

Las fichas `configs/release/<modelo>.json` se calculan con `janonet.release` sobre las
predicciones de referencia completas: umbrales A y B idénticos a `r2_umbrales.json`; el
ensamble reproduce AP REAL 0,897 y recall 0,941 en el umbral B (resumen del documento).
La escalera añade, para cada π ∈ {0,005, 0,01, 0,02, 0,05, 0,10}, los umbrales de
fricción 3–20 y recall 0,80–0,95 elegidos en validación con su TPR, FPR y fricción en
validación y en prueba (`analisis2.py:66-80` generalizado a todas las π).
