# Datos

JanoNet se entrenó y evaluó con **94 234 videos** de cuatro datasets públicos, tras
aplicar las reglas de limpieza a 140 423 videos. Este repositorio **no contiene videos
ni cuadros** de DFDC, FaceForensics++ ni FakeAVCeleb: solo manifiestos sin medios
(`manifests/`) e instrucciones para obtener cada dataset de su fuente oficial.

> Cada dataset tiene sus propios términos de uso, todos limitados a investigación no
> comercial y con acceso previa solicitud. Léelos y acéptalos en la fuente oficial antes
> de descargar nada. Lo que sigue es un resumen, no sustituye el texto oficial.

## 1. Datasets y subconjuntos usados

| Dataset | Qué se usa | Videos procesados | Videos limpios | Cita |
|---|---|---|---|---|
| DFDC (conjunto de entrenamiento completo) | partes `dfdc_train_part_00` a `49`, REAL y FAKE | 119 146 | 79 992 | Dolhansky et al., 2020 |
| FaceForensics++ (c23) | `manipulated_sequences/{DeepFakeDetection, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures}/c23` | 8 068 | 5 937 | Rössler et al., ICCV 2019 |
| FaceForensics++ (c23), originales | `original_sequences/youtube/c23` (REAL; son los videos objetivo de DeeperForensics) | 1 000 | 835 | Rössler et al., ICCV 2019 |
| DeeperForensics-1.0 | `manipulated_videos/{end_to_end, reenact_postprocess}` | 2 000 | 1 683 | Jiang et al., CVPR 2020 |
| FakeAVCeleb v1.2 | `FakeVideo-RealAudio` (FAKE) y `RealVideo-RealAudio` (REAL) | 10 209 | 5 787 | Khalid et al., NeurIPS 2021 |

En las tablas del estudio los originales de FF++ figuran con `dataset = deeper-forensics`
(entraron como los REAL de DeeperForensics). FaceForensics++ como tal no aporta ningún
video REAL.

## 2. Dónde ubicarlos

Cada dataset se busca en una variable de entorno o, si no está definida, en
`$JANONET_DATA/<carpeta>` (`configs/data.yaml`):

| Variable | Carpeta por defecto | Layout esperado (relativo a la raíz) |
|---|---|---|
| `JANONET_DFDC_ROOT` | `dfdc` | los mp4 en cualquier subcarpeta (layout oficial por partes o aplanado) y los `metadata.json` de cada parte |
| `JANONET_FFPP_ROOT` | `ffpp` | `manipulated_sequences/<Método>/c23/videos/*.mp4` y `original_sequences/youtube/c23/videos/*.mp4` |
| `JANONET_DF_ROOT` | `deeperforensics` | `manipulated_videos/<variante>/*.mp4` (también se acepta `FAKE/<variante>/`) |
| `JANONET_FAVC_ROOT` | `fakeavceleb` | `FakeVideo-RealAudio/...` y `RealVideo-RealAudio/...` como en la descarga v1.2 (también `FAKE/` y `REAL/`), más `meta_data.csv` |
| `JANONET_PROCESSED` | `processed` | salida: videos de rostro 224×224 |
| `JANONET_CACHE` | `frames_cache` | salida: 32 cuadros por video en `.bin` (≈ 4,8 MB cada uno; 423 GB para los 94 234) |

Los nombres de fichero se conservan tal como vienen en cada descarga.

## 3. Flujo de preparación

```text
01_build_manifest.py sources   inventario: video_id, dataset, etiqueta, método
02_preprocess.py faces         RetinaFace + limpieza + recorte + resize -> processed/
01_build_manifest.py split     grafo de identidades + partición + tabla maestra
02_preprocess.py cache         32 cuadros centrados por video -> frames_cache/
```

Con los manifiestos versionados (`manifests/`) se puede saltar directamente a
`02_preprocess.py faces` (sobre la lista de `manifests/samples.csv.gz`) y luego a
`cache`: la partición y los `sample_id` ya están fijados. `01_build_manifest.py split
--check` reconstruye la partición desde los manifiestos y verifica que coincide columna
a columna con la del estudio.

### Filas de FakeAVCeleb: se regeneran localmente

Los términos de uso de FakeAVCeleb prohíben distribuir datos derivados del dataset, así
que sus filas **no están en git**. Quien tenga acceso a FakeAVCeleb v1.2 las regenera
con el mismo código (`make favc` o `.\tasks.ps1 favc`):

```bash
python scripts/01_build_manifest.py sources --datasets fakeavceleb --out outputs/fakeavceleb_sources.csv
python scripts/02_preprocess.py faces --videos outputs/fakeavceleb_sources.csv \
    --out manifests/local/fakeavceleb_preprocessing.csv
python scripts/01_build_manifest.py split --check --write-local
```

El último paso reconstruye la partición con los cuatro datasets, verifica que coincide
con la publicada y guarda las filas de FakeAVCeleb de la tabla maestra en
`manifests/local/` (ignorada por git). Sin ellas:

- los `sample_id` y la partición publicados de los otros tres datasets siguen siendo
  los del estudio, y se puede evaluar, preprocesar y entrenar con ellos;
- no se puede **reconstruir** la partición (el reparto se optimiza sobre los cuatro
  datasets a la vez) ni reproducir las cifras que incluyen FakeAVCeleb.

**Preprocesamiento** (`configs/preprocess.yaml`): RetinaFace (score ≥ 0,9, lado ≥ 60 px)
cada 5 cuadros; recorte cuadrado de margen 0 sobre la última caja válida;
redimensionado bicúbico a 224×224; video reescrito con el códec `mp4v`. Un video se
descarta si algún cuadro revisado no tiene rostro, si < 95 % de los cuadros revisados
tiene exactamente un rostro, si > 5 % tiene dos o más, o si > 20 % de los cuadros
válidos no es frontal.

**Partición** (`configs/data.yaml`): por grupos de identidad, 80/10/10 estratificado por
(dataset, etiqueta), semilla 42. DFDC se agrupa por parte del dataset; FF++ y
DeeperForensics por id de video objetivo (uniendo origen y destino de cada
manipulación); FakeAVCeleb por `idNNNNN`; DeepFakeDetection por componentes de
actores. **Limitación conocida:** los 100 actores donantes de DeeperForensics no se
agrupan (colapsaría el reparto) y 81 de ellos aparecen en más de una partición.

## 4. Manifiestos

| Fichero | Filas | Contenido |
|---|---|---|
| `manifests/preprocessing.csv.gz` | 130 214 | `video_id`, dataset, etiqueta, método, parte/original de DFDC, resultado del preprocesamiento y de las reglas de limpieza |
| `manifests/samples.csv.gz` | 88 447 | `sample_id`, partición, `group_id`, tipo de identidad, acción y % de redimensionamiento, duración del clip y `window_start` |
| `manifests/local/fakeavceleb_*.csv.gz` (fuera de git) | 10 209 / 5 787 | las mismas columnas para FakeAVCeleb, regeneradas localmente |

`video_id` es la ruta relativa a la raíz del dataset con prefijo (`dfdc/`, `ffpp/`,
`deeperforensics/`, `fakeavceleb/`). `sample_id` es la clave de todas las predicciones.

## 5. Citas y términos de uso

| Dataset | Referencia | Acceso y términos |
|---|---|---|
| DFDC | B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang y C. Canton Ferrer, "The DeepFake Detection Challenge (DFDC) Dataset", arXiv:2006.07397, 2020. | Acceso con licencia del dataset aceptada en <https://ai.meta.com/datasets/dfdc/>. |
| FaceForensics++ | A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies y M. Nießner, "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019. | Formulario y términos de uso (<https://github.com/ondyari/FaceForensics>): investigación y docencia no comerciales. |
| DeeperForensics-1.0 | L. Jiang, R. Li, W. Wu, C. Qian y C. C. Loy, "DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection", CVPR 2020. | Términos de uso y formulario (<https://github.com/EndlessSora/DeeperForensics-1.0>): investigación no comercial. Los actores fuente firmaron consentimiento. |
| FakeAVCeleb | H. Khalid, S. Tariq, M. Kim y S. S. Woo, "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset", NeurIPS 2021 Datasets and Benchmarks. | Formulario de solicitud (<https://github.com/DASH-Lab/FakeAVCeleb>): investigación y docencia no comerciales. |

> **Material de DeeperForensics-1.0 en este repositorio** (ejemplos de la demo y figuras):
> se muestra con la cita anterior. En los videos falsos de DeeperForensics solo la cara
> es del actor fuente; el fondo, el cuerpo y los rótulos provienen del video original
> de FaceForensics++.
