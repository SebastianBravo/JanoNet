# examples/

Videos de ejemplo para la demo (`app/streamlit_app.py`), descritos en `examples.json`.

| Fichero | Etiqueta | Origen |
|---|---|---|
| `deeperforensics_end_to_end_001_W101.mp4` | FAKE | DeeperForensics-1.0, `end_to_end/001_W101.mp4` (partición de prueba) |
| `deeperforensics_end_to_end_100_M121.mp4` | FAKE | DeeperForensics-1.0, `end_to_end/100_M121.mp4` (partición de prueba) |
| `deeperforensics_end_to_end_117_W132.mp4` | FAKE | DeeperForensics-1.0, `end_to_end/117_W132.mp4` (partición de prueba) |
| `deeperforensics_end_to_end_017_W006.mp4` | FAKE | DeeperForensics-1.0, `end_to_end/017_W006.mp4` (partición de validación) |
| `real_propio.mp4` | REAL | video real del autor (fuera de git hasta que se agregue) |

Los cuatro falsos son los ficheros originales del dataset, **sin recodificar** (1,0–4,4 MB),
de cuatro actores fuente distintos. Tres son de la partición de prueba; `017_W006`, de la de
validación (no se usó para entrenar, pero sí para elegir umbrales), y es el par de la figura
del ejemplo SRM del documento. En cada uno **la cara es la del actor de DeeperForensics y el fondo, el cuerpo
y los rótulos provienen del video original de FaceForensics++** sobre el que se hizo el
intercambio.

**Cita:** L. Jiang, R. Li, W. Wu, C. Qian y C. C. Loy, "DeeperForensics-1.0: A Large-Scale
Dataset for Real-World Face Forgery Detection", CVPR 2020. Uso no comercial de
investigación, sujeto a los términos de uso de DeeperForensics-1.0
(<https://github.com/EndlessSora/DeeperForensics-1.0>).
