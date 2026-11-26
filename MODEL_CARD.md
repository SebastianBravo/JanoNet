# Ficha de los modelos JanoNet

Detectores binarios (real / falso) de deepfakes de tipo *face swap* en video sin audio,
del trabajo de grado *JanoNet: Modelo RGB-SRM con atención cruzada para la detección de
deepfakes de tipo face swap en video bajo prevalencia realista de clases* (Juan S. Bravo,
Pontificia Universidad Javeriana; director: Wilson J. Arenas).

| Modelo | Qué es | Parámetros | Pesos |
|---|---|---|---|
| `u100` | Rama RGB sola: EfficientNet-B0 100 % entrenable, módulo de media | 4,0 M | `u100.pt`, 16 MB |
| `rs_ca` | RGB + SRM (ResNet-18 sobre 3 filtros fijos), atención cruzada bidireccional, módulo de media | 16,2 M | `rs_ca.pt`, 65 MB |
| `rs_ca_tr` | **JanoNet completo**: RGB + SRM, atención cruzada bidireccional, Transformer temporal | 17,9 M | `rs_ca_tr.pt`, 72 MB |
| `ensemble` | Media de logits de `u100` y `rs_ca_tr`, sin entrenamiento | — | usa los de sus miembros |
| `e6` *(opcional)* | Como `rs_ca_tr` pero con la RGB congelada salvo la conv final (10,3 %): poca memoria de entrenamiento | 17,9 M (14,3 M entrenables) | `e6.pt`, 72 MB |

Cada modelo tiene su ficha técnica en `configs/release/<modelo>.json`: config, umbrales,
escalera de operación por prevalencia, métricas de prueba y SHA256 de los pesos.

## Uso previsto

**Control complementario** en procesos de verificación remota de identidad (p. ej.
registro de clientes, eKYC): señalar videos de rostro para **revisión humana**, con un
punto de operación elegido según el coste de revisión aceptable (fricción) y la
prevalencia esperada de deepfakes.

Entrada: un video con **un solo rostro, frontal y visible** durante la captura. El video
pasa por el mismo preprocesamiento del entrenamiento (RetinaFace, recorte cuadrado,
224 × 224, `configs/preprocess.yaml`); si no cumple las reglas de limpieza (sin rostro,
varios rostros o pose no frontal) el sistema no da puntaje y se debe pedir una nueva
captura. Se evalúa el **clip central** de 16 cuadros (≈ 0,5 s).

Salida: un **puntaje de falsedad** en [0, 1] y la decisión en el umbral elegido. **El
puntaje no es una probabilidad calibrada**: solo sirve para ordenar videos y compararlo
con los umbrales de la ficha.

## Usos no permitidos

- Tomar decisiones automáticas con efectos sobre una persona (rechazo de un trámite,
  bloqueo de una cuenta, denuncia) **sin revisión humana**.
- Vigilancia, identificación o seguimiento de personas.
- Atribuir autoría o tomar decisiones legales o periodísticas sobre un video concreto.
- Usar el detector para entrenar o ajustar generadores que lo eludan.
- Cualquier uso comercial de los pesos (licencia no comercial, ver abajo).

## Datos de entrenamiento

94 234 videos de DFDC, FaceForensics++ (c23), DeeperForensics-1.0 y FakeAVCeleb, tras
aplicar reglas de limpieza a 140 423 (ver `DATA.md`). Partición 80/10/10 **por grupos de
identidad**: 75 467 de entrenamiento (11 218 REAL / 64 249 FAKE), 9 169 de validación y
9 598 de prueba (1 464 REAL / 8 134 FAKE). Pérdida BCE con
`pos_weight = n_REAL/n_FAKE = 0,1746`. Entrenamiento en una Tesla V100 con AMP
(receta en `configs/models/*.yaml`).

## Métricas en prueba

Dos regímenes, con umbrales **elegidos en validación** y aplicados a prueba.

**Régimen estándar** (prevalencia natural de prueba, 85 % de falsos; umbral A = máximo F1
de la clase REAL):

| Modelo | AUC | AP REAL | Exactitud | F1 REAL | F1 FAKE |
|---|---|---|---|---|---|
| `u100` | 0,967 | 0,870 | 0,941 | 0,798 | 0,965 |
| `rs_ca` | 0,963 | 0,867 | 0,935 | 0,777 | 0,962 |
| `rs_ca_tr` | 0,968 | 0,881 | 0,937 | 0,780 | 0,963 |
| `ensemble` | 0,975 | 0,897 | 0,944 | 0,805 | 0,967 |
| `e6` | 0,935 | 0,789 | 0,908 | 0,695 | 0,946 |

**Régimen productivo** (2 % de deepfakes, π = 0,02; umbral B = menor umbral de validación
con fricción ≤ 5). La **fricción** es el número de alertas por cada deepfake detectado,
`1 + 49 · FPR/TPR`:

| Modelo | Recall FAKE | FPR | Precisión FAKE | Fricción | Alertas por 1000 videos |
|---|---|---|---|---|---|
| `u100` | 0,923 | 0,114 | 0,142 | 7,05 | 130 |
| `rs_ca` | 0,869 | 0,085 | 0,172 | 5,82 | 101 |
| `rs_ca_tr` | 0,909 | 0,096 | 0,162 | 6,19 | 113 |
| `ensemble` | 0,941 | 0,109 | 0,150 | 6,66 | 125 |
| `e6` | 0,796 | 0,082 | 0,165 | 6,05 | 96 |

**Sensibilidad a la prevalencia** (umbral con fricción ≤ 5 elegido en validación a cada
π; recall y fricción en prueba):

| Modelo | π = 0,5 % | π = 2 % | π = 10 % |
|---|---|---|---|
| `u100` | 0,795 / 10,9 | 0,923 / 7,05 | 0,991 / 5,29 |
| `rs_ca` | 0,712 / 8,63 | 0,869 / 5,82 | 0,991 / 4,97 |
| `rs_ca_tr` | 0,723 / 6,83 | 0,909 / 6,19 | 0,997 / 5,46 |
| `ensemble` | 0,810 / 7,04 | 0,941 / 6,66 | 0,994 / 5,36 |

En el régimen estándar las variantes no se distinguen; en el productivo definen puntos
de operación distintos: la RGB sola maximiza la detección, la fusión con SRM reduce la
fricción y el ensamble combina ambos. La escalera completa (fricción 3–20 y recall
0,80–0,95, para π de 0,5 % a 10 %) está en cada `configs/release/<modelo>.json`.

Estas cifras son las del documento y se reproducen con `scripts/05_evaluate.py --compare`.

## Limitaciones

- **Una sola semilla** por modelo: las diferencias entre modelos menores que el ruido
  entre semillas (≈ 0,03 de AP REAL) no son concluyentes.
- **Clip central único** de 16 cuadros (≈ 0,5 s): una manipulación que no aparezca en ese
  tramo no se ve.
- **Datasets públicos sin canal real**: no hay videos capturados ni inyectados en una
  aplicación de verificación real (cámara del móvil, compresión del canal, ataques de
  inyección), ni manipulaciones posteriores a la fecha de los datasets.
- **El puntaje no es una probabilidad** y no está calibrado.
- **Deriva del umbral entre validación y prueba**: el umbral elegido con fricción ≤ 5 en
  validación da entre 5,8 y 7,1 de fricción en prueba. Antes de operar hay que
  recalibrar con datos del canal real.
- **Prevalencia supuesta**: la fricción depende de π; si la prevalencia real es otra, hay
  que elegir el punto de operación de esa π.
- **Identidad**: los 100 actores donantes de DeeperForensics no se pudieron agrupar en la
  partición (81 aparecen en más de un conjunto), y en DFDC el grupo de identidad es la
  parte del dataset, no la persona.
- **Precisión numérica**: los modelos con rama SRM se entrenaron con el residuo en
  float16 (AMP) y deben ejecutarse igual (`srm.residual_precision: float16`, ya en sus
  configs); en float32 puro sus puntajes se alejan de los publicados.
- Solo se evaluó *face swap* en video sin audio; no se evaluaron otras manipulaciones
  (síntesis completa, edición de atributos) ni el audio.

## Licencias

- **Código**: MIT (`LICENSE`).
- **Pesos**: **CC BY-NC 4.0**, además sujetos a los términos de uso de los datasets con
  que se entrenaron (DFDC, FaceForensics++, DeeperForensics-1.0 y FakeAVCeleb), todos
  limitados a investigación no comercial. Quien use los pesos debe cumplir también esos
  términos (ver `DATA.md`).

## Cómo citar

Ver la sección de cita del `README.md` (trabajo de grado y datasets).
