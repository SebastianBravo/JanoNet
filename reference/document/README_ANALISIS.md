# Análisis que generaron las cifras del documento (régimen productivo, ensamble, bootstrap por grupo)

Estos scripts se ejecutaron fuera del repositorio de desarrollo para producir las tablas y figuras de resultados del documento (v12). Sirven como **referencia de paridad** para la Fase 3 de JanoNet: el código nuevo debe reproducir los CSV de `resultados/`.

## Entradas
- `pv.parquet` = copia de `export_resultados/predicciones_por_video.parquet` (se filtra `seed == 20260910`). Columna de puntaje: `score_oficial` (en configuraciones con Transformer es la salida por video, `prob_video`; con módulo de media, `prob_mean`).
- `m.pkl` (solo `analisis3.py`) = manifiesto limpio (`manifest_all_datasets_with_labels` + reglas de limpieza), usado para atributos de segmento.
- `EX = export_resultados/` (solo en scripts de figuras: `barrido_rgb.csv`, `metricas_globales.csv`, `curvas_recall_friccion.csv`, `segmentos_pares.csv`).
Las rutas están escritas para el entorno donde se corrieron (`/home/claude/...`); hay que ajustarlas.

## Definiciones (las que usa el documento)
- π = 0.02. Fricción = 1 + (1 − π)/π · FPR/TPR (`fr`).
- Umbral A = máximo F1 de la clase REAL en validación (`thr_A`).
- Umbral B = **menor umbral de validación con fricción ≤ 5 a π = 0.02, SIN el factor 1.173** (`thr_B`).
- Escalera: mismo criterio de `thr_B` con fricción objetivo 3, 4, 5, 6, 8, 10; y umbrales de recall objetivo 0.80, 0.85, 0.90, 0.95 elegidos en validación.
- Precisión FAKE(π) = π·TPR / (π·TPR + (1 − π)·FPR). F1 REAL a π usa TNR y (1 − TPR).
- Sensibilidad a π: para cada π ∈ {0.005, 0.01, 0.02, 0.05, 0.10} se elige de nuevo el umbral con fricción ≤ 5 en validación a ese π.
- Ensamble: media de logits de U100 y RS-ca-tr, con logit(clip(p, 1e-6, 1 − 1e-6)) sobre `score_oficial`. Reproduce AP REAL de test 0.897.
- IC binomiales exactos (Clopper–Pearson) para FPR y TPR (`cp`).
- Bootstrap por video: estratificado por clase. Bootstrap por grupo: remuestrea `group_id` completos (146 grupos en test). Comparaciones emparejadas: ambas configuraciones sobre los mismos índices.

## Scripts, semillas y salidas
| Script | Semilla | Réplicas | Salida | Uso en el documento |
|---|---|---|---|---|
| `analisis2.py` | 20261123 | — | `r2_global.csv`, `r2_escalera.csv`, `r2_prevalencia.csv`, `r2_umbrales.json` | Tablas VIII, X, XII, escalera (Fig. 9), Fig. 14 |
| `analisis3.py` (reutiliza `analisis2`) | 7 | 1,000 por video y por grupo | `r3_boot.csv`, `r3_subsets.csv`, `r3_macro.csv`, `r3_errores.csv`, `r3_fps.csv`, `r3_demografia.csv` | Diferencias de AP REAL por video/grupo, errores compartidos (Fig. 11a), subconjuntos |
| `analisis4.py` (reutiliza `analisis3`, misma secuencia del generador) | continúa la de `analisis3` | 1,000 por video y por grupo | `r4_point.csv`, `r4_ci.csv` | Ensamble, diferencias productivas emparejadas a π = 0.02 y 0.005 (Fig. 10) |
| `analisis5.py` | — | — | `r5_segmentos.csv` | Recall por fuente/método y FPR/recall por redimensionamiento en el umbral B (Figs. 12 y 13) |
| `analisis5b.py` | 11 | 2,000 por video (dentro del segmento) | impresión en consola | IC de las diferencias por segmento (FaceShifter, upsample, downsample, DFDC) |
| `figs_res.py`, `figs2.py`, `fig_forest2.py`, `fig_seg.py`, `fig_barrido.py` | — | — | figuras | Figs. 8–14 |

## Recomendación de paridad
- Estimaciones puntuales (métricas, umbrales, escalera, fricción por π, conteos de errores): deben coincidir exactamente.
- Intervalos bootstrap: coinciden exactamente solo si se respeta la misma semilla y el mismo orden de llamadas al generador (`analisis3` → `analisis4`). Si el código nuevo reorganiza el orden, comparar con tolerancia (±0.005 en métricas de proporción, ±0.10 en fricción) y documentarlo.
