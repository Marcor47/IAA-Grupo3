# Guía para trabajar en este repositorio (personas y asistentes de IA)

Proyecto del curso IA Aplicada (1INF62, PUCP 2026-2), Grupo 3. Léelo antes de tomar un issue.

## El problema en una línea por tarea
- **Tarea 1 · Pronóstico de riesgo.** Panel departamento × día (`data/processed/panel_depto_dia.csv`). Objetivos:
  `N_ACC`, el conteo de accidentes, y `HAY_FATAL` (0/1).
- **Tarea 2 · Triaje de gravedad.** Un accidente por fila (`data/processed/accidentes.csv`). Objetivo: `GRAVEDAD`
  (0 sin víctimas, 1 heridos, 2 fatal).

El detalle está en `README.md` §1 y §4. El estado actual son los baselines del README §5.

## Entorno
```
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
.venv/bin/python -m src.preprocessing        # regenera data/processed/ desde las cachés de data/raw/ (≈ 1 min, sin internet)
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_modelos.ipynb
```

## Dónde va cada cosa
| Qué | Dónde |
|---|---|
| Limpieza, cruces y variables nuevas | `src/preprocessing.py`. Las decisiones se documentan en `notebooks/00_preprocesamiento.ipynb` |
| Grupos de variables, encoding y factorías de modelos | `src/models.py` (`GRUPOS_T1`, `GRUPOS_T2`, `ORDINALES`, `preprocesador`, `poisson`, `logit`, `rf_conteo`, `gbm_conteo`, `random_forest`, `gbm_clasificador`) |
| Métricas y validación | `src/evaluation.py` (`cv_temporal`, `CV_REPETIDA`, `cv_puntajes`, `metricas_*`, `comparacion_pareada`) |
| Experimentos | `notebooks/02_modelos.ipynb`, o un notebook nuevo `notebooks/03_<tema>.ipynb` |
| Resultados | Filas nuevas en `results/metrics.csv` (mismo esquema: `tarea, modelo, validacion, <métricas>`), figuras `results/plots/modelo_XX_<tema>.png` y la lectura en `results/model_comparison.txt` |

## Reglas
1. **No cambies las decisiones de preprocesamiento ni de encoding** (variables eliminadas, ordinales, one-hot) sin
   que el grupo las apruebe. Si crees que hace falta, propónlo en el issue con los números que lo justifiquen.
2. **Validación fija**, para que todo sea comparable con el baseline:
   - **Tarea 1:** `E.cv_temporal(tr.FECHA)` sobre el train (`FECHA < M.INICIO_HOLDOUT`). El holdout de abr–sep 2021
     se usa **solo** para el modelo final. Nunca hagas particiones aleatorias en la Tarea 1.
   - **Tarea 2:** `E.CV_REPETIDA` (5 × 5, `random_state=42`) para comparar y `E.CV` para el ajuste de hiperparámetros.
3. **Métricas principales:**
   - Tarea 1 · `N_ACC`: D² de Poisson.
   - Tarea 1 · `HAY_FATAL`: PR-AUC.
   - Tarea 2: macro-F1, con la PR-AUC de la clase fatal como secundaria.

   La exactitud no se usa. Todo modelo se reporta **junto a su referencia** (ingenuo o Dummy, y el baseline) y con
   `E.comparacion_pareada`.
4. **Sin fuga de información:**
   - En la Tarea 1, toda variable de historia usa datos hasta *t − 1*.
   - En la Tarea 2, `NUM_FALLECIDOS` y `NUM_HERIDOS` forman parte del objetivo y nunca son variables de entrada.
5. **No subas datos nuevos sin documentar su fuente y licencia** en el README §2. Las descargas van con caché en `data/raw/`.
6. **Reporta lo que sale, aunque un modelo no gane:** que no supere al baseline también es un resultado. No ajustes
   hiperparámetros mirando el holdout ni el test.
7. **Commits pequeños**, en español y en imperativo, como el historial. Abre un PR que cite el issue (`Closes #N`).
