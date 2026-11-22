# Entregable final (Semana 15)

> **Estado al 12/11/2026: replanteo + baseline.**
> Tras la presentación de avances, el proyecto se replanteó como dos problemas de ML (pronóstico de riesgo y triaje
> de gravedad; README §1). Están listos el preprocesamiento, el EDA y los baselines. Los modelos de ensamble, el ajuste
> de hiperparámetros, la ablación, el informe y la presentación están en los issues del repositorio.

| Requisito del enunciado | Estado | Dónde está |
|---|---|---|
| Código limpio, comentado y reproducible (se ejecuta sin errores) | 🟡 En curso | `00_preprocesamiento`, `01_eda`, `02_modelos`, `03_random_forest`, `04_gradient_boosting`, `05_binomial_negativa` y `06_mlp` corren de punta a punta. Falta la revisión final tras los modelos nuevos |
| `requirements.txt` con todas las dependencias | ✅ | [`requirements.txt`](requirements.txt) |
| Modelos implementados (1 por integrante) en `src/models.py` | 🟡 | Baselines: Poisson y regresión logística (binaria y multinomial). Binomial negativa implementada (`05_binomial_negativa.ipynb`). MLP implementado y ajustado en las dos tareas (`06_mlp.ipynb`). Random Forest implementado y ajustado en las dos tareas (`03_random_forest.ipynb`). Gradient boosting ajustado en las dos tareas (`04_gradient_boosting.ipynb`). Ensamble por votación (`M.votacion`) en las tres variables (`06_mlp.ipynb` §4) |
| Métricas y evaluación en `src/evaluation.py` | ✅ | [`src/evaluation.py`](src/evaluation.py): CV temporal, CV 5 × 5, métricas de conteo, binarias y multiclase, comparación pareada |
| Tabla comparativa de métricas (`results/metrics.csv`) | 🟡 | Baselines de las dos tareas, binomial negativa (Tarea 1), MLP, Random Forest y ensamble por votación (dos tareas). Incluye todos los modelos; falta la comparación final (`08_comparacion`) |
| Gráficos en `results/plots/` | 🟡 | `prep_*`, `eda_*`, `modelo_01` a `modelo_03`, `modelo_04` a `modelo_06` (Random Forest), `modelo_09` (binomial negativa) `modelo_10` a `modelo_12` (MLP), `modelo_13` a `modelo_15` (gradient boosting) y `modelo_16` a `modelo_18` (ensamble por votación) |
| `results/model_comparison.txt` con resumen y análisis detallado | 🟡 Versión baseline | [`results/model_comparison.txt`](results/model_comparison.txt) |
| Análisis de por qué un modelo funcionó mejor, limitaciones e insights | 🟡 | README §5.3 y `model_comparison.txt` §3–8, con el gradient boosting y el ensamble por votación. Falta la ablación §3–7, con el ensamble por votación. Falta la sección del gradient boosting |
| Paper/reporte académico (3-5 páginas) en `report.pdf` | ⬜ Pendiente | Issue |
| Presentación (máx. 5 min) en `presentation.pdf` | ⬜ Pendiente | Issue. La de avances (planteamiento anterior) está archivada en `presentación/avances/` |
| `ENTREGABLE_FINAL.md` confirmando completitud | 🟡 | Este archivo |

## Resultado actual (baseline)

| Tarea | Referencia sin ML | Baseline | Métrica |
|---|---|---|---|
| 1 · `N_ACC` | 0.217 (media de 28 días) | **0.256** (Poisson) | D² de Poisson, CV temporal |
| 1 · `HAY_FATAL` | 0.112 (fatales de 28 días) | **0.133** (logística) | PR-AUC, CV temporal (azar 0.058) |
| 2 · `GRAVEDAD` | 0.228 (Dummy) | **0.419** (logística multinomial) | Macro-F1, CV 5 × 5 |
