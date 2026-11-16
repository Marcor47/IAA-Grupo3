# Entregable final (Semana 15)

> **Estado al 12/11/2026: replanteo + baseline.**
> Tras la presentación de avances, el proyecto se replanteó como dos problemas de ML (pronóstico de riesgo y triaje
> de gravedad; README §1). Están listos el preprocesamiento, el EDA y los baselines. Los modelos de ensamble, el ajuste
> de hiperparámetros, la ablación, el informe y la presentación están en los issues del repositorio.

| Requisito del enunciado | Estado | Dónde está |
|---|---|---|
| Código limpio, comentado y reproducible (se ejecuta sin errores) | 🟡 En curso | `00_preprocesamiento`, `01_eda`, `02_modelos` y `05_binomial_negativa` corren de punta a punta. Falta la revisión final tras los modelos nuevos |
| `requirements.txt` con todas las dependencias | ✅ | [`requirements.txt`](requirements.txt) |
| Modelos implementados (1 por integrante) en `src/models.py` | 🟡 | Baselines: Poisson y regresión logística (binaria y multinomial). Binomial negativa implementada (`05_binomial_negativa.ipynb`). Pendientes: Random Forest, gradient boosting y MLP (issues) |
| Métricas y evaluación en `src/evaluation.py` | ✅ | [`src/evaluation.py`](src/evaluation.py): CV temporal, CV 5 × 5, métricas de conteo, binarias y multiclase, comparación pareada |
| Tabla comparativa de métricas (`results/metrics.csv`) | 🟡 | Baselines de las dos tareas y binomial negativa (Tarea 1). Faltan Random Forest, gradient boosting y MLP |
| Gráficos en `results/plots/` | 🟡 | `prep_*`, `eda_*`, `modelo_01` a `modelo_03` y `modelo_09` (binomial negativa) |
| `results/model_comparison.txt` con resumen y análisis detallado | 🟡 Versión baseline | [`results/model_comparison.txt`](results/model_comparison.txt) |
| Análisis de por qué un modelo funcionó mejor, limitaciones e insights | 🟡 | README §5.3 y `model_comparison.txt` §3–4. Falta la versión con los modelos de ensamble |
| Paper/reporte académico (3-5 páginas) en `report.pdf` | ⬜ Pendiente | Issue |
| Presentación (máx. 5 min) en `presentation.pdf` | ⬜ Pendiente | Issue. La de avances (planteamiento anterior) está archivada en `presentación/avances/` |
| `ENTREGABLE_FINAL.md` confirmando completitud | 🟡 | Este archivo |

## Resultado actual (baseline)

| Tarea | Referencia sin ML | Baseline | Métrica |
|---|---|---|---|
| 1 · `N_ACC` | 0.217 (media de 28 días) | **0.256** (Poisson) | D² de Poisson, CV temporal |
| 1 · `HAY_FATAL` | 0.112 (fatales de 28 días) | **0.133** (logística) | PR-AUC, CV temporal (azar 0.058) |
| 2 · `GRAVEDAD` | 0.228 (Dummy) | **0.419** (logística multinomial) | Macro-F1, CV 5 × 5 |
