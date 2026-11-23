# Entregable final (Semana 15)

> **Estado al 22/11/2026: versión final del código y de los resultados.**
> El proyecto se plantea como dos problemas de ML (pronóstico de riesgo y triaje de gravedad; README §1). Están
> listos el preprocesamiento, el EDA, los baselines, los cinco modelos de la literatura con ajuste de
> hiperparámetros, la ablación y la comparación final. Un clon limpio (22/11/2026, Python 3.14) ejecuta los 9
> notebooks sin errores siguiendo solo el README §6, en ≈ 1 h 20 min. El informe y la presentación están en la raíz.
> **Antes de entregar:** completar en `report.docx` los códigos y correos, la declaración de uso de IA y las
> contribuciones de Marco, Angel y Fernando, y volver a exportar `report.pdf`; ensayar la presentación con el guion.

| Requisito del enunciado | Estado | Dónde está |
|---|---|---|
| Código limpio, comentado y reproducible (se ejecuta sin errores) | ✅ | Los 9 notebooks (`00_preprocesamiento` a `08_comparacion`) corren de punta a punta en un clon limpio siguiendo solo el README §6 |
| `requirements.txt` con todas las dependencias | ✅ | [`requirements.txt`](requirements.txt) |
| Modelos implementados (1 por integrante) en `src/models.py` | ✅ | [`src/models.py`](src/models.py): baselines (Poisson y regresión logística binaria y multinomial), binomial negativa (`binomial_negativa`), Random Forest (`rf_conteo`, `random_forest`), gradient boosting (`gbm_conteo`, `gbm_clasificador`), MLP (`mlp_conteo`, `mlp_clasificador`) y ensamble por votación (`votacion`). Configuraciones elegidas en `CONFIG_AJUSTADA` y modelos finales en `modelos_ajustados` |
| Métricas y evaluación en `src/evaluation.py` | ✅ | [`src/evaluation.py`](src/evaluation.py): CV temporal, CV 5 × 5, métricas de conteo, binarias y multiclase, comparación pareada |
| Tabla comparativa de métricas (`results/metrics.csv`) | ✅ | [`results/metrics.csv`](results/metrics.csv): referencias, baselines, binomial negativa, MLP, Random Forest, gradient boosting y ensamble, con CV y holdout/test. Tabla I y comparación pareada en `08_comparacion.ipynb` |
| Gráficos en `results/plots/` | ✅ | `prep_*`, `eda_*`, `modelo_01`–`modelo_03` (baseline), `modelo_04`–`modelo_06` (Random Forest), `modelo_07`–`modelo_08` (ablación), `modelo_09` (binomial negativa), `modelo_10`–`modelo_12` (MLP), `modelo_13`–`modelo_15` (gradient boosting), `modelo_16`–`modelo_18` (ensamble) y `modelo_19`–`modelo_22` (comparación final) |
| `results/model_comparison.txt` con resumen y análisis detallado | ✅ | [`results/model_comparison.txt`](results/model_comparison.txt): Tabla I, una sección por modelo, comparación final, ablación y conclusiones |
| Análisis de por qué un modelo funcionó mejor, limitaciones e insights | ✅ | `model_comparison.txt` §9–12 (comparación final, ablación, conclusiones por variable objetivo y limitaciones) y README §5.3 |
| Paper/reporte académico (3-5 páginas) en `report.pdf` | ✅ | [`report.pdf`](report.pdf) (5 páginas, plantilla del curso `Plantilla_reporte_TA.docx`): Introducción, Trabajo relacionado, Dataset y EDA, Metodología, Resultados (Tabla I de métricas en prueba y Tabla II de hiperparámetros), Discusión, Conclusiones, Declaración de uso de IA, Contribuciones y Referencias IEEE. Fuente editable: [`report.docx`](report.docx) |
| Presentación en `presentation.pdf` | ✅ | [`presentation.pdf`](presentation.pdf): 11 diapositivas (≈ 10 min, cada integrante expone su modelo; una figura por diapositiva y poco texto). Fuente, guion y respuestas a preguntas probables con tiempos en [`presentación/final/`](presentación/final/). La de avances está archivada en `presentación/avances/` |
| `ENTREGABLE_FINAL.md` confirmando completitud | ✅ | Este archivo |

## Resultado final (métrica principal; `model_comparison.txt` §2 y §11)

| Variable objetivo | Referencia sin ML | Baseline | Modelo elegido | Métrica |
|---|---|---|---|---|
| 1 · `N_ACC` | 0.217 (media de 28 días) | 0.256 (Poisson) | **Ensamble por votación: 0.268** (holdout 0.296) | D² de Poisson, CV temporal |
| 1 · `HAY_FATAL` | 0.112 (fatales de 28 días) | **0.133** (logística) | **Regresión logística** (holdout 0.161; todos los modelos empatan) | PR-AUC, CV temporal (azar 0.058) |
| 2 · `GRAVEDAD` | 0.228 (Dummy) | 0.419 (logística multinomial) | **Random Forest: 0.452**; ensamble (0.447) si importa no dejar pasar fatales | Macro-F1, CV 5 × 5 |
