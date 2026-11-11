# Entregable final (Semana 15)

> Estado al 11/11/2026: **avance**. Modelo 1 entrenado y comparado contra el baseline; presentación lista.

| Requisito del enunciado | Estado | Dónde está |
|---|---|---|
| Código limpio, comentado y reproducible (se ejecuta sin errores) | 🟡 En curso | `notebooks/01_eda.ipynb` y `notebooks/02_modelos.ipynb` se ejecutan de punta a punta; falta la revisión final |
| `requirements.txt` con todas las dependencias | ✅ | [`requirements.txt`](requirements.txt) |
| Modelos implementados (mínimo 2) en `src/models.py` | ✅ | [`src/models.py`](src/models.py): baseline (regresión logística, Set A), **Modelo 1** (regresión logística, departamento + hora + altitud), Random Forest (respaldo) y regresión de Poisson (Tarea B) |
| Métricas y evaluación en `src/evaluation.py` | ✅ | [`src/evaluation.py`](src/evaluation.py): métricas, umbral por CV, CV repetida 5 × 5 y comparación pareada |
| Tabla comparativa de métricas (`results/metrics.csv`) | ✅ | [`results/metrics.csv`](results/metrics.csv) (filas de test 80/20 y de CV 5 × 5) |
| Gráficos en `results/plots/` | ✅ | [`results/plots/`](results/plots/): `eda_*` y `modelo_01` a `modelo_04` |
| `results/model_comparison.txt` con resumen y análisis detallado | 🟡 Versión de avance | [`results/model_comparison.txt`](results/model_comparison.txt) |
| Análisis de por qué un modelo funcionó mejor, limitaciones e insights | 🟡 Versión de avance | `model_comparison.txt` §5–6 y `02_modelos.ipynb` §7 |
| Paper/reporte académico (3-5 páginas) en `report.pdf` | ⬜ Pendiente | |
| Presentación (máx. 5 min, resultados y conclusiones) en `presentation.pdf` | ✅ | [`presentation.pdf`](presentation.pdf) (7 diapositivas); fuente en `presentación/presentacion_final.html`, figuras con `presentación/figuras.py` y guion en `presentación/GUION_FINAL.md` |
| `ENTREGABLE_FINAL.md` confirmando completitud | 🟡 | Este archivo |

## Resultado principal

| Modelo | PR-AUC CV 5×5 | PR-AUC test | ROC-AUC test | F1 test |
|---|---|---|---|---|
| Baseline · Reg. logística (Set A, 9 variables) | 0.165 ± 0.014 | 0.186 | 0.616 | 0.225 |
| **Modelo 1 · Reg. logística (depto + hora + altitud)** | **0.168 ± 0.015** | 0.184 | **0.632** | **0.255** |

El Modelo 1 iguala al baseline con 3 variables en lugar de 9 (+0.003 de PR-AUC, mejor en 15 de 25 particiones). Agregarle la lluvia no cambia nada (+0.000).

## Pendiente
- `report.pdf`.
- Revisión final de README y de la ejecución completa desde cero.
