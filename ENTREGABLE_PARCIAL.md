# Entregable parcial (Semana 8)

| Requisito del enunciado | Estado | Dónde está |
|---|---|---|
| `README.md` completo con descripción del problema, objetivo y metodología | ✅ | [README §1](README.md#1-problema-y-objetivo) y [§4](README.md#4-metodología-y-propuesta-de-modelos) |
| Carpeta `papers/` con los papers revisados (mínimo 1 por integrante) | ✅ | [`papers/`](papers/): `paper_integrante1.pdf` … `paper_integrante4.pdf`, índice en [`papers/README.md`](papers/README.md) y resúmenes en [README §3](README.md#3-revisión-de-literatura) |
| Dataset en `data/raw/` con descripción de origen, características y permisos | ✅ | [`data/raw/`](data/raw/) (`accidentes_transito_carreteras.csv`, `diccionario_de_datos.docx`, `clima_open_meteo.csv`); descripción en [README §2](README.md#2-dataset) |
| EDA en `notebooks/01_eda.ipynb` con visualizaciones | ✅ | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (ejecutado, con la celda "Hallazgos del EDA"); figuras en [`results/plots/`](results/plots/) |
| Propuesta de modelos identificados en la literatura | ✅ | [README §3.1 y §4.2](README.md#42-modelos-candidatos) y sección 6 de `02_modelos.ipynb` |
| Baseline implementado y evaluado en `notebooks/02_modelos.ipynb` | ✅ | [`notebooks/02_modelos.ipynb`](notebooks/02_modelos.ipynb): regresión logística (clasificación) y Poisson (conteo) frente a un `Dummy`; métricas en [`results/metrics.csv`](results/metrics.csv) y [README §5](README.md#5-resultados-parciales) |
| Archivo `ENTREGABLE_PARCIAL.md` | ✅ | Este archivo |

## Observaciones
- La lluvia se mide en la capital de cada departamento (el dataset no trae coordenadas).
- Los datos cubren enero 2020 – septiembre 2021.
- Los notebooks se ejecutan en orden desde `notebooks/` (ver [README §6](README.md#6-cómo-ejecutar)); usan el CSV local y la caché de clima, así que no requieren internet si esos archivos existen.
- `src/` queda vacío por ahora: los módulos `preprocessing.py`, `models.py` y `evaluation.py` son parte del entregable final.
