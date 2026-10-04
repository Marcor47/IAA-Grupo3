# Entregable parcial (Semana 8)

## Documentación
- [x] `README.md` con descripción del problema, objetivo y metodología (falta escribir los apellidos de los integrantes)
- [x] Descripción del dataset en el README: origen, características (diccionario de datos) y permisos
- [x] Resúmenes de los 4 papers en el README, con las técnicas encontradas (cada integrante debe contrastar el suyo con el PDF)
- [ ] `papers/` con los 4 PDF (nombres en `papers/README.md`)
- [x] Propuesta de modelos (README, sección 4)

## Datos
- [x] `data/raw/accidentes_transito_carreteras.csv` subido al repo (archivo completo)
- [x] `data/raw/diccionario_de_datos.docx` (diccionario oficial de SUTRAN)
- [ ] `data/raw/clima_open_meteo.csv` generado al ejecutar `01_eda.ipynb`

## Código y resultados
- [ ] `notebooks/01_eda.ipynb` ejecutado completo, con salidas visibles y la celda "Hallazgos del EDA" completada
- [ ] `notebooks/02_modelos.ipynb` ejecutado completo (baseline evaluado) y la sección "Lectura del baseline" completada
- [ ] `results/metrics.csv` y `results/plots/` generados y subidos
- [ ] N de registros y resultados reales copiados al README (sección 2.1)

## Pendientes antes de la entrega
- [ ] Completar nombres de los integrantes en el README
- [ ] Confirmar con el profesor si Random Forest / XGBoost se aceptan como extensión de árboles de decisión
- [ ] Confirmar con el profesor cómo se entrega el repo (enlace público o acceso para el profesor)

## Observaciones
- La lluvia se mide en la capital de cada departamento (el dataset no trae coordenadas).
- Los datos cubren 2020-2021.
- El CSV real usa `;`, latin-1 y columnas `CODIGO_VM-MA`, `FALLECIDOS`, `HERIDOS`; `01_eda.ipynb` las normaliza.
