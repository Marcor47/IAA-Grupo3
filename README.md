# Predicción del riesgo y la gravedad de accidentes en carreteras del Perú con datos abiertos y clima

**Curso:** Inteligencia Artificial Aplicada (1INF62) · PUCP · 2026-2
**Integrantes:** Marco Rodriguez, Angel Cerdán, Raul Malaver, Fernando De La Cruz

## Contenido
1. [Problema y objetivo](#1-problema-y-objetivo)
2. [Dataset](#2-dataset)
3. [Revisión de literatura](#3-revisión-de-literatura)
4. [Metodología y modelos](#4-metodología-y-modelos)
5. [Resultados](#5-resultados)
6. [Cómo ejecutar](#6-cómo-ejecutar)
7. [Estructura del repositorio](#7-estructura-del-repositorio)

---

## 1. Problema y objetivo

Los accidentes en carreteras causan muertes y heridos cada año en el Perú, y los recursos para atenderlos
(patrullas de la PNP, ambulancias, personal del Centro de Gestión y Monitoreo de SUTRAN) son limitados. Para
repartirlos hacen falta dos cosas: saber **dónde y cuándo habrá más accidentes** y, cuando llega un reporte,
saber **qué tan grave es** para priorizar el despacho.

> **Replanteo (noviembre de 2026).** En la presentación de avances, el proyecto preguntaba "¿cuánta mortalidad hay
> según la lluvia?" y lo respondía con asociaciones (chi², odds ratios, p-values): un problema de estadística
> inferencial. Ahora se plantea como **dos problemas de aprendizaje supervisado** con un uso operativo. El EDA y la
> literatura se usan para diseñar las variables, y la lluvia pasa a ser una variable más, cuyo aporte se mide con
> ablación.

**Objetivo general:** construir y comparar modelos de aprendizaje automático que (1) pronostiquen el riesgo diario
de accidentes por departamento y (2) clasifiquen la gravedad de un accidente al reportarse, integrando el registro
de SUTRAN con fuentes abiertas de clima y de infraestructura vial.

**Objetivos específicos**
- Integrar el registro de accidentes de SUTRAN con el clima diario y horario (Open-Meteo) y con los atributos de la
  Red Vial Nacional (MTC), para reconstruir las variables que la bibliografía asocia a la siniestralidad.
- Construir un panel departamento × día que incluya los días sin accidentes, para modelar la frecuencia.
- Entrenar y ajustar modelos propuestos en la literatura (Poisson, binomial negativa, regresión logística, Random
  Forest, gradient boosting, perceptrón multicapa y un ensamble por votación) y compararlos con referencias ingenuas,
  con validación temporal o cruzada repetida.
- Medir con ablación e importancia por permutación qué factores (tipo de choque, vía, clima, luz, calendario, lugar)
  aportan a la predicción.

| Tarea | Unidad | Pregunta (uso) | Objetivo | Se conoce |
|---|---|---|---|---|
| **1 · Pronóstico de riesgo** | departamento × día (15 336) | ¿Cuántos accidentes habrá mañana en cada departamento y habrá alguno fatal? (asignar patrullas y ambulancias) | `N_ACC` (conteo) y `HAY_FATAL` (binaria) | Historia hasta el día anterior, pronóstico de lluvia y calendario |
| **2 · Triaje de gravedad** | accidente (8 100) | Llegó un reporte: ¿es sin víctimas, con heridos o fatal? (priorizar el despacho) | `GRAVEDAD` ∈ {sin víctimas, heridos, fatal} | Tipo de choque, lugar, vía, clima y luz al momento del reporte |

---

## 2. Dataset

### 2.1 Origen y permisos
- **Nombre:** Accidentes de Tránsito en Carreteras (2020-2021).
- **Fuente:** Superintendencia de Transporte Terrestre de Personas, Carga y Mercancías (**SUTRAN**), publicado en la Plataforma Nacional de Datos Abiertos del Estado Peruano: <https://www.datosabiertos.gob.pe/dataset/accidentes-de-tr%C3%A1nsito-en-carreteras>
- **Contenido:** accidentes en vías nacionales y departamentales reportados por la Policía Nacional del Perú (PNP) y el Centro de Gestión y Monitoreo (CGM) de SUTRAN. Cada fila es un accidente.
- **Corte de la información:** 22/12/2021 (`FECHA_CORTE = 20211222`); el portal indica última modificación el 28/01/2022.
- **Licencia:** Open Data Commons Attribution (ODC-By). Es un dato público sin datos personales; se puede usar citando la fuente. No se requiere consentimiento adicional.
- **Tamaño:** 8 155 registros y 9 columnas en el CSV original. En la limpieza se descartan 3 sin dato de fallecidos, 7 sin departamento (sin departamento no se puede asignar lluvia) y 35 duplicados exactos; quedan **8 110 registros**, todos con dato de lluvia. El dataset limpio de la Tarea 2 (`data/processed/accidentes.csv`) excluye además 5 accidentes cuya gravedad no se conoce (sin dato de heridos y sin fallecidos): **8 105 accidentes × 33 columnas**. El panel de la Tarea 1 (`data/processed/panel_depto_dia.csv`) usa los 8 110.
- **Periodo real de los accidentes:** 01/01/2020 – 30/09/2021. La fecha de corte es el 22/12/2021, pero el archivo no contiene accidentes posteriores al 30/09/2021.
- **Archivo en el repo:** `data/raw/accidentes_transito_carreteras.csv` y su diccionario oficial en `data/raw/diccionario_de_datos.docx`.

### 2.2 Variables (según el diccionario de datos oficial)

| Variable | Descripción | Tipo | Tamaño | Observaciones del diccionario |
|---|---|---|---|---|
| `FECHA_CORTE` | Fecha de corte de la información | Numérico (AAAAMMDD) | 8 | |
| `FECHA` | Fecha del accidente de tránsito | Numérico (AAAAMMDD) | 8 | |
| `HORA` | Hora del accidente (HH:MM) | Numérico | 5 | "N.I." = no identificado: 88 accidentes sin hora |
| `DEPARTAMENTO` | Departamento donde ocurrió el accidente | Texto | 13 | 7 accidentes sin departamento |
| `CODIGO_VIA` | Código de la vía donde se reportó el accidente | Alfanumérico | 6 | 46 accidentes sin código de vía |
| `KILOMETRO` | Kilómetro de la vía donde se reportó el accidente | Numérico | 4 | 45 accidentes sin kilómetro |
| `MODALIDAD` | Atropello, choque, despiste, especial, volcadura o N.I. | Texto | 9 | 28 accidentes sin modalidad |
| `NUM_FALLECIDOS` | Número de personas reportadas como fallecidas | Numérico | 2 | 3 accidentes sin dato de fallecidos |
| `NUM_HERIDOS` | Número de personas reportadas como heridas | Numérico | 2 | 10 accidentes sin dato de heridos |

**Diferencias entre el diccionario y el archivo real** (el notebook las corrige automáticamente):
- En el CSV, las columnas se llaman `CODIGO_VM-MA` (encabezado con la tilde dañada, equivale a `CODIGO_VIA`), `FALLECIDOS` y `HERIDOS`.
- Separador `;`, codificación ISO-8859-1 (latin-1) y saltos de línea CRLF.
- Fecha como `AAAAMMDD` y hora como `HH:MM`.
- Los valores "N.I." se tratan como datos faltantes.

### 2.3 Datasets complementarios: ubicación, vía y clima
El registro de accidentes no tiene clima ni coordenadas, pero sí vía (`CODIGO_VIA`) y kilómetro.

- **Ubicación — Red Vial Nacional del MTC** (al 31/12/2014), publicada como servicio de mapas en el catálogo geoespacial de INGEMMET (<https://geocatmin.ingemmet.gob.pe/arcgis/rest/services/SERV_OTRAS_FUENTES/MapServer/23>). Cada tramo trae su ruta, su km de inicio y fin y su departamento. Con la vía y el kilómetro de cada accidente se interpola su posición sobre el tramo; el punto se acepta solo si el tramo está en el mismo departamento del accidente (o a ≤ 10 km de su límite). Así se ubica el **80.2 %** de los accidentes; el resto (vías departamentales, km "N.I." o km que no coincide con el kilometraje del MTC) usa la capital de su departamento. Copia en `data/raw/red_vial_nacional_mtc.json`.
- **Atributos de la vía — misma capa del MTC.** Del tramo en el que cae cada accidente se toman la topografía
  (plano / ondulado / montañoso), la superficie de rodadura (dominio de la capa: asfaltado, asfaltado económico,
  afirmado, sin afirmar, trocha, proyectada), su estado, el número de carriles, el ancho de calzada, la velocidad
  promedio del tramo y la región natural. Además se calcula la **sinuosidad** en ±1 km del punto (largo del camino /
  distancia en línea recta), como medida del alineamiento.
- **Precipitación — API histórica de Open-Meteo** (<https://open-meteo.com/en/docs/historical-weather-api>), gratuita para uso no comercial, sin API key y con atribución. Se descarga la precipitación diaria (`precipitation_sum`, en mm) en la **celda de 0.1° (~11 km)** que contiene a cada accidente y se une por **celda + fecha**; la API también da la **altitud** de la celda (`ELEVACION`). Copia en `data/raw/clima_celdas_open_meteo.csv`. Para cada accidente con hora se descarga también el **clima por hora** en su celda (`data/raw/clima_horario_celdas.csv`): código WMO agrupado (despejado / nublado / neblina / llovizna / lluvia), lluvia de la hora y de las 3 horas alrededor, nubosidad, temperatura, viento, neblina (temperatura − punto de rocío < 2 °C) y, con la salida y la puesta del sol del día, si era de día, de noche o crepúsculo. Se conserva además la lluvia en la capital del departamento (`PRECIP_CAPITAL`, `data/raw/clima_open_meteo.csv`) para comparar.

### 2.4 Limitaciones del dataset
- **Resolución espacial de la lluvia:** el dataset no trae coordenadas; la ubicación se reconstruye con la vía y el kilómetro. El 19.8 % de los accidentes no se puede ubicar y usa la lluvia de la capital de su departamento, y la lluvia se mide en celdas de ~11 km, no en el punto exacto.
- **Kilometraje:** la capa de vías es de 2014 y en algunas rutas el kilómetro reportado por SUTRAN no coincide con el kilometraje oficial del MTC (1 059 accidentes se dejan con la capital por esa razón).
- **Origen de la lluvia:** los valores de la API provienen de datos de reanálisis, no de estaciones; pueden diferir de las mediciones de SENAMHI, sobre todo en la zona andina.
- **Solo accidentes:** el registro no tiene días sin accidentes ni tráfico. Los días sin accidentes se reconstruyen en el panel departamento × día (Tarea 1); la exposición (vehículos que circulan) sigue sin conocerse.
- **Sin datos del vehículo ni de las personas:** tipo de vehículo, uso de cinturón, edad o alcohol, que son los factores más importantes en Babaoglu y Babaoglu (2021) y Emu et al. (2022), no existen en el registro.
- **Periodo enero 2020 – septiembre 2021:** coincide con la pandemia (restricciones de movilidad); en el EDA, abril de 2020 cae a ≈ 20 fallecidos, coherente con la cuarentena.
- **Datos faltantes y posible subregistro** (valores "N.I.").

---

## 3. Revisión de literatura

Cada integrante revisó un artículo revisado por pares sobre predicción de fatalidad o gravedad, o sobre búsqueda de patrones en accidentes viales, que no usa el mismo dataset que nosotros. Los PDF están en [`papers/`](papers/) (índice en [`papers/README.md`](papers/README.md)).

### Paper 1 · Marco Rodriguez · [`paper_integrante1.pdf`](papers/paper_integrante1.pdf)
**K. V. Mhetre y A. D. Thube**, "Count Data Modeling for Predicting Crash Severity on Indian Highways", *Engineering, Technology & Applied Science Research*, vol. 13, n.º 5, pp. 11816-11820, 2023. DOI: [10.48084/etasr.6172](https://doi.org/10.48084/etasr.6172)

- **Problema:** predecir la fatalidad de accidentes en una carretera nacional rural de la India (NH-48, Maharashtra, ~265 km) según la naturaleza del choque, el horario (AM/PM) y el clima.
- **Datos:** registros de la National Highways Authority of India, 2016-2020. División 70 % entrenamiento / 30 % validación.
- **Técnica:** regresión **binomial negativa** (extiende Poisson a conteos con sobredispersión). Cuatro modelos: naturaleza del choque, horario, condición climática (fino, nublado, llovizna, lluvia fuerte, neblina) y choque + horario. Validación con log-verosimilitud, AIC, BIC, MAD, MSE, RMSE y MAPE.
- **Resultados:** la naturaleza del choque y el clima resultaron significativos; lluvia ligera y fuerte, neblina, tiempo fino y nublado aparecen asociados positivamente con la fatalidad. Los modelos 2 y 4 ajustaron mejor.
- **Aporte a nuestro proyecto:** respalda modelar `NUM_FALLECIDOS` como **conteo** (Poisson como baseline y binomial negativa como modelo propuesto, §4.2) y confirma el clima como variable relevante. **Crítica:** usa el clima como categorías y no aísla un efecto propio de la lluvia; nosotros usamos precipitación continua en mm y evaluamos en datos de prueba.

### Paper 2 · Angel Cerdán · [`paper_integrante2.pdf`](papers/paper_integrante2.pdf) (resumen de una página: [`paper_integrante2_resumen.pdf`](papers/paper_integrante2_resumen.pdf))
**E. G. Muñoz Muñoz, D. A. Verduga Alcívar, Y. F. Guerrero Alcívar, M. A. Lapo Palacios y O. Zorrilla Briones**, "Búsqueda de patrones con machine learning en datos de siniestros de tránsito", *Ciencia Latina Revista Científica Multidisciplinar*, vol. 8, n.º 2, pp. 1617-1637, 2024. DOI: [10.37811/cl_rcm.v8i2.10592](https://doi.org/10.37811/cl_rcm.v8i2.10592)

- **Problema:** identificar patrones en los siniestros de tránsito (lugar, hora, causa, condiciones del entorno) para orientar intervenciones de seguridad vial focalizadas. Es un análisis **no supervisado**: no hay etiqueta que predecir.
- **Datos:** 21 352 siniestros de la base de la Agencia Nacional de Tránsito (organizada por provincia y cantón, como en Ecuador). 12 columnas: año, mes, día, hora, provincia, cantón, zona, clase de siniestro, causa, número de fallecidos, número de lesionados y total de víctimas. El artículo no precisa el periodo cubierto.
- **Técnicas:**
  - *Preprocesamiento:* limpieza (faltantes, errores tipográficos en categóricas, duplicados), normalización de las variables numéricas y **codificación binaria** de las categóricas (de 11 a 39 variables).
  - *EDA:* distribuciones por mes, día de la semana, zona, clase, causa y hora; **mapa coroplético** de siniestros por zona y **diagrama de Sankey** tiempo → causa/ubicación → víctimas.
  - *Clustering:* **K-Means** (minimiza la suma de distancias cuadradas dentro del cluster, WCSS). El número de clusters se eligió con el **método del codo** (inflexión entre 3 y 5) y el **análisis de silueta**; se fijó K = 4.
  - *Reducción de dimensionalidad:* **PCA** de 39 a 2 y 3 componentes para visualizar los clusters.
  - *Complemento supervisado:* **Random Forest** con gráfico de **importancia de variables** y visualización de un **árbol de decisión** individual del bosque.
- **Resultados:** cuatro clusters: (0) urbano, tarde/noche, distracción del conductor (5 563 casos); (1) rural, fines de semana, noche/madrugada, exceso de velocidad y alcohol (3 560); (2) intersecciones urbanas en horas punta, incumplimiento de señales (6 317); (3) carreteras y zonas periurbanas con mal estado de la vía y **clima adverso** (5 912). En el Random Forest las variables más importantes fueron el número de lesionados y el total de víctimas, seguidas de provincia, cantón, mes y hora.
- **Aporte a nuestro proyecto:** el cluster 3 (carreteras + clima adverso) es justamente nuestro caso de estudio y respalda incluir el clima y la ubicación. Tomamos dos ideas: (a) agrupar con **K-Means** los departamentos (o vías) según su régimen de lluvia y su siniestralidad, y usar el cluster como variable espacial más compacta que el one-hot de 25 departamentos; (b) usar **PCA** en el EDA para visualizar la estructura de los accidentes y la **importancia de variables** de modelos de árboles para ordenar los predictores. También valida el preprocesamiento que ya hacemos (limpieza de "N.I.", codificación de categóricas, escalado).

### Paper 3 · Raul Malaver · [`paper_integrante3.pdf`](papers/paper_integrante3.pdf)
**L. Babaoglu y C. Babaoglu**, "Prediction of Fatalities in Vehicle Collisions in Canada", *Promet – Traffic&Transportation*, vol. 33, n.º 5, 2021. DOI: [10.7307/ptt.v33i5.3782](https://doi.org/10.7307/ptt.v33i5.3782)

- **Problema:** predecir si un accidente tendrá fallecidos y descubrir las causas de la fatalidad en las carreteras de Canadá.
- **Datos:** National Collision Database de Canadá, 1999-2017.
- **Técnicas:** análisis exploratorio con minería de datos y **reglas de asociación** para hallar factores clave; luego dos clasificadores supervisados, **regresión Lasso** (regresión con penalización L1 que selecciona variables) y **XGBoost** (árboles con *boosting*). Se interpretan las variables más importantes del modelo ganador.
- **Resultados:** XGBoost fue el mejor clasificador con 83 % de exactitud. Las variables más importantes fueron la configuración de la colisión y el uso de dispositivos de seguridad, por encima de año del vehículo, hora, edad o sexo. Los choques frontales fuera de intersecciones y sin control de tránsito resultaron los más letales, y **la mayoría de las muertes ocurre con clima y condiciones de vía no extremos**.
- **Aporte a nuestro proyecto:** ejemplo de pipeline EDA + Lasso + XGBoost y de interpretación por importancia de variables. Su hallazgo sobre clima no extremo es un contrapunto a nuestra hipótesis: conviene controlar por exposición y por departamento antes de atribuir efectos a la lluvia.
- **Limitación:** la exactitud sola no es suficiente con clases desbalanceadas.

### Paper 4 · Fernando De La Cruz · [`paper_integrante4.pdf`](papers/paper_integrante4.pdf)
**M. Emu et al.**, "Fatality Prediction for Motor Vehicle Collisions: Mining Big Data Using Deep Learning and Ensemble Methods", *IEEE Open Journal of Intelligent Transportation Systems*, 2022. DOI: [10.1109/OJITS.2022.3160404](https://doi.org/10.1109/OJITS.2022.3160404)

- **Problema:** predecir si el resultado de una colisión será fatal o no, como base de un posible sistema de alerta en tiempo real.
- **Datos:** base de choques de Canadá con 5.8 millones de registros.
- **Técnicas:** ensambles por **votación mayoritaria y soft voting** para tratar el desbalance de clases, y **redes neuronales convolucionales (CNN)**; además un análisis del contenido de información de cada atributo para identificar los factores que más distinguen choques fatales de no fatales.
- **Resultados:** exactitud cercana al 75 % con CNN. Los factores más influyentes incluyen las características de la vía y **las condiciones climáticas al momento del choque**, tipo de vehículo, hora, clase y posición del usuario, uso de dispositivos de seguridad y estado del control de tránsito.
- **Aporte a nuestro proyecto:** respalda usar redes neuronales (MLP) y ensambles, e incluir el clima como variable de entrada; sugiere medir la importancia de cada variable.
- **Limitación:** conjunto de datos y contexto muy distintos (Canadá, millones de registros); la exactitud ~75 % sola no refleja el desbalance.

### 3.1 Técnicas encontradas

| Técnica | Dónde aparece | Para qué sirve |
|---|---|---|
| Regresión binomial negativa / Poisson | Mhetre y Thube | Modelar conteo de fallecidos con sobredispersión |
| Lasso (regresión con penalización L1) | Babaoglu | Selección de variables y clasificación |
| XGBoost (*boosting*) | Babaoglu | Mejor clasificador del paper (83 % de exactitud) |
| Random Forest y árbol de decisión | Muñoz et al. | Clasificación e interpretación (importancia de variables, lectura de un árbol) |
| CNN y ensambles por votación | Emu et al. | Clasificación con grandes volúmenes y desbalance |
| K-Means + método del codo y silueta | Muñoz et al. | Agrupar accidentes o zonas con perfiles similares; elegir K |
| PCA | Muñoz et al. | Reducir dimensionalidad y visualizar en 2D/3D |
| Codificación de categóricas y normalización | Muñoz et al. | Preparar variables para modelos basados en distancia |
| Mapa coroplético y diagrama de Sankey | Muñoz et al. | EDA espacial y de flujos tiempo → causa → víctimas |
| Reglas de asociación | Babaoglu | Descubrir factores de riesgo en el EDA |
| Importancia de variables / contenido de información | Muñoz et al., Babaoglu, Emu et al. | Ordenar variables por su relación con la fatalidad |
| AIC, BIC, MAD, RMSE, MAPE | Mhetre y Thube | Evaluar modelos de conteo |

---

## 4. Metodología y modelos

**Flujo**
1. **Preprocesamiento** (`00_preprocesamiento.ipynb`; las mismas funciones están en `src/preprocessing.py`, que se
   ejecuta con `python -m src.preprocessing`):
   - limpieza del CSV de SUTRAN;
   - ubicación de cada accidente por vía y km sobre la Red Vial Nacional, con los atributos de su tramo;
   - clima diario y por hora en su celda, y la condición de luz;
   - variables derivadas;
   - selección de variables y encoding.

   Salen dos datasets limpios:
   - `accidentes.csv`: 8 105 × 33, Tarea 2;
   - `panel_depto_dia.csv`: 15 336 × 25, Tarea 1.
2. **EDA** (`01_eda.ipynb`): trabaja solo sobre los datasets limpios. Cubre la gravedad frente al tipo de choque,
   el lugar, la hora, la vía, la luz y el clima, y el panel (ceros, memoria, cuarentena, lluvia). Cada hallazgo se
   traduce en una decisión de modelado.
3. **Baseline** (`02_modelos.ipynb`): por cada tarea, una referencia sin aprendizaje y un modelo lineal con todas las
   variables, sin ajuste de hiperparámetros.
4. **Modelos de la literatura** (un notebook por modelo, `03` a `06`): cada uno se ajusta con la misma validación y
   las mismas métricas que el baseline, y se compara con él pliegue a pliegue.
5. **Ablación** (`07_ablacion.ipynb`): con el mejor modelo de cada objetivo, se quita un grupo de variables a la vez
   y se mide cuánto empeora; se complementa con la importancia por permutación.
6. **Comparación** (`08_comparacion.ipynb`): Tabla I con todos los modelos a partir de `results/metrics.csv`, y
   reevaluación pliegue a pliegue (`M.modelos_ajustados`) para la comparación pareada, las figuras finales y el
   veredicto por variable objetivo.

### 4.1 Variables (factores de la bibliografía → columnas)

Los grupos están en `src/models.py` (`GRUPOS_T1`, `GRUPOS_T2`) y son la unidad de la ablación.

| Grupo | Factor en la bibliografía | Variables | Tarea |
|---|---|---|---|
| Tipo de choque | Naturaleza / configuración del choque (Mhetre; Babaoglu; Emu) | `MODALIDAD` | 2 |
| Vía | Alineamiento, superficie, configuración, velocidad (Babaoglu; Emu) | `TOPOGRAFIA`ᵒ, `SUPERFICIE`ᵒ, `ESTADO_VIA`ᵒ, `SINUOSIDAD`, `CARRILES`, `ANCHO_CALZADA`, `VEL_PROMEDIO_TRAMO`, `TIPO_VIA`, `UBICACION` | 2 |
| Clima | Condición climática por categorías (Mhetre; Babaoglu; Emu) | Diario: `PRECIP_LOG`, `HORAS_LLUVIA`; por hora: `CLIMA_HORA`, `PRECIP_3H_LOG`, `NEBLINA`. Tarea 1: lluvia del día y acumulada de 3 y 7 días en la capital | 1 y 2 |
| Luz y hora | Condición de luz y hora (Emu; Muñoz) | `ES_DE_DIA`, `CREPUSCULO`, `HORA_SIN`/`HORA_COS` | 2 |
| Calendario | Mes, fin de semana, feriados (Muñoz, cluster 1) | `MES_SIN`/`MES_COS`, `FIN_DE_SEMANA`, `FERIADO`, `PUENTE`, `CUARENTENA` | 1 y 2 |
| Lugar | Provincia / cantón, zona (Muñoz) | `DEPARTAMENTO`, `REGION_NATURAL`, `ELEVACION`, `DIST_CAPITAL_KM`; Tarea 1: `ALTITUD_MEDIA` | 1 y 2 |
| Historia | Variación semanal de los choques (Mhetre, trabajos relacionados) | `N_ACC_AYER`, `N_ACC_SEMANA_PASADA`, `N_ACC_M7`, `N_ACC_M28`, `N_FATALES_M28`, `N_HERIDOS_M28`, `P_ATROPELLO_M28`, `P_CHOQUE_M28` (todas hasta *t − 1*) | 1 |

ᵒ = ordinal (una columna: plano < ondulado < montañoso; muy buena → mala; asfaltado → trocha).

**Decisiones de selección y encoding** (12/11/2026; detalle en `00_preprocesamiento.ipynb` §7). Para que el encoding no
sea excesivo, ya que antes eran 135 columnas en la Tarea 2:
- **Salen del modelo:** `CODIGO_VIA` (174 vías, cubiertas por los atributos del tramo), `FRANJA` (repite la hora),
  `DIA_SEMANA` (ya está `FIN_DE_SEMANA`), `TEMPERATURA`, `VIENTO` y `NUBOSIDAD` (sin respaldo ni señal), y `LAT`/`LON`
  (se superponen con el departamento).
- **Codificación:**
  - topografía, estado y superficie son ordinales;
  - `DEPARTAMENTO`, `MODALIDAD`, `CLIMA_HORA`, `REGION_NATURAL` y `TIPO_VIA` van en one-hot, con las categorías de
    < 30 casos agrupadas;
  - las binarias ocupan una sola columna.

  El encoding se aplica dentro del pipeline de cada modelo (`src/models.py::preprocesador`), así que se ajusta solo con
  el train de cada pliegue.
- **Resultado:** **67 columnas** en la Tarea 2 y 43 en la Tarea 1.

**Sin fuga de información.**
- En la Tarea 1, toda la historia se desplaza un día. El preprocesamiento lo verifica con un `assert`.
- En la Tarea 2, `NUM_HERIDOS` y `NUM_FALLECIDOS` solo forman parte del objetivo (`GRAVEDAD`), nunca son variables de entrada.

### 4.2 Modelos

Todos los modelos se construyen con las factorías de `src/models.py` (`poisson`, `binomial_negativa`, `logit`,
`rf_conteo`, `random_forest`, `gbm_conteo`, `gbm_clasificador`, `mlp_conteo`, `mlp_clasificador` y `votacion`); la
configuración elegida por cada búsqueda está en `M.CONFIG_AJUSTADA`.

| Modelo | Tarea | Respaldo | Estado |
|---|---|---|---|
| Referencias sin ML: media de 28 días, semana pasada, fatales de 28 días, Dummy | 1 y 2 | Lo que hay que superar | **Implementado** |
| Regresión de Poisson | 1 · `N_ACC` | Mhetre y Thube (2023) | **Implementado (baseline)** |
| Regresión binomial negativa (NB2) | 1 · `N_ACC` y `HAY_FATAL` (vía `N_FATALES`) | Mhetre y Thube (2023) | **Implementado** (`05_binomial_negativa.ipynb`) |
| Regresión logística (binaria / multinomial) | 1 · `HAY_FATAL`, 2 | Curso; Babaoglu (Lasso) | **Implementado (baseline)** |
| Random Forest (`criterion="poisson"` en la Tarea 1) | 1 y 2 | Muñoz et al. (2024) | **Implementado y ajustado** (`03_random_forest.ipynb`) |
| Gradient boosting (`HistGradientBoosting`, pérdida de Poisson en la Tarea 1, categóricas nativas) | 1 y 2 | Babaoglu y Babaoglu (2021) (XGBoost) | **Implementado y ajustado** (`04_gradient_boosting.ipynb`) |
| MLP (perceptrón multicapa; pérdida de Poisson en `N_ACC`, pesos de clase balanceados en la Tarea 2) | 1 y 2 | Emu et al. (2022) | **Implementado y ajustado** (`06_mlp.ipynb`) |
| Ensamble por votación (`M.votacion`: promedio de las predicciones de Poisson/logística, MLP, Random Forest y gradient boosting; votación suave en clasificación) | 1 y 2 | Emu et al. (2022) | **Implementado** (`06_mlp.ipynb` §4) |
| Comparación final (`M.modelos_ajustados`: todos los modelos con su configuración elegida) | 1 y 2 | — | **Hecho** (`08_comparacion.ipynb`) |
| Ablación por grupos + importancia por permutación | 1 y 2 | Muñoz; Babaoglu; Emu (importancia de variables) | **Hecho** (`07_ablacion.ipynb`) |

### 4.3 Validación y métricas

| Tarea | Validación | Métrica principal | Otras |
|---|---|---|---|
| 1 · `N_ACC` | Origen móvil por fechas (`E.cv_temporal`: 5 bloques tras 180 días de entrenamiento) y **holdout abr–sep 2021** | **D² de Poisson** (fracción de devianza explicada; 0 = predecir la media) | Devianza de Poisson, MAE, RMSE |
| 1 · `HAY_FATAL` | Igual | **PR-AUC** (al azar = 0.058) | ROC-AUC, Brier (calibración) |
| 2 · `GRAVEDAD` | CV estratificada **repetida 5 × 5** (los mismos pliegues para todos) y test 80/20 | **Macro-F1** | Balanced accuracy, PR-AUC de la clase fatal (al azar = 0.117), log-loss, matriz de confusión |

- **La exactitud no se usa:** el 66 % de los departamento-días tiene 0 accidentes y solo el 11.7 % de los accidentes es fatal.
- **El MAE no decide en la Tarea 1:** premia predecir 0.
- **La validación de la Tarea 1 es temporal** porque la serie tiene memoria y cambia de nivel con la cuarentena. Una partición aleatoria mezclaría el futuro con el pasado.
- **Comparación entre modelos:** se hace pliegue a pliegue (`E.comparacion_pareada`).

---

## 5. Resultados

### 5.1 EDA (`01_eda.ipynb`)
- **Datos:** 8 110 accidentes (ene 2020 – sep 2021), de los cuales:
  - 36 % sin víctimas, 52 % con heridos y 11.7 % fatales;
  - 80.2 % ubicados por vía y km;
  - todos con clima en su celda, y los que tienen hora, con clima por hora.
- **Panel:** el 66 % de los departamento-días no tiene accidentes y el 5.8 % tiene alguno fatal. La serie tiene
  memoria y cae con la cuarentena (abril de 2020). Con 5–10 mm de lluvia en la capital hay ≈ 23 % más accidentes que
  el promedio del departamento. La relación es débil y no monótona, pero existe, a diferencia de lo que pasa con la gravedad.
- **Gravedad (% de accidentes fatales):**
  - **Tipo de choque:** atropello 42.9 %, despiste 7.4 %.
  - **Luz:** noche 14.1 %, día 10.0 %.
  - **Alineamiento:** rectas 13.4 %, curvas cerradas 8.1 %.
  - **Relieve:** plano 13.9 %, montañoso 7.5 %. Coincide con Babaoglu (2021), que encuentra la mayoría de muertes
    en vías rectas y niveladas; una explicación probable es la velocidad.
  - **Clima por categorías:** casi no cambia el % fatal.

  ![Variables de la bibliografía y gravedad](results/plots/eda_03_variables_bibliografia.png)

### 5.2 Modelos (`results/metrics.csv`)

**Tarea 1 · `N_ACC`** (D² de Poisson; más es mejor)

| Modelo | CV temporal (5 pliegues) | Holdout abr–sep 2021 |
|---|---|---|
| Ingenuo · semana pasada | −4.650 | −4.610 |
| Ingenuo · media 28 días | 0.217 | **0.277** |
| **Baseline · Poisson** | **0.256** | 0.267 |
| Binomial negativa (`05_binomial_negativa.ipynb`) | 0.256 | 0.268 |
| Random Forest ajustado (`03_random_forest.ipynb`) | 0.254 | **0.293** |
| MLP ajustado (`06_mlp.ipynb`) | 0.250 | 0.292 |
| Gradient boosting ajustado (`04_gradient_boosting.ipynb`) | 0.267 | 0.289 |
| Ensamble por votación (`06_mlp.ipynb` §4) | **0.268** | **0.296** |

**Tarea 1 · `HAY_FATAL`** (PR-AUC; al azar ≈ 0.06)

| Modelo | CV temporal | Holdout | ROC-AUC holdout |
|---|---|---|---|
| Ingenuo · fatales últimos 28 días | 0.112 | 0.131 | 0.689 |
| **Baseline · Reg. logística** | **0.133** | **0.161** | 0.734 |
| Binomial negativa (vía `N_FATALES`) | 0.130 | 0.160 | 0.731 |
| Random Forest ajustado | **0.140** | 0.158 | 0.725 |
| MLP ajustado | 0.138 | 0.157 | 0.722 |
| Gradient boosting ajustado | 0.137 | 0.160 | **0.735** |
| Ensamble por votación | 0.135 | 0.160 | 0.733 |

**Tarea 2 · `GRAVEDAD`** (CV 5 × 5)

| Modelo | Macro-F1 | Balanced acc. | PR-AUC fatal | Log-loss |
|---|---|---|---|---|
| Dummy (proporciones) | 0.228 | 0.333 | 0.117 | 0.960 |
| Modelo 2 del avance (depto + hora + altitud + modalidad) | 0.421 | **0.469** | **0.269** | 1.037 |
| **Baseline · Reg. logística multinomial (67 columnas)** | 0.419 | 0.466 | 0.266 | 1.037 |
| Random Forest ajustado (`03_random_forest.ipynb`) | **0.452** | 0.458 | 0.263 | 0.980 |
| MLP ajustado (`06_mlp.ipynb`, mismas 67 columnas) | 0.422 | 0.460 | 0.254 | 1.039 |
| Gradient boosting ajustado (`04_gradient_boosting.ipynb`) | 0.432 | 0.441 | 0.236 | 1.012 |
| Ensamble por votación (`06_mlp.ipynb` §4) | 0.447 | 0.466 | 0.268 | 0.993 |

![Matriz de confusión del baseline](results/plots/modelo_03_t2_baseline_test.png)

### 5.3 Lectura de los resultados
- **Tarea 1 · conteo:** el Poisson le gana a la media móvil en los 5 pliegues de la CV (+0.039 de D², IC 95 % aprox.
  +0.025 a +0.053). En el holdout queda empatado o un poco por debajo (0.267 frente a 0.277). Hay señal
  aprendible más allá de la inercia, pero un modelo lineal apenas la captura; los modelos no lineales de abajo la
  aprovechan mejor en el holdout.
- **Tarea 1 · `HAY_FATAL`:** la logística mejora a la referencia en CV (+0.021, mejor en 4 de 5 pliegues) y en el holdout
  (PR-AUC 0.161, 2.4 veces el azar; ROC-AUC 0.734).
- **Tarea 2:** pasar de las 4 variables del avance a todas (67 columnas tras el encoding) con un modelo **lineal**
  deja el resultado igual: −0.002 de macro-F1, mejor en 10 de 25 pliegues. Con 135 columnas, antes de la reducción,
  quedaba algo peor (−0.004). Sin interacciones, las variables nuevas no suman; con árboles sí (Random Forest, abajo).
- **El triaje sigue siendo difícil:** en el test, el baseline detecta el 51 % de los fatales con una precisión de 0.21. Los heridos
  y los sin víctimas se confunden entre sí, porque el registro no trae vehículo, ocupantes ni velocidad.
- **Binomial negativa:** empata con el Poisson en `N_ACC` (D² −0.001, IC −0.002 a +0.000) y con la logística en
  `HAY_FATAL`. La sobredispersión es significativa pero pequeña (α = 0.10; AIC 16 799 frente a 16 816) y en
  `N_FATALES` es nula (α ≈ 0): las variables explican casi toda la variación extra, así que el Poisson basta.
- **MLP** (Emu et al., 2022; hiperparámetros ajustados con la misma validación, sin mirar el holdout ni el test):
  - **`N_ACC`:** empata con el Poisson en la CV (−0.006, mejor en 1 de 5 pliegues), pero en el holdout es el único
    modelo que supera a la media de 28 días (0.292; 0.283 ± 0.006 con 5 semillas). Sigue mejor la subida de
    accidentes tras la cuarentena (figura `modelo_10`): señal a favor de las interacciones no lineales, que los
    árboles confirman.
  - **`HAY_FATAL`:** sin mejora frente a la logística (IC con el 0 en la CV; por debajo en el holdout con todas las semillas).
  - **Tarea 2:** empata en macro-F1 (+0.003, IC con el 0), pero ordena peor a los fatales (PR-AUC −0.011, peor en 21
    de 25 pliegues) y en el test queda por debajo con todas las semillas (0.423 ± 0.009 frente a 0.436). Con 8 105
    accidentes y sin variables del vehículo ni de las personas, la red no aprovecha más que el modelo lineal; Emu et
    al. usan 5.8 millones de registros.

- **Random Forest** (Muñoz et al., 2024; `RandomizedSearchCV` de 20 combinaciones con la misma validación):
  - **`N_ACC`:** empata con el Poisson en la CV (−0.002, IC −0.029 a +0.025) y es el **mejor en el holdout** (0.293,
    frente a 0.277 del ingenuo y 0.292 del MLP). Confirma lo del MLP: los modelos no lineales siguen mejor la subida
    de accidentes tras la cuarentena (figura `modelo_04`).
  - **`HAY_FATAL`:** sin mejora frente a la logística (+0.006 en la CV con IC con el 0; 0.158 frente a 0.161 en el
    holdout). Probabilidades bien calibradas (Brier 0.060, figura `modelo_05`).
  - **Tarea 2:** primer modelo que **supera a la logística en macro-F1**: +0.033 (IC +0.029 a +0.036), mejor en los 25
    pliegues. La ganancia viene de separar sin víctimas y heridos; la clase fatal no mejora (PR-AUC 0.263 frente a
    0.266; en el test, recall 0.27 con precisión 0.25).


- **Gradient boosting** (Babaoglu y Babaoglu, 2021; `GridSearchCV` con la misma validación):
  - **`N_ACC`:** el mejor promedio en la CV de los modelos individuales (0.267; +0.011 frente al Poisson, IC −0.013
    a +0.035) y por encima del ingenuo en el holdout (0.289 frente a 0.277), como el RF y el MLP (figura `modelo_13`).
    La grilla eligió el modelo más simple y regularizado: la señal más allá de la historia reciente es débil.
  - **`HAY_FATAL`:** empate con la logística (+0.004 en la CV con IC con el 0; 0.160 frente a 0.161 en el holdout).
  - **Tarea 2:** supera a la logística en macro-F1 (+0.013, IC +0.008 a +0.018), pero ordena peor a los fatales
    (PR-AUC −0.030) y detecta menos (recall 0.28 en el test, frente a 0.51); queda por debajo del RF (0.432 frente
    a 0.452).

- **Ensamble por votación** (Emu et al., 2022; promedio de Poisson/logística, MLP, RF y GBM con la configuración de
  cada uno, pesos iguales):
  - **`N_ACC`:** el mejor en la CV (0.268) y en el holdout (0.296), pero empata con el GBM (+0.001, IC −0.001 a
    +0.003): los miembros predicen casi lo mismo (correlación 0.96–0.98) y promediarlos aporta poco (figura `modelo_16`).
  - **`HAY_FATAL`:** sin mejora frente a la logística ni al RF (IC con el 0 en la CV; 0.160 frente a 0.161 en el holdout).
  - **Tarea 2:** combina en parte las fortalezas. Supera a la logística en macro-F1 (+0.028, IC +0.024 a +0.032, 25 de
    25 pliegues) sin perder en balanced accuracy ni en PR-AUC de fatales (0.268, la más alta). Queda algo por debajo
    del RF en macro-F1 (−0.005), pero ordena mejor a los fatales (+0.005). En el test tiene el mejor macro-F1 (0.447) y
    detecta el 37 % de los fatales (RF 27 %, logística 51 %) (figura `modelo_18`). Es el modelo más equilibrado: en ninguna
    variable queda por debajo del baseline.

- **Comparación final** (`08_comparacion.ipynb`; figuras `modelo_19` a `modelo_22`):
  - **`N_ACC`:** todos los modelos con aprendizaje superan a la media de 28 días en los 5 pliegues, pero ninguno al
    Poisson con un IC sin el 0. En el holdout, los cuatro no lineales (0.289–0.296) superan a la media de 28 días y los
    lineales no (0.267–0.268). Se elige el **ensamble**: mejor en la CV (0.268) y en el holdout (0.296).
  - **`HAY_FATAL`:** empate entre todos (PR-AUC 0.130–0.140 en la CV, 0.157–0.161 en el holdout). Se elige la
    **logística**: la más simple, calibrada y con la mejor PR-AUC en el holdout.
  - **`GRAVEDAD`:** el RF (+0.033), el ensamble (+0.028) y el GBM (+0.014) superan a la logística en macro-F1 con un
    IC sin el 0, pero ninguno la supera en la clase fatal. Por la métrica principal gana el **RF**; si importa no dejar
    pasar fatales, la logística; el **ensamble** es el punto medio (figura `modelo_21`).

  ![Comparación final](results/plots/modelo_19_comparacion_metrica_principal.png)

- **Ablación e importancia por permutación** (`07_ablacion.ipynb`; GBM en `N_ACC`, RF en `HAY_FATAL` y `GRAVEDAD`;
  figuras `modelo_07` y `modelo_08`):
  - **Clima:** aporta poco, pero no cero. En `N_ACC`, quitarlo baja el D² en 0.001 (IC ± 0.001, peor en 4 de 5
    pliegues); en `HAY_FATAL` y en `GRAVEDAD` no aporta (caída 0.000 con IC que incluye el 0).
  - **Tarea 1:** el **lugar** es el único grupo cuya ablación empeora en todos los pliegues (D² −0.022). La **historia**
    es lo que más usa el modelo (permutación: D² −0.21), pero quitarla no cambia la métrica: lleva la misma
    información que el departamento.
  - **Tarea 2:** domina el **tipo de choque** (macro-F1 −0.042, 25 de 25 pliegues), seguido del lugar (−0.016) y de la
    luz y la hora (−0.009); la vía y el calendario aportan poco (≈ 0.003).

  ![Ablación por grupos](results/plots/modelo_07_ablacion.png)

Detalle de cada modelo, la comparación final, la ablación y las conclusiones en
[`results/model_comparison.txt`](results/model_comparison.txt) (§3–12).

### 5.4 Siguientes pasos
- **Exposición:** agregar el flujo vehicular (peajes) si existe en datos abiertos; hoy los modelos no saben cuánto
  tráfico hay.
- **Más de un holdout:** ampliar el periodo para validar fuera de la salida de la cuarentena (abr–sep 2021).
- **Triaje por costo:** elegir umbrales de decisión por clase según el costo de dejar pasar un fatal.

---

## 6. Cómo ejecutar

### Requisitos
- Python 3.10 o superior (`python --version`). Probado con 3.12 y 3.14.
- Los datos procesados y las descargas externas ya están en el repo: los notebooks corren sin internet.

### 1. Crear el entorno e instalar las dependencias (una sola vez)

**Windows · cmd**
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**Windows · PowerShell**
```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Linux / Mac**
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

> Al activar el entorno, la consola muestra `(.venv)` al inicio de la línea. Si no aparece, `pip` instalará en el
> Python global. Para confirmar qué Python se usa: `where python` (Windows) o `which python` (Linux/Mac).

**Sin entorno virtual (Python global).** Omite la creación y activación de `.venv` y ejecuta solo
`python -m pip install -r requirements.txt`; luego abre directamente `jupyter notebook`. Las librerías quedan
compartidas con tus otros proyectos y pueden entrar en conflicto si alguno necesita otras versiones; por eso se
recomienda el entorno virtual. Si ya las tienes instaladas, puedes omitir este paso.

### 2. Comprobar la instalación
```
python -c "from src import models as M; print('OK', M.cargar_panel().shape)"
```

### 3. Trabajar con los notebooks
Cada vez que trabajes, activa el entorno y abre Jupyter con `jupyter notebook`. En Jupyter o VS Code, elige el kernel
de `.venv`. Para ejecutar un notebook: *Kernel → Restart & Run All*.

| Notebook | Qué hace |
|---|---|
| `00_preprocesamiento.ipynb` | Limpieza, ubicación, atributos de la vía, clima, variables, selección y encoding. Guarda `data/processed/accidentes.csv` y `panel_depto_dia.csv`. Equivale a `python -m src.preprocessing` |
| `01_eda.ipynb` | EDA sobre los datasets limpios (figuras `eda_*`) |
| `02_modelos.ipynb` | Baselines de las dos tareas, `results/metrics.csv` y figuras `modelo_01`–`modelo_03` |
| `03_random_forest.ipynb` | Random Forest para las dos tareas, con ajuste de hiperparámetros (Muñoz et al., 2024); figuras `modelo_04`–`modelo_06` (≈ 15 min) |
| `04_gradient_boosting.ipynb` | Gradient boosting para las dos tareas, con ajuste de hiperparámetros en grilla (Babaoglu y Babaoglu, 2021); figuras `modelo_13`–`modelo_15` (15–30 min) |
| `05_binomial_negativa.ipynb` | Binomial negativa para `N_ACC` y `HAY_FATAL` (Mhetre y Thube, 2023); figura `modelo_09` |
| `06_mlp.ipynb` | MLP para las dos tareas, con ajuste de hiperparámetros y sensibilidad a la semilla (Emu et al., 2022), y ensamble por votación (§4); figuras `modelo_10`–`modelo_12` y `modelo_16`–`modelo_18` |
| `07_ablacion.ipynb` | Ablación por grupos de variables e importancia por permutación con el mejor modelo de cada objetivo; figuras `modelo_07` y `modelo_08` (≈ 10 min) |
| `08_comparacion.ipynb` | Comparación final: Tabla I, comparación pareada de todos los modelos, figuras `modelo_19`–`modelo_22` y veredicto por variable objetivo (≈ 20 min) |

### 4. Reproducir todos los resultados
Los notebooks se ejecutan en orden. El 02 reescribe `results/metrics.csv` con los baselines; los siguientes agregan
sus filas, así que el 02 va siempre antes que ellos. Desde la carpeta `notebooks/` (igual en Windows, Linux y Mac):
```
jupyter nbconvert --to notebook --execute --inplace 00_preprocesamiento.ipynb 01_eda.ipynb 02_modelos.ipynb 03_random_forest.ipynb 04_gradient_boosting.ipynb 05_binomial_negativa.ipynb 06_mlp.ipynb 07_ablacion.ipynb 08_comparacion.ipynb
```
El 06 va después del 03 y el 04 porque su ensamble usa el Random Forest y el gradient boosting, y el 08 va al final
porque lee `results/metrics.csv` completo. En total tarda alrededor de 1 h 20 min (probado en un clon limpio con Python 3.14, scikit-learn 1.9 y pandas 3.1;
las cifras coinciden con `results/metrics.csv` en ± 0.002, salvo el macro-F1 del test de GRAVEDAD de los modelos
con muchos árboles, que puede variar en el tercer decimal entre máquinas).

Las descargas externas están en `data/raw/` como cachés: red vial con atributos, clima por celda (diario y por hora)
y clima por capital. Si se borran, se vuelven a descargar: el clima por hora tarda ~1.5 h por los límites de la API
gratuita de Open-Meteo y retoma donde quedó si se interrumpe.

### Problemas comunes
| Mensaje | Causa | Solución |
|---|---|---|
| `"source" no se reconoce…` | Es un comando de Linux/Mac | En Windows usa `.venv\Scripts\activate` |
| `"..\" no se reconoce…` o `.venv/bin` no existe | Ruta de Linux/Mac | En Windows la carpeta es `.venv\Scripts` |
| `"uv" no se reconoce…` | `uv` no viene con Python | Usa `python -m venv`, como arriba |
| `Activate.ps1 no se puede cargar…` | Política de PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar |
| `ModuleNotFoundError: No module named 'sklearn'` (u otro) | El entorno no está activado, o el kernel de Jupyter no es el de `.venv` | Activa el entorno o elige el kernel correcto |
| `ModuleNotFoundError: No module named 'src'` | Jupyter se abrió fuera de la carpeta del repo | Abre Jupyter desde la raíz del repo |

---

## 7. Estructura del repositorio
```
README.md              descripción del proyecto, resultados y cómo ejecutar
AGENTS.md              guía para quien trabaje en el repo (personas o asistentes de IA): convenciones y cómo validar
ENTREGABLE_PARCIAL.md  checklist del entregable de la semana 8
ENTREGABLE_FINAL.md    checklist del entregable de la semana 15
report.pdf             informe académico (plantilla del curso); report.docx es su fuente editable
presentation.pdf       presentación final (≈ 10 min + preguntas)
Plantilla_reporte_TA.docx  plantilla del informe entregada por el curso
requirements.txt       dependencias
data/raw/              accidentes SUTRAN + diccionario, red vial del MTC (con atributos) y cachés de clima
data/processed/        accidentes.csv (Tarea 2, 8 105 × 33) y panel_depto_dia.csv (Tarea 1, 15 336 × 25)
notebooks/             00_preprocesamiento, 01_eda, 02_modelos, 03_random_forest, 04_gradient_boosting, 05_binomial_negativa, 06_mlp, 07_ablacion, 08_comparacion
results/               metrics.csv (todas las métricas), model_comparison.txt (análisis por modelo y conclusiones), plots/
src/                   preprocessing.py (datos y variables), models.py (grupos de variables, encoding y modelos), evaluation.py (métricas y validación)
papers/                paper_integrante1..4.pdf, resúmenes e índice
presentación/          template del curso; final/ = fuente (HTML) y guion de la presentación final; avances/ = planteamiento anterior (archivado)
```
