# IAA-Grupo3
Tarea Académica de Inteligencia Artifical Aplicada

# Mortalidad en carreteras del Perú según el nivel de precipitación

**Curso:** Inteligencia Artificial Aplicada (1INF62) · PUCP · 2026-2
**Integrantes:** Marco Rodriguez, [integrante 2], [integrante 3], [integrante 4]

## Contenido
1. [Problema y objetivo](#1-problema-y-objetivo)
2. [Dataset](#2-dataset)
3. [Revisión de literatura](#3-revisión-de-literatura)
4. [Metodología y propuesta de modelos](#4-metodología-y-propuesta-de-modelos)
5. [Cómo ejecutar](#5-cómo-ejecutar)
6. [Estructura del repositorio](#6-estructura-del-repositorio)

---

## 1. Problema y objetivo

Los accidentes en carreteras causan muertes y heridos cada año en el Perú. El país combina costa árida, sierra y selva, con regímenes de lluvia muy distintos, y la precipitación suele asociarse a menor fricción del pavimento y menor visibilidad. Aun así, no está claro **cuánto cambia la mortalidad de un accidente cuando llueve**, ni si ese efecto se mantiene al comparar zonas con climas distintos.

**Pregunta:** ¿cuánta mortalidad hay en una carretera del Perú según el nivel de precipitación?

**Objetivo general:** construir y comparar modelos de inteligencia artificial que estimen la gravedad de los accidentes en carreteras peruanas a partir de la precipitación y de variables de contexto (lugar, hora, época del año).

**Objetivos específicos**
- Integrar el registro de accidentes de SUTRAN con la precipitación diaria de una API climática histórica.
- Caracterizar mediante EDA la relación entre lluvia y fatalidad, controlando por departamento.
- Implementar un baseline y, para el entregable final, modelos propuestos por la literatura.
- Comparar los modelos con métricas adecuadas para datos desbalanceados y discutir limitaciones.

**Formulación.** El dataset contiene solo accidentes ya ocurridos, así que el modelo no estima cuántos accidentes habrá, sino qué tan graves son una vez ocurridos:

| Tarea | Pregunta | Variable objetivo | Baseline |
|---|---|---|---|
| A · Clasificación | Dado un accidente con cierta lluvia, lugar y hora, ¿es fatal? | `ES_FATAL` (`NUM_FALLECIDOS ≥ 1`) | Regresión logística |
| B · Regresión de conteo | ¿Cuántos fallecidos se esperan en ese accidente? | `NUM_FALLECIDOS` | Regresión de Poisson |

---

## 2. Dataset

### 2.1 Origen y permisos
- **Nombre:** Accidentes de Tránsito en Carreteras (2020-2021).
- **Fuente:** Superintendencia de Transporte Terrestre de Personas, Carga y Mercancías (**SUTRAN**), publicado en la Plataforma Nacional de Datos Abiertos del Estado Peruano: <https://www.datosabiertos.gob.pe/dataset/accidentes-de-tr%C3%A1nsito-en-carreteras>
- **Contenido:** accidentes en vías nacionales y departamentales reportados por la Policía Nacional del Perú (PNP) y el Centro de Gestión y Monitoreo (CGM) de SUTRAN. Cada fila es un accidente.
- **Corte de la información:** 22/12/2021 (`FECHA_CORTE = 20211222`); el portal indica última modificación el 28/01/2022.
- **Licencia:** Open Data Commons Attribution (ODC-By). Es un dato público sin datos personales; se puede usar citando la fuente. No se requiere consentimiento adicional.
- **Tamaño:** [completar con el número de filas tras ejecutar `01_eda.ipynb`] registros, 9 columnas.
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

### 2.3 Dataset complementario: precipitación
El registro de accidentes no tiene clima ni coordenadas. La lluvia se obtiene de la **API histórica de Open-Meteo** (<https://open-meteo.com/en/docs/historical-weather-api>), que entrega precipitación diaria por coordenadas, sin API key. Para cada departamento se descarga la precipitación diaria acumulada (`precipitation_sum`, en mm) en la capital del departamento y se une con los accidentes por **departamento + fecha**. Open-Meteo es gratuito para uso no comercial y exige atribución (revisar los términos vigentes).

### 2.4 Limitaciones del dataset
- **Resolución espacial de la lluvia:** el dataset no trae coordenadas, solo departamento, vía y kilómetro. La lluvia se mide en la capital del departamento, lo que es un proxy grueso en departamentos extensos (Loreto, Cusco, Puno, Junín).
- **Origen de la lluvia:** los valores de la API provienen de datos de reanálisis, no de estaciones; pueden diferir de las mediciones de SENAMHI, sobre todo en la zona andina.
- **Solo accidentes:** no hay información de tráfico ni de días sin accidentes, por lo que no se puede estimar la probabilidad de que ocurra un accidente, solo su gravedad.
- **Periodo 2020-2021:** coincide con la pandemia (restricciones de movilidad).
- **Datos faltantes y posible subregistro** (valores "N.I.").

---

## 3. Revisión de literatura

Cada integrante revisó un artículo revisado por pares sobre predicción de fatalidad o gravedad de accidentes viales, que no usa el mismo dataset que nosotros. Los PDF están en [`papers/`](papers/). Los resúmenes se basan en el texto o resumen de cada paper; antes de la entrega, cada integrante debe contrastar el suyo con el PDF.

### Paper 1 · Marco
**K. V. Mhetre y A. D. Thube**, "Count Data Modeling for Predicting Crash Severity on Indian Highways", *Engineering, Technology & Applied Science Research*, vol. 13, n.º 5, pp. 11816-11820, 2023. DOI: [10.48084/etasr.6172](https://doi.org/10.48084/etasr.6172)

- **Problema:** predecir la fatalidad de accidentes en una carretera nacional rural de la India (NH-48, Maharashtra, ~265 km) según la naturaleza del choque, el horario (AM/PM) y el clima.
- **Datos:** registros de la National Highways Authority of India, 2016-2020. División 70 % entrenamiento / 30 % validación.
- **Técnica:** regresión **binomial negativa** (extiende Poisson a conteos con sobredispersión). Cuatro modelos: naturaleza del choque, horario, condición climática (fino, nublado, llovizna, lluvia fuerte, neblina) y choque + horario. Validación con log-verosimilitud, AIC, BIC, MAD, MSE, RMSE y MAPE.
- **Resultados:** la naturaleza del choque y el clima resultaron significativos; lluvia ligera y fuerte, neblina, tiempo fino y nublado aparecen asociados positivamente con la fatalidad. Los modelos 2 y 4 ajustaron mejor.
- **Aporte a nuestro proyecto:** respalda modelar `NUM_FALLECIDOS` como **conteo** (usamos Poisson como baseline y proponemos binomial negativa para el final) y confirma el clima como variable relevante. **Crítica:** usa el clima como categorías y no aísla un efecto propio de la lluvia; nosotros usamos precipitación continua en mm y evaluamos en datos de prueba.

### Paper 2 · [integrante 2]
**A. J. Ghandour, H. Hammoud y S. Al-Hajj**, "Analyzing Factors Associated with Fatal Road Crashes: A Machine Learning Approach", *International Journal of Environmental Research and Public Health*, vol. 17, n.º 11, art. 4111, 2020. DOI: [10.3390/ijerph17114111](https://doi.org/10.3390/ijerph17114111)

- **Problema:** identificar y ordenar los factores asociados a que un accidente sea fatal en Líbano.
- **Datos:** Lebanese Road Accidents Platform (LRAP), 8 482 accidentes (feb. 2015 - feb. 2019); solo ~5 % son fatales (relación 1:19). Nueve variables de entrada (mes, día, día de la semana, hora, AM/PM, tipo de choque, severidad de lesión, tipo de vía y un *cluster* espacial creado con K-means) y salida fatal / no fatal.
- **Técnicas:**
  - *Desbalance:* SMOTE (sobremuestreo sintético) combinado con submuestreo de la clase mayoritaria, solo sobre el entrenamiento.
  - *Modelos individuales:* SMO (variante de SVM), Random Forest, red neuronal (ANN), regresión logística y Naive Bayes.
  - *Modelo final:* ensamble híbrido que vota SMO con *bagging* de 100 árboles J48.
  - *Validación:* 20 % de prueba y 10-fold cross-validation. Métricas: F1, AUC-PR y Kappa de Cohen (el artículo explica que accuracy y AUC-ROC engañan con clases desbalanceadas).
  - *Importancia de variables:* ranking por chi-cuadrado y análisis de sensibilidad.
- **Resultados:** el ensamble fue el mejor; en prueba obtuvo F1 = 0.435, AUC-PR = 0.368 y Kappa = 0.407 (desempeño "moderado"). Siete de las nueve variables se asociaron con la fatalidad: tipo de choque (atropello y camión-moto, los más letales), severidad, cluster espacial y hora (madrugada).
- **Aporte a nuestro proyecto:** métricas para clases desbalanceadas (F1, AUC-PR, Kappa), SMOTE, comparación de modelos de clase y ranking de variables con chi-cuadrado. Usa una idea de agrupación espacial que podemos imitar por departamento o vía. **Limitación:** no incluye clima.

### Paper 3 · [integrante 3]
**L. Babaoglu y C. Babaoglu**, "Prediction of Fatalities in Vehicle Collisions in Canada", *Promet – Traffic&Transportation*, vol. 33, n.º 5, 2021. DOI: [10.7307/ptt.v33i5.3782](https://doi.org/10.7307/ptt.v33i5.3782)

- **Problema:** predecir si un accidente tendrá fallecidos y descubrir las causas de la fatalidad en las carreteras de Canadá.
- **Datos:** National Collision Database de Canadá, 1999-2017.
- **Técnicas:** análisis exploratorio con minería de datos y **reglas de asociación** para hallar factores clave; luego dos clasificadores supervisados, **regresión Lasso** (regresión con penalización L1 que selecciona variables) y **XGBoost** (árboles con *boosting*). Se interpretan las variables más importantes del modelo ganador.
- **Resultados:** XGBoost fue el mejor clasificador con 83 % de exactitud. Las variables más importantes fueron la configuración de la colisión y el uso de dispositivos de seguridad, por encima de año del vehículo, hora, edad o sexo. Los choques frontales fuera de intersecciones y sin control de tránsito resultaron los más letales, y **la mayoría de las muertes ocurre con clima y condiciones de vía no extremos**.
- **Aporte a nuestro proyecto:** ejemplo de pipeline EDA + Lasso + XGBoost y de interpretación por importancia de variables. Su hallazgo sobre clima no extremo es un contrapunto a nuestra hipótesis: conviene controlar por exposición y por departamento antes de atribuir efectos a la lluvia. **Limitación:** la exactitud sola no es suficiente con clases desbalanceadas.

### Paper 4 · [integrante 4]
**M. Emu et al.**, "Fatality Prediction for Motor Vehicle Collisions: Mining Big Data Using Deep Learning and Ensemble Methods", *IEEE Open Journal of Intelligent Transportation Systems*, 2022. DOI: [10.1109/OJITS.2022.3160404](https://doi.org/10.1109/OJITS.2022.3160404)

- **Problema:** predecir si el resultado de una colisión será fatal o no, como base de un posible sistema de alerta en tiempo real.
- **Datos:** base de choques de Canadá con 5.8 millones de registros.
- **Técnicas:** ensambles por **votación mayoritaria y soft voting** para tratar el desbalance de clases, y **redes neuronales convolucionales (CNN)**; además un análisis del contenido de información de cada atributo para identificar los factores que más distinguen choques fatales de no fatales.
- **Resultados:** exactitud cercana al 75 % con CNN. Los factores más influyentes incluyen las características de la vía y **las condiciones climáticas al momento del choque**, tipo de vehículo, hora, clase y posición del usuario, uso de dispositivos de seguridad y estado del control de tránsito.
- **Aporte a nuestro proyecto:** respalda usar redes neuronales (MLP) y ensambles, e incluir el clima como variable de entrada; sugiere medir la importancia de cada variable. **Limitación:** conjunto de datos y contexto muy distintos (Canadá, millones de registros); la exactitud ~75 % sola no refleja el desbalance.

### 3.1 Técnicas encontradas

| Técnica | Dónde aparece | Para qué sirve |
|---|---|---|
| Regresión binomial negativa / Poisson | Mhetre y Thube | Modelar conteo de fallecidos con sobredispersión |
| Regresión logística | Ghandour (modelo individual) | Baseline interpretable de fatal / no fatal |
| Árboles de decisión (J48) y *bagging* | Ghandour | Clasificación robusta; base del ensamble final |
| Random Forest, ANN, Naive Bayes, SMO (SVM) | Ghandour | Modelos de comparación |
| Lasso (regresión con penalización L1) | Babaoglu | Selección de variables y clasificación |
| XGBoost (*boosting*) | Babaoglu | Mejor clasificador del paper (83 % de exactitud) |
| CNN y ensambles por votación | Emu et al. | Clasificación con grandes volúmenes y desbalance |
| SMOTE + submuestreo | Ghandour | Balancear clases solo en entrenamiento |
| Reglas de asociación | Babaoglu | Descubrir factores de riesgo en el EDA |
| Chi-cuadrado, análisis de sensibilidad, importancia de variables | Ghandour, Babaoglu, Emu et al. | Ordenar variables por su relación con la fatalidad |
| K-means para variable espacial | Ghandour | Resumir la ubicación en un *cluster* |
| F1, AUC-PR, Kappa | Ghandour | Métricas para clases desbalanceadas |
| AIC, BIC, MAD, RMSE, MAPE | Mhetre y Thube | Evaluar modelos de conteo |

---

## 4. Metodología y propuesta de modelos

**Metodología**
1. Limpieza y tipificación del CSV de accidentes (`01_eda.ipynb`).
2. Descarga de precipitación diaria por departamento y unión por departamento + fecha.
3. EDA: lluvia vs. fatalidad, controlando por departamento.
4. Baseline con partición estratificada 80/20, validación cruzada de 5 particiones y partición temporal de robustez (`02_modelos.ipynb`).
5. Dos conjuntos de variables: **Set A** (lluvia, hora, mes, día, departamento, vía, kilómetro; todo conocido antes del accidente) y **Set B** (A + `MODALIDAD`, que solo se conoce después).
6. Métricas: ROC-AUC, PR-AUC, F1, recall y balanced accuracy (clasificación); MAE, RMSE y devianza de Poisson (conteo).

**Modelos**

| Modelo | Tarea | Respaldo | Estado |
|---|---|---|---|
| Regresión logística (`class_weight` balanceado) | A | Curso; Ghandour (2020) | **Implementado (baseline)** |
| Regresión de Poisson | B | Mhetre y Thube (2023) | **Implementado (baseline)** |
| Árbol de decisión | A y B | Curso; Ghandour (2020) | Propuesto |
| KNN | A y B | Curso | Propuesto |
| MLP | A y B | Curso; Emu et al. (2022) | Propuesto |
| Regresión binomial negativa | B | Mhetre y Thube (2023) | Propuesto |
| SMOTE / `class_weight` | A | Ghandour (2020) | Parcial (`class_weight`) |
| Regresión logística con penalización L1 (Lasso) | A | Babaoglu y Babaoglu (2021) | Propuesto |
| Ranking de variables por chi-cuadrado | A | Ghandour (2020) | Propuesto |

*Nota:* los papers usan también Random Forest y XGBoost; se evaluará con el profesor si se permiten como extensión de árboles.

---

## 5. Cómo ejecutar

```bash
python -m venv .venv && source .venv/bin/activate    # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Ejecutar en orden desde la carpeta `notebooks/`:
1. `01_eda.ipynb`: lee `data/raw/accidentes_transito_carreteras.csv` (si no existe, lo descarga), descarga el clima, genera `data/processed/accidentes_clima.csv` y las figuras.
2. `02_modelos.ipynb`: baseline y `results/metrics.csv`.

Requiere internet en la primera ejecución (Open-Meteo). El clima queda en caché en `data/raw/clima_open_meteo.csv`.

---

## 6. Estructura del repositorio
```
data/raw/        CSV de accidentes, diccionario de datos y caché de clima
data/processed/  accidentes_clima.csv
notebooks/       01_eda.ipynb, 02_modelos.ipynb
results/         metrics.csv, plots/
papers/          PDF de los 4 papers revisados
```
Estado de la entrega: ver [`ENTREGABLE_PARCIAL.md`](ENTREGABLE_PARCIAL.md).
