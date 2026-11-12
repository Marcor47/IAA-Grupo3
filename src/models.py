"""Variables y modelos de las dos tareas del proyecto.

Tarea 1 · Pronóstico de riesgo (panel departamento × día): ¿cuántos accidentes habrá mañana en cada departamento
(N_ACC) y habrá alguno fatal (HAY_FATAL)?
Tarea 2 · Triaje de gravedad (un accidente recién reportado): ¿sin víctimas, con heridos o fatal (GRAVEDAD)?

Uso desde un notebook (con la raíz del repo en sys.path):
    from src import models as M
    panel = M.cargar_panel()
    modelo = M.gbm_conteo(M.columnas(M.GRUPOS_T1))
    acc = M.cargar_accidentes()
    modelo = M.gbm_clasificador(M.columnas(M.GRUPOS_T2))
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
F_DATASET = ROOT / "data" / "processed" / "accidentes.csv"
F_PANEL = ROOT / "data" / "processed" / "panel_depto_dia.csv"
SEED = 42
INICIO_HOLDOUT = "2021-04-01"   # Tarea 1: se entrena hasta marzo de 2021 y se pronostica abril-septiembre

# Cada grupo es (numéricas, categóricas). Los grupos siguen los factores de la bibliografía, para medir con
# ablación cuál aporta (README §4).
GRUPOS_T1 = {
    "clima": (["PRECIP_LOG", "HORAS_LLUVIA", "PRECIP_PREV3_LOG", "PRECIP_PREV7_LOG"], []),
    "historia": (["N_ACC_AYER", "N_ACC_SEMANA_PASADA", "N_ACC_M7", "N_ACC_M28", "N_FATALES_M28", "N_HERIDOS_M28",
                  "P_ATROPELLO_M28", "P_CHOQUE_M28"], []),
    "calendario": (["MES_SIN", "MES_COS", "FIN_DE_SEMANA", "FERIADO", "PUENTE", "CUARENTENA"], []),
    "lugar": (["ALTITUD_MEDIA"], ["DEPARTAMENTO"]),
}
# Decisiones de encoding (12/11/2026): fuera CODIGO_VIA (174 vías; la vía ya entra por los atributos del tramo),
# FRANJA (repite la hora seno/coseno), DIA_SEMANA (ya está FIN_DE_SEMANA), TEMPERATURA, VIENTO, NUBOSIDAD (sin respaldo
# en la bibliografía ni señal en el EDA) y LAT/LON (se superponen con el departamento). Ordinales en ORDINALES.
GRUPOS_T2 = {
    "tipo de choque": ([], ["MODALIDAD"]),
    "vía": (["CARRILES", "ANCHO_CALZADA", "VEL_PROMEDIO_TRAMO", "SINUOSIDAD"],
            ["TOPOGRAFIA", "SUPERFICIE", "ESTADO_VIA", "TIPO_VIA", "UBICACION"]),
    "clima": (["PRECIP_LOG", "PRECIP_3H_LOG", "HORAS_LLUVIA", "NEBLINA"], ["CLIMA_HORA"]),
    "luz y hora": (["HORA_SIN", "HORA_COS", "ES_DE_DIA", "CREPUSCULO"], []),
    "calendario": (["MES_SIN", "MES_COS", "FIN_DE_SEMANA", "FERIADO", "PUENTE", "CUARENTENA"], []),
    "lugar": (["ELEVACION", "DIST_CAPITAL_KM"], ["DEPARTAMENTO", "REGION_NATURAL"]),
}
# Categóricas con orden natural: una sola columna (0, 1, 2…) en vez de one-hot. Lo que no está en la lista
# (SIN TRAMO, SIN INFORMACION, PROYECTADA) queda vacío y se imputa con la mediana; UBICACION ya marca "sin tramo".
ORDINALES = {
    "TOPOGRAFIA": ["PLANO", "ONDULADO", "MONTANOSO"],
    "ESTADO_VIA": ["MUY BUENA", "BUENA", "REGULAR", "MALA"],
    "SUPERFICIE": ["ASFALTADO", "ASFALTADO ECONOMICO", "AFIRMADO", "SIN AFIRMAR", "TROCHA"],
}
CLASES_T2 = ["sin víctimas", "heridos", "fatal"]


def columnas(grupos, sin=()):
    """(numéricas, categóricas) de todos los grupos menos los de `sin` (para la ablación)."""
    num = [c for g, (n, _) in grupos.items() if g not in sin for c in n]
    cat = [c for g, (_, k) in grupos.items() if g not in sin for c in k]
    return num, cat


def _ciclicas(df, col, periodo, prefijo):
    df[f"{prefijo}_SIN"] = np.sin(2 * np.pi * df[col] / periodo)
    df[f"{prefijo}_COS"] = np.cos(2 * np.pi * df[col] / periodo)


def cargar_panel(path=F_PANEL):
    """Panel de la Tarea 1 (lo genera src/preprocessing.py). Se descartan los primeros 7 días de cada
    departamento, que no tienen historia de una semana."""
    p = pd.read_csv(path, parse_dates=["FECHA"])
    p = p[p["FECHA"] >= p["FECHA"].min() + pd.Timedelta(days=7)].reset_index(drop=True)
    for c in ("PRECIP", "PRECIP_PREV3", "PRECIP_PREV7"):
        p[f"{c}_LOG"] = np.log1p(p[c])   # la lluvia está muy sesgada
    return p


def cargar_accidentes(path=F_DATASET):
    """Dataset limpio de la Tarea 2 (lo genera src/preprocessing.py) con las transformaciones de los modelos:
    lluvia en escala log (está muy sesgada) y hora y mes cíclicos (23 h queda junto a 0 h)."""
    df = pd.read_csv(path, parse_dates=["FECHA"])
    df["PRECIP_LOG"] = np.log1p(df["PRECIP_DIA"])
    df["PRECIP_3H_LOG"] = np.log1p(df["PRECIP_3H"])
    _ciclicas(df, "HORA", 24, "HORA")
    _ciclicas(df, "MES", 12, "MES")
    cat = [c for _, k in GRUPOS_T2.values() for c in k]
    df[cat] = df[cat].fillna("SIN DATO").astype(str)
    return df


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
def _separar(cat_cols):
    """Categóricas nominales (one-hot / nativas en el árbol) y ordinales (una columna numérica)."""
    return [c for c in cat_cols if c not in ORDINALES], [c for c in cat_cols if c in ORDINALES]


def _ordinal(cols):
    return OrdinalEncoder(categories=[ORDINALES[c] for c in cols], handle_unknown="use_encoded_value", unknown_value=np.nan)


def preprocesador(num_cols, cat_cols):
    """Para modelos lineales y Random Forest:
    - numéricas: mediana + estandarización;
    - ordinales (ORDINALES): 0, 1, 2… + mediana + estandarización;
    - nominales: one-hot; las binarias en una sola columna y las categorías con < 30 casos como 'infrecuente'."""
    nom, ords = _separar(cat_cols)
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    ordp = Pipeline([("ord", _ordinal(ords)), ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, drop="if_binary", sparse_output=False)
    return ColumnTransformer([("num", num, num_cols), ("ord", ordp, ords), ("cat", cat, nom)])


def preprocesador_arboles(num_cols, cat_cols):
    """Para HistGradientBoosting: nominales a enteros (categóricas nativas del modelo, van primero), ordinales como
    número y numéricas sin tocar (el modelo maneja los NaN)."""
    nom, ords = _separar(cat_cols)
    cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan,
                         min_frequency=30)
    return ColumnTransformer([("cat", cat, nom), ("ord", _ordinal(ords), ords), ("num", "passthrough", num_cols)])


def n_columnas(cols, datos):
    """Cuántas columnas recibe el modelo lineal tras el encoding (para reportarlo)."""
    return preprocesador(*cols).fit(datos[cols[0] + cols[1]]).transform(datos[cols[0] + cols[1]]).shape[1]


# ---------------------------------------------------------------------------
# Tarea 1 · conteo (N_ACC) y clasificación binaria (HAY_FATAL)
# ---------------------------------------------------------------------------
def poisson(cols):
    """Modelo lineal de conteo: regresión de Poisson (Mhetre y Thube, 2023)."""
    return Pipeline([("pre", preprocesador(*cols)), ("reg", PoissonRegressor(alpha=1e-3, max_iter=2000))])


def gbm_conteo(cols, **kw):
    """Gradient boosting con pérdida de Poisson (boosting: Babaoglu y Babaoglu, 2021)."""
    params = dict(loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=False, random_state=SEED) | kw
    return Pipeline([("pre", preprocesador_arboles(*cols)),
                     ("reg", HistGradientBoostingRegressor(categorical_features=list(range(len(_separar(cols[1])[0]))), **params))])


def rf_conteo(cols, **kw):
    params = dict(n_estimators=400, criterion="poisson", min_samples_leaf=20, max_features=0.5, n_jobs=-1,
                  random_state=SEED) | kw
    return Pipeline([("pre", preprocesador(*cols)), ("reg", RandomForestRegressor(**params))])


# ---------------------------------------------------------------------------
# Clasificadores (Tarea 1 HAY_FATAL y Tarea 2 GRAVEDAD)
# ---------------------------------------------------------------------------
def logit(cols, **kw):
    """Regresión logística (binaria o multinomial) con pesos de clase balanceados."""
    params = dict(max_iter=3000, class_weight="balanced") | kw
    return Pipeline([("pre", preprocesador(*cols)), ("clf", LogisticRegression(**params))])


def gbm_clasificador(cols, **kw):
    """Gradient boosting con categóricas nativas y pesos de clase balanceados."""
    params = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                  class_weight="balanced", early_stopping=False, random_state=SEED) | kw
    return Pipeline([("pre", preprocesador_arboles(*cols)),
                     ("clf", HistGradientBoostingClassifier(categorical_features=list(range(len(_separar(cols[1])[0]))), **params))])


def random_forest(cols, **kw):
    """Random Forest (Muñoz et al., 2024)."""
    params = dict(n_estimators=500, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample",
                  n_jobs=-1, random_state=SEED) | kw
    return Pipeline([("pre", preprocesador(*cols)), ("clf", RandomForestClassifier(**params))])
