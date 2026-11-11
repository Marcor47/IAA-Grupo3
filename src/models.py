"""Variables y modelos del proyecto.

Uso desde un notebook (con la raíz del repo en sys.path):
    from src import models as M
    df = M.cargar_datos()
    modelo = M.modelo_1()                       # Pipeline de scikit-learn sin entrenar
    modelo.fit(df[M.COLS_MODELO_1], df[M.Y])
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
F_DATASET = ROOT / "data" / "processed" / "accidentes_clima.csv"
SEED = 42
Y = "ES_FATAL"

# Set A (baseline, ex-ante, 9 variables): lluvia, altitud, hora, mes, día, departamento y vía.
# KILOMETRO no entra como número: solo tiene sentido dentro de cada vía (01_eda.ipynb, sección 1.3);
# se usa para ubicar el accidente (src/preprocessing.py) y CODIGO_VIA ya está en CAT_A.
NUM_A = ["PRECIP_LOG", "ELEVACION", "HORA_SIN", "HORA_COS", "MES_SIN", "MES_COS"]
CAT_A = ["DEPARTAMENTO", "DIA_SEMANA", "CODIGO_VIA"]
COLS_A = NUM_A + CAT_A

# Modelo 1: solo departamento + hora (versión continua de la franja horaria) + altitud
NUM_1 = ["ELEVACION", "HORA_SIN", "HORA_COS"]
CAT_1 = ["DEPARTAMENTO"]
COLS_MODELO_1 = NUM_1 + CAT_1

# Modelo 2 (triaje): Modelo 1 + tipo de accidente. MODALIDAD se conoce cuando se reporta el accidente,
# no antes: el modelo sirve para priorizar la atención, no para prevenir.
CAT_2 = CAT_1 + ["MODALIDAD"]
COLS_MODELO_2 = NUM_1 + CAT_2


def cargar_datos(path=F_DATASET):
    """Lee el dataset procesado (lo genera src/preprocessing.py) y construye las variables de los modelos."""
    df = pd.read_csv(path, parse_dates=["FECHA"])
    df = df.dropna(subset=["precipitation_sum"]).reset_index(drop=True)
    df["PRECIP_LOG"] = np.log1p(df["precipitation_sum"])             # la lluvia está muy sesgada
    df["HORA_SIN"] = np.sin(2 * np.pi * df["HORA_INT"] / 24)         # hora cíclica: 23 h queda junto a 0 h
    df["HORA_COS"] = np.cos(2 * np.pi * df["HORA_INT"] / 24)
    df["MES_SIN"] = np.sin(2 * np.pi * df["MES"] / 12)
    df["MES_COS"] = np.cos(2 * np.pi * df["MES"] / 12)
    df["DIA_SEMANA"] = df["DIA_SEMANA"].astype(str)
    # Versiones por categorías (solo para describir grupos; los modelos usan las continuas)
    df["FRANJA"] = pd.cut(df["HORA_INT"], [-1, 5, 11, 17, 24],
                          labels=["madrugada", "mañana", "tarde", "noche"]).astype(object).fillna("sin hora")
    df["ALTITUD"] = pd.cut(df["ELEVACION"], [-1, 500, 2500, 9000], labels=["costa (<500 m)", "media", "alta (>2500 m)"]).astype(str)
    return df


def preprocesador(num_cols, cat_cols):
    """Numéricas: mediana + estandarización. Categóricas: one-hot; categorías con < 30 casos van a 'infrecuente'."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False)
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def logit(num_cols, cat_cols):
    """Regresión logística con pesos de clase balanceados (≈ 12 % de accidentes fatales)."""
    return Pipeline([("pre", preprocesador(num_cols, cat_cols)),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])


def baseline():
    """Baseline del entregable parcial: regresión logística con el Set A."""
    return logit(NUM_A, CAT_A)


def modelo_1(con_lluvia=False):
    """Modelo 1: regresión logística con departamento + hora + altitud (3 variables).

    con_lluvia=True agrega PRECIP_LOG, para medir si la lluvia aporta algo sobre estas tres.
    """
    return logit(NUM_1 + (["PRECIP_LOG"] if con_lluvia else []), CAT_1)


def modelo_2(con_lluvia=False):
    """Modelo 2 (triaje): regresión logística con departamento + hora + altitud + tipo de accidente."""
    return logit(NUM_1 + (["PRECIP_LOG"] if con_lluvia else []), CAT_2)


def poisson():
    """Baseline de la Tarea B (conteo de fallecidos): regresión de Poisson con el Set A."""
    return Pipeline([("pre", preprocesador(NUM_A, CAT_A)), ("reg", PoissonRegressor(alpha=1e-3, max_iter=1000))])


def random_forest(num_cols=NUM_A, cat_cols=CAT_A, **kw):
    """Random Forest: el mejor algoritmo no lineal en la CV de train (02_modelos.ipynb, sección 5)."""
    params = dict(n_estimators=500, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample",
                  n_jobs=-1, random_state=SEED) | kw
    return Pipeline([("pre", preprocesador(num_cols, cat_cols)), ("clf", RandomForestClassifier(**params))])
