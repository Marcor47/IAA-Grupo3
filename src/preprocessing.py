"""Preparación de datos: lectura y limpieza de accidentes (SUTRAN), descarga de
precipitación (Open-Meteo) y unión de ambos datasets por departamento + fecha.

Uso desde la raíz del repo:
    python -m src.preprocessing      # genera data/processed/accidentes_clima.csv

Uso desde un notebook:
    from src import preprocessing as pp
    raw = pp.leer_accidentes()
    df, bitacora = pp.limpiar_accidentes(raw)
    clima = pp.cargar_clima(df)
    data = pp.unir_clima(df, clima)
"""
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
F_ACC = RAW / "accidentes_transito_carreteras.csv"
F_CLIMA = RAW / "clima_open_meteo.csv"
F_DATASET = PROC / "accidentes_clima.csv"

URL_ACCIDENTES = ("https://www.datosabiertos.gob.pe/sites/default/files/"
                  "Accidentes%20de%20tr%C3%A1nsito%20en%20carreteras-2020-2021-Sutran.csv")
API_CLIMA = "https://archive-api.open-meteo.com/v1/archive"   # API histórica gratuita, sin API key
VARIABLES_CLIMA = "precipitation_sum,rain_sum,precipitation_hours"

NI = "N.I."   # "no identificado" en el dataset de SUTRAN

# Coordenadas APROXIMADAS de la capital de cada departamento (lat, lon).
# El dataset no trae coordenadas, así que este es el punto de referencia de lluvia por departamento.
COORD = {
    "AMAZONAS": (-6.23, -77.87), "ANCASH": (-9.53, -77.53), "APURIMAC": (-13.64, -72.88),
    "AREQUIPA": (-16.40, -71.54), "AYACUCHO": (-13.16, -74.22), "CAJAMARCA": (-7.16, -78.51),
    "CALLAO": (-12.05, -77.12), "CUSCO": (-13.53, -71.97), "HUANCAVELICA": (-12.79, -74.98),
    "HUANUCO": (-9.93, -76.24), "ICA": (-14.07, -75.73), "JUNIN": (-12.07, -75.21),
    "LA LIBERTAD": (-8.11, -79.03), "LAMBAYEQUE": (-6.77, -79.84), "LIMA": (-12.05, -77.04),
    "LORETO": (-3.75, -73.25), "MADRE DE DIOS": (-12.59, -69.19), "MOQUEGUA": (-17.19, -70.93),
    "PASCO": (-10.68, -76.26), "PIURA": (-5.19, -80.63), "PUNO": (-15.84, -70.02),
    "SAN MARTIN": (-6.03, -76.97), "TACNA": (-18.01, -70.25), "TUMBES": (-3.57, -80.45),
    "UCAYALI": (-8.38, -74.55),
}

# Variantes de nombre de departamento que se unifican antes de cruzar con el clima
ALIAS_DEPARTAMENTO = {"LIMA METROPOLITANA": "LIMA", "LIMA PROVINCIAS": "LIMA", "PROV. CONST. DEL CALLAO": "CALLAO",
                      "PROVINCIA CONSTITUCIONAL DEL CALLAO": "CALLAO"}


# ---------------------------------------------------------------------------
# Accidentes (SUTRAN)
# ---------------------------------------------------------------------------
def quitar_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")


def descargar_accidentes(path=F_ACC):
    """Descarga el CSV de SUTRAN solo si no existe en data/raw/."""
    if path.exists():
        return path
    r = requests.get(URL_ACCIDENTES, timeout=120)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return path


def leer_csv(path):
    # El CSV de SUTRAN usa ';' como separador y codificación ISO-8859-1 (latin-1).
    for enc in ("utf-8-sig", "latin-1"):
        try:
            out = pd.read_csv(path, sep=";", encoding=enc, dtype=str)
            if out.shape[1] == 1:      # por si algún día cambian el separador
                out = pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
            return out
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el CSV con utf-8 ni latin-1")


def leer_accidentes(path=F_ACC):
    """Lee el CSV crudo (todo como texto) y alinea los nombres de columna con el diccionario de datos."""
    raw = leer_csv(descargar_accidentes(path))
    raw.columns = [quitar_tildes(c).strip().upper().replace(" ", "_") for c in raw.columns]
    # El archivo real difiere del diccionario de datos:
    #   CODIGO_VM-MA / CODIGO_VIA (encabezado con la tilde dañada) -> CODIGO_VIA
    #   FALLECIDOS / HERIDOS                                        -> NUM_FALLECIDOS / NUM_HERIDOS
    renombrar = {c: "CODIGO_VIA" for c in raw.columns if c.startswith("CODIGO_V")}
    renombrar.update({"FALLECIDOS": "NUM_FALLECIDOS", "HERIDOS": "NUM_HERIDOS"})
    raw = raw.rename(columns=renombrar)
    esperadas = {"FECHA", "HORA", "DEPARTAMENTO", "CODIGO_VIA", "KILOMETRO", "MODALIDAD", "NUM_FALLECIDOS",
                 "NUM_HERIDOS"}
    faltan = esperadas - set(raw.columns)
    assert not faltan, f"Faltan columnas en el CSV: {faltan}. Columnas encontradas: {list(raw.columns)}"
    return raw


def resumen_faltantes(raw):
    """Faltantes por columna en el CSV crudo. Los faltantes vienen como texto 'N.I.', no como celdas vacías,
    por eso `isna()` sobre el crudo da 0 % en todas las columnas."""
    v = raw.apply(lambda s: s.astype(str).str.strip().str.upper())
    out = pd.DataFrame({"vacios": raw.isna().sum() + (v == "").sum(), "N.I.": (v == NI).sum()})
    out["% faltante"] = (100 * (out["vacios"] + out["N.I."]) / len(raw)).round(2)
    return out


def parse_fecha(s):
    s = s.astype(str).str.strip().str.split(" ").str[0]
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
        m = out.isna()
        if not m.any():
            break
        out[m] = pd.to_datetime(s[m], format=fmt, errors="coerce")
    return out


def parse_hora(s):
    # Devuelve la hora entera (0-23). Acepta 'HH:MM:SS', 'HH:MM' o numérico HHMM / HHMMSS; 'N.I.' -> NaN.
    s = s.astype(str).str.strip()
    t = pd.to_datetime(s, format="%H:%M:%S", errors="coerce")
    m = t.isna()
    t[m] = pd.to_datetime(s[m], format="%H:%M", errors="coerce")
    h = t.dt.hour.astype("float")
    m = h.isna() & s.str.fullmatch(r"\d{1,6}")
    if m.any():
        num = pd.to_numeric(s[m])
        largo = s[m].str.len()
        h[m] = np.where(largo <= 4, num // 100, num // 10000)
    h[(h < 0) | (h > 23)] = np.nan
    return h


def parse_km(s):
    # Toma el primer número de la cadena (admite coma decimal); 'N.I.' -> NaN
    s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


def limpiar_accidentes(raw):
    """Tipifica, normaliza y filtra los accidentes.

    Devuelve (df, bitacora): el DataFrame limpio y una tabla con los registros que se descartan en cada paso.
    Decisiones (justificadas en notebooks/01_eda.ipynb):
      - Se descartan los accidentes sin dato de fallecidos (es la variable objetivo).
      - Se descartan los accidentes sin departamento (sin departamento no se les puede asignar lluvia).
      - Se descartan los duplicados exactos (todas las columnas iguales).
      - Se conservan los accidentes sin hora, vía, kilómetro o modalidad: la lluvia es diaria y se une
        por departamento + fecha, así que esos campos no impiden el cruce.
    """
    bitacora = []

    def registrar(paso, antes, despues, decision):
        bitacora.append({"paso": paso, "antes": antes, "descartados": antes - despues, "despues": despues,
                         "decision": decision})

    df = raw.copy()
    df["FECHA"] = parse_fecha(df["FECHA"])
    df["HORA_INT"] = parse_hora(df["HORA"])
    df["KILOMETRO"] = parse_km(df["KILOMETRO"])
    for c in ("NUM_FALLECIDOS", "NUM_HERIDOS"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)
    df = df.dropna(subset=["FECHA", "NUM_FALLECIDOS"]).copy()
    registrar("Fecha inválida o sin dato de fallecidos", n, len(df), "Descartar: sin variable objetivo")

    # HERIDOS es consecuencia del accidente y no entra a los modelos; sus 'N.I.' se dejan en 0 solo para
    # poder sumar heridos en el EDA.
    df["NUM_HERIDOS"] = df["NUM_HERIDOS"].fillna(0)
    df["NUM_FALLECIDOS"] = df["NUM_FALLECIDOS"].astype(int)

    # Texto normalizado: sin tildes, en mayúsculas (el crudo mezcla 'AREQUIPA' y 'Arequipa')
    df["DEPARTAMENTO"] = df["DEPARTAMENTO"].map(lambda x: quitar_tildes(x).strip().upper()).replace(ALIAS_DEPARTAMENTO)
    df["MODALIDAD"] = df["MODALIDAD"].fillna(NI).map(lambda x: quitar_tildes(x).strip().upper())
    df["CODIGO_VIA"] = df["CODIGO_VIA"].fillna(NI).astype(str).str.strip().str.upper()

    # Variables derivadas
    df["ANIO"] = df["FECHA"].dt.year
    df["MES"] = df["FECHA"].dt.month
    df["DIA_SEMANA"] = df["FECHA"].dt.dayofweek          # 0 = lunes
    df["ES_FATAL"] = (df["NUM_FALLECIDOS"] > 0).astype(int)

    n = len(df)
    df = df[df["DEPARTAMENTO"] != NI]
    registrar("Departamento 'N.I.'", n, len(df), "Descartar: no se puede asignar lluvia")

    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    registrar("Duplicados exactos", n, len(df), "Descartar: mismo registro repetido")

    return df, pd.DataFrame(bitacora)


# ---------------------------------------------------------------------------
# Clima (Open-Meteo)
# ---------------------------------------------------------------------------
def bajar_clima(nombre, lat, lon, ini, fin, intentos=5):
    """Precipitación diaria de un punto. timezone=America/Lima para que el 'día' coincida con FECHA del accidente."""
    params = dict(latitude=lat, longitude=lon, start_date=ini, end_date=fin,
                  daily=VARIABLES_CLIMA, timezone="America/Lima")
    for k in range(intentos):
        r = requests.get(API_CLIMA, params=params, timeout=60)
        if r.status_code == 200:
            out = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "FECHA"})
            out["DEPARTAMENTO"] = nombre
            return out
        time.sleep(3 * (k + 1))          # espera creciente si la API limita las peticiones
    raise RuntimeError(f"Open-Meteo falló para {nombre}: {r.status_code} {r.text[:200]}")


def cargar_clima(df, path=F_CLIMA):
    """Lee la caché de clima; si no existe, la descarga para los departamentos y el periodo de `df`."""
    if path.exists():
        return pd.read_csv(path, parse_dates=["FECHA"])
    ini, fin = df["FECHA"].min().strftime("%Y-%m-%d"), df["FECHA"].max().strftime("%Y-%m-%d")
    partes = []
    for dep in sorted(set(df["DEPARTAMENTO"]) & set(COORD)):
        lat, lon = COORD[dep]
        partes.append(bajar_clima(dep, lat, lon, ini, fin))
        print("Clima descargado:", dep)
        time.sleep(1)
    clima = pd.concat(partes, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    clima.to_csv(path, index=False)
    clima["FECHA"] = pd.to_datetime(clima["FECHA"])
    return clima


def unir_clima(df, clima):
    """Une cada accidente con la lluvia de su departamento ese día (left join: no se pierde ningún accidente)."""
    return df.merge(clima, on=["DEPARTAMENTO", "FECHA"], how="left")


def construir_dataset(guardar=True):
    """Pipeline completo: crudo -> limpio -> + clima -> data/processed/accidentes_clima.csv."""
    df, bitacora = limpiar_accidentes(leer_accidentes())
    data = unir_clima(df, cargar_clima(df))
    if guardar:
        PROC.mkdir(parents=True, exist_ok=True)
        data.to_csv(F_DATASET, index=False)
    return data, bitacora


if __name__ == "__main__":
    data, bitacora = construir_dataset()
    print(bitacora.to_string(index=False))
    print(f"Accidentes con dato de lluvia: {data.precipitation_sum.notna().mean():.1%}")
    print("Guardado:", F_DATASET, data.shape)
