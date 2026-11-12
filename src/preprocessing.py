"""Preprocesamiento: del CSV crudo de SUTRAN a los dos datasets limpios del proyecto.

1. Lectura y limpieza de accidentes (SUTRAN).
2. Ubicación de cada accidente con su vía y kilómetro sobre la Red Vial Nacional del MTC, con los atributos de su
   tramo (topografía, superficie, carriles, velocidad, sinuosidad…).
3. Clima en la celda de cada accidente (Open-Meteo): diario y por hora (condición, lluvia de 3 h, luz).
4. Variables derivadas (calendario, distancia a la capital, tipo de vía).
5. Salidas:
   - data/processed/accidentes.csv      Tarea 2 · una fila por accidente, solo las COLUMNAS_FINALES (33)
   - data/processed/panel_depto_dia.csv Tarea 1 · departamento x día, con los días sin accidentes

Uso desde la raíz del repo:
    python -m src.preprocessing

El paso a paso, con cada decisión justificada, está en notebooks/00_preprocesamiento.ipynb.
"""
import json
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
F_DATASET = PROC / "accidentes.csv"   # dataset limpio de la Tarea 2 (una fila por accidente)

URL_ACCIDENTES = ("https://www.datosabiertos.gob.pe/sites/default/files/"
                  "Accidentes%20de%20tr%C3%A1nsito%20en%20carreteras-2020-2021-Sutran.csv")
API_CLIMA = "https://archive-api.open-meteo.com/v1/archive"   # API histórica gratuita, sin API key
URL_VIAS = "https://geocatmin.ingemmet.gob.pe/arcgis/rest/services/SERV_OTRAS_FUENTES/MapServer/23/query"
F_VIAS = RAW / "red_vial_nacional_mtc.json"
F_CLIMA_PUNTOS = RAW / "clima_celdas_open_meteo.csv"
TOLERANCIA_FRONTERA_KM = 10
RES_CELDA = 0.1   # grados (~11 km): resolución de ERA5-Land; IFS HRES es de ~9 km
VARIABLES_CLIMA = "precipitation_sum,rain_sum,precipitation_hours"

NI = "N.I."   # "no identificado" en el dataset de SUTRAN

# Atributos de la Red Vial Nacional (MTC) que se copian a cada accidente ubicado por km: campo -> nombre
ATRIBUTOS_VIA = {"CTOPOGRAFI": "topografia", "DSUPERFI_7": "superficie", "CESTADOD_3": "estado",
                 "DNROCARRIL": "carriles", "DANCCALZAD": "ancho", "DVELPROTRA": "velocidad", "CREGION": "region"}
# Dominio WS2_DOM_MTC_SUPERFICIE de la capa (consultado en el servicio el 12/11/2026)
SUPERFICIE = {1: "ASFALTADO", 11: "ASFALTADO ECONOMICO", 2: "AFIRMADO", 3: "SIN AFIRMAR", 4: "TROCHA",
              9: "PROYECTADA", 99: "SIN INFORMACION"}
FERIADOS = pd.to_datetime([  # feriados nacionales del Perú en el periodo de los datos
    "2020-01-01", "2020-04-09", "2020-04-10", "2020-05-01", "2020-06-29", "2020-07-28", "2020-07-29", "2020-08-30",
    "2020-10-08", "2020-11-01", "2020-12-08", "2020-12-25", "2021-01-01", "2021-04-01", "2021-04-02", "2021-05-01",
    "2021-06-29", "2021-07-28", "2021-07-29", "2021-08-30"])
CUARENTENA = ("2020-03-16", "2020-06-30")   # cuarentena nacional obligatoria
SIN_TRAMO = "SIN TRAMO"   # accidentes ubicados en la capital: no se conoce el tramo, no se imputa

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

    # HERIDOS es consecuencia del accidente: no es variable de entrada, pero forma parte de GRAVEDAD. Sus
    # 'N.I.' se dejan en 0 para poder sumar heridos en el EDA.
    sin_heridos = df["NUM_HERIDOS"].isna()
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
    # Objetivo de la Tarea 2 (triaje): 0 sin víctimas, 1 solo heridos, 2 al menos un fallecido. Si falta el dato
    # de heridos y no hubo fallecidos, la clase es desconocida (NaN) y ese accidente no entra a la Tarea 2.
    df["GRAVEDAD"] = np.select([df["NUM_FALLECIDOS"] > 0, df["NUM_HERIDOS"] > 0], [2, 1], 0).astype(float)
    df.loc[sin_heridos & (df["NUM_FALLECIDOS"] == 0), "GRAVEDAD"] = np.nan

    n = len(df)
    df = df[df["DEPARTAMENTO"] != NI]
    registrar("Departamento 'N.I.'", n, len(df), "Descartar: no se puede asignar lluvia")

    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    registrar("Duplicados exactos", n, len(df), "Descartar: mismo registro repetido")

    return df, pd.DataFrame(bitacora)


# ---------------------------------------------------------------------------
# Ubicación: vía + kilómetro -> coordenadas (Red Vial Nacional del MTC)
# ---------------------------------------------------------------------------
def descargar_red_vial(path=F_VIAS):
    """Descarga la capa de la Red Vial Nacional (MTC, al 31/12/2014) solo si no existe en data/raw/.

    Cada tramo trae su ruta (CCODRUTA_4, p. ej. 'PE-22'), su km de inicio y fin, su departamento y los
    atributos de la vía que la bibliografía asocia a la gravedad: topografía, superficie de rodadura y su
    estado, número de carriles, ancho de calzada, velocidad promedio del tramo y región natural.
    Si la caché es de la versión anterior (sin atributos), se vuelve a descargar.
    El servicio no admite paginación, así que se pide por rangos de OBJECTID. Su certificado HTTPS
    tiene la cadena incompleta, por eso verify=False (la capa es pública y queda guardada en el repo).
    """
    if path.exists() and "topografia" in path.read_text(encoding="utf-8")[:3000]:
        return path
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ids = sorted(requests.get(URL_VIAS, params=dict(where="1=1", returnIdsOnly="true", f="json"),
                              verify=False, timeout=120).json()["objectIds"])
    tramos = []
    for i in range(0, len(ids), 500):
        lote = ids[i:i + 500]
        r = requests.get(URL_VIAS, verify=False, timeout=300, params=dict(
            where=f"OBJECTID>={lote[0]} AND OBJECTID<={lote[-1]}",
            outFields="CCODRUTA_4,DKMINICIO,DKMFINAL,CDEPARTAME," + ",".join(ATRIBUTOS_VIA), returnGeometry="true", outSR=4326,
            maxAllowableOffset=0.0002, geometryPrecision=5, f="json"))   # simplifica la línea a ~20 m
        for f in r.json()["features"]:
            a = f["attributes"]
            partes = (f.get("geometry") or {}).get("paths", [])
            tramos.append({"ruta": a["CCODRUTA_4"].strip().upper(), "km_ini": a["DKMINICIO"], "km_fin": a["DKMFINAL"],
                           "departamento": quitar_tildes(a["CDEPARTAME"]).strip().upper(),
                           **{nombre: a[campo] for campo, nombre in ATRIBUTOS_VIA.items()},
                           "coords": [p for parte in partes for p in parte]})   # [lon, lat]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tramos), encoding="utf-8")
    return path


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def _orientar(tramos):
    """Algunos tramos están dibujados al revés (del km final al inicial). Se corrige con los tramos vecinos
    de la misma ruta: el extremo que toca al tramo anterior es el km de inicio; si no hay anterior, el
    extremo que toca al siguiente es el km final. Sin vecinos, se deja como viene."""
    por_ruta = {}
    for t in tramos:
        por_ruta.setdefault(t["ruta"], []).append(t)
    invertidos = 0
    for grupo in por_ruta.values():
        for t in grupo:
            c = t["coords"]
            ant = [u for u in grupo if u is not t and abs(u["km_fin"] - t["km_ini"]) < 0.5]
            sig = [u for u in grupo if u is not t and abs(u["km_ini"] - t["km_fin"]) < 0.5]

            def dist(p, vecinos):
                return min(haversine_km(p[0], p[1], q[0], q[1]) for u in vecinos for q in (u["coords"][0], u["coords"][-1]))

            if ant:
                invertir = dist(c[-1], ant) < dist(c[0], ant)
            elif sig:
                invertir = dist(c[0], sig) < dist(c[-1], sig)
            else:
                invertir = False
            if invertir:
                t["coords"] = c[::-1]
                invertidos += 1
    return invertidos


def cargar_red_vial(path=F_VIAS):
    """Lee la red vial, orienta los tramos y precalcula la distancia acumulada a lo largo de cada línea."""
    tramos = [t for t in json.loads(descargar_red_vial(path).read_text(encoding="utf-8"))
              if len(t["coords"]) >= 2 and t["km_fin"] > t["km_ini"]]
    _orientar(tramos)
    for t in tramos:
        c = np.array(t["coords"])
        t["lon"], t["lat"] = c[:, 0], c[:, 1]
        t["dist"] = np.concatenate([[0], np.cumsum(haversine_km(c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1]))])
    return tramos


def ubicar_accidentes(df, tramos, tolerancia_km=TOLERANCIA_FRONTERA_KM):
    """Asigna coordenadas a cada accidente.

    UBICACION = 'km': la vía y el kilómetro caen en un tramo de la Red Vial Nacional del MTC. El punto se
    interpola a lo largo del tramo en proporción al kilómetro. Se acepta solo si el tramo está en el mismo
    departamento del accidente, o si la ruta entra a ese departamento a <= `tolerancia_km` del kilómetro
    (accidentes cerca de un límite departamental). Si no, el kilómetro de SUTRAN no coincide con el
    kilometraje oficial del MTC (p. ej., hitos antiguos) y el punto no es confiable.
    UBICACION = 'capital': cualquier otro caso (vía departamental, vía o km 'N.I.', ruta o km fuera de la
    capa, o departamento inconsistente). Se usa la capital del departamento, como en la versión anterior.

    Devuelve un DataFrame alineado con `df`: LAT, LON, UBICACION, MOTIVO (por qué cayó en cada caso) y los
    atributos del tramo (COLS_VIA); los accidentes ubicados en la capital llevan SIN_TRAMO / NaN.
    """
    por_ruta = {}
    for t in tramos:
        por_ruta.setdefault(t["ruta"], []).append(t)

    filas = []
    for via, km, dep in zip(df["CODIGO_VIA"], df["KILOMETRO"], df["DEPARTAMENTO"]):
        motivo, punto = None, None
        if not str(via).startswith("PE-"):
            motivo = "vía N.I." if via == NI else "vía departamental o vecinal"
        elif pd.isna(km):
            motivo = "kilómetro N.I."
        elif via not in por_ruta:
            motivo = "ruta no está en la capa del MTC"
        else:
            cand = [t for t in por_ruta[via] if t["km_ini"] <= km <= t["km_fin"]]
            mismo = [t for t in cand if t["departamento"] == dep]
            cerca = any(min(abs(t["km_ini"] - km), abs(t["km_fin"] - km)) <= tolerancia_km
                        for t in por_ruta[via] if t["departamento"] == dep)
            if not cand:
                motivo = "kilómetro fuera de los tramos de la ruta"
            elif mismo or cerca:
                t = (mismo or cand)[0]
                d = (km - t["km_ini"]) / (t["km_fin"] - t["km_ini"]) * t["dist"][-1]
                punto = (np.interp(d, t["dist"], t["lat"]), np.interp(d, t["dist"], t["lon"]))
                motivo = "tramo en el departamento" if mismo else "tramo en el departamento vecino (frontera)"
            else:
                motivo = "tramo en otro departamento (km no coincide con el MTC)"
        if punto is None:
            lat, lon = COORD[dep]
            filas.append((lat, lon, "capital", motivo) + ATRIB_VACIOS)
        else:
            filas.append((punto[0], punto[1], "km", motivo) + atributos_via(t, d))
    return pd.DataFrame(filas, columns=["LAT", "LON", "UBICACION", "MOTIVO_UBICACION"] + COLS_VIA, index=df.index)


COLS_VIA = ["TOPOGRAFIA", "SUPERFICIE", "ESTADO_VIA", "CARRILES", "ANCHO_CALZADA", "VEL_PROMEDIO_TRAMO",
            "REGION_NATURAL", "SINUOSIDAD"]
ATRIB_VACIOS = (SIN_TRAMO, SIN_TRAMO, SIN_TRAMO, np.nan, np.nan, np.nan, SIN_TRAMO, np.nan)


def _texto_via(x):
    x = str(x).strip()
    try:
        x = x.encode("cp1252").decode("utf-8")   # el servicio devuelve "MONTAÃ‘OSO" (UTF-8 leído como cp1252)
    except UnicodeError:
        pass
    return quitar_tildes(x).upper() or "SIN INFORMACION"


def sinuosidad(t, d, ventana_km=1.0):
    """Largo del camino / distancia en línea recta entre los puntos a ±`ventana_km` del accidente sobre el
    tramo (1 = recta; > 1.5 = curvas cerradas). Proxy del alineamiento de la vía (C_RALN en Babaoglu y Emu)."""
    a, b = max(d - ventana_km, 0), min(d + ventana_km, t["dist"][-1])
    if b - a < 0.2:   # tramo demasiado corto para medir
        return np.nan
    p = [(np.interp(x, t["dist"], t["lon"]), np.interp(x, t["dist"], t["lat"])) for x in (a, b)]
    recta = haversine_km(p[0][0], p[0][1], p[1][0], p[1][1])
    # ponytail: tope en 3 para que unos pocos tramos mal dibujados (máx. 12) no dominen el escalado
    return min((b - a) / recta, 3.0) if recta > 0.01 else np.nan


def atributos_via(t, d):
    num = lambda x: x if x is not None and x > 0 else np.nan   # 0 = sin dato en la capa
    return (_texto_via(t["topografia"]), SUPERFICIE.get(t["superficie"], "SIN INFORMACION"), _texto_via(t["estado"]),
            num(t["carriles"]), num(t["ancho"]), num(t["velocidad"]), _texto_via(t["region"]), sinuosidad(t, d))


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


def asignar_celda(lat, lon, res=RES_CELDA):
    """Celda de la grilla (centro redondeado a `res` grados) a la que se pide la lluvia."""
    return np.round(np.round(lat / res) * res, 2), np.round(np.round(lon / res) * res, 2)


def descargar_por_bloques(faltan, ini, fin, path, variables, parsear, por_minuto=450, por_hora=4500, margen_dias=0):
    """Descarga de Open-Meteo los pares (CELDA_LAT, CELDA_LON, BLOQUE) de `faltan`, con bloques de 14 días desde
    `ini` (+ `margen_dias` a cada lado), hasta 100 celdas por petición y respetando los límites gratuitos por
    minuto y por hora. `parsear(clat, clon, json_de_la_celda, bloque)` devuelve las filas que se agregan a la
    caché `path` (CSV en modo append, así una descarga interrumpida se retoma donde quedó)."""
    historial = []   # (instante, llamadas) para respetar los límites por minuto y por hora

    def esperar(n):
        while True:
            ahora = time.time()
            ult_min = sum(k for t, k in historial if ahora - t < 60)
            ult_hora = sum(k for t, k in historial if ahora - t < 3600)
            if ult_min + n <= por_minuto and ult_hora + n <= por_hora:
                return
            time.sleep(5)

    n_bloques = (fin - ini).days // 14 + 1
    for bloque, grupo in faltan.groupby("BLOQUE"):
        b_ini = ini + pd.Timedelta(days=14 * int(bloque) - margen_dias)
        b_fin = min(ini + pd.Timedelta(days=14 * int(bloque) + 13), fin) + pd.Timedelta(days=int(margen_dias))
        for i in range(0, len(grupo), 100):   # hasta 100 coordenadas por petición
            lote = grupo.iloc[i:i + 100]
            esperar(len(lote))
            params = dict(latitude=",".join(map(str, lote.CELDA_LAT)), longitude=",".join(map(str, lote.CELDA_LON)),
                          start_date=b_ini.strftime("%Y-%m-%d"), end_date=b_fin.strftime("%Y-%m-%d"),
                          timezone="America/Lima", **variables)
            for intento in range(20):   # con 429 por límite horario se espera hasta ~1 h y media
                try:
                    r = requests.get(API_CLIMA, params=params, timeout=120)
                except requests.RequestException:   # timeout o corte de red: se reintenta
                    time.sleep(30)
                    continue
                if r.status_code == 200:
                    break
                time.sleep(300 if r.status_code == 429 else 5 * (intento + 1))
            else:
                raise RuntimeError("Open-Meteo falló 20 veces seguidas")
            historial.append((time.time(), len(lote)))
            respuesta = r.json()
            respuesta = respuesta if isinstance(respuesta, list) else [respuesta]
            nuevo = pd.concat([parsear(clat, clon, loc, bloque)   # la respuesta viene en el mismo orden que el pedido
                               for clat, clon, loc in zip(lote.CELDA_LAT, lote.CELDA_LON, respuesta)], ignore_index=True)
            nuevo.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"  bloque {int(bloque) + 1}/{n_bloques} listo")


def cargar_clima_puntos(data, path=F_CLIMA_PUNTOS, por_minuto=450, por_hora=4500):
    """Precipitación diaria en la celda de cada accidente (columnas CELDA_LAT, CELDA_LON y FECHA de `data`).

    Open-Meteo cobra una llamada por cada punto y cada 14 días pedidos (límite gratuito: 600/min, 5 000/hora,
    10 000/día). Pedir los 639 días para ~1 000 celdas costaría ~46 000 llamadas, así que el periodo se parte
    en bloques de 14 días y para cada celda solo se piden los bloques en los que tuvo accidentes. Cada bloque
    descargado se agrega a la caché, de modo que una descarga interrumpida se retoma donde quedó.
    """
    ini = data["FECHA"].min()
    fin = data["FECHA"].max()
    pedidos = data[["CELDA_LAT", "CELDA_LON"]].assign(BLOQUE=(data["FECHA"] - ini).dt.days // 14).drop_duplicates()

    hechos = set()
    if path.exists():
        cache = pd.read_csv(path, parse_dates=["FECHA"])
        cache["BLOQUE"] = (cache["FECHA"] - ini).dt.days // 14
        hechos = set(zip(cache.CELDA_LAT.round(2), cache.CELDA_LON.round(2), cache.BLOQUE))
    faltan = pedidos[[k not in hechos for k in zip(pedidos.CELDA_LAT, pedidos.CELDA_LON, pedidos.BLOQUE)]]
    if len(faltan):
        print(f"Clima por celda: faltan {len(faltan)} de {len(pedidos)} pares celda-bloque")

    def parsear(clat, clon, loc, bloque):
        parte = pd.DataFrame(loc["daily"]).rename(columns={"time": "FECHA"})
        parte["CELDA_LAT"], parte["CELDA_LON"], parte["ELEVACION"] = clat, clon, loc.get("elevation")
        return parte

    descargar_por_bloques(faltan, ini, fin, path, dict(daily=VARIABLES_CLIMA), parsear, por_minuto, por_hora)

    clima = pd.read_csv(path, parse_dates=["FECHA"])
    clima["CELDA_LAT"], clima["CELDA_LON"] = clima.CELDA_LAT.round(2), clima.CELDA_LON.round(2)
    return clima.drop_duplicates(["CELDA_LAT", "CELDA_LON", "FECHA"])


F_CLIMA_HORARIO = RAW / "clima_horario_celdas.csv"
VARIABLES_HORARIAS = "precipitation,weather_code,cloud_cover,temperature_2m,dew_point_2m,wind_speed_10m,is_day"   # la caché guarda todas; se usan lluvia, código WMO y neblina


def _horas_necesarias(data):
    """(celda, instante) de la hora del accidente y de la anterior y la siguiente (ventana de 3 h)."""
    d = data.dropna(subset=["HORA_INT"])
    t = d["FECHA"] + pd.to_timedelta(d["HORA_INT"], unit="h")
    return pd.concat([pd.DataFrame({"CELDA_LAT": d.CELDA_LAT, "CELDA_LON": d.CELDA_LON, "TIEMPO": t + pd.Timedelta(hours=k)})
                      for k in (-1, 0, 1)]).drop_duplicates()


def cargar_clima_horario(data, path=F_CLIMA_HORARIO, por_minuto=390, por_hora=3900):
    """Clima por hora en la celda de cada accidente (Open-Meteo, mismas celdas de 0.1° que la lluvia diaria),
    para las variables de la bibliografía: condición climática en categorías, superficie mojada (lluvia de las
    últimas horas) y condición de luz. Se piden bloques de 14 días + 1 día de margen a cada lado (para la
    ventana de 3 h en la medianoche), pero la caché solo guarda las horas de la ventana de cada accidente.
    Los límites por hora se bajan a 3 900 porque 16 días por celda cuentan como ~1.14 llamadas."""
    ini, fin = data["FECHA"].min(), data["FECHA"].max()
    d = data.dropna(subset=["HORA_INT"])
    pedidos = d[["CELDA_LAT", "CELDA_LON"]].assign(BLOQUE=(d["FECHA"] - ini).dt.days // 14).drop_duplicates()
    necesarias = _horas_necesarias(data)
    necesarias = set(zip(necesarias.CELDA_LAT, necesarias.CELDA_LON, necesarias.TIEMPO))

    hechos = set()
    if path.exists():
        cache = pd.read_csv(path, usecols=["CELDA_LAT", "CELDA_LON", "BLOQUE"])
        hechos = set(zip(cache.CELDA_LAT.round(2), cache.CELDA_LON.round(2), cache.BLOQUE))
    faltan = pedidos[[k not in hechos for k in zip(pedidos.CELDA_LAT, pedidos.CELDA_LON, pedidos.BLOQUE)]]
    if len(faltan):
        print(f"Clima por hora: faltan {len(faltan)} de {len(pedidos)} pares celda-bloque")

    def parsear(clat, clon, loc, bloque):
        h = pd.DataFrame(loc["hourly"]).rename(columns={"time": "TIEMPO"})
        h["TIEMPO"] = pd.to_datetime(h["TIEMPO"])
        sol = pd.DataFrame(loc["daily"]).rename(columns={"time": "FECHA"})
        h["FECHA"] = h["TIEMPO"].dt.strftime("%Y-%m-%d")
        h = h.merge(sol, on="FECHA").drop(columns="FECHA")
        h["CELDA_LAT"], h["CELDA_LON"], h["BLOQUE"] = clat, clon, bloque
        return h[[k in necesarias for k in zip(h.CELDA_LAT, h.CELDA_LON, h.TIEMPO)]]

    descargar_por_bloques(faltan, ini, fin, path, dict(hourly=VARIABLES_HORARIAS, daily="sunrise,sunset"), parsear,
                          por_minuto, por_hora, margen_dias=1)
    clima = pd.read_csv(path, parse_dates=["TIEMPO", "sunrise", "sunset"])
    clima["CELDA_LAT"], clima["CELDA_LON"] = clima.CELDA_LAT.round(2), clima.CELDA_LON.round(2)
    return clima.drop(columns="BLOQUE").drop_duplicates(["CELDA_LAT", "CELDA_LON", "TIEMPO"])


def grupo_wmo(codigo):
    """Código WMO de Open-Meteo -> categorías como las de Mhetre y Thube (fino, nublado, neblina, llovizna, lluvia)."""
    if pd.isna(codigo):
        return np.nan
    c = int(codigo)
    if c <= 1:
        return "DESPEJADO"
    if c <= 3:
        return "NUBLADO"
    if c in (45, 48):
        return "NEBLINA"
    if 51 <= c <= 57:
        return "LLOVIZNA"
    if 71 <= c <= 77 or c in (85, 86):
        return "NIEVE"
    return "LLUVIA"   # 61-67, 80-82 y tormentas (95-99)


def variables_clima_horario(data, clima):
    """Variables por accidente a partir del clima por hora (NaN para los 88 accidentes sin hora).

    Convención de Open-Meteo: el valor de las HH:00 resume la hora previa, así que la hora del accidente
    (HH:MM con HH = h) se describe con el registro de las h+1."""
    c = clima.set_index(["CELDA_LAT", "CELDA_LON", "TIEMPO"])
    t = data["FECHA"] + pd.to_timedelta(data["HORA_INT"], unit="h")

    def en(k):
        idx = pd.MultiIndex.from_arrays([data.CELDA_LAT, data.CELDA_LON, t + pd.Timedelta(hours=k)])
        return c.reindex(idx).reset_index(drop=True).set_index(data.index)

    h0, h1, hm1 = en(0), en(1), en(-1)
    out = pd.DataFrame(index=data.index)
    out["PRECIP_HORA"] = h1["precipitation"]
    out["PRECIP_3H"] = hm1["precipitation"] + h0["precipitation"] + h1["precipitation"]   # proxy de vía mojada
    out["CLIMA_HORA"] = h1["weather_code"].map(grupo_wmo)
    out["NEBLINA"] = (h1["temperature_2m"] - h1["dew_point_2m"] < 2).astype(float).where(h1["temperature_2m"].notna())
    # Luz: con la hora y minuto exactos frente a la salida y la puesta del sol de ese día y lugar
    hhmm = pd.to_datetime(data["FECHA"].dt.strftime("%Y-%m-%d ") + data["HORA"].astype(str).str.strip().str[:5],
                          format="%Y-%m-%d %H:%M", errors="coerce")
    sale, pone = h1["sunrise"], h1["sunset"]
    out["ES_DE_DIA"] = ((hhmm >= sale) & (hhmm < pone)).astype(float).where(hhmm.notna() & sale.notna())
    a_sol = pd.concat([(hhmm - sale).abs(), (hhmm - pone).abs()], axis=1).min(axis=1)
    out["CREPUSCULO"] = (a_sol <= pd.Timedelta(hours=1)).astype(float).where(hhmm.notna() & sale.notna())
    return out


def variables_calendario(fechas):
    """Fin de semana, feriado, puente (día dentro de >= 3 días seguidos no laborables) y cuarentena."""
    dias = pd.date_range(fechas.min(), fechas.max())
    libre = pd.Series((dias.dayofweek >= 5) | dias.isin(FERIADOS), index=dias)
    racha = libre.groupby((~libre).cumsum()).transform("sum").where(libre, 0)
    return pd.DataFrame({"FIN_DE_SEMANA": (fechas.dt.dayofweek >= 5).astype(int),
                         "FERIADO": fechas.isin(FERIADOS).astype(int),
                         "PUENTE": fechas.map(racha >= 3).astype(int),
                         "CUARENTENA": fechas.between(*CUARENTENA).astype(int)}, index=fechas.index)


def variables_derivadas(data):
    """Variables de la bibliografía que se derivan sin fuentes externas."""
    out = variables_calendario(data["FECHA"])
    cap = np.array([COORD[d] for d in data["DEPARTAMENTO"]])
    out["DIST_CAPITAL_KM"] = haversine_km(data["LON"], data["LAT"], cap[:, 1], cap[:, 0])
    out["TIPO_VIA"] = np.select([data.CODIGO_VIA.str.startswith("PE-", na=False), data.CODIGO_VIA.str.contains("-", na=False)],
                                ["NACIONAL", "DEPARTAMENTAL"], "SIN DATO")
    return out


# Dataset limpio: solo identificación, objetivo y variables de la bibliografía, con nombres legibles.
# columna final -> (columna intermedia, descripción). La descripción es el diccionario de datos del README.
COLUMNAS_FINALES = {
    # identificación
    "FECHA": ("FECHA", "Fecha del accidente"),
    "HORA": ("HORA_INT", "Hora entera 0-23 (vacía si SUTRAN no la registró)"),
    "MES": ("MES", "Mes 1-12"),
    "DIA_SEMANA": ("DIA_SEMANA", "0 = lunes … 6 = domingo"),
    "DEPARTAMENTO": ("DEPARTAMENTO", "Departamento (SUTRAN, normalizado)"),
    "CODIGO_VIA": ("CODIGO_VIA", "Código de la vía (SUTRAN)"),
    "TIPO_VIA": ("TIPO_VIA", "NACIONAL / DEPARTAMENTAL / SIN DATO"),
    "UBICACION": ("UBICACION", "km = ubicado sobre la vía; capital = respaldo en la capital del departamento"),
    # objetivo
    "NUM_FALLECIDOS": ("NUM_FALLECIDOS", "Fallecidos (SUTRAN)"),
    "NUM_HERIDOS": ("NUM_HERIDOS", "Heridos (SUTRAN)"),
    "GRAVEDAD": ("GRAVEDAD", "Objetivo Tarea 2: 0 sin víctimas, 1 heridos, 2 al menos un fallecido"),
    # tipo de choque
    "MODALIDAD": ("MODALIDAD", "Choque, despiste, atropello, volcadura, especial (SUTRAN)"),
    # vía (MTC)
    "TOPOGRAFIA": ("TOPOGRAFIA", "Plano / ondulado / montañoso (tramo del MTC)"),
    "SUPERFICIE": ("SUPERFICIE", "Superficie de rodadura del tramo"),
    "ESTADO_VIA": ("ESTADO_VIA", "Estado de la superficie del tramo"),
    "CARRILES": ("CARRILES", "Número de carriles del tramo"),
    "ANCHO_CALZADA": ("ANCHO_CALZADA", "Ancho de calzada del tramo (m)"),
    "VEL_PROMEDIO_TRAMO": ("VEL_PROMEDIO_TRAMO", "Velocidad promedio del tramo (km/h)"),
    "SINUOSIDAD": ("SINUOSIDAD", "Largo del camino / línea recta en ±1 km (1 = recta)"),
    # clima (Open-Meteo, celda de 0.1°)
    "PRECIP_DIA": ("precipitation_sum", "Lluvia del día en la celda (mm)"),
    "HORAS_LLUVIA": ("precipitation_hours", "Horas con lluvia en el día"),
    "PRECIP_3H": ("PRECIP_3H", "Lluvia en las 3 h alrededor del accidente (mm; proxy de vía mojada)"),
    "CLIMA_HORA": ("CLIMA_HORA", "Despejado / nublado / llovizna / lluvia / nieve a la hora del accidente"),
    "NEBLINA": ("NEBLINA", "1 si temperatura − punto de rocío < 2 °C"),
    # luz y hora
    "ES_DE_DIA": ("ES_DE_DIA", "1 si fue entre la salida y la puesta del sol"),
    "CREPUSCULO": ("CREPUSCULO", "1 si fue a ±1 h de la salida o la puesta del sol"),
    # calendario
    "FIN_DE_SEMANA": ("FIN_DE_SEMANA", "1 si sábado o domingo"),
    "FERIADO": ("FERIADO", "1 si feriado nacional"),
    "PUENTE": ("PUENTE", "1 si es parte de ≥ 3 días seguidos no laborables"),
    "CUARENTENA": ("CUARENTENA", "1 si fue en la cuarentena nacional (16/03 – 30/06/2020)"),
    # lugar
    "REGION_NATURAL": ("REGION_NATURAL", "Costa / sierra / selva (tramo del MTC)"),
    "ELEVACION": ("ELEVACION", "Altitud de la celda (m)"),
    "DIST_CAPITAL_KM": ("DIST_CAPITAL_KM", "Distancia a la capital del departamento (km)"),
}


def dataset_final(data):
    """Selecciona y renombra las columnas finales. Deja fuera las crudas e intermedias (FECHA_CORTE, texto de la
    hora, KILOMETRO, celda, motivo de la ubicación, rain_sum, lluvia de la capital…) y las descartadas en las
    decisiones de encoding (franja, temperatura, viento, nubosidad, LAT/LON) y los 5 accidentes sin
    GRAVEDAD conocida (sin dato de heridos y sin fallecidos). Las categóricas sin dato quedan como 'SIN DATO'."""
    out = data[[c for c, _ in COLUMNAS_FINALES.values()]].copy()
    out.columns = list(COLUMNAS_FINALES)
    out = out.dropna(subset=["GRAVEDAD"]).reset_index(drop=True)
    out[["GRAVEDAD", "NUM_HERIDOS"]] = out[["GRAVEDAD", "NUM_HERIDOS"]].astype(int)
    out["MODALIDAD"] = out["MODALIDAD"].replace({NI: "SIN DATO"})
    out["CODIGO_VIA"] = out["CODIGO_VIA"].replace({NI: "SIN DATO"})
    out["CLIMA_HORA"] = out["CLIMA_HORA"].fillna("SIN HORA")
    return out


def construir_dataset(guardar=True):
    """Pipeline completo: crudo -> limpio -> ubicación y atributos de la vía -> clima diario y por hora ->
    variables derivadas. Devuelve (data, bitacora): `data` con TODAS las columnas intermedias (lo usa el panel y el
    notebook 00); lo que se guarda en data/processed/accidentes.csv es `dataset_final(data)`.
    """
    df, bitacora = limpiar_accidentes(leer_accidentes())
    data = df.join(ubicar_accidentes(df, cargar_red_vial()))
    data["CELDA_LAT"], data["CELDA_LON"] = asignar_celda(data["LAT"], data["LON"])

    capital = cargar_clima(df)[["DEPARTAMENTO", "FECHA", "precipitation_sum"]]
    data = data.merge(capital.rename(columns={"precipitation_sum": "PRECIP_CAPITAL"}),
                      on=["DEPARTAMENTO", "FECHA"], how="left")
    data = data.merge(cargar_clima_puntos(data), on=["CELDA_LAT", "CELDA_LON", "FECHA"], how="left")
    data = data.join(variables_clima_horario(data, cargar_clima_horario(data))).join(variables_derivadas(data))
    if guardar:
        PROC.mkdir(parents=True, exist_ok=True)
        dataset_final(data).to_csv(F_DATASET, index=False)
    return data, bitacora


F_PANEL = PROC / "panel_depto_dia.csv"


def construir_panel(data, clima_capital, guardar=True):
    """Panel departamento x día para la Tarea 1 (pronóstico de riesgo), CON los días sin accidentes (N_ACC = 0).

    Objetivos del día t: N_ACC, N_FATALES (accidentes con fallecidos) y HAY_FATAL. Variables de entrada, sin fuga
    de información: la historia (conteos, heridos, mezcla de modalidades) usa solo días <= t-1; el clima del día t
    es la lluvia en la capital (supuesto: se conoce por el pronóstico) y el calendario se conoce de antemano.
    Callao (6 accidentes) se une a Lima, que lo rodea.
    """
    acc = data.assign(DEPARTAMENTO=data["DEPARTAMENTO"].replace({"CALLAO": "LIMA"}))
    conteos = acc.groupby(["DEPARTAMENTO", "FECHA"]).agg(
        N_ACC=("ES_FATAL", "size"), N_FATALES=("ES_FATAL", "sum"), N_HERIDOS=("NUM_HERIDOS", "sum"),
        N_ATROPELLO=("MODALIDAD", lambda m: (m == "ATROPELLO").sum()),
        N_CHOQUE=("MODALIDAD", lambda m: (m == "CHOQUE").sum()))
    panel = (clima_capital[clima_capital["DEPARTAMENTO"] != "CALLAO"]
             .rename(columns={"precipitation_sum": "PRECIP", "precipitation_hours": "HORAS_LLUVIA"})
             [["DEPARTAMENTO", "FECHA", "PRECIP", "HORAS_LLUVIA"]]
             .merge(conteos.reset_index(), on=["DEPARTAMENTO", "FECHA"], how="left")
             .fillna({c: 0 for c in conteos.columns})
             .sort_values(["DEPARTAMENTO", "FECHA"]).reset_index(drop=True))
    panel["HAY_FATAL"] = (panel["N_FATALES"] > 0).astype(int)

    g = panel.groupby("DEPARTAMENTO")
    previo = lambda col, k: g[col].transform(lambda x: x.shift(1).rolling(k, min_periods=1).sum())
    panel["PRECIP_PREV3"], panel["PRECIP_PREV7"] = previo("PRECIP", 3), previo("PRECIP", 7)
    panel["N_ACC_AYER"] = g["N_ACC"].shift(1)
    panel["N_ACC_SEMANA_PASADA"] = g["N_ACC"].shift(7)   # mismo día de la semana anterior
    for k in (7, 28):
        panel[f"N_ACC_M{k}"] = previo("N_ACC", k) / k
    for col in ("N_FATALES", "N_HERIDOS"):
        panel[f"{col}_M28"] = previo(col, 28) / 28
    total28 = previo("N_ACC", 28)
    for col in ("N_ATROPELLO", "N_CHOQUE"):   # mezcla reciente de modalidades (sin fuga: hasta t-1)
        panel[f"P_{col[2:]}_M28"] = (previo(col, 28) / total28).where(total28 > 0, 0)

    panel = panel.join(variables_calendario(panel["FECHA"]))
    panel["DIA_SEMANA"] = panel["FECHA"].dt.dayofweek
    panel["MES_SIN"], panel["MES_COS"] = np.sin(2 * np.pi * panel.FECHA.dt.month / 12), np.cos(2 * np.pi * panel.FECHA.dt.month / 12)
    panel["ALTITUD_MEDIA"] = panel["DEPARTAMENTO"].map(acc.groupby("DEPARTAMENTO")["ELEVACION"].median())
    panel = panel.drop(columns=["N_HERIDOS", "N_ATROPELLO", "N_CHOQUE"])   # son del día t: solo sus versiones previas
    if guardar:
        panel.to_csv(F_PANEL, index=False)
    return panel


if __name__ == "__main__":
    data, bitacora = construir_dataset()
    print(bitacora.to_string(index=False))
    print(data["UBICACION"].value_counts().to_string())
    print(f"Accidentes con dato de lluvia: {data.precipitation_sum.notna().mean():.1%}")
    print("Guardado:", F_DATASET, dataset_final(data).shape)
    panel = construir_panel(data, cargar_clima(data))
    print("Guardado:", F_PANEL, panel.shape)
