"""Preparación de datos: lectura y limpieza de accidentes (SUTRAN), ubicación de cada accidente
con su vía y kilómetro (Red Vial Nacional del MTC), descarga de precipitación (Open-Meteo) en la
celda de cada accidente y unión por celda + fecha.

Uso desde la raíz del repo:
    python -m src.preprocessing      # genera data/processed/accidentes_clima.csv

Uso desde un notebook:
    from src import preprocessing as pp
    raw = pp.leer_accidentes()
    df, bitacora = pp.limpiar_accidentes(raw)
    df = df.join(pp.ubicar_accidentes(df, pp.cargar_red_vial()))
    df["CELDA_LAT"], df["CELDA_LON"] = pp.asignar_celda(df.LAT, df.LON)
    clima = pp.cargar_clima_puntos(df)
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
F_DATASET = PROC / "accidentes_clima.csv"

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
# Ubicación: vía + kilómetro -> coordenadas (Red Vial Nacional del MTC)
# ---------------------------------------------------------------------------
def descargar_red_vial(path=F_VIAS):
    """Descarga la capa de la Red Vial Nacional (MTC, al 31/12/2014) solo si no existe en data/raw/.

    Cada tramo trae su ruta (CCODRUTA_4, p. ej. 'PE-22'), su km de inicio y fin y su departamento.
    El servicio no admite paginación, así que se pide por rangos de OBJECTID. Su certificado HTTPS
    tiene la cadena incompleta, por eso verify=False (la capa es pública y queda guardada en el repo).
    """
    if path.exists():
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
            outFields="CCODRUTA_4,DKMINICIO,DKMFINAL,CDEPARTAME", returnGeometry="true", outSR=4326,
            maxAllowableOffset=0.0002, geometryPrecision=5, f="json"))   # simplifica la línea a ~20 m
        for f in r.json()["features"]:
            a = f["attributes"]
            partes = (f.get("geometry") or {}).get("paths", [])
            tramos.append({"ruta": a["CCODRUTA_4"].strip().upper(), "km_ini": a["DKMINICIO"], "km_fin": a["DKMFINAL"],
                           "departamento": quitar_tildes(a["CDEPARTAME"]).strip().upper(),
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

    Devuelve un DataFrame alineado con `df`: LAT, LON, UBICACION y MOTIVO (por qué cayó en cada caso).
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
            filas.append((lat, lon, "capital", motivo))
        else:
            filas.append((punto[0], punto[1], "km", motivo))
    return pd.DataFrame(filas, columns=["LAT", "LON", "UBICACION", "MOTIVO_UBICACION"], index=df.index)


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

    historial = []   # (instante, llamadas) para respetar los límites por minuto y por hora

    def esperar(n):
        while True:
            ahora = time.time()
            ult_min = sum(k for t, k in historial if ahora - t < 60)
            ult_hora = sum(k for t, k in historial if ahora - t < 3600)
            if ult_min + n <= por_minuto and ult_hora + n <= por_hora:
                return
            time.sleep(5)

    for bloque, grupo in faltan.groupby("BLOQUE"):
        b_ini = ini + pd.Timedelta(days=14 * int(bloque))
        b_fin = min(b_ini + pd.Timedelta(days=13), fin)
        for i in range(0, len(grupo), 100):   # hasta 100 coordenadas por petición
            lote = grupo.iloc[i:i + 100]
            esperar(len(lote))
            params = dict(latitude=",".join(map(str, lote.CELDA_LAT)), longitude=",".join(map(str, lote.CELDA_LON)),
                          start_date=b_ini.strftime("%Y-%m-%d"), end_date=b_fin.strftime("%Y-%m-%d"),
                          daily=VARIABLES_CLIMA, timezone="America/Lima")
            for intento in range(6):
                r = requests.get(API_CLIMA, params=params, timeout=120)
                if r.status_code == 200:
                    break
                time.sleep(60 if r.status_code == 429 else 5 * (intento + 1))
            else:
                raise RuntimeError(f"Open-Meteo falló: {r.status_code} {r.text[:200]}")
            historial.append((time.time(), len(lote)))
            respuesta = r.json()
            respuesta = respuesta if isinstance(respuesta, list) else [respuesta]
            partes = []
            for (clat, clon), loc in zip(zip(lote.CELDA_LAT, lote.CELDA_LON), respuesta):   # mismo orden que el pedido
                parte = pd.DataFrame(loc["daily"]).rename(columns={"time": "FECHA"})
                parte["CELDA_LAT"], parte["CELDA_LON"], parte["ELEVACION"] = clat, clon, loc.get("elevation")
                partes.append(parte)
            nuevo = pd.concat(partes, ignore_index=True)
            nuevo.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"  bloque {int(bloque) + 1}/{(fin - ini).days // 14 + 1} listo")

    clima = pd.read_csv(path, parse_dates=["FECHA"])
    clima["CELDA_LAT"], clima["CELDA_LON"] = clima.CELDA_LAT.round(2), clima.CELDA_LON.round(2)
    return clima.drop_duplicates(["CELDA_LAT", "CELDA_LON", "FECHA"])


def construir_dataset(guardar=True):
    """Pipeline completo: crudo -> limpio -> ubicación -> + clima -> data/processed/accidentes_clima.csv.

    La lluvia principal (precipitation_sum, rain_sum, precipitation_hours) es la de la celda del accidente;
    PRECIP_CAPITAL es la de la capital del departamento (versión anterior), que se guarda para comparar.
    """
    df, bitacora = limpiar_accidentes(leer_accidentes())
    data = df.join(ubicar_accidentes(df, cargar_red_vial()))
    data["CELDA_LAT"], data["CELDA_LON"] = asignar_celda(data["LAT"], data["LON"])

    capital = cargar_clima(df)[["DEPARTAMENTO", "FECHA", "precipitation_sum"]]
    data = data.merge(capital.rename(columns={"precipitation_sum": "PRECIP_CAPITAL"}),
                      on=["DEPARTAMENTO", "FECHA"], how="left")
    data = data.merge(cargar_clima_puntos(data), on=["CELDA_LAT", "CELDA_LON", "FECHA"], how="left")
    if guardar:
        PROC.mkdir(parents=True, exist_ok=True)
        data.to_csv(F_DATASET, index=False)
    return data, bitacora


if __name__ == "__main__":
    data, bitacora = construir_dataset()
    print(bitacora.to_string(index=False))
    print(data["UBICACION"].value_counts().to_string())
    print(f"Accidentes con dato de lluvia: {data.precipitation_sum.notna().mean():.1%}")
    print("Guardado:", F_DATASET, data.shape)
