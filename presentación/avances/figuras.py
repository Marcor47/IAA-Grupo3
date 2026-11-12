"""Figuras para las presentaciones: avances (semana 11, figuras 01-07) y final (semana 15, figuras 01-10).

Ejecutar desde la raíz del repo, después de los notebooks:
    python presentación/figuras.py
Genera presentación/figuras/*.png: gráficos simples (dona y barras con el valor escrito)
cuyo título es el hallazgo.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = Path(__file__).resolve().parent / "figuras"
SALIDA.mkdir(exist_ok=True)
for viejo in SALIDA.glob("*.png"):
    viejo.unlink()

# Paleta de referencia (skill dataviz): azul = lluvia/modelo, naranja = fatal, gris = resto
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#c9c8c2"
TINTA, TINTA2, FONDO = "#1b1f24", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "font.size": 15, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 16, "axes.labelcolor": TINTA2, "xtick.color": TINTA, "ytick.color": TINTA2,
    "xtick.labelsize": 15, "axes.edgecolor": GRIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": False, "text.color": TINTA, "legend.frameon": False,
})

df = pd.read_csv(RAIZ / "data/processed/accidentes_clima.csv")
metricas = pd.read_csv(RAIZ / "results/metrics.csv")
km = df[df.UBICACION == "km"]
PREV = 100 * df.ES_FATAL.mean()


def guardar(fig, nombre, nota=None):
    if nota:
        fig.text(0.01, -0.02, nota, fontsize=12, color=TINTA2, va="top")
    fig.savefig(SALIDA / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)


def dona(ax, valores, colores, centro, sub):
    ax.pie(valores, colors=colores, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.32, edgecolor=FONDO, linewidth=3))
    ax.text(0, 0.08, centro, ha="center", va="center", fontsize=40, fontweight="bold")
    ax.text(0, -0.22, sub, ha="center", va="center", fontsize=15, color=TINTA2)
    ax.set_aspect("equal")


def barras(ax, etiquetas, valores, colores, fmt="{:.1f} %", ymax=None):
    x = np.arange(len(valores))
    ax.bar(x, valores, color=colores, width=0.62)
    for xi, v in zip(x, valores):
        ax.text(xi, v + (ymax or max(valores)) * 0.02, fmt.format(v), ha="center", va="bottom",
                fontsize=18, fontweight="bold", bbox=dict(fc=FONDO, ec="none", pad=1))
    ax.set_xticks(x, etiquetas)
    ax.set_yticks([])
    ax.set_ylim(0, ymax or max(valores) * 1.2)


# 1 · Dona: fatales vs. no fatales
fig, ax = plt.subplots(figsize=(6, 6))
n_fatal = int(df.ES_FATAL.sum())
dona(ax, [n_fatal, len(df) - n_fatal], [NARANJA, GRIS], f"{PREV:.1f} %", "son fatales")
ax.set_title("Solo 1 de cada 9 accidentes es fatal", loc="center")
ax.legend([f"Fatal · {n_fatal:,}".replace(",", " "), f"No fatal · {len(df) - n_fatal:,}".replace(",", " ")],
          loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=15, handlelength=1)
guardar(fig, "01_dona_fatales.png")

# 2 · Mapa: dónde ocurren
fig, ax = plt.subplots(figsize=(6.5, 8))
ax.scatter(km.LON, km.LAT, s=3, c=GRIS, alpha=0.7, lw=0, label="No fatal")
f = km[km.ES_FATAL == 1]
ax.scatter(f.LON, f.LAT, s=5, c=NARANJA, alpha=0.85, lw=0, label="Fatal")
ax.set_aspect("equal")
ax.set_xticks([]), ax.set_yticks([])
ax.spines["bottom"].set_visible(False)
ax.set_title("Cada accidente ubicado en la red vial")
ax.legend(loc="lower left", markerscale=4, fontsize=15)
guardar(fig, "02_mapa_accidentes.png", f"{len(km) / len(df):.0%} de los accidentes, ubicados por vía y km (Red Vial Nacional MTC).")

# 3 · Dona: la lluvia de la capital se equivoca
llovio_punto, llovio_cap = km.precipitation_sum > 0.1, km.PRECIP_CAPITAL > 0.1
difiere = 100 * (llovio_punto != llovio_cap).mean()
fig, ax = plt.subplots(figsize=(6, 6))
dona(ax, [difiere, 100 - difiere], [AZUL, GRIS], f"{difiere:.0f} %", "de los accidentes")
ax.set_title("La capital se equivoca sobre si llovió\nen 1 de cada 5 accidentes", loc="center")
ax.legend(["Capital y punto difieren", "Coinciden"], loc="lower center", bbox_to_anchor=(0.5, -0.12),
          ncol=2, fontsize=15, handlelength=1)
guardar(fig, "03_dona_capital.png")

# 4 · Lluvia: % fatal en día seco vs. con lluvia
seco = df.precipitation_sum <= 0.1
niveles = pd.cut(df.precipitation_sum, [-1, 0.1, 5, 20, np.inf], labels=["Seco", "Ligera\n0–5 mm", "Moderada\n5–20 mm", "Fuerte\n> 20 mm"])
t = df.groupby(niveles, observed=True).ES_FATAL.agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(8, 5.2))
barras(ax, t.index, 100 * t["mean"], [GRIS, AZUL, AZUL, "#9fc2ec"], ymax=20)
ax.axhline(PREV, color=TINTA2, lw=1.2, ls="--", zorder=0)
ax.text(3.5, PREV - 0.9, f"promedio\n{PREV:.1f} %", fontsize=13, color=TINTA2, ha="left")
ax.text(3, 100 * t["mean"].iloc[3] + 2.6, "pocos casos", ha="center", fontsize=13, color=TINTA2)
for xi, n in enumerate(t["size"]):
    ax.text(xi, 0.6, f"{n:,} acc.".replace(",", " "), ha="center", fontsize=12, color=FONDO, fontweight="bold")
ax.set_title("Con lluvia, los accidentes no son más fatales")
guardar(fig, "04_lluvia_fatalidad.png", "% de accidentes con ≥ 1 fallecido según la lluvia del día en el punto. \"Fuerte\" tiene pocos casos (107): diferencia no concluyente.")

# 5 · Altitud: menos fatal y más lluviosa
zonas = pd.cut(km.ELEVACION, [-np.inf, 500, 2500, 3500, np.inf], labels=["Costa\n< 500 m", "500–\n2 500 m", "2 500–\n3 500 m", "Altiplano\n> 3 500 m"])
fatal = 100 * km.groupby(zonas, observed=True).ES_FATAL.mean()
lluvia = 100 * (km.precipitation_sum > 0.1).groupby(zonas, observed=True).mean()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
barras(a1, fatal.index, fatal, NARANJA, ymax=16)
a1.set_title("% de accidentes fatales", fontsize=17)
barras(a2, lluvia.index, lluvia, AZUL, fmt="{:.0f} %", ymax=90)
a2.set_title("% de accidentes en día con lluvia", fontsize=17)
fig.suptitle("A más altura, más lluvia pero menos muertes", x=0.01, ha="left", fontsize=22, fontweight="bold", y=1.06)
guardar(fig, "05_altitud.png", "Por eso la lluvia parecía \"proteger\": en la regresión, al controlar por altitud su efecto desaparece (p = 0.67).")

# 6 · Baseline: PR-AUC frente al azar
cl = metricas[(metricas.tarea == "A_clasificacion") & (metricas.split == "aleatorio 80/20")].set_index("modelo")
cv = metricas[metricas.split.str.startswith("CV")].set_index("modelo")
BASE, M1 = "Baseline · Reg. logística · Set A", "Modelo 1 · Reg. logística (depto + hora + altitud)"
SET_B = "Referencia · Reg. logística · Set B (+ MODALIDAD, ex-post)"
fila_a = cl.loc[BASE]
vals = [fila_a.prevalencia_test, fila_a.pr_auc, cl.loc[SET_B].pr_auc]
fig, ax = plt.subplots(figsize=(8, 5.2))
barras(ax, ["Azar", "Nuestro\nbaseline", "Con modalidad\n(post-accidente)"], vals, [GRIS, AZUL, "#9fc2ec"], fmt="{:.2f}", ymax=0.32)
for xi, v in enumerate(vals[1:], start=1):
    ax.text(xi, v / 2, f"{v / vals[0]:.1f}×", ha="center", fontsize=20, fontweight="bold", color=FONDO)
ax.set_title("El modelo detecta fatales 1.6× mejor que el azar")
guardar(fig, "06_baseline.png", "PR-AUC de la clase fatal (test 80/20). La modalidad solo se conoce después del accidente: es referencia, no modelo.")

# 7 · Qué significa en la práctica: recall y precisión
fig, ax = plt.subplots(figsize=(8, 3.6))
y = [1, 0]
v = [100 * fila_a.recall, 100 * fila_a.precision]
ax.barh(y, [100, 100], color="#ecebe6", height=0.55)
ax.barh(y, v, color=[AZUL, NARANJA], height=0.55)
for yi, vi in zip(y, v):
    ax.text(vi + 1.5, yi, f"{vi:.0f} de 100", va="center", fontsize=18, fontweight="bold")
ax.set_yticks(y, ["Fatales que detecta", "Alertas que aciertan"], fontsize=16)
ax.set_xticks([]), ax.set_xlim(0, 100)
ax.tick_params(axis="y", labelsize=17, length=0)
ax.spines["bottom"].set_visible(False)
ax.set_title(f"Encuentra {round(10 * fila_a.recall)} de cada 10 fatales, con muchas falsas alarmas")
guardar(fig, "07_recall_precision.png", "Regresión logística balanceada, umbral elegido por CV en train, test 80/20 (1 622 accidentes).")

# 8 · Techo de señal: el grupo más letal que se puede identificar antes y después del accidente
grupos = [("Depar-\ntamento", ["DEPARTAMENTO"]), ("+ franja\nhoraria", ["DEPARTAMENTO", "FRANJA"]),
          ("+ altitud", ["DEPARTAMENTO", "FRANJA", "ALTITUD"]), ("Tipo de\nchoque", ["MODALIDAD"]),
          ("Choque + depto.\n+ franja", ["MODALIDAD", "DEPARTAMENTO", "FRANJA"])]
df["FRANJA"] = pd.cut(df.HORA_INT, [-1, 5, 11, 17, 24], labels=["madrugada", "mañana", "tarde", "noche"]).astype(object).fillna("sin hora")
df["ALTITUD"] = pd.cut(df.ELEVACION, [-1, 500, 2500, 9000], labels=["costa", "media", "alta"]).astype(str)
maximo = []
for _, cols in grupos:
    g = df.groupby(cols, dropna=False, observed=True).ES_FATAL.agg(["mean", "size"])
    maximo.append(100 * g[g["size"] >= 30]["mean"].max())
fig, ax = plt.subplots(figsize=(11, 5.4))
barras(ax, [n for n, _ in grupos], maximo, [NARANJA] * 3 + [GRIS] * 2, fmt="{:.0f} %", ymax=70)
ax.axvline(2.5, color=TINTA2, lw=1, ls=":")
ax.text(1, 64, "Se sabe ANTES del accidente", ha="center", fontsize=14, color=NARANJA, fontweight="bold")
ax.text(3.5, 64, "Solo se sabe DESPUÉS", ha="center", fontsize=14, color=TINTA2, fontweight="bold")
ax.set_title("Antes del accidente no hay situación de alto riesgo")
guardar(fig, "08_techo.png", "% fatal del grupo más letal (grupos con ≥ 30 accidentes). Ni la peor combinación de lugar, hora y altitud llega a 3 de cada 10.")

# 9 · Modelo 1 vs. baseline (CV 5×5): igual desempeño con 3 variables; la lluvia no suma
vals = [100 * df.ES_FATAL.mean() / 100, cv.loc[BASE].pr_auc, cv.loc[M1].pr_auc, cv.loc["Modelo 1 + lluvia del día"].pr_auc]
fig, ax = plt.subplots(figsize=(10, 5.2))
barras(ax, ["Azar", "Baseline\n9 variables", "Modelo 1\n3 variables", "Modelo 1\n+ lluvia"], vals,
       [GRIS, "#9fc2ec", AZUL, "#9fc2ec"], fmt="{:.3f}", ymax=0.22)
for xi, v in enumerate(vals[1:], start=1):
    ax.text(xi, v / 2, f"{v / vals[0]:.1f}×", ha="center", fontsize=20, fontweight="bold", color=FONDO)
ax.set_title("Con 3 variables predecimos igual que con 9, y la lluvia no suma")
guardar(fig, "09_modelo1.png", "PR-AUC promedio en validación cruzada 5 × 5 (25 particiones). Modelo 1: departamento + hora + altitud.")

# 10 · Hora: la madrugada es la franja más letal
orden = ["madrugada", "mañana", "tarde", "noche"]
franja = 100 * df[df.FRANJA != "sin hora"].groupby("FRANJA").ES_FATAL.mean().reindex(orden)
fig, ax = plt.subplots(figsize=(8, 5.2))
barras(ax, ["Madrugada\n0–5 h", "Mañana\n6–11 h", "Tarde\n12–17 h", "Noche\n18–23 h"], franja,
       [NARANJA, GRIS, GRIS, "#f3a582"], ymax=20)
ax.axhline(PREV, color=TINTA2, lw=1.2, ls="--", zorder=0)
ax.set_title("De madrugada, los accidentes son más fatales")
guardar(fig, "10_franja.png", "% de accidentes con ≥ 1 fallecido según la hora del accidente.")

# 11 · Dos escenarios: prevención (Modelo 1) vs. triaje (Modelo 2)
M2 = "Modelo 2 · Reg. logística (depto + hora + altitud + tipo de accidente)"
vals = [df.ES_FATAL.mean(), cv.loc[M1].pr_auc, cv.loc[M2].pr_auc, cv.loc["Modelo 2 + lluvia del día"].pr_auc]
fig, ax = plt.subplots(figsize=(10, 5.2))
barras(ax, ["Azar", "Modelo 1\nprevención", "Modelo 2\ntriaje", "Modelo 2\n+ lluvia"], vals,
       [GRIS, "#9fc2ec", AZUL, "#9fc2ec"], fmt="{:.3f}", ymax=0.34)
for xi, v in enumerate(vals[1:], start=1):
    ax.text(xi, v / 2, f"{v / vals[0]:.1f}×", ha="center", fontsize=20, fontweight="bold", color=FONDO)
ax.set_title("Con el tipo de accidente, el modelo pasa de 1.4× a 2.3× el azar")
guardar(fig, "11_modelo2.png", "PR-AUC promedio en validación cruzada 5 × 5. Modelo 2 = Modelo 1 + tipo de accidente (se conoce al reportarse).")

# 12 · % fatal por tipo de accidente
t = df.dropna(subset=["MODALIDAD"]).groupby("MODALIDAD").ES_FATAL.agg(["mean", "size"])
t = t[t["size"] >= 100].sort_values("mean", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5.2))
barras(ax, [m.capitalize() for m in t.index], 100 * t["mean"], [NARANJA] + [GRIS] * (len(t) - 1), ymax=50)
for xi, n in enumerate(t["size"]):
    ax.text(xi, 0.8, f"{n:,} acc.".replace(",", " "), ha="center", fontsize=12, color=FONDO if xi == 0 else TINTA, fontweight="bold")
ax.axhline(PREV, color=TINTA2, lw=1.2, ls="--", zorder=0)
ax.set_title("Un atropello es 6 veces más letal que un despiste")
guardar(fig, "12_tipo_accidente.png", "% de accidentes con ≥ 1 fallecido según el tipo de accidente (tipos con ≥ 100 casos).")

print("Figuras en", SALIDA)
for p in sorted(SALIDA.glob("*.png")):
    print(" ", p.name)
