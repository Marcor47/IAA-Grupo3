"""Figuras para la presentación de avances (semana 10-11).

Ejecutar desde la raíz del repo, después de los notebooks:
    python presentación/figuras.py
Genera presentación/figuras/*.png (título = hallazgo principal).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = Path(__file__).resolve().parent / "figuras"
SALIDA.mkdir(exist_ok=True)

# Paleta de referencia (skill dataviz): azul y naranja validados; tinta en grises
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#b9b8b3"
TINTA, TINTA2, FONDO = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "font.size": 13, "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14, "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.edgecolor": GRIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.color": TINTA, "legend.frameon": False,
})

df = pd.read_csv(RAIZ / "data/processed/accidentes_clima.csv")
metricas = pd.read_csv(RAIZ / "results/metrics.csv")
PREV = df.ES_FATAL.mean()


def guardar(fig, nombre, nota=None):
    if nota:
        fig.text(0.01, -0.02, nota, fontsize=10, color=TINTA2, va="top")
    fig.savefig(SALIDA / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ic_wilson(k, n, z=1.96):
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return c - m, c + m


def tasa_por(grupos, d=df):
    t = d.groupby(grupos, observed=True).ES_FATAL.agg(["sum", "count"])
    t["pct"] = 100 * t["sum"] / t["count"]
    lo, hi = ic_wilson(t["sum"], t["count"])
    t["lo"], t["hi"] = 100 * lo, 100 * hi
    return t


# 1 · Mapa: dónde ocurren los accidentes y cuáles son fatales (slide 1)
fig, ax = plt.subplots(figsize=(6.5, 8))
km = df[df.UBICACION == "km"]
ax.scatter(km.LON, km.LAT, s=3, c=GRIS, alpha=0.6, lw=0, label="No fatal")
f = km[km.ES_FATAL == 1]
ax.scatter(f.LON, f.LAT, s=5, c=NARANJA, alpha=0.8, lw=0, label="Fatal (≥ 1 fallecido)")
ax.set_aspect("equal")
ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"8 110 accidentes en la red vial;\n1 de cada 9 es fatal ({100 * PREV:.1f} %)")
ax.legend(loc="lower left", markerscale=3)
guardar(fig, "01_mapa_accidentes.png",
        f"SUTRAN, ene 2020 – sep 2021. Se muestran los {len(km) / len(df):.0%} ubicados por vía y km (Red Vial Nacional MTC).")

# 2 · Lluvia en el punto vs. en la capital (justifica ubicar cada accidente)
fig, ax = plt.subplots(figsize=(7, 5))
m = km.precipitation_sum.clip(upper=40), km.PRECIP_CAPITAL.clip(upper=40)
ax.scatter(m[1], m[0], s=6, c=AZUL, alpha=0.25, lw=0)
ax.plot([0, 40], [0, 40], color=TINTA2, lw=1, ls="--")
r = km[["precipitation_sum", "PRECIP_CAPITAL"]].corr().iloc[0, 1]
desacuerdo = ((km.precipitation_sum > 0.1) != (km.PRECIP_CAPITAL > 0.1)).mean()
ax.set_title(f"La lluvia de la capital no representa al accidente\n(r = {r:.2f}; difieren en si llovió en el {desacuerdo:.0%})")
ax.set_xlabel("Lluvia en la capital del departamento (mm)")
ax.set_ylabel("Lluvia en el punto del accidente (mm)")
guardar(fig, "02_lluvia_punto_vs_capital.png", "Por eso se ubica cada accidente por vía + km y se usa la lluvia de su celda de ~11 km (Open-Meteo).")

# 3 · Hallazgo 1: la lluvia no aumenta la fatalidad
bins = [-0.01, 0, 1, 5, 10, 20, np.inf]
etiq = ["0", "0–1", "1–5", "5–10", "10–20", "> 20"]
df["LLUVIA_CAT"] = pd.cut(df.precipitation_sum, bins, labels=etiq)
t = tasa_por("LLUVIA_CAT")
fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(len(t))
ax.axhline(100 * PREV, color=TINTA2, lw=1, ls="--")
ax.text(len(t) - 0.6, 100 * PREV + 0.3, f"promedio {100 * PREV:.1f} %", color=TINTA2, fontsize=11, ha="right")
ax.errorbar(x, t.pct, yerr=[t.pct - t.lo, t.hi - t.pct], fmt="o-", color=AZUL, lw=2, ms=8, capsize=4)
for xi, (p, n) in enumerate(zip(t.pct, t["count"])):
    ax.text(xi, 2.2, f"n={n:,}".replace(",", " "), ha="center", fontsize=10, color=TINTA2)
ax.set_xticks(x, etiq)
ax.set_ylim(0, 26)
ax.set_xlabel("Lluvia diaria en el punto del accidente (mm)")
ax.set_ylabel("% de accidentes fatales (IC 95 %)")
ax.set_title("Más lluvia no hace más fatales los accidentes")
guardar(fig, "03_lluvia_vs_fatalidad.png", "Spearman ρ = −0.031: asociación negativa y muy débil. Solo > 20 mm sube, con IC muy ancho (n = 107).")

# 4 · Hallazgo 2: la altitud explica el signo de la lluvia
# Solo accidentes ubicados por km: la altitud de la capital no es la del accidente (igual que 01_eda)
km = km.assign(ALT_CAT=pd.cut(km.ELEVACION, [-np.inf, 500, 2500, 3500, np.inf],
                              labels=["< 500 m", "500–2 500", "2 500–3 500", "> 3 500 m"]))
t = tasa_por("ALT_CAT", km)
llueve = 100 * km.assign(L=km.precipitation_sum > 0.1).groupby("ALT_CAT", observed=True).L.mean()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
x = np.arange(len(t))
a1.errorbar(x, t.pct, yerr=[t.pct - t.lo, t.hi - t.pct], fmt="o-", color=NARANJA, lw=2, ms=8, capsize=4)
a1.set_xticks(x, t.index), a1.set_ylim(0, 16)
a1.set_ylabel("% de accidentes fatales"), a1.set_title("Arriba, menos fatalidad…", fontsize=15)
a2.bar(x, llueve, color=AZUL, width=0.6)
for xi, v in enumerate(llueve):
    a2.text(xi, v + 1.5, f"{v:.0f} %", ha="center", fontsize=12)
a2.set_xticks(x, llueve.index), a2.set_ylim(0, 85), a2.grid(axis="x", visible=False)
a2.set_ylabel("% de accidentes en día con lluvia"), a2.set_title("…y más lluvia", fontsize=15)
fig.suptitle("La altitud, no la lluvia, explica la menor fatalidad en días lluviosos",
             x=0.01, ha="left", fontsize=17, fontweight="bold", y=1.04)
guardar(fig, "04_altitud.png", "Accidentes ubicados por vía + km; altitud de su celda (Open-Meteo). La sierra concentra los días de lluvia y tiene accidentes menos fatales.")

# 5 · Odds ratios: al controlar por altitud la lluvia deja de asociarse
# ponytail: valores copiados de la salida de 02_modelos.ipynb (logit statsmodels); re-copiar si cambia el modelo
ors = [("Lluvia · control depto + hora", 0.929, 0.847, 1.018, AZUL),
       ("Lluvia · + altitud", 0.980, 0.891, 1.077, AZUL),
       ("Altitud (por cada 1 000 m)", 0.855, 0.796, 0.918, NARANJA)]
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, (nom, o, lo, hi, c) in enumerate(ors[::-1]):
    ax.plot([lo, hi], [i, i], color=c, lw=3, solid_capstyle="round")
    ax.plot(o, i, "o", color=c, ms=11, mec=FONDO, mew=2)
    ax.text(hi + 0.01, i, f"OR {o:.2f}  [{lo:.2f}–{hi:.2f}]", va="center", fontsize=11, color=TINTA2)
ax.axvline(1, color=TINTA2, lw=1, ls="--")
ax.set_yticks(range(3), [o[0] for o in ors[::-1]])
ax.set_xlim(0.75, 1.2), ax.grid(axis="y", visible=False)
ax.set_xlabel("Odds ratio de accidente fatal (IC 95 %)   ← menos fatal | más fatal →")
ax.set_title("Controlando por altitud, la lluvia ya no se asocia con la fatalidad")
guardar(fig, "05_odds_ratios.png", "Regresión logística con efectos de departamento y hora. Lluvia como log(1 + mm).")

# 6 · Baseline: PR-AUC frente al azar
cl = metricas[metricas.tarea == "A_clasificacion"].copy()
cl["nombre"] = ["Dummy (azar)", "Logística · Set A\n(antes del accidente)",
                "Logística · Set B\n(+ modalidad, después)", "Logística · Set A\npartición temporal"]
cl = cl.iloc[[0, 3, 1, 2]]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
colores = [GRIS, AZUL, AZUL, NARANJA]
y = np.arange(len(cl))[::-1]
ax.barh(y, cl.pr_auc, color=colores, height=0.6)
for yi, v, p in zip(y, cl.pr_auc, cl.prevalencia_test):
    ax.text(v + 0.004, yi, f"{v:.3f}  ({v / p:.1f}× azar)", va="center", fontsize=12)
ax.set_yticks(y, cl.nombre)
ax.set_xlim(0, 0.33), ax.grid(axis="y", visible=False)
ax.set_xlabel("PR-AUC (clase fatal)")
ax.set_title("El baseline supera al azar 1.6×, pero aún predice poco")
guardar(fig, "06_baseline_pr_auc.png",
        "Test 80/20 estratificado, 1 622 accidentes (11.7 % fatales). Set B es referencia: la modalidad se conoce después del accidente.")

print("Figuras en", SALIDA)
for p in sorted(SALIDA.glob("*.png")):
    print(" ", p.name)
