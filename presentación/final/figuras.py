"""Figuras de la presentación final: barras simples, sin título (el título lo pone la diapositiva).

Ejecutar desde la raíz del repo:  python presentación/final/figuras.py
Lee results/metrics.csv. El recall de fatales y la ablación vienen de results/model_comparison.txt.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = Path(__file__).resolve().parent / "figuras"
SALIDA.mkdir(exist_ok=True)

# Misma fuente que el HTML (IBM Plex Sans, OFL, en fuentes/) para que figuras y texto combinen
for ttf in (Path(__file__).resolve().parent / "fuentes").glob("*.ttf"):
    font_manager.fontManager.addfont(str(ttf))

# Misma paleta que el avance: azul = lo que se destaca, naranja = fatal / costo, gris = resto
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#c9c8c2"
TINTA, TINTA2, FONDO = "#1b1f24", "#55595f", "#ffffff"
plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "font.family": "IBM Plex Sans", "font.size": 16, "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14, "axes.titlecolor": TINTA2, "xtick.color": TINTA, "ytick.color": TINTA,
    "ytick.labelsize": 16, "xtick.labelsize": 16, "axes.edgecolor": GRIS, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.bottom": False, "text.color": TINTA,
})

m = pd.read_csv(RAIZ / "results/metrics.csv")


def met(tarea, modelo, val, col):
    f = m[(m.tarea == tarea) & (m.modelo == modelo) & m.validacion.str.startswith(val)]
    return float(f[col].iloc[0])


def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)


def hbarras(ax, etiquetas, valores, colores, fmt="{:.3f}", titulo=None, xmax=None):
    """Barras horizontales con el valor escrito al final; la primera etiqueta queda arriba."""
    y = np.arange(len(valores))[::-1]
    ax.barh(y, valores, color=colores, height=0.62)
    xmax = xmax or max(valores) * 1.22
    for yi, v in zip(y, valores):
        ax.text(max(v, 0) + xmax * 0.015, yi, fmt.format(v), va="center", fontsize=17, fontweight="bold")
    ax.set_yticks(y, etiquetas)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.set_xlim(min(0, min(valores) * 1.3), xmax)
    if titulo:
        ax.set_title(titulo)


def resaltar(etiquetas, cual, color=AZUL):
    return [color if e == cual else GRIS for e in etiquetas]


C, H = "1_conteo_N_ACC", "1_clasificacion_HAY_FATAL"
G = "2_triaje_GRAVEDAD"
NOMBRES_C = {"Media de 28 días": "Ingenuo · media 28 días", "Poisson (baseline)": "Baseline · Poisson",
             "Random Forest": "Random Forest (ajustado)", "MLP": "MLP (ajustado)",
             "Gradient boosting": "Gradient boosting (ajustado)", "Ensamble": "Ensamble (votación)"}
NOMBRES_G = {"Dummy": "Dummy (proporciones)", "Logística (baseline)": "Baseline · Reg. logística multinomial",
             "MLP": "MLP (ajustado)", "Gradient boosting": "Gradient boosting (ajustado)",
             "Ensamble": "Ensamble (votación)", "Random Forest": "Random Forest (ajustado)"}
# Recall de la clase fatal en el test 80/20 (results/model_comparison.txt)
RECALL = {"Logística (baseline)": 51, "MLP": 49, "Ensamble": 37, "Gradient boosting": 28, "Random Forest": 27}

# 1 · Binomial negativa = Poisson
et = ["Media de 28 días", "Poisson (baseline)", "Binomial negativa"]
va = [met(C, "Ingenuo · media 28 días", "CV", "d2_poisson"), met(C, "Baseline · Poisson", "CV", "d2_poisson"),
      met(C, "Binomial negativa", "CV", "d2_poisson")]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
hbarras(ax, et, va, [GRIS, AZUL, AZUL], titulo="Accidentes por día · D² de Poisson (más alto = mejor)", xmax=0.33)
guardar(fig, "01_binomial.png")

# 2 · Random Forest: gana en general, pierde fatales
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.55})
et = list(NOMBRES_G)
hbarras(a1, et, [met(G, NOMBRES_G[e], "CV", "macro_f1") for e in et], resaltar(et, "Random Forest"),
        titulo="Acierto en las 3 clases · macro-F1", xmax=0.56)
et = list(RECALL)
hbarras(a2, et, [RECALL[e] for e in et], resaltar(et, "Random Forest", NARANJA), fmt="{:.0f} %",
        titulo="Fatales detectados (test)", xmax=62)
guardar(fig, "02_rf.png")

# 3 · Gradient boosting en el conteo: CV y holdout
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.55})
et = [e for e in NOMBRES_C if e != "Ensamble"]
for ax, val, t in [(a1, "CV", "Validación cruzada temporal"), (a2, "holdout", "Futuro no visto (abr–sep 2021)")]:
    hbarras(ax, et, [met(C, NOMBRES_C[e], val, "d2_poisson") for e in et], resaltar(et, "Gradient boosting"),
            titulo=t, xmax=0.36)
guardar(fig, "03_gbm.png")

# 4 · Ensamble: el equilibrio entre acierto general y fatales detectados
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for e, r in RECALL.items():
    f1 = met(G, NOMBRES_G[e], "test", "macro_f1")
    c = AZUL if e == "Ensamble" else GRIS
    ax.scatter(r, f1, s=380 if e == "Ensamble" else 220, color=c, zorder=3, edgecolor=FONDO, lw=2)
    ax.annotate(e, (r, f1), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=15,
                fontweight="bold" if e == "Ensamble" else "normal", color=TINTA if e == "Ensamble" else TINTA2)
ax.set_xlabel("Fatales detectados (%) · más es mejor", fontsize=15)
ax.set_ylabel("Acierto en las 3 clases (macro-F1)", fontsize=15)
ax.set_xlim(20, 58), ax.set_ylim(0.422, 0.452)
ax.spines["bottom"].set_visible(True)
ax.grid(color="#ecebe7", lw=1)
ax.set_axisbelow(True)
ax.text(57, 0.4505, "ideal: arriba a la derecha", ha="right", fontsize=14, color=TINTA2)
guardar(fig, "04_ensamble.png")

# 5 · Comparación final en una sola escala: cuánto mejora el modelo elegido sobre el baseline lineal (CV).
# Significancia de la comparación pareada (model_comparison.txt): solo GRAVEDAD excluye el 0.
filas = [
    ("¿Cuántos accidentes mañana?", C, "Baseline · Poisson", "Ensamble (votación)", "d2_poisson",
     "Ensamble: de {a:.3f} a {b:.3f} · no significativa en la CV (en el futuro no visto, +11 %)"),
    ("¿Habrá un fatal?", H, "Baseline · Reg. logística", "Baseline · Reg. logística", "pr_auc",
     "Todos empatan: se queda la logística, que es el baseline"),
    ("¿Qué tan grave es?", G, "Baseline · Reg. logística multinomial", "Random Forest (ajustado)", "macro_f1",
     "Random Forest: de {a:.3f} a {b:.3f} · significativa, mejor en 25 de 25 pruebas"),
]
fig, ax = plt.subplots(figsize=(13, 4.2))
y = np.arange(len(filas))[::-1]
for yi, (preg, t, ref, mod, col, nota) in zip(y, filas):
    a, b = met(t, ref, "CV", col), met(t, mod, "CV", col)
    mejora = 100 * (b / a - 1)
    ax.barh(yi, mejora, color=AZUL if t == G else GRIS, height=0.6)
    ax.text(mejora + 0.15, yi + 0.1, f"+{mejora:.0f} %" if round(mejora) else "0 %", va="center", fontsize=22, fontweight="bold")
    ax.text(mejora + 0.15, yi - 0.22, nota.format(a=a, b=b), va="center", fontsize=13, color=TINTA2)
ax.set_yticks(y, [f[0] for f in filas], fontsize=17)
ax.tick_params(axis="y", length=0)
ax.set_xticks([]), ax.set_xlim(0, 16)
ax.set_title("Mejora del modelo elegido sobre el baseline lineal (validación cruzada)", fontsize=16)
guardar(fig, "05_comparacion.png")

# 6 · Ablación: cuánto empeora el modelo si se quita cada grupo (model_comparison.txt §ablación)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.5})
et = ["Lugar", "Calendario", "Clima (lluvia)", "Historia"]
hbarras(a1, et, [0.022, 0.006, 0.001, -0.001], resaltar(et, "Clima (lluvia)"), fmt="{:+.3f}",
        titulo="Accidentes por día · caída del D²", xmax=0.03)
et = ["Tipo de choque", "Lugar", "Luz y hora", "Vía", "Calendario", "Clima (lluvia)"]
hbarras(a2, et, [0.042, 0.016, 0.009, 0.004, 0.003, 0.000], resaltar(et, "Clima (lluvia)"), fmt="{:+.3f}",
        titulo="Gravedad · caída del macro-F1", xmax=0.055)
for ax in (a1, a2):  # la barra del clima es ~0: se marca la etiqueta
    for t in ax.get_yticklabels():
        if t.get_text().startswith("Clima"):
            t.set_color(AZUL), t.set_fontweight("bold")
guardar(fig, "06_ablacion.png")
# 7 · Mapa y 8 · dona (diapositiva 3). Las ubicaciones se recalculan desde las cachés de data/raw (≈ 2 s).
import sys  # noqa: E402
sys.path.insert(0, str(RAIZ))
import src.preprocessing as P  # noqa: E402

acc, _ = P.limpiar_accidentes(P.leer_accidentes())
acc = acc.join(P.ubicar_accidentes(acc, P.cargar_red_vial()))
km = acc[acc.UBICACION == "km"]
fig, ax = plt.subplots(figsize=(6, 7.5))
ax.scatter(km.LON, km.LAT, s=4, c=GRIS, alpha=0.8, lw=0, label="Sin fallecidos")
f = km[km.ES_FATAL == 1]
ax.scatter(f.LON, f.LAT, s=7, c=NARANJA, alpha=0.9, lw=0, label="Fatal")
ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([]), ax.spines["left"].set_visible(False)
ax.legend(loc="lower left", markerscale=4, fontsize=15, frameon=False)
guardar(fig, "07_mapa.png")

fig, ax = plt.subplots(figsize=(6, 6.4))
n_fatal = int(acc.ES_FATAL.sum())
ax.pie([n_fatal, len(acc) - n_fatal], colors=[NARANJA, GRIS], startangle=90, counterclock=False,
       wedgeprops=dict(width=0.32, edgecolor=FONDO, linewidth=3))
ax.text(0, 0.1, f"{100 * n_fatal / len(acc):.1f} %", ha="center", va="center", fontsize=40, fontweight="bold")
ax.text(0, -0.2, "son fatales", ha="center", va="center", fontsize=17, color=TINTA2)
ax.legend([f"Fatal · {n_fatal}", f"Sin fallecidos · {len(acc) - n_fatal:,}".replace(",", " ")], loc="lower center",
          bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=15, handlelength=1, frameon=False)
ax.set_aspect("equal")
guardar(fig, "08_dona.png")
print("figuras en", SALIDA)
