# Brief para Claude Design · Presentación de avances (semana 11)

Pegar este archivo en Claude Design junto con las imágenes de `presentación/figuras/`.

## Encargo
Presentación de **3 diapositivas, 16:9, máximo 5 minutos**, para el curso Inteligencia Artificial Aplicada (PUCP). Objetivo: comunicar los avances del proyecto y **justificar las decisiones** tomadas. Estilo de directorio: muy visual, poco texto, un mensaje por bloque, cifras grandes. Sin código ni párrafos.

- **Título del proyecto:** Mortalidad en carreteras del Perú según el nivel de precipitación
- **Grupo 3:** Marco Rodriguez · Angel Cerdán · Raul Malaver · Fernando De La Cruz
- **Estilo:** fondo claro (#fcfcfb), texto casi negro, acento azul #2a78d6 (lluvia / modelo) y naranja #eb6834 (fatalidad / altitud). Son los colores de los gráficos: no introducir otros. Tipografía sans sobria.
- **Reglas del curso:** cada gráfico con un título que diga el hallazgo (ya lo traen); explicar qué significa cada gráfico o métrica; decisiones justificadas.
- Las figuras ya traen título y nota al pie: no repetir ese texto en la diapositiva.

---

## Diapositiva 1 · Problema y datos

**Titular:** ¿Un accidente en carretera es más mortal cuando llueve?

**Bloque izquierdo, problema (3 líneas como máximo):**
- Perú combina costa árida, sierra y selva: regímenes de lluvia muy distintos.
- La lluvia reduce la fricción y la visibilidad, pero no se sabe cuánto cambia la mortalidad.
- **Qué predecimos:** dado un accidente, ¿es fatal? (clasificación, `ES_FATAL`) y ¿cuántos fallecidos? (conteo).

**Tres KPI grandes:**
- **8 110** accidentes (SUTRAN, ene 2020 – sep 2021)
- **11.7 %** fatales → clases desbalanceadas
- **80 %** ubicados por vía + km

**Diagrama de flujo de los datos (dibujarlo, 3 cajas → 1):**
`Accidentes SUTRAN (vía, km, fecha)` + `Red Vial Nacional MTC (ubicación)` + `Open-Meteo (lluvia diaria y altitud, celda ~11 km)` → `Dataset unido: 8 110 accidentes × lluvia en el punto`

**Figura principal:** `01_mapa_accidentes.png` (grande, a la derecha).
**Figura de apoyo (pequeña, justifica la decisión):** `02_lluvia_punto_vs_capital.png` con la etiqueta *"Decisión: ubicar cada accidente, porque la lluvia de la capital no lo representa"*.

**Variables (una línea, chips):** lluvia (mm) · altitud · hora · día · mes · departamento · vía/km · *modalidad (solo después del accidente)*.

---

## Diapositiva 2 · Hallazgos y baseline

**Titular:** La lluvia no explica la fatalidad; la altitud sí

**Fila superior, dos hallazgos del EDA:**
1. `03_lluvia_vs_fatalidad.png` · etiqueta: *"El % fatal no sube con la lluvia (ρ = −0.03)"*
2. `05_odds_ratios.png` · etiqueta: *"Al controlar por altitud, el efecto de la lluvia desaparece (OR 0.98, p = 0.67); cada 1 000 m reducen las odds de fatalidad un 15 %"*
   (alternativa si cabe mejor: `04_altitud.png`, que lo muestra de forma descriptiva)

**Fila inferior, baseline:** `06_baseline_pr_auc.png` y, a su lado, tres bullets cortos:
- **Modelo:** regresión logística balanceada (clasificación) y Poisson (conteo). *Por qué:* interpretable, da odds ratios para medir la lluvia y es el punto de partida de la literatura.
- **Métrica:** PR-AUC, no exactitud: un modelo que siempre dice "no fatal" acierta el 88 %.
- **Lectura:** 1.6× el azar y estable en el tiempo (1.5×), pero precisión baja (≈ 15 de cada 100 alertas son fatales). La modalidad sube a 2.2×, pero solo se conoce después del accidente.

---

## Diapositiva 3 · Literatura y siguientes pasos

**Titular:** De la literatura al modelo final

**Tabla visual de 4 filas (paper → técnica → qué tomamos):**

| Paper | Técnica | Qué aplicamos |
|---|---|---|
| Mhetre y Thube (2023) · India | Poisson / binomial negativa | Fallecidos como conteo; binomial negativa por sobredispersión |
| Muñoz et al. (2024) · Ecuador | K-Means + PCA, Random Forest | Agrupar departamentos por régimen de lluvia; importancia de variables |
| Babaoglu y Babaoglu (2021) · Canadá | Lasso + XGBoost | Selección de variables; aviso: la mayoría de muertes ocurre con clima no extremo |
| Emu et al. (2022) · Canadá | CNN + ensambles por votación | MLP y ensambles para el desbalance |

**Siguientes pasos (4 íconos o tarjetas en fila):**
1. **Modelos no lineales:** árbol de decisión, KNN, MLP, binomial negativa
2. **Interacciones:** lluvia × altitud, lluvia × departamento
3. **Variable espacial:** clusters K-Means de departamentos
4. **Desbalance:** ajuste de umbral y remuestreo; comparar todo en las 3 particiones (aleatoria, CV-5, temporal)

**Cierre (una línea):** *"Hipótesis inicial: la lluvia aumenta la mortalidad. Lo que muestran los datos hasta ahora: el riesgo depende más de dónde ocurre el accidente."*

---

## Guion de 5 minutos (notas del orador)
- **D1 (1:30):** problema → qué predecimos → la decisión clave: ubicar cada accidente por km para medir la lluvia donde ocurrió.
- **D2 (2:00):** el % fatal no sube con la lluvia → el signo negativo venía de la geografía (la sierra llueve más y sus accidentes son menos fatales) → baseline: supera al azar, pero falta poder predictivo; por eso usamos PR-AUC.
- **D3 (1:30):** qué tomamos de cada paper y el plan para el entregable final.
