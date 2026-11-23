# Guion · Presentación final (≈ 10 min + 5 min de preguntas)

Fuente: `presentacion_final.html` (se exporta a `presentation.pdf` en la raíz del repo; ver al final). Once
diapositivas y dos de anexo para las preguntas. Cada título dice la conclusión de la diapositiva: si el tiempo
aprieta, basta con leer el título y señalar la cifra grande.

## Reparto y tiempos

| # | Diapositiva | Expone | Tiempo | Acumulado |
|---|---|---|---|---|
| 1 | Portada: la pregunta | Raul Malaver | 0:20 | 0:20 |
| 2 | El problema: dos preguntas | Raul Malaver | 0:30 | 0:50 |
| 3 | Datos: tres fuentes | Raul Malaver | 0:30 | 1:20 |
| 4 | Cómo medimos | Angel Cerdán | 1:00 | 2:20 |
| 5 | Binomial negativa | Marco Rodriguez | 1:00 | 3:20 |
| 6 | Random Forest | Angel Cerdán | 1:30 | 4:50 |
| 7 | Gradient boosting | Raul Malaver | 1:20 | 6:10 |
| 8 | MLP y ensamble por votación | Fernando De La Cruz | 1:40 | 7:50 |
| 9 | Comparación final | Marco Rodriguez | 0:50 | 8:40 |
| 10 | Ablación: ¿importa la lluvia? | Marco Rodriguez | 0:50 | 9:30 |
| 11 | Recomendación | Fernando De La Cruz | 0:50 | 10:20 |

Por integrante: Raul 2:40 · Angel 2:30 · Marco 2:40 · Fernando 2:30. Si el ensayo pasa de 10:30, recortar primero la
diapositiva 3 (solo el título) y la 9 (solo el título).

La barra de arriba de cada diapositiva (Problema → Datos → Cómo medimos → Modelos → Resultados → Recomendación)
marca en qué parte vamos: al cambiar de bloque, nombrarlo («pasamos a los modelos»). Las figuras se generan con
`python presentación/final/figuras.py`.

## Qué decir

1. **Portada (Raul).** Los recursos para atender accidentes en carreteras son limitados. Queremos saber dónde habrá
   accidentes mañana y, cuando llega un reporte, qué tan grave es. 8 105 accidentes de SUTRAN; el 66 % de los
   departamento-días no tiene ninguno y solo el 11.7 % de los accidentes es fatal.
2. **El problema (Raul).** Dos tareas: pronóstico por departamento y día (cuántos accidentes, si habrá un fatal) y
   triaje de un accidente (sin víctimas, heridos o fatal). Las barras muestran el desbalance.
3. **Datos (Raul).** El registro no trae coordenadas ni clima: ubicamos cada accidente por vía y km sobre la Red Vial
   del MTC (80 %) y le pegamos el clima por hora de Open-Meteo.
4. **Cómo medimos (Angel).** En la Tarea 1 entrenamos con el pasado y probamos en un futuro que el modelo no
   vio (abr–sep 2021). No usamos exactitud: decir «nunca fatal» acierta el 88 %. Métricas: D² de Poisson, PR-AUC y
   macro-F1. El baseline lineal le gana a las referencias, pero con 67 columnas no mejora a las 4 variables del
   avance: sin interacciones, las variables nuevas no suman.
5. **Binomial negativa (Marco).** Los conteos tienen más varianza que media (α = 0.10, significativo), pero el modelo
   da el mismo D² que el Poisson: el departamento, la historia y el calendario ya explican esa variación. En los
   fatales α ≈ 0.
6. **Random Forest (Angel).** El mejor en el triaje: +0.033 de macro-F1 en los 25 pliegues, porque la gravedad depende
   de combinaciones (atropello de noche en una recta). También el mejor en el holdout del conteo entre los modelos
   individuales. Su costo: detecta solo el 27 % de los fatales.
7. **Gradient boosting (Raul).** Con pérdida de Poisson es el mejor modelo individual en la CV del conteo (0.267) y en
   el holdout supera a la media de 28 días; la grilla eligió la configuración más regularizada. En el triaje gana en
   macro-F1 (+0.013) pero detecta menos fatales (28 %).
8. **MLP y ensamble (Fernando).** La red sola empata con la logística: con 8 105 accidentes no hay datos para
   aprovecharla (Emu et al. usan 5.8 millones) y varía con la semilla. El ensamble promedia los cuatro modelos: es el
   mejor en el conteo (0.268 / 0.296) y en el triaje queda en medio (37 % de fatales) sin bajar del baseline.
9. **Comparación (Marco).** Cada panel va de la regla sin ML al baseline y al modelo elegido. El salto grande
   está en la gravedad (+98 % sobre el Dummy). En HAY_FATAL todos los modelos quedan entre 0.13 y 0.14.
10. **Ablación (Marco).** Quitamos un grupo de variables a la vez. Sin el clima el modelo casi no cambia (+0.001 de D²);
   sin el tipo de choque el triaje cae 0.042. La lluvia del día no agrega información una vez que se conocen el lugar
   y la historia.
11. **Recomendación (Fernando).** Ensamble para el conteo, logística para anticipar un fatal, Random Forest o ensamble
    para el triaje según el costo de perder un fatal. El límite está en los datos: sin vehículo, ocupantes ni tráfico.

## Preguntas probables (anexo A1–A2)

| Pregunta | Responde | Respuesta corta |
|---|---|---|
| ¿Por qué validación temporal? | Angel | La serie tiene memoria y cambia de nivel con la cuarentena; con pliegues aleatorios el modelo vería el futuro. La historia se calcula hasta *t − 1*. |
| ¿Cómo trataron el desbalance? | Angel | PR-AUC y macro-F1, pesos balanceados en GRAVEDAD; en HAY_FATAL sin pesos para que las probabilidades queden calibradas (Brier 0.060). |
| ¿Por qué no exactitud? | Angel | «Nunca fatal» acierta el 88 % y «0 accidentes» el 66 % de los días. |
| ¿Hubo fuga de información? | Raul | Encoding dentro de cada pliegue, víctimas solo en el objetivo, hiperparámetros sin mirar holdout ni test. |
| ¿Son significativas las diferencias? | Marco | Comparación pareada con IC 95 %: sí en GRAVEDAD (RF, ensamble, GBM); no en N_ACC ni HAY_FATAL en la CV. |
| ¿Por qué ganó el RF en el triaje? | Angel | Interacciones entre tipo de choque, lugar y luz (la ablación lo confirma). |
| ¿Por qué el ensamble no gana en todo? | Fernando | En el conteo los miembros predicen casi lo mismo (correlación 0.96–0.98); en fatales discrepan y ahí sí combina. |
| ¿Por qué la lluvia no importa? | Marco | Reanálisis en celdas de ~11 km; el efecto queda absorbido por la estación y el lugar. Babaoglu también halla muertes con clima no extremo. |
| ¿Por qué el GBM eligió pocos árboles? | Raul | Con 10 776 departamento-días, más flexibilidad sobreajusta; las 5 mejores configuraciones están a < 0.01. |
| ¿Por qué el MLP no rinde? | Fernando | Pocos datos (8 105 frente a 5.8 M) y sin variables del vehículo; varía ± 0.009 con la semilla. |

## Ensayo

Pendiente: ensayo cronometrado con los cuatro integrantes. Anotar aquí el tiempo real y los ajustes.

## Exportar a PDF

Desde la raíz del repo, con Chrome:

```
chrome --headless=new --no-pdf-header-footer --allow-file-access-from-files --print-to-pdf=presentation.pdf "file:///<ruta al repo>/presentación/final/presentacion_final.html"
```
