# Guion · Presentación final (máx. 5 min)

Hilo: una pregunta → los datos → el giro del EDA → el baseline → por qué no da más → el Modelo 1 → la respuesta.

| # | Diapositiva | Tiempo | Qué decir |
|---|---|---|---|
| 0 | Portada: la pregunta | 0:20 | Perú tiene costa, sierra y selva; la lluvia baja la fricción y la visibilidad. ¿Eso vuelve más mortales los accidentes? 8 110 accidentes, 1 de cada 9 fatal. |
| 1 | Problema y datos | 0:45 | El registro no tiene coordenadas ni clima: ubicamos cada accidente por vía y km y medimos la lluvia en ese punto (la capital se equivocaba en 1 de cada 5). Predecimos si un accidente es fatal. |
| 2 | EDA | 0:50 | Con lluvia, los accidentes no son más fatales. La sierra llueve más y sus accidentes son menos letales; al controlar por altitud, la lluvia deja de importar (OR 0.98, p = 0.67). |
| 3 | Baseline | 0:45 | Regresión logística con 9 variables. Usamos PR-AUC porque decir siempre "no fatal" acierta el 88 %. Da 1.4 veces el azar: encuentra 8 de cada 10 fatales, pero solo 13 de cada 100 alertas aciertan. ¿Por qué no más? |
| 4 | Techo de los datos | 0:50 | Ni la peor combinación de lugar, hora y altitud pasa de 29 % de fatales: antes del accidente no hay una situación de alto riesgo. Lo que decide es cómo ocurre el choque (57 % con el tipo de choque). Por eso los modelos complejos no ganan. |
| 5 | Modelo 1 | 0:45 | Nos quedamos con las 3 variables que tienen la señal: departamento, hora y altitud. Igual desempeño que con 9 (15 de 25 particiones mejor) y la lluvia suma +0.000. Pesa la madrugada, la costa y algunos departamentos. |
| 6 | Respuesta | 0:25 | La lluvia no vuelve más mortales los accidentes; importan dónde, cuándo y, sobre todo, cómo ocurre el choque. Para ir más lejos hacen falta datos del choque y del tráfico. |
