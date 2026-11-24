---
title: "Artículo 4 · Mixed Effect Gradient Boosting (MEGB) para estimación en áreas pequeñas"
author: "Messer y Schmid · arXiv 2025"
lang: es
---

*Resumen de lectura. Messer, P. y Schmid, T. (2025). Gradient Boosting for Hierarchical Data in Small Area Estimation. arXiv:2406.04256v2 [stat.ME], 21 de enero de 2025 (preprint, Universidad de Bamberg). Extiende al boosting el MERF para áreas pequeñas de Krennmair y Schmid (2022, *JRSS C*). La réplica reducida está en `ejemplos/megb_articulo.py` (`xgboost`, `gpboost` 1.8.0, `scikit-learn`).*

# La idea en un párrafo

En **estimación en áreas pequeñas** (SAE) se quiere la media (o el total) de una variable en muchas áreas, por ejemplo municipios. La encuesta tiene pocas observaciones por área, o ninguna, y el censo tiene las covariables de toda la población. El estándar a nivel unidad es el modelo de Battese, Harter y Fuller (BHF): un modelo lineal mixto con intercepto aleatorio por área. MEGB sustituye la parte lineal por **XGBoost** y estima el modelo con el bucle alternado del MERF: ajusta el boosting con el efecto de área restado, ajusta un modelo mixto a los residuos y repite. Luego promedia las predicciones sobre el censo y estima el ECM por área con un **bootstrap por bloques de efectos aleatorios**.

# El problema de SAE en dos minutos

- **Población** $U$ con $D$ áreas $U_d$ de tamaño $N_d$. El censo da $x_{di}$ para todas las unidades; $y_{di}$ solo se observa en la muestra $s_d$, con $n_d$ pequeño (o $n_d=0$ en áreas fuera de muestra).
- **Objetivo:** $\mu_d=N_d^{-1}\sum_{i\in U_d}y_{di}$.
- **Estimador directo** (Horvitz-Thompson o la media muestral): insesgado bajo el diseño, pero con varianza enorme si $n_d$ es chico, e inexistente si $n_d=0$.
- **Estimadores basados en modelo:** se “toma fuerza” de las demás áreas a través del modelo. BHF:

$$y_{di}=x_{di}^\top\beta+v_d+e_{di},\qquad v_d\sim N(0,\sigma_v^2),\quad e_{di}\sim N(0,\sigma_e^2)$$

El EBLUP de $\mu_d$ combina la parte fija con el efecto de área predicho, $\hat v_d=\gamma_d(\bar y_d-\bar x_d^\top\hat\beta)$ con $\gamma_d=\sigma_v^2/(\sigma_v^2+\sigma_e^2/n_d)$. Es el mismo encogimiento por tamaño de grupo que en las notas 9 a 11.

- **Lo que falla:** si la relación $x\to y$ no es lineal o tiene interacciones, la parte fija está mal especificada. Transformar (EBP con Box-Cox) ayuda con la asimetría, no con las interacciones.

# El modelo y el algoritmo

$$y_d=f(X_d)+Z_dv_d+\epsilon_d,\qquad v_d\sim N(0,\sigma_v^2),\quad \epsilon_d\sim N(0,\sigma_\epsilon^2 I_{n_d})$$

con $f$ un XGBoost y un solo intercepto aleatorio por área ($Z_d=\mathbf 1_{n_d}$).

**Algoritmo 1 del artículo.** Inicializar $\hat v=0$ y $\beta_0=0$. Repetir:

1. $y^*=y-\beta_0-Z\hat v$.
2. Ajustar XGBoost a $y^*$, con el número de árboles por paro temprano en una submuestra de validación: $\hat f$.
3. $y^{**}=y-\hat f(X)$.
4. Ajustar un modelo lineal mixto con intercepto a $y^{**}$: da $\beta_0$, $\hat\sigma_v^2$, $\hat\sigma_\epsilon^2$ y los BLUP $\hat v_d$.
5. Parar si el cambio relativo de la log-verosimilitud generalizada (GLL de Hajjem et al.) es menor que una tolerancia.

**Media de área:**

$$\hat\mu_d^{\text{MEGB}}=N_d^{-1}\sum_{i\in U_d}\hat f(x_{di})+\hat v_d$$

promediando sobre **todo el censo** del área. Para áreas no muestreadas, $\hat v_d=0$ y queda solo el promedio de $\hat f$.

**Qué es este algoritmo, en realidad.** El artículo lo llama EM, pero el apéndice A de GPBoost (nota 9) ya señaló que el algoritmo de MERF no es un EM propiamente dicho. Se entiende mejor como **descenso por bloques** sobre la verosimilitud conjunta de $(f,v,\theta)$:

$$\tfrac1{\sigma_\epsilon^2}\lVert y-f-Zv\rVert^2+\tfrac1{\sigma_v^2}\lVert v\rVert^2+\log\det(\cdot)$$

El paso 2 minimiza en $f$ con $v$ fijo, usando mínimos cuadrados **ordinarios**, porque con $v$ fijo los errores son independientes. El paso 4 minimiza en $(v,\theta)$ con $f$ fijo. Diferencias con GPBoost:

| | MEGB / MERF | GPBoost |
|:--|:--|:--|
| ajuste de $f$ | un modelo completo **por iteración externa** | un solo boosting; un árbol por iteración |
| $\theta$ | se reestima tras cada modelo completo | se reestima antes de cada árbol |
| costo | iteraciones × (boosting completo + paro temprano) | un boosting + optimizaciones de $\theta$ con arranque en caliente |
| criterio de paro | cambio relativo de la GLL | número de árboles (paro temprano) |

Hay **dos errores de notación** en el artículo. El estimador de los efectos aparece como $\hat\vartheta=GZ^\top G^{-1}(y-\hat f)$; el correcto es $GZ^\top V^{-1}(y-\hat f)$, con $V=ZGZ^\top+R$. Y el error unitario se escribe $\epsilon_i\sim N(0,G)$, cuando $G$ es la varianza del efecto de área.

# Incertidumbre: bootstrap por bloques de efectos aleatorios (REBB)

Adaptado de Chambers y Chandra (2013):

1. **Residuos marginales:** $\hat e_{di}=y_{di}-\hat f(x_{di})$. **Nivel 2:** $\bar r_d=n_d^{-1}\sum_i\hat e_{di}$. **Nivel 1:** $\hat r_{di}=\hat e_{di}-\bar r_d$.
2. Reescalar los residuos de nivel 1 para que su varianza sea $\hat\sigma_\epsilon^2$ y los de nivel 2 para que sea $\hat\sigma_v^2$, y centrarlos.
3. Para $b=1,\dots,B$: muestrear con reemplazo $D$ residuos de nivel 2 y, dentro de cada área, residuos de nivel 1. Generar una **población** bootstrap $y^{(b)}_{di}=\hat f(x_{di})+\bar r^{(b)}_d+\hat r^{(b)}_{di}$ y su media verdadera $\mu_d^{(b)}$. Extraer una muestra con los mismos $n_d$, reajustar MEGB y obtener $\hat\mu_d^{(b)}$.
4. $\widehat{\mathrm{ECM}}_d=B^{-1}\sum_b(\hat\mu_d^{(b)}-\mu_d^{(b)})^2$.

No supone normalidad, pero **no refleja la incertidumbre de haber elegido $\hat f$**: $\hat f$ se toma como la verdad que genera las poblaciones bootstrap. Cada réplica reajusta MEGB completo con los hiperparámetros fijos: $B=100$ réplicas por 10 a 25 iteraciones externas.

# Resultados

**Simulación basada en modelo** (la de Krennmair y Schmid 2022). $D=50$ áreas de $N_d=1000$, muestra estratificada de $n=1410$ ($n_d$ entre 6 y 49), 200 repeticiones. Cuatro escenarios combinan lineal o complejo ($-500x_1x_2-250x_2^2$) con error normal o Pareto(3, 800), asimétrico.

XGBoost con hiperparámetros **fijos** (tasa 0.01, profundidad 3, `min_child_weight` 3, `subsample` 0.5, $\lambda=1$, $\gamma=0.9$); solo el número de árboles por paro temprano.

RRMSE de las medias de área (%, media sobre áreas):

| | Lineal-Normal | Complejo-Normal | Lineal-Pareto | Complejo-Pareto |
|:--|:--|:--|:--|:--|
| BHF | **3.91** | 3.65 | 3.77 | 2.76 |
| EBP-BC | 3.99 | 3.66 | **3.51** | 2.63 |
| MERF | 4.19 | 2.17 | 4.02 | 1.37 |
| MEGB | 3.98 | **1.54** | 3.77 | **1.09** |

- Con $f$ lineal, BHF y EBP ganan, como debe ser, pero MEGB queda cerca y por delante de MERF.
- Con $f$ compleja, MEGB reduce el RRMSE a menos de la mitad del de BHF y entre 21 y 29 % respecto a MERF.
- El error Pareto no afecta a los métodos de árboles; EBP-BC gana en Lineal-Pareto gracias a la transformación Box-Cox.
- Sesgos relativos por debajo de 0.6 % en todos.

**ECM bootstrap.** Sesgo relativo de la raíz del ECM estimado prácticamente nulo (entre −0.07 % y −0.12 %), pero con un RRMSE alto: 16–17 % con error normal y 37–40 % con Pareto. **En promedio sobre áreas acierta; para un área concreta es ruidoso.**

**Simulación basada en diseño: ingreso laboral en Nuevo León.** El censo tiene 54 848 hogares en 51 municipios. La encuesta (ingreso y gasto de los hogares) tiene 1435 hogares en 21 municipios (de 5 a 342 por municipio). Quedan 30 municipios **fuera de muestra**. 100 repeticiones.

| RMSE (media) | en muestra | fuera de muestra | total |
|:--|:--|:--|:--|
| EBP-BC | 316.7 | 417.6 | 376.0 |
| MERF | 256.6 | 316.2 | 291.6 |
| MEGB | **219.9** | **257.4** | **242.0** |

MEGB gana en media en todos los grupos y también tiene el menor sesgo. En la **mediana** en muestra gana MERF (128.0 frente a 138.8). El ECM bootstrap es más impreciso aquí: RRMSE de 57 % en muestra y 77 % fuera de muestra. Fuera de muestra no hay $\hat v_d$ que ayude, y es esperable.

**Ajuste de hiperparámetros** (apéndice: búsqueda secuencial en rejilla, 20 repeticiones en el diseño basado en modelo y 50 en el basado en diseño): mejora poco, por ejemplo de 1.09 a 1.01 en Complejo-Pareto. Ojo: las tablas del apéndice ponen los resultados de 200 repeticiones sin ajuste junto a los de 20 con ajuste, así que la comparación no es del todo pareja.

# Réplica reducida

`ejemplos/megb_articulo.py`: escenarios Lineal-Normal y Complejo-Normal, 20 repeticiones, $n_d$ de 6 a 49 ($n=1375$). Estimadores:

- **Directo:** media muestral del área.
- **BHF:** modelo lineal mixto.
- **MEGB:** el algoritmo 1 con los hiperparámetros del artículo.
- **MERF:** el mismo bucle con un bosque aleatorio.
- **GPBoost** (nota 9): estimación conjunta, mismos árboles salvo el submuestreo, que GPBoost no admite.

RRMSE de las medias de área (%, media / mediana sobre las 50 áreas):

| | Lineal-Normal | Complejo-Normal |
|:--|:--|:--|
| Directo | 9.58 / 8.88 | 7.09 / 6.52 |
| BHF | **3.77** / 3.46 | 3.66 / 3.63 |
| MERF | 4.02 / 3.72 | 2.35 / 2.29 |
| MEGB | 3.90 / 3.64 | **1.54** / 1.44 |
| GPBoost | 3.92 / 3.64 | 1.61 / 1.55 |

Se reproduce el artículo casi al decimal. En Lineal-Normal: 3.77 / 3.90 / 4.02 para BHF / MEGB / MERF, frente a 3.91 / 3.98 / 4.19. En Complejo-Normal: BHF 3.66 y MEGB 1.54, frente a 3.65 y 1.54; mi MERF sale algo peor que el del artículo (2.35 frente a 2.17), probablemente por los hiperparámetros del bosque. GPBoost, estimado de forma conjunta en una sola pasada, queda a la par de MEGB: un poco peor en el escenario complejo (1.61) e igual en el lineal.

Tres cosas que salieron al implementarlo:

- **El criterio de paro casi nunca se cumple.** Con tolerancia relativa de $10^{-6}$ en la verosimilitud, MEGB agotó las 25 iteraciones en **todas** las repeticiones; MERF paró en 17 iteraciones en promedio en el escenario lineal y en 24 en el complejo. La razón: con `subsample=0.5` y una submuestra de validación nueva en cada iteración, el XGBoost de cada vuelta es distinto aunque nada haya cambiado, y la verosimilitud oscila sin converger. En la práctica conviene fijar la semilla o usar un número fijo de iteraciones externas.
- **$\gamma=0.9$ no hace nada.** En XGBoost, $\gamma$ es la reducción mínima de la **pérdida** para aceptar un corte. Con $y$ del orden de $10^4$, las ganancias se miden en millones, así que 0.9 equivale a $\gamma=0$.
- **La escala importa para GPBoost.** Con $y$ en miles y tasa 0.01, el paso de gradiente de GPBoost ($\Psi^{-1}(y-F)$, del orden de $1/\sigma^2\approx10^{-6}$) casi no movía los árboles: el RMSE salía igual al de un modelo sin covariables. Estandarizar $y$ lo resolvió. Es justo la advertencia de la nota 9: el paso de gradiente no es invariante a la escala; el de Newton sí.

# Lectura crítica

- **El método es MERF con XGBoost.** Lo nuevo es empírico: que el boosting le gane al bosque en este contexto, más la adaptación del bootstrap. No hay teoría de convergencia ni de consistencia.
- **Falta la comparación natural: GPBoost** (2020; publicado en 2022), que estima exactamente el mismo modelo de forma conjunta y más barata. En la réplica, GPBoost y MEGB quedan cerca.
- **Hiperparámetros fijos por diseño** (salvo el número de árboles). Es honesto como evaluación, pero algunos valores no tienen sentido en la escala de los datos, como $\gamma$.
- **El estimador (4) ignora las $y$ observadas.** Promedia $\hat f$ sobre todo el censo, incluidas las unidades muestreadas. Con fracciones de muestreo de 0.6–5 % no importa; con fracciones grandes convendría el estimador “censo menos muestra más observados”, como en el EBLUP.
- **El ECM por área es poco preciso** (RRMSE de 16 % a 77 %). Para publicar mapas con intervalos por municipio, eso pesa. Una alternativa sin modelo para la variabilidad de cada unidad es la conformal de la nota 8. Para medias de área no es directa, pero el problema de fondo es el mismo: garantías por grupo con pocos datos.
- **No hay intervalos para áreas fuera de muestra más allá del bootstrap**, que ahí es el más impreciso.

# Para conectar con tus notas

- **Notas 9 y 10:** mismo modelo; MEGB es la versión “por bloques con modelos completos” y GPBoost la versión “conjunta árbol por árbol”.
- **Nota 11:** el BLUP $\hat v_d=\gamma_d(\bar y_d-\bar f_d)$ es HS con un solo nivel: encoger la media del grupo hacia la predicción global con un factor que depende de $n_d$.
- **Nota 8:** SAE es “una garantía por tipo de producto” llevada a medias en lugar de observaciones individuales. En tu caso, estimar el precio medio por modelo y región con pocas ventas por celda, o con celdas sin ventas, es literalmente un problema de áreas pequeñas.
- **Nota 7:** el promedio sobre el censo traslada el error de $\hat f$ a la media del área. El sesgo de $\hat f$ en una región del espacio de covariables no se promedia a cero, y por eso el sesgo relativo de MEGB no es exactamente cero.
