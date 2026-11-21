---
title: "CatBoost: estadísticas de target ordenadas, boosting ordenado y árboles simétricos"
author: Javier
lang: es
---

*Cuarta nota. Supone la nota de fundamentos y la de XGBoost. Los experimentos se corrieron con `catboost` 1.2.10 en CPU; el script está en `ejemplos/catboost_experimentos.py` del repositorio.*

# El problema que ataca CatBoost

Prokhorenkova et al. (2018) identifican un mismo defecto en dos lugares del gradient boosting habitual: **se usa la respuesta $y_k$ de una observación para construir algo que después se evalúa sobre esa misma observación**. Eso produce un desplazamiento entre la distribución en entrenamiento y la distribución en datos nuevos.

1. En la **codificación de categóricas por target**: si el valor codificado de $x_k$ usa $y_k$, el modelo ve una variable que en entrenamiento “sabe” la respuesta.
2. En el **cálculo de gradientes**: el gradiente $g_k$ se evalúa con un modelo $F^{t-1}$ que ya usó $y_k$ para ajustarse, así que $g_k$ está sesgado respecto al gradiente en un punto nuevo. Los autores lo llaman ***prediction shift***.

La solución en ambos casos es la misma idea: **imponer un orden aleatorio a los datos y, para cada observación, usar solo información de las anteriores**. Es la lógica del aprendizaje en línea aplicada dentro del entrenamiento.

Además, CatBoost usa por defecto **árboles simétricos** (*oblivious trees*) y construye **combinaciones de categóricas**.

# Estadísticas de target (TS)

Una variable categórica $x^{i}$ con muchos niveles se reemplaza por un número: una estimación de $\mathbb E[Y\mid x^i = c]$ para el nivel $c$. Es una *estadística de target* (TS). La forma general, con suavizado hacia una media previa $p$ con peso $a>0$, es

$$\hat x^{i}_k = \frac{\sum_{j\in \mathcal D_k} \mathbf 1\{x^i_j = x^i_k\}\, y_j + a\,p}{\sum_{j\in \mathcal D_k} \mathbf 1\{x^i_j = x^i_k\} + a}$$

donde $\mathcal D_k$ es el conjunto de observaciones que se permite usar para codificar la observación $k$. Todo depende de esa elección.

## TS codicioso: $\mathcal D_k$ = todo el entrenamiento

Es la codificación por media más común. El problema es la **fuga de target**: el valor $\hat x_k$ contiene a $y_k$.

**Caso extremo.** Sea un nivel que aparece una sola vez. Su TS es $\hat x_k=\frac{y_k + ap}{1+a}$, una función creciente de $y_k$. Un árbol con un solo corte en $\hat x$ separa perfectamente esas observaciones en entrenamiento. En datos nuevos, el nivel aporta $p$ o un valor sin relación con la nueva $y$.

La condición que se rompe es

$$\mathbb E\big[\hat x^i \mid y\big] \;=\; \mathbb E\big[\hat x^i_k \mid y_k\big]$$

es decir, que la variable codificada tenga la misma relación con $y$ en entrenamiento que en prueba.

## TS *leave-one-out*: $\mathcal D_k$ = todo menos $k$

Quitar a $k$ parece arreglarlo, pero no. Tomemos un nivel con $n_c$ observaciones y suma de respuestas $S_c$. Entonces $\hat x_k = \frac{S_c - y_k + ap}{n_c - 1 + a}$. Dentro del nivel, $S_c$ es fijo, y $\hat x_k$ es una función **decreciente** de $y_k$: toma un valor para las $y_k=1$ y otro, mayor, para las $y_k=0$. Un corte en $\hat x$ recupera $y_k$ perfectamente dentro de cada nivel. La fuga cambió de signo, pero sigue ahí.

## TS ordenado (CatBoost)

Se toma una permutación aleatoria $\sigma$ de los datos de entrenamiento y se usa solo la “historia”:

$$\mathcal D_k = \{\, j : \sigma(j) < \sigma(k)\,\}$$

Ahora $\hat x_k$ depende solo de respuestas anteriores, que son independientes de $y_k$ (si las observaciones son independientes). La relación entre $\hat x$ e $y$ en entrenamiento es la misma que en prueba, donde se usa **todo** el entrenamiento como historia.

El costo es la varianza: las primeras observaciones de la permutación tienen poca historia y una TS ruidosa. CatBoost usa varias permutaciones (`permutation_count=4` por defecto) para distintos pasos del entrenamiento.

**Detalles de la implementación por defecto** (de `get_all_params()` en la versión 1.2.10):

- Para clasificación binaria, se calculan TS con **tres** previas $p\in\{0,\,0.5,\,1\}$, cada una con peso $a=1$ (`Prior=0/1`, `Prior=0.5/1`, `Prior=1/1`). Cada una es una variable nueva.
- También se calcula un **contador** (`Counter`): la frecuencia del nivel, sin usar $y$.
- En regresión, el target se **binariza** antes de calcular la TS (`TargetBorderCount=1`, `TargetBorderType=MinEntropy`): la TS estima la probabilidad de estar por arriba de un umbral elegido.
- Variables con a lo sumo `one_hot_max_size` niveles (2 por defecto en CPU) se codifican *one-hot* en lugar de TS.

## Experimento: fuga con una categórica de puro ruido

Clasificación binaria con $P(y=1)=\sigma(1.5x_1-x_2)$, más una categórica de 1 000 niveles **sin relación con $y$**. Entrenamiento con 4 000 filas (unas 4 por nivel), prueba con 20 000. Las TS se calcularon a mano con $a=1$ y se pasaron a LightGBM (300 árboles, `num_leaves=15`).

| Método | AUC ent. | AUC prueba | log-loss prueba | peso de la categórica |
|:--|:--|:--|:--|:--|
| Sin la categórica | 0.897 | 0.828 | 0.520 | — |
| TS codicioso | 0.952 | **0.708** | **0.938** | 35 % |
| TS *leave-one-out* | 0.911 | 0.827 | 0.521 | 8 % |
| CatBoost (TS ordenado) | 0.868 | 0.838 | 0.498 | 12 % |

“Peso” es la fracción de la importancia total que el modelo asigna a la categórica (ganancia en LightGBM, `PredictionValuesChange` en CatBoost).

- El TS codicioso es desastroso: sube el AUC de entrenamiento y hunde el de prueba. El modelo le da un tercio de la importancia a una variable que es ruido.
- El *leave-one-out* casi no dañó aquí. La fuga existe, pero explotarla requiere cortes dentro de cada nivel, y con ~4 observaciones por nivel y árboles de 15 hojas el modelo no la aprovechó mucho. No es una garantía.
- CatBoost tuvo el mejor resultado de prueba, pero también es otro algoritmo (árboles simétricos, otro muestreo), así que la comparación no aísla solo el efecto de la TS.
- **Importancia en prueba.** Con `get_feature_importance(type="LossFunctionChange")` sobre los datos de **prueba**, la categórica obtiene $-0.004$ (empeora ligeramente la pérdida), frente a $0.141$ para $x_1$ y $0.079$ para $x_2$. La importancia basada en el uso del modelo (12 %) no es lo mismo que la utilidad predictiva. Volveremos a esto en las notas de explicabilidad.

# *Prediction shift* y boosting ordenado

En la etapa $t$, el gradiente de la observación $k$ es $g_k = \partial \ell(y_k, F)/\partial F$ evaluado en $F^{t-1}(\mathbf x_k)$. Pero $F^{t-1}$ se entrenó con $(\mathbf x_k, y_k)$, así que

la distribución de $F^{t-1}(\mathbf x_k)$ dado $\mathbf x_k$ no es la misma que la de $F^{t-1}(\mathbf x)$ dado $\mathbf x$ para un punto de prueba $\mathbf x$. Los residuos de entrenamiento son, en promedio, más pequeños que los de prueba, y el árbol siguiente se ajusta a una señal sesgada. El artículo analiza un caso simple de regresión con dos variables binarias y muestra que el sesgo es del orden de $1/n$.

**Boosting ordenado (idealizado).** Con una permutación $\sigma$, mantener $n$ modelos: $M_j$ se entrena solo con las primeras $j$ observaciones. El residuo de la observación $k$ se calcula con $M_{\sigma(k)-1}$, que nunca vio a $k$. Es costoso: $n$ modelos.

**Implementación práctica.** El artículo mantiene modelos de apoyo solo para prefijos de longitud $2^j$, es decir $O(\log n)$ por permutación, y comparte la **estructura** del árbol: la estructura se elige con los gradientes “honestos”, y los valores de hoja del modelo final se calculan con todos los datos. Se activa con `boosting_type="Ordered"`.

**Valor por defecto.** En la versión 1.2.10 en CPU, `get_all_params()` reportó `boosting_type="Plain"` tanto con 2 000 como con 60 000 filas. La documentación indica que en GPU se usa `Ordered` con 50 000 objetos o menos (salvo multiclase). La TS ordenada se usa siempre, sin importar `boosting_type`.

**Experimento.** Friedman #1 (solo numéricas), 500 árboles, `learning_rate=0.05`, `depth=6`, 8 semillas, prueba con 10 000:

| $n$ | Plain | Ordered | Ordered − Plain (pareado) |
|:--|:--|:--|:--|
| 200 | 1.916 ± 0.108 | 1.913 ± 0.093 | −0.004 ± 0.026 |
| 1000 | 1.216 ± 0.016 | 1.229 ± 0.017 | +0.013 ± 0.016 |

Con 20 000 filas y 300 árboles: 0.7 s en `Plain` frente a 2.0 s en `Ordered`.

En este experimento, sin categóricas, el boosting ordenado **no ayudó** y triplicó el tiempo. El artículo reporta mejoras en conjuntos pequeños con categóricas; el sesgo de $O(1/n)$ explica por qué el efecto es pequeño cuando $n$ crece. Pruébalo con validación cruzada en tus datos en lugar de asumirlo.

# Árboles simétricos

Un árbol **simétrico** (*oblivious*) usa **la misma condición en todos los nodos de un mismo nivel**. Un árbol de profundidad $d$ es una lista de $d$ condiciones $(j_1,s_1),\dots,(j_d,s_d)$ y una tabla de $2^d$ valores:

$$f(\mathbf x) = w\Big[\textstyle\sum_{l=1}^{d} 2^{\,l-1}\,\mathbf 1\{x_{j_l} > s_l\}\Big]$$

El índice de la hoja es el número binario formado por los resultados de las $d$ comparaciones.

**Consecuencias.**

- **Predicción muy rápida**: $d$ comparaciones y un acceso a tabla, sin ramificaciones.
- **Fuerte regularización estructural**: con $d=6$ hay 64 hojas pero solo 6 decisiones. El árbol no puede refinar una región sin partir todas las demás con la misma condición.
- **Interacciones**: cada hoja combina las $d$ condiciones, así que todas las variables del árbol interactúan entre sí.

**Ejemplo real.** El primer árbol de un modelo con `depth=6` entrenado sobre Friedman #1 tiene 6 cortes y 64 valores de hoja. Los cortes (índice de variable, umbral) son: $(3, 0.514)$, $(0, 0.320)$, $(1, 0.298)$, $(4, 0.478)$, $(2, 0.139)$, $(3, 0.354)$. Las variables 0 a 4 son las relevantes en Friedman #1, y la 3 aparece dos veces con umbrales distintos.

**Cómo se elige cada nivel.** Para cada candidato $(j,s)$ se parten **todas** las hojas actuales y se evalúa el criterio sumando sobre ellas. La función de evaluación por defecto en CPU es `score_function="Cosine"`, que mide el coseno entre el vector de gradientes y el vector de predicciones del árbol candidato (es el criterio del algoritmo del artículo); `L2` es la alternativa en CPU.

`grow_policy` admite `Depthwise` y `Lossguide` (crecimiento por hojas, con `max_leaves`) para árboles no simétricos. La documentación advierte no usar `Cosine` con `Lossguide`.

# Combinaciones de categóricas

Una variable categórica sola no captura efectos que dependen de **pares** de niveles. CatBoost construye combinaciones de forma voraz: al elegir los cortes de un árbol, combina las categóricas ya usadas en ese árbol con otras, y calcula una TS ordenada para la combinación como si fuera una nueva categórica. `max_ctr_complexity` limita cuántas categóricas puede tener una combinación.

**Experimento: efecto solo por interacción.** Dos categóricas de 6 niveles cada una, con efecto sobre el log-odds dado por una tabla $6\times6$ con **medias cero por fila y por columna** (no hay efectos principales), más $0.5x$. Entrenamiento 3 000, prueba 30 000. Log-loss de prueba (sin información: 0.6935):

| `max_ctr_complexity` | `one_hot_max_size=2` | `one_hot_max_size=10` |
|:--|:--|:--|
| 1 | 0.6554 | 0.6078 |
| 2 | 0.6020 | 0.6078 |
| 4 | 0.6019 | 0.6078 |

- Sin combinaciones (`max_ctr_complexity=1`) y con TS, cada categórica sola casi no tiene señal: 0.655.
- Permitir pares baja la pérdida a 0.602.
- Con *one-hot* (6 niveles $\le$ 10), el árbol puede formar la interacción con cortes sucesivos y el parámetro deja de importar. Las combinaciones solo aplican a categóricas codificadas por TS.
- El valor efectivo del parámetro lo ajusta CatBoost: en este experimento `get_all_params()` reportó 4, y en pruebas de regresión con pocas iteraciones reportó 1. Revisa el valor efectivo en tu modelo.

# Valores de hoja y regularización

- **Valores de hoja.** Se estiman con pasos de Newton o de gradiente (`leaf_estimation_method`) con penalización L2 `l2_leaf_reg` (3 por defecto). En la versión 1.2.10, `get_all_params()` reportó `Newton` con una iteración para RMSE y para Logloss, y `Exact` para MAE.
- **Ruido en la elección de cortes.** `random_strength` (1 por defecto, solo CPU) multiplica la varianza de un ruido normal de media cero que se suma a los puntajes de los cortes. Esa varianza disminuye a lo largo del entrenamiento.
- **Submuestreo.** En CPU, para objetivos distintos de multiclase, el defecto es `bootstrap_type="MVS"` (*minimal variance sampling*, Ibragimov y Gusev 2019) con `subsample=0.8`. MVS muestrea con probabilidad creciente en la magnitud del gradiente regularizado y repondera, en el mismo espíritu que GOSS. La alternativa `Bayesian` asigna pesos aleatorios con `bagging_temperature` (1 equivale a pesos exponenciales; 0, a pesos iguales).
- **Tasa de aprendizaje automática.** Si no se fija `learning_rate` (ni `leaf_estimation_iterations`, `leaf_estimation_method` o `l2_leaf_reg`), CatBoost la elige según el número de iteraciones para Logloss, MultiClass y RMSE. En mis pruebas: 0.031 con 500 iteraciones en clasificación y 0.078 con 1 000 en regresión con 60 000 filas.
- **Orden temporal.** Con `has_time=True` se usa el orden de entrada en lugar de permutaciones aleatorias. Si los datos tienen orden temporal, es la opción correcta: la “historia” es el pasado real.

## Experimento: `depth` y `l2_leaf_reg`

Friedman #1, entrenamiento 2 000, validación 2 000 con parada temprana, prueba 20 000. RMSE de prueba:

| `depth` | `l2_leaf_reg`=1 | 3 | 30 |
|:--|:--|:--|:--|
| 2 | 1.128 | 1.125 | 1.136 |
| 4 | 1.097 | **1.096** | 1.105 |
| 6 | 1.110 | 1.110 | 1.127 |
| 8 | 1.158 | 1.160 | 1.171 |
| 10 | 1.302 | 1.287 | 1.248 |

- La profundidad es el parámetro principal: aquí el óptimo está en 4, y con 10 (1 024 hojas por árbol) el error sube mucho.
- `l2_leaf_reg` alto exige más iteraciones (unas 1 400 frente a 700 con `depth=4`) y solo ayuda cuando el árbol es demasiado profundo.
- Comparado con LightGBM en los mismos datos (nota 3), CatBoost con valores razonables queda en 1.10, mejor que el 1.18 de LightGBM con 4 hojas. Los árboles simétricos funcionan bien en un problema casi aditivo como este. Es una comparación de un solo problema.

# Exploración del modelo entrenado

CatBoost incluye herramientas propias para inspeccionar el modelo:

| Herramienta | Qué da |
|:--|:--|
| `get_feature_importance(type="PredictionValuesChange")` | Cuánto cambia la predicción, en promedio, al cambiar la variable (defecto) |
| `get_feature_importance(type="LossFunctionChange", data=pool)` | Cuánto empeora la pérdida en `pool` sin la variable |
| `get_feature_importance(type="Interaction")` | Fuerza de interacción por pares de variables |
| `get_feature_importance(type="ShapValues", data=pool)` | Contribuciones SHAP por observación |
| `calc_feature_statistics(X, y, feature)` | Media de $y$, de la predicción y número de datos por intervalo de una variable |
| `plot_tree(i)` | Dibujo del árbol $i$ |
| `save_model(..., format="json")` | Estructura completa: cortes, valores de hoja y TS |

# Mapa de hiperparámetros

| Parámetro (defecto CPU) | Efecto |
|:--|:--|
| `iterations` (1000) | Número de árboles; usar con parada temprana |
| `learning_rate` (automática) | Ver arriba |
| `depth` (6) | Tamaño del árbol simétrico: $2^d$ hojas |
| `l2_leaf_reg` (3) | Encoge los valores de hoja |
| `random_strength` (1) | Ruido en la elección de cortes |
| `bootstrap_type` (MVS), `subsample` (0.8) | Submuestreo |
| `one_hot_max_size` (2) | Niveles hasta los que se usa one-hot en lugar de TS |
| `max_ctr_complexity` | Tamaño máximo de las combinaciones de categóricas |
| `border_count` (254) | Intervalos para numéricas |
| `boosting_type` (Plain en CPU) | `Ordered` corrige el *prediction shift*, más lento |
| `grow_policy` (SymmetricTree) | `Depthwise` o `Lossguide` para árboles no simétricos |
| `has_time` (False) | Usar el orden de los datos como historia |

# Ejercicios

1. **Fuga del *leave-one-out*.** Para un nivel con $n_c$ observaciones, $m$ de ellas con $y=1$, calcula los dos valores que toma la TS *leave-one-out* y muestra que un corte entre ellos clasifica perfectamente el nivel.
2. **Insesgadez de la TS ordenada.** Bajo independencia entre observaciones, muestra que $\mathbb E[\hat x_k \mid y_k, x_k=c]$ no depende de $y_k$.
3. **Árboles simétricos.** ¿Cuántas funciones distintas puede representar un árbol simétrico de profundidad 2 sobre dos variables binarias, comparado con un árbol general de profundidad 2?
4. **Experimento.** Repite el experimento de fuga con 100 niveles en lugar de 1 000. ¿Sigue siendo grave el TS codicioso?

# Referencias

- Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features*. NeurIPS 31.
- Dorogush, A. V., Ershov, V. y Gulin, A. (2018). *CatBoost: gradient boosting with categorical features support*. arXiv:1810.11363.
- Ibragimov, B. y Gusev, G. (2019). *Minimal Variance Sampling in Stochastic Gradient Boosting*. NeurIPS 32.
- Documentación de CatBoost: *Training parameters*, *Feature importance*.
