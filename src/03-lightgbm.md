---
title: "LightGBM: histogramas, crecimiento por hojas, GOSS y EFB"
author: Javier
lang: es
---

*Tercera nota. Supone las dos anteriores: el objetivo de segundo orden, el peso de hoja $-G/(H+\lambda)$ y la ganancia de un corte se reutilizan sin cambios. Los experimentos se corrieron con `lightgbm` 4.7.0 en 2 núcleos; el script está en `ejemplos/lightgbm_experimentos.py` del repositorio.*

# Qué es LightGBM respecto a XGBoost

Ke et al. (2017) parten del mismo modelo que XGBoost: árboles sumados por etapas, ajustados con gradiente $g_i$ y hessiano $h_i$ de la pérdida. La pregunta del artículo es de **eficiencia**: el costo de entrenar un árbol es, aproximadamente,

$$\text{costo} \;\propto\; (\text{n.º de observaciones}) \times (\text{n.º de variables})$$

porque para cada variable hay que recorrer los datos para encontrar el mejor corte. LightGBM ataca los dos factores y además cambia cómo crece el árbol:

| Idea | Qué reduce | Cambia el modelo |
|:--|:--|:--|
| Histogramas | costo por variable | poco (umbrales discretos) |
| Crecimiento por hojas | — | sí: forma del árbol |
| GOSS | observaciones | sí: muestreo sesgado corregido |
| EFB | variables | no (agrupa sin pérdida si no hay conflictos) |
| Categóricas nativas | — | sí: cortes por grupos de categorías |

Lo que **no** cambia: el peso de hoja y la ganancia. En el código, la ganancia de un corte es

$$\text{gain} = \frac{G_L^2}{H_L+\lambda_2} + \frac{G_R^2}{H_R+\lambda_2} - \frac{G^2}{H+\lambda_2}$$

sin el factor $\tfrac12$, igual que en XGBoost. Lo comprobé con un *stump* sobre los datos del ejemplo de la nota 1: con $\lambda_2=0$ LightGBM reporta `split_gain = 18.75` y la fórmula da $18.75$; con $\lambda_2=1$, ambos dan $13.33$. Los valores de hoja que reporta `trees_to_dataframe()` incluyen la predicción inicial (por ejemplo $3 + (-1.25) = 1.75$).

# Histogramas

Antes de entrenar, cada variable continua se **discretiza** en a lo sumo `max_bin` intervalos (255 por defecto). Los límites se calculan a partir de una muestra de hasta `bin_construct_sample_cnt` filas (200 000 por defecto) buscando intervalos con frecuencias parecidas, y cada intervalo debe tener al menos `min_data_in_bin` observaciones (3 por defecto). Desde ese momento cada valor se guarda como el índice de su intervalo (un entero pequeño).

En cada nodo, para cada variable, se construye un histograma: para cada intervalo $b$,

$$G_b=\sum_{i\in \text{nodo},\; x_i\in b} g_i, \qquad H_b=\sum_{i\in \text{nodo},\; x_i\in b} h_i$$

y se evalúa la ganancia en los $B-1$ límites entre intervalos, acumulando $G_L$ y $H_L$ de izquierda a derecha. El costo pasa de $O(n_{\text{nodo}})$ por variable con ordenamiento a $O(n_{\text{nodo}})$ sumas baratas más $O(B)$ evaluaciones de ganancia.

**Resta de histogramas.** Como $G_b^{\text{padre}} = G_b^{\text{izq}} + G_b^{\text{der}}$, basta construir el histograma del hijo con menos datos; el del otro hijo sale por resta en $O(B)$.

**Efecto de `max_bin` en un umbral.** Con $x\sim U(0,1)$, $y = 2\cdot\mathbf 1\{x>0.37\}+\varepsilon$, $\varepsilon\sim N(0, 0.5^2)$, un solo corte, y $n=20\,000$:

| `max_bin` | umbral elegido | RMSE de prueba |
|:--|:--|:--|
| 4 | 0.334 | 0.620 |
| 8 | 0.428 | 0.669 |
| 16 | 0.400 | 0.601 |
| 64 | 0.366 | 0.515 |
| 255 | 0.372 | 0.507 |

El RMSE mínimo posible es $0.5$ (el ruido). Con pocos intervalos el corte solo puede caer en un límite de intervalo, y el error es grande cerca del salto. Con 64 o más ya casi no hay pérdida. Un `max_bin` bajo actúa como regularización (umbrales más gruesos), pero también como sesgo.

# Crecimiento por hojas

XGBoost (por defecto) crece **por niveles**: parte todas las hojas de un nivel antes de pasar al siguiente, hasta `max_depth`. LightGBM crece **por hojas** (*best-first*): en cada paso parte **la hoja cuya mejor ganancia es la mayor**, sin importar su profundidad, hasta tener `num_leaves` hojas.

> **Algoritmo (un árbol).**
>
> 1. Raíz = una hoja. Calcular su mejor corte y ganancia.
>
> 2. Mientras haya menos de `num_leaves` hojas: tomar la hoja con mayor ganancia; si esa ganancia no supera `min_gain_to_split`, parar. Partirla; calcular el mejor corte de sus dos hijos (uno con histograma, el otro por resta).

Con el mismo número de hojas, el crecimiento por hojas reduce más la pérdida de entrenamiento, porque cada corte es el mejor disponible en todo el árbol. El costo es que puede producir **ramas muy profundas** con pocos datos.

**Forma real de un árbol.** Un árbol sobre Friedman #1 ($n=20\,000$, 10 variables) tiene estas profundidades de hoja:

| `num_leaves` | profundidades de las hojas (profundidad: n.º de hojas) |
|:--|:--|
| 31 | 5: 5, 6: 18, 7: 8 |
| 255 | 7: 13, 8: 44, 9: 70, 10: 70, 11: 34, 12: 17, 13: 5, 14: 2 |

Un árbol completo por niveles con 32 hojas tendría todas a profundidad 5, y con 256, todas a profundidad 8. Aquí, con 255 hojas, hay ramas hasta profundidad 14.

**Relación entre `num_leaves` y `max_depth`.** Un árbol binario de profundidad $d$ tiene a lo sumo $2^d$ hojas. Si se fija `max_depth`, conviene `num_leaves` $< 2^{d}$; de lo contrario la restricción efectiva es la profundidad. `max_depth=-1` (por defecto) significa sin límite. Aunque se fije `max_depth`, el árbol **sigue creciendo por hojas**.

## Experimento: `num_leaves` y `min_data_in_leaf` con pocos datos

Friedman #1: $y = 10\sin(\pi x_1x_2)+20(x_3-0.5)^2+10x_4+5x_5+\varepsilon$, con 5 variables de ruido adicionales y $\varepsilon\sim N(0,1)$. Entrenamiento con 2 000 filas, validación con 2 000 (parada temprana de 100 rondas, `learning_rate=0.05`), prueba con 20 000. RMSE de prueba (el mínimo posible es $1$):

| `num_leaves` | `min_data_in_leaf`=5 | 20 | 100 |
|:--|:--|:--|:--|
| 4 | 1.195 | 1.183 | 1.183 |
| 8 | 1.248 | 1.238 | 1.226 |
| 16 | 1.287 | 1.270 | 1.259 |
| 31 | 1.322 | 1.310 | 1.258 |
| 63 | 1.390 | 1.360 | 1.258 |
| 127 | 1.510 | 1.366 | 1.258 |
| 255 | 1.637 | 1.366 | 1.258 |

Lecturas:

- Friedman #1 es aditivo salvo una interacción de orden 2, así que árboles chicos bastan. Más hojas = más varianza.
- Con `min_data_in_leaf=5` el error crece sin freno con `num_leaves`: las hojas pequeñas memorizan ruido.
- Con `min_data_in_leaf=100` las columnas a partir de 31 son idénticas: con 2 000 filas no caben más de unas 20 hojas de 100 datos, así que `num_leaves` deja de ser la restricción activa.
- Con datos pequeños, `min_data_in_leaf` (o `min_sum_hessian_in_leaf`) es tan importante como `num_leaves`.

# GOSS: muestreo de una sola cara basado en gradientes

**Idea.** Las observaciones con gradiente pequeño ya están bien ajustadas y aportan poco a la ganancia. GOSS se queda con todas las de gradiente grande y con una muestra aleatoria de las demás, **reponderando** la muestra para no sesgar las sumas.

Sea $n$ el número de observaciones, $a$ = `top_rate` y $b$ = `other_rate`:

1. Ordenar por magnitud y tomar el conjunto $A$ con las $an$ mayores.
2. Del resto $A^c$, con $(1-a)n$ elementos, tomar al azar un subconjunto $B$ de tamaño $bn$.
3. Multiplicar $g_i$ y $h_i$ de los elementos de $B$ por $\dfrac{1-a}{b}$.
4. Construir el árbol solo con $A\cup B$, es decir, con una fracción $a+b$ de los datos.

**Por qué el factor $(1-a)/b$.** Cada elemento de $A^c$ entra en $B$ con probabilidad $\frac{bn}{(1-a)n}=\frac{b}{1-a}$. Entonces, para cualquier suma sobre un nodo,

$$\begin{aligned} \mathbb E\Big[\sum_{i\in A} g_i + \tfrac{1-a}{b}\sum_{i\in B} g_i\Big] &= \sum_{i\in A} g_i + \tfrac{1-a}{b}\cdot\tfrac{b}{1-a}\sum_{i\in A^c} g_i \\ &= \sum_{i=1}^{n} g_i \end{aligned}$$

Las sumas $G$ y $H$ son **insesgadas**. La ganancia, que es un cociente de cuadrados, no lo es exactamente; el artículo acota el error de aproximación de la ganancia y argumenta que es pequeño cuando $n$ es grande.

**Lo que hace la implementación** (leído en `src/boosting/goss.hpp`, rama principal):

- La magnitud para ordenar es $\lvert g_i\, h_i\rvert$, no $\lvert g_i\rvert$ como en el artículo.
- El factor es `(cnt - top_k) / other_k`, que es $(1-a)/b$, y se aplica a gradientes **y** hessianos.
- Durante las primeras $\lfloor 1/\eta\rfloor$ iteraciones no se muestrea (con $\eta=0.1$, las primeras 10).
- Desde la versión 4.0 se activa con `data_sample_strategy="goss"` (antes era `boosting="goss"`). Por defecto `top_rate=0.2` y `other_rate=0.1`.

**Experimento.** Friedman #1 con $n=300\,000$, 300 árboles, `learning_rate=0.1`, `num_leaves=31`:

| Muestreo | segundos | RMSE de validación |
|:--|:--|:--|
| todos los datos | 2.5 | 1.038 |
| *bagging* aleatorio 30 % | 3.3 | 1.044 |
| GOSS 20 % + 10 % | 4.6 | 1.042 |

La pérdida de precisión de GOSS es mínima, pero aquí **no** fue más rápido: con solo 10 variables, construir histogramas es barato y el costo de ordenar y muestrear domina. La ventaja aparece cuando la construcción de histogramas domina (muchas variables y muchas filas). No asumas aceleración: mídela en tus datos.

# EFB: agrupación de variables exclusivas

En datos dispersos (por ejemplo, one-hot o indicadores), muchas variables **casi nunca son distintas de cero al mismo tiempo**. Dos variables son *mutuamente exclusivas* si nunca lo son.

**Idea.** Fusionar variables exclusivas en una sola, desplazando sus intervalos para que no se encimen. Si $f_1$ ocupa los intervalos $0$–$9$ y $f_2$ los $0$–$19$, la variable agrupada usa $0$–$9$ para $f_1$ y $10$–$29$ para $f_2$. Un histograma de la variable agrupada contiene la información de las dos.

**Encontrar los grupos.** Construir un grafo con una arista entre cada par de variables que chocan (son distintas de cero a la vez), con peso igual al número de choques. Agrupar con el menor número de grupos es un problema de coloración de grafos, NP-difícil. LightGBM usa un algoritmo voraz: ordena las variables por grado y asigna cada una al primer grupo donde los choques no excedan una tolerancia pequeña.

Se controla con `enable_bundle` (activado por defecto). En mi prueba (50 000 filas, 3 000 columnas indicadoras exclusivas, 100 árboles) la diferencia fue de 1.1 frente a 1.2 segundos. LightGBM también tiene optimizaciones para datos dispersos independientes de EFB, y el beneficio depende mucho del formato y la estructura de los datos.

# Variables categóricas

Con una variable categórica de $k$ niveles, un corte binario es una partición de los niveles en dos grupos: hay $2^{k-1}-1$ posibilidades.

**Resultado de Fisher (1958).** Para error cuadrático, la mejor partición en dos grupos es **contigua** cuando los niveles se ordenan por la media de $y$. Basta probar $k-1$ cortes sobre la lista ordenada: el costo baja a $O(k\log k)$.

LightGBM aplica la versión de segundo orden. En cada nodo (leído en `src/treelearner/feature_histogram.cpp`):

1. Si el número de niveles es $\le$ `max_cat_to_onehot` (4 por defecto), prueba cortes **uno contra el resto**.
2. Si no, conserva los niveles con al menos `cat_smooth` observaciones aproximadas (10 por defecto) y los ordena por
$$\frac{G_c}{H_c + \texttt{cat\_smooth}}$$
que es, salvo el signo, un paso de Newton por nivel encogido hacia cero. Los niveles excluidos nunca van al lado izquierdo del corte.
3. Recorre la lista ordenada **desde ambos extremos**, considerando como máximo `max_cat_threshold` niveles (32) y la mitad de los niveles disponibles.
4. Solo evalúa un corte cuando el grupo acumulado desde el último corte probado tiene al menos `min_data_per_group` datos (100), y para estos cortes usa $\lambda_2 +$ `cat_l2` (10) en lugar de $\lambda_2$.

Como el orden se recalcula en cada nodo con los gradientes actuales, es una codificación por gradientes **dentro del entrenamiento**, no una codificación fija de antemano.

**Riesgo.** Con muchos niveles raros, ordenar por gradiente medio sobreajusta: la partición se adapta al ruido de los niveles con pocos datos. Los cuatro parámetros anteriores son la defensa.

**Experimento.** Una categórica con 300 niveles de frecuencia tipo Zipf (226 niveles con menos de 10 observaciones en entrenamiento), efecto por nivel $\sim N(0,1)$, más $x_1 + 0.5x_2^2$ y una variable de ruido. Entrenamiento 5 000, validación 5 000, prueba 50 000. RMSE mínimo posible: $1$.

| Variante | iteraciones | RMSE de prueba |
|:--|:--|:--|
| Nativa, valores por defecto | 107 | **1.113** |
| Nativa, sin suavizar (`cat_smooth=0.1`, `cat_l2=0`, `min_data_per_group=1`) | 75 | 1.119 |
| Nativa, más suavizado (`cat_smooth=50`, `min_data_per_group=200`) | 96 | 1.202 |
| Uno contra el resto (`max_cat_to_onehot=1000`) | 116 | 1.151 |
| Código entero tratado como numérica | 203 | 1.146 |

- El manejo nativo con valores por defecto fue el mejor.
- Suavizar de más fue lo peor: con `cat_smooth=50` solo los niveles frecuentes pueden aislarse.
- Uno contra el resto necesita muchos cortes para separar grupos de niveles.
- Tratar el código como número funcionó razonablemente, porque con suficientes cortes un árbol puede aislar rangos de códigos, pero necesitó el doble de iteraciones.
- Es un solo experimento con una semilla. Úsalo como intuición, no como regla.

**Detalles prácticos.**

- Declara las categóricas con `categorical_feature` o usa columnas `pandas.Categorical`.
- Los valores se convierten a enteros de 32 bits; los negativos se tratan como faltantes.
- Asegúrate de que el mapeo nivel → código sea el mismo en entrenamiento y predicción. Con `pandas.Categorical` hay que fijar las mismas categorías en ambos.

# Valores faltantes

Como en XGBoost, en cada corte se aprende una dirección por defecto para los faltantes (`use_missing=true`). Con `zero_as_missing=true`, los ceros también se tratan como faltantes (útil en matrices dispersas donde un cero no observado significa “sin dato”).

# Regularización y controles del árbol

**Peso de hoja.** Con L1 ($\lambda_1$ = `lambda_l1`) y L2 ($\lambda_2$ = `lambda_l2`), el peso es la umbralización suave de la nota 2:

$$w^*=-\frac{T_{\lambda_1}(G)}{H+\lambda_2}$$

Ambas valen 0 por defecto, a diferencia de XGBoost, donde $\lambda=1$.

**Suavizado hacia el padre** (`path_smooth`, $\alpha$). Si $\alpha>0$, el peso de cada nodo con $n$ observaciones se reemplaza por

$$w_{\text{nodo}} = \frac{n}{n+\alpha}\, w + \frac{\alpha}{n+\alpha}\, w_{\text{padre}}$$

que es la fórmula de la documentación reescrita. Es un promedio entre la estimación del nodo y la del padre, con más peso al padre cuando el nodo tiene pocos datos. Tiene la misma forma que la media posterior de un modelo jerárquico normal. La ganancia se calcula con los pesos suavizados.

**Otros controles.**

| Parámetro (defecto) | Efecto |
|:--|:--|
| `num_leaves` (31) | Complejidad principal del árbol |
| `max_depth` (−1) | Tope de profundidad; sin límite por defecto |
| `min_data_in_leaf` (20) | Datos mínimos por hoja (aproximados vía hessiano) |
| `min_sum_hessian_in_leaf` (1e-3) | Como `min_child_weight` de XGBoost |
| `min_gain_to_split` (0) | Como `gamma` de XGBoost, sobre la ganancia sin $\tfrac12$ |
| `lambda_l1`, `lambda_l2` (0, 0) | Encogen pesos de hoja |
| `path_smooth` (0) | Encoge hacia el padre |
| `max_delta_step` (0) | Acota la salida de la hoja a `learning_rate`·`max_delta_step` |
| `feature_fraction` (1) | Fracción de variables por árbol |
| `feature_fraction_bynode` (1) | Fracción de variables por nodo; no acelera |
| `bagging_fraction`, `bagging_freq` (1, 0) | *Bagging*; solo actúa si `bagging_freq`>0 y fracción<1 |
| `extra_trees` (false) | Un umbral aleatorio por variable en cada nodo |
| `max_bin` (255) | Resolución de umbrales |

**Restricciones estructurales.**

- `monotone_constraints`: una entrada por variable ($-1$, $0$ o $+1$). El método `basic` (defecto) no hace más lento el entrenamiento pero restringe de más; `intermediate` y `advanced` restringen menos a cambio de tiempo.
- `interaction_constraints`: listas de variables que pueden aparecer juntas en una misma rama. Dos variables solo comparten rama si alguna lista contiene a ambas.
- `linear_tree`: cada hoja ajusta una regresión lineal en las variables usadas en la rama, en lugar de una constante. El primer árbol usa hojas constantes, y aumenta bastante la memoria.

**Experimento con `linear_tree`.** $y = 3x_1$ si $x_1>0$, $-x_1$ si no, más $0.5x_2$ y ruido $N(0, 0.3^2)$. `learning_rate=0.3`:

| `num_leaves` | árboles | constante | lineal |
|:--|:--|:--|:--|
| 2 | 5 | 0.899 | 0.748 |
| 4 | 5 | 0.649 | 0.557 |
| 31 | 5 | 0.444 | 0.428 |
| 2 | 50 | 0.356 | 0.310 |
| 4 | 50 | 0.320 | 0.308 |

Con relaciones lineales por tramos ayuda, sobre todo con árboles chicos. Con suficientes árboles la diferencia casi desaparece (el ruido es $0.3$).

**DART** (`boosting="dart"`): en cada iteración se descarta al azar una fracción de los árboles previos (`drop_rate=0.1`) para calcular los gradientes, y luego se reescalan. Reduce la dependencia de los primeros árboles, pero hace incompatible la parada temprana estándar, porque los árboles anteriores cambian.

# LightGBM frente a XGBoost, en valores por defecto

| | XGBoost 3.4 | LightGBM 4.7 |
|:--|:--|:--|
| Crecimiento | por niveles (`depthwise`) | por hojas |
| Tamaño del árbol | `max_depth=6` | `num_leaves=31`, sin tope de profundidad |
| Tasa de aprendizaje | 0.3 | 0.1 |
| L2 en hojas | 1 | 0 |
| Mínimo por hoja | hessiano ≥ 1 | 20 datos y hessiano ≥ 1e-3 |
| Intervalos | 256 | 255 |
| Categóricas | opcional (`enable_categorical`) | nativas |

Los valores de XGBoost los leí de `save_config()` de un modelo entrenado con la versión 3.4.1. XGBoost también puede crecer por hojas con `grow_policy="lossguide"` y `max_leaves`; las diferencias de diseño son menores que las de valores por defecto.

# Receta práctica de ajuste

1. Fija `learning_rate` entre 0.03 y 0.1 y usa parada temprana con un conjunto de validación.
2. Ajusta la complejidad: `num_leaves` (8–255) junto con `min_data_in_leaf` (10–200). Si los datos son pocos, empieza por árboles chicos.
3. Añade aleatoriedad: `feature_fraction` 0.5–0.9 y `bagging_fraction` 0.5–0.9 con `bagging_freq=1`.
4. Regulariza si aún sobreajusta: `lambda_l2`, `min_gain_to_split`, `path_smooth`.
5. Con categóricas de alta cardinalidad, revisa `cat_smooth`, `min_data_per_group` y `max_cat_threshold`.
6. Al final, baja `learning_rate` y sube el número de rondas para ganar un poco más.

# Ejercicios

1. **Sesgo de la ganancia con GOSS.** Muestra con un ejemplo de dos observaciones que $\mathbb E\big[\tilde G^2\big]\neq G^2$ aunque $\mathbb E[\tilde G]=G$. ¿Hacia dónde va el sesgo?
2. **Fisher.** Demuestra el resultado de Fisher para dos grupos: si una partición óptima no es contigua en el orden por medias, intercambiar dos niveles mejora o no empeora la suma de cuadrados.
3. **`path_smooth`.** Muestra que la fórmula de la documentación, $w\frac{n/\alpha}{n/\alpha+1}+\frac{w_p}{n/\alpha+1}$, es igual a la de esta nota.
4. **Experimento.** Repite el experimento de `num_leaves` con 20 000 filas de entrenamiento. ¿Qué cambia?

# Referencias

- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 30.
- Fisher, W. D. (1958). *On grouping for maximum homogeneity*. JASA, 53(284).
- Documentación de LightGBM 4.7: *Parameters* y *Features*; código fuente: `src/boosting/goss.hpp`, `src/treelearner/feature_histogram.cpp`.
- Rashmi, K. V. y Gilad-Bachrach, R. (2015). *DART: Dropouts meet Multiple Additive Regression Trees*. AISTATS.
