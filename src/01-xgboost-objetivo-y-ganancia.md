---
title: "XGBoost: derivación del objetivo y la ganancia"
author: Javier
lang: es
---

# Idea general

Un modelo de *boosting* construye la predicción como una suma de árboles, añadiendo uno nuevo en cada ronda $t$ para corregir lo que el ensamble previo no explica:

$$\hat y_i^{(t)} = \hat y_i^{(t-1)} + \eta\, f_t(\mathbf{x}_i)$$

donde $\eta$ es la tasa de aprendizaje (*shrinkage*) y $f_t$ es el árbol nuevo. Cada árbol se describe por una función de estructura $q$ que asigna cada observación a una hoja, y por un vector de pesos $w \in \mathbb{R}^T$ con $T$ hojas: $f(\mathbf{x}) = w_{q(\mathbf{x})}$.

# Objetivo regularizado

En la ronda $t$ se minimiza

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} \ell\!\left(y_i,\ \hat y_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)$$

con la penalización de complejidad

$$\Omega(f) = \gamma\, T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Aquí $\gamma$ castiga cada hoja adicional y $\lambda$ encoge los pesos de hoja (regularización L2).

# Aproximación de Taylor de 2.º orden

Se expande la pérdida alrededor de $\hat y_i^{(t-1)}$:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ \ell(y_i, \hat y_i^{(t-1)}) + g_i\, f_t(\mathbf{x}_i) + \tfrac{1}{2} h_i\, f_t(\mathbf{x}_i)^2 \right] + \Omega(f_t)$$

con gradiente y hessiano de la pérdida respecto a la predicción previa:

$$g_i = \frac{\partial\, \ell(y_i, \hat y)}{\partial \hat y}\bigg|_{\hat y = \hat y_i^{(t-1)}}, \qquad h_i = \frac{\partial^2 \ell(y_i, \hat y)}{\partial \hat y^2}\bigg|_{\hat y = \hat y_i^{(t-1)}}$$

El primer término no depende de $f_t$, así que se descarta.

# Agrupar por hoja

Sea $I_j = \{\, i : q(\mathbf{x}_i) = j \,\}$ el conjunto de observaciones en la hoja $j$, y defina $G_j = \sum_{i \in I_j} g_i$ y $H_j = \sum_{i \in I_j} h_i$. Entonces

$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} \left[ G_j\, w_j + \tfrac{1}{2}\,(H_j + \lambda)\, w_j^2 \right] + \gamma\, T$$

Cada hoja es una parábola en $w_j$, así que se minimiza por separado. Derivando e igualando a cero:

$$w_j^{*} = -\frac{G_j}{H_j + \lambda}$$

Sustituyendo, el valor óptimo de la pérdida para una estructura fija es

$$\tilde{\mathcal{L}}^{*} = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma\, T$$

Esta cantidad funciona como un *score* de calidad de la estructura: menor es mejor.

# Ganancia de un corte

Al dividir una hoja en izquierda ($L$) y derecha ($R$), la reducción de pérdida es

$$\begin{aligned}\text{Gain} &= \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} \right.\\ &\qquad \left. - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma\end{aligned}$$

Si $\text{Gain} < 0$ el corte no compensa el costo $\gamma$ y se poda. Esto da una poda natural controlada por $\gamma$.

# Casos concretos de $g_i$ y $h_i$

| Pérdida | $\ell$ | $g_i$ | $h_i$ |
|:--|:--|:--|:--|
| Error cuadrático | $\tfrac12 (y - \hat y)^2$ | $\hat y - y$ | $1$ |
| Logística | log-loss (abajo) | $p - y$ | $p(1-p)$ |

La pérdida logística es

$$\ell = -\left[\, y \ln p + (1-y)\ln(1-p) \,\right]$$

y $p = \sigma(\hat y) = 1/(1+e^{-\hat y})$ y $\hat y$ está en escala de log-odds.

Con error cuadrático, $w_j^* = \sum_{i\in I_j}(y_i - \hat y_i) \,/\, (n_j + \lambda)$: el promedio de residuos de la hoja, encogido hacia cero por $\lambda$.

# Efecto de los hiperparámetros

- **$\eta$ (`learning_rate`)**: escala cada $f_t$; valores bajos exigen más árboles pero generalizan mejor.
- **$\lambda$ (`reg_lambda`)**: aparece en el denominador de $w_j^*$ y de la ganancia; reduce el peso de hojas con poco hessiano.
- **$\gamma$ (`gamma`, `min_split_loss`)**: umbral mínimo de ganancia para aceptar un corte.
- **`min_child_weight`**: exige $H_j$ mínimo en cada hoja, es decir, suma de hessianos y no número de observaciones.
- **`max_depth`**: limita la profundidad y con ello $T$.

# Resumen

> Cada árbol se ajusta a $(g_i, h_i)$, no a los residuos crudos. Con la estructura fija, el peso óptimo de hoja es $-G_j/(H_j+\lambda)$. La ganancia de un corte sale de comparar los *scores* antes y después, menos $\gamma$.
