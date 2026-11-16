---
title: "Fundamentos de Gradient Boosting"
author: Javier
lang: es
---

*Nota introductoria. Objetivo: entender de dónde sale el algoritmo antes de ver XGBoost, LightGBM y CatBoost. Todo lo que sigue es previo a cualquier implementación concreta.*

# El problema de aprendizaje

Tenemos datos $(\mathbf{x}_i, y_i)$, $i=1,\dots,n$, que modelamos como observaciones independientes de una distribución $P$, y una función de pérdida $\ell(y, F)$ que mide el costo de predecir $F$ cuando el valor real es $y$. El **riesgo** de una función $F$ es

$$R(F) = \mathbb{E}\big[\ell\big(Y, F(X)\big)\big], \qquad F^{*} = \underset{F}{\arg\min}\; R(F)$$

El minimizador $F^*$ depende de la pérdida:

- **Error cuadrático**, $\ell = \tfrac12 (y-F)^2$: $F^*(\mathbf{x}) = \mathbb{E}[Y \mid X=\mathbf{x}]$.
- **Devianza binomial**, $y\in\{0,1\}$ y $\ell(y,F) = -yF + \ln(1+e^{F})$: $F^*(\mathbf{x}) = \ln \dfrac{P(Y=1\mid \mathbf{x})}{P(Y=0\mid \mathbf{x})}$, el log-odds. Esta $\ell$ es la log-verosimilitud negativa con $p=\sigma(F)=1/(1+e^{-F})$.

Como $P$ es desconocida, minimizamos el **riesgo empírico** sobre una clase de funciones $\mathcal{F}$:

$$\hat R(F) = \frac{1}{n}\sum_{i=1}^{n} \ell\big(y_i, F(\mathbf{x}_i)\big)$$

Si $\mathcal{F}$ es demasiado rica, $\hat R$ llega a cero y el modelo memoriza. Si es demasiado pobre, hay sesgo. Boosting propone una clase de funciones construida **por sumas** de piezas simples, y una forma controlada de recorrerla.

# Árboles de regresión como piezas básicas

Un árbol de regresión con $J$ hojas parte el espacio de variables en regiones disjuntas $R_1,\dots,R_J$ y es constante en cada una:

$$h(\mathbf{x}) = \sum_{j=1}^{J} c_j\, \mathbf{1}\{\mathbf{x}\in R_j\}$$

Dadas las regiones, el mejor valor de hoja para error cuadrático es la media de $y$ en la región: $c_j = \bar y_{R_j}$.

Las regiones se construyen con **particiones binarias recursivas voraces** (CART): en cada nodo se prueba cada variable $k$ y cada umbral $s$, y se elige el corte que más reduce la suma de cuadrados. Para un nodo $R$ con $n$ observaciones que se parte en $R_L$ y $R_R$ (con $n_L$ y $n_R$ observaciones):

$$\mathrm{SSE}(R) - \mathrm{SSE}(R_L) - \mathrm{SSE}(R_R) = \frac{n_L\, n_R}{n}\,\big(\bar y_L - \bar y_R\big)^2$$

> **Demostración.** Descomponiendo la varianza dentro y entre grupos, $\mathrm{SSE}(R)=\mathrm{SSE}(R_L)+\mathrm{SSE}(R_R)+n_L(\bar y_L-\bar y)^2+n_R(\bar y_R-\bar y)^2$. Como $\bar y = (n_L\bar y_L + n_R \bar y_R)/n$, se tiene $\bar y_L-\bar y = n_R(\bar y_L-\bar y_R)/n$ y $\bar y_R-\bar y=-n_L(\bar y_L-\bar y_R)/n$. Sustituyendo, los dos últimos términos suman $\frac{n_L n_R^2 + n_R n_L^2}{n^2}(\bar y_L-\bar y_R)^2 = \frac{n_L n_R}{n}(\bar y_L-\bar y_R)^2$. $\blacksquare$

Esta identidad reaparecerá en la nota de XGBoost: la ganancia de un corte generaliza esta reducción de suma de cuadrados.

**Por qué árboles y por qué pequeños.** Un árbol profundo tiene sesgo bajo y varianza alta. Boosting hace lo contrario: usa piezas de sesgo alto y varianza baja (árboles chicos) y reduce el sesgo sumando muchas. Además, los árboles son invariantes a transformaciones monótonas de cada variable, no necesitan escalar y capturan interacciones sin especificarlas.

**Tamaño del árbol e interacciones.** Un árbol con $J$ hojas solo puede representar interacciones de orden a lo sumo $J-1$ entre variables (Hastie, Tibshirani y Friedman, cap. 10). Con $J=2$ (un *stump*) el modelo final es puramente aditivo, $F(\mathbf{x})=\sum_k f_k(x_k)$. En la práctica, valores de $J$ entre 4 y 8 suelen funcionar bien en boosting.

# Modelos aditivos y ajuste por etapas

Buscamos $F$ de la forma

$$F_M(\mathbf{x}) = F_0 + \sum_{m=1}^{M} \beta_m\, h(\mathbf{x};\boldsymbol{\gamma}_m)$$

donde $h(\mathbf{x};\boldsymbol{\gamma})$ es un árbol con parámetros $\boldsymbol{\gamma}$ (estructura y valores de hoja) y $\beta_m$ es un coeficiente. Optimizar todos los términos a la vez es difícil. El **ajuste por etapas** (*forward stagewise*) los agrega de uno en uno, sin reajustar los anteriores:

$$(\beta_m, \boldsymbol{\gamma}_m) = \underset{\beta,\boldsymbol{\gamma}}{\arg\min}\; \sum_{i=1}^{n} \ell\big(y_i,\; F_{m-1}(\mathbf{x}_i) + \beta\, h(\mathbf{x}_i;\boldsymbol{\gamma})\big)$$

Con error cuadrático el paso se simplifica. Como $y_i - F_{m-1}(\mathbf{x}_i) - \beta h(\mathbf{x}_i)$ es el residuo actual menos la contribución nueva, la etapa consiste en **ajustar $h$ a los residuos** $y_i - F_{m-1}(\mathbf{x}_i)$. Para otras pérdidas no hay forma cerrada, y ahí entra el gradiente.

# Boosting como descenso de gradiente en el espacio de funciones

Idea de Friedman (2001). El riesgo empírico depende de $F$ solo a través del vector $\mathbf{F}=\big(F(\mathbf{x}_1),\dots,F(\mathbf{x}_n)\big)\in\mathbb{R}^n$. Podemos hacer descenso de gradiente sobre ese vector:

$$\mathbf{F}_m = \mathbf{F}_{m-1} - \rho_m\, \mathbf{g}_m, \qquad g_{im} = \left.\frac{\partial\, \ell(y_i, F)}{\partial F}\right|_{F=F_{m-1}(\mathbf{x}_i)}$$

El problema es que $\mathbf{g}_m$ solo está definido en los puntos de entrenamiento y no generaliza. La solución es **aproximar el gradiente negativo con un árbol**: se ajusta $h_m$ por mínimos cuadrados a los **pseudo-residuos**

$$r_{im} = -g_{im} = -\left.\frac{\partial\, \ell(y_i, F)}{\partial F}\right|_{F=F_{m-1}(\mathbf{x}_i)}$$

Así $h_m$ es la función de la clase que mejor imita la dirección de máximo descenso, pero definida para cualquier $\mathbf{x}$.

> **Algoritmo (Gradient Tree Boosting, Friedman 2001).**
>
> 1. Inicializar $F_0(\mathbf{x}) = \arg\min_c \sum_i \ell(y_i, c)$.
>
> 2. Para $m=1,\dots,M$:
>
> (a) Calcular los pseudo-residuos $r_{im}$ para $i=1,\dots,n$.
>
> (b) Ajustar un árbol de regresión con $J$ hojas a los $r_{im}$; esto da las regiones $R_{jm}$.
>
> (c) Para cada hoja, $\gamma_{jm} = \arg\min_{\gamma} \sum_{\mathbf{x}_i\in R_{jm}} \ell\big(y_i, F_{m-1}(\mathbf{x}_i)+\gamma\big)$.
>
> (d) Actualizar $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \sum_{j} \gamma_{jm}\,\mathbf{1}\{\mathbf{x}\in R_{jm}\}$, con $\nu\in(0,1]$.

Dos observaciones sobre el paso (b) y (c):

- Las **regiones** se eligen mirando el gradiente (aproximación de primer orden), pero los **valores de hoja** se recalculan con la pérdida verdadera en el paso (c). Eso distingue a TreeBoost de ajustar el árbol a los residuos y quedarse con sus medias.
- El factor $\nu$ es la **tasa de aprendizaje** o *shrinkage*: se da solo una fracción del paso.

## El puente hacia XGBoost

Si en el paso (c) aproximamos la pérdida de cada hoja con Taylor de segundo orden alrededor de $\gamma=0$,

$$\sum_{i\in R_{jm}} \ell\big(y_i, F_{m-1}(\mathbf{x}_i)+\gamma\big) \approx \text{cte} + \gamma \sum_{i} g_i + \tfrac12 \gamma^2 \sum_{i} h_i$$

con $h_i=\dfrac{\partial^2 \ell}{\partial F^2}\Big|_{F_{m-1}(\mathbf{x}_i)}$ el hessiano. Al minimizar en $\gamma$ se obtiene el **paso de Newton**

$$\gamma_{jm} = -\frac{\sum_{i\in R_{jm}} g_i}{\sum_{i\in R_{jm}} h_i}$$

XGBoost lleva esta idea más lejos: usa $g_i$ y $h_i$ también para **elegir** los cortes, y añade regularización explícita. Ese es el tema de la siguiente nota.

# Casos particulares de pérdida

Los pseudo-residuos y los valores de hoja para pérdidas comunes:

| Pérdida | $\ell(y,F)$ | $r = -\partial \ell/\partial F$ | Valor de hoja $\gamma$ |
|:--|:--|:--|:--|
| Cuadrática | $\tfrac12(y-F)^2$ | $y-F$ | media de $r$ |
| Absoluta (L1) | $\lvert y-F\rvert$ | $\operatorname{sign}(y-F)$ | mediana de $y-F$ |
| Binomial | $-yF+\ln(1+e^F)$ | $y-p$ | $\dfrac{\sum r_i}{\sum p_i(1-p_i)}$ (Newton) |
| Poisson | $e^{F}-yF$ | $y-e^{F}$ | $\dfrac{\sum r_i}{\sum e^{F_i}}$ (Newton) |

Notas:

- En el caso **binomial**, $p=\sigma(F)$. La derivada es $\partial\ell/\partial F = p-y$ y la segunda derivada es $p(1-p)$. No existe forma cerrada para el $\gamma$ exacto, por lo que se usa un paso de Newton.
- En el caso **Poisson** con enlace log, $\mu = e^{F}$. La pérdida es la log-verosimilitud negativa sin el término constante $\ln y!$. Aquí $g=\mu-y$ y $h=\mu$.
- En **multiclase** con softmax y $K$ clases, hay una función $F_k$ por clase y $p_k = e^{F_k}/\sum_l e^{F_l}$. Los pseudo-residuos son $r_{ik}=\mathbf{1}\{y_i=k\}-p_{ik}$ y se ajusta **un árbol por clase** en cada iteración.

# Conexión con AdaBoost

AdaBoost es el caso particular de ajuste por etapas con **pérdida exponencial**. Sea $y\in\{-1,+1\}$, $\ell(y,F)=e^{-yF}$ y $h(\mathbf{x})\in\{-1,+1\}$. El problema de la etapa $m$ es

$$\underset{\beta, h}{\arg\min} \sum_{i} w_i^{(m)}\, e^{-\beta\, y_i h(\mathbf{x}_i)}, \qquad w_i^{(m)} = e^{-y_i F_{m-1}(\mathbf{x}_i)}$$

Separando aciertos y errores, con $W_c$ la suma de pesos de los aciertos y $W_e$ la de los errores:

$$e^{-\beta} W_c + e^{\beta} W_e$$

Para $\beta$ fijo, minimizar equivale a elegir el $h$ que minimice el error ponderado. Derivando en $\beta$ e igualando a cero, $e^{2\beta}=W_c/W_e$, es decir

$$\beta_m = \tfrac12 \ln\frac{1-\mathrm{err}_m}{\mathrm{err}_m}, \qquad \mathrm{err}_m = \frac{W_e}{W_c+W_e}$$

Los pesos se actualizan como $w_i^{(m+1)}=w_i^{(m)}e^{-\beta_m y_i h_m(\mathbf{x}_i)}$: los errores pesan más en la ronda siguiente. Es el resultado de Friedman, Hastie y Tibshirani (2000). Gradient boosting generaliza el esquema a **cualquier pérdida diferenciable**. El minimizador poblacional de la pérdida exponencial es $F^*(\mathbf{x})=\tfrac12\ln\frac{P(Y=1\mid\mathbf{x})}{P(Y=-1\mid\mathbf{x})}$.

# Regularización: cómo no sobreajustar

La pérdida de entrenamiento baja con cada iteración. El error de validación suele bajar y luego subir. Hay varias palancas:

- **Número de iteraciones $M$.** Es el regularizador más básico: detenerse a tiempo (*early stopping*) con un conjunto de validación.
- **Shrinkage $\nu$.** Valores pequeños (por ejemplo $\nu\le 0.1$) suelen generalizar mejor, a cambio de necesitar más iteraciones. Como regla heurística, $\nu$ y $M$ se compensan.
- **Submuestreo** (*stochastic gradient boosting*, Friedman 2002). En cada iteración se ajusta el árbol con una fracción de las observaciones, sin reemplazo. Reduce la varianza y el costo.
- **Tamaño del árbol $J$.** Controla el orden de interacción permitido.
- **Penalización explícita** de la complejidad del árbol (número de hojas y tamaño de los valores de hoja), que es lo que añade XGBoost.

Con $\nu$ pequeño, boosting se comporta de forma parecida a la regresión *forward stagewise*, que está relacionada con la trayectoria del lasso (Hastie, Tibshirani y Friedman, §10.12). Es una conexión conceptual, no una igualdad exacta.

# Ejemplo numérico: dos iteraciones con pérdida cuadrática

Datos: $x=1,\dots,6$ e $y=(1,2,1,3,6,5)$. Usamos *stumps* ($J=2$) y $\nu=0.5$.

**Inicialización.** $F_0=\bar y = 3$. El ECM inicial es $3.667$.

**Iteración 1.** Residuos $r=(-2,-1,-2,0,3,2)$. La SSE de cada corte candidato:

| Corte | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 |
|:--|:--|:--|:--|:--|:--|
| SSE | 17.20 | 15.25 | 5.33 | **3.25** | 17.20 |

El mejor corte es $x<4.5$: media izquierda $-1.25$, media derecha $2.5$. Entonces $F_1=3+0.5\cdot(-1.25)=2.375$ para $x\le 4$ y $F_1 = 3+0.5\cdot 2.5=4.25$ para $x\ge 5$. El ECM baja a $1.323$.

**Iteración 2.** Residuos $r=(-1.375,-0.375,-1.375,0.625,1.75,0.75)$.

| Corte | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 |
|:--|:--|:--|:--|:--|:--|
| SSE | 5.67 | 5.64 | **1.43** | 3.25 | 7.26 |

Ahora el mejor corte es distinto, $x<3.5$: medias $-1.0417$ y $1.0417$. Resulta
$F_2=(1.854,\,1.854,\,1.854,\,2.896,\,4.771,\,4.771)$ y el ECM baja a $0.509$.

Dos stumps con cortes distintos ya producen una función escalonada de tres peldaños: cada iteración puede refinar una zona distinta del espacio.

# Implementación mínima

Gradient boosting con pérdida cuadrática, usando `DecisionTreeRegressor` de scikit-learn como pieza base:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def gb_fit(X, y, M=100, nu=0.1, depth=3):
    F0 = y.mean()
    F = np.full(len(y), F0)
    trees = []
    for m in range(M):
        r = y - F                      # pseudo-residuo (pérdida L2)
        t = DecisionTreeRegressor(max_depth=depth).fit(X, r)
        F += nu * t.predict(X)         # la hoja ya es la media de r
        trees.append(t)
    return F0, trees

def gb_predict(X, F0, trees, nu=0.1):
    return F0 + nu * sum(t.predict(X) for t in trees)
```

Con los datos del ejemplo, `depth=1`, `nu=0.5` y `M=2` reproduce exactamente $F_2$ y el ECM de $0.509$ de arriba (lo verifiqué ejecutándolo).

# Resumen

- Boosting es **ajuste aditivo por etapas**: $F_m = F_{m-1} + \nu\, h_m$.
- Gradient boosting interpreta cada etapa como un **paso de descenso de gradiente en el espacio de funciones**, aproximando $-\partial \ell/\partial F$ con un árbol.
- Las regiones salen del gradiente. Los valores de hoja pueden recalcularse con la pérdida verdadera, y con un paso de Newton se obtiene $-\sum g/\sum h$.
- AdaBoost es el caso de la pérdida exponencial.
- Se regulariza con $M$ (parada temprana), $\nu$, submuestreo y tamaño del árbol.
- **Siguiente nota:** XGBoost, que convierte el paso de Newton en el criterio de construcción del árbol y añade regularización explícita.

# Referencias

- Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics, 29(5).
- Friedman, J. H., Hastie, T. y Tibshirani, R. (2000). *Additive logistic regression: a statistical view of boosting*. Annals of Statistics, 28(2).
- Friedman, J. H. (2002). *Stochastic gradient boosting*. Computational Statistics & Data Analysis, 38(4).
- Freund, Y. y Schapire, R. (1997). *A decision-theoretic generalization of on-line learning and an application to boosting*. Journal of Computer and System Sciences, 55(1).
- Hastie, T., Tibshirani, R. y Friedman, J. *The Elements of Statistical Learning*, cap. 10.
