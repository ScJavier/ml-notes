---
title: "XGBoost: derivación del objetivo y la ganancia"
author: Javier
lang: es
---

*Segunda nota. Parte de la nota «Fundamentos de Gradient Boosting»: aquí se deriva con detalle qué cambia en XGBoost y por qué. Los ejemplos numéricos se verificaron con `xgboost` 3.4.1.*

# Qué cambia respecto al gradient boosting clásico

Chen y Guestrin (2016) mantienen el esquema aditivo por etapas, pero modifican cuatro cosas:

1. **Objetivo regularizado explícito.** La complejidad del árbol entra en lo que se optimiza, no solo como un parámetro externo.
2. **Segundo orden.** Se usa la aproximación de Taylor con gradiente $g_i$ y hessiano $h_i$ por observación. El valor óptimo de cada hoja tiene forma cerrada y la ganancia de un corte se calcula sin ajustar nada.
3. **Búsqueda de cortes escalable.** Algoritmo exacto, algoritmo aproximado con cuantiles ponderados por $h_i$ y manejo nativo de valores faltantes.
4. **Ingeniería de sistemas.** Bloques de columnas ordenados, uso de caché, paralelismo. No lo veremos aquí.

# Modelo y objetivo

Un árbol con $T$ hojas se escribe $f(\mathbf{x}) = w_{q(\mathbf{x})}$, donde $q:\mathbb{R}^d\to\{1,\dots,T\}$ asigna cada observación a una hoja y $w\in\mathbb{R}^T$ son los pesos de hoja. El modelo es

$$\hat y_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i)$$

y el objetivo regularizado es

$$\mathcal{L}(\phi) = \sum_{i=1}^{n} \ell(y_i, \hat y_i) + \sum_{k=1}^{K} \Omega(f_k)$$

con la penalización de complejidad

$$\Omega(f) = \gamma\, T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Aquí $\gamma\ge 0$ castiga cada hoja adicional y $\lambda\ge 0$ es una penalización L2 sobre los pesos. Como los parámetros son funciones, no se puede optimizar con métodos euclídeos tradicionales. Se entrena **por etapas**: en la ronda $t$ se agrega $f_t$ a la predicción $\hat y_i^{(t-1)}$ ya existente:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} \ell\!\left(y_i,\ \hat y_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)$$

# Aproximación de segundo orden

Para una función $\ell$ de $\hat y$, el desarrollo de Taylor de orden 2 en $\delta = f_t(\mathbf{x}_i)$ es

$$\ell(y_i, \hat y_i^{(t-1)} + \delta) \approx \ell(y_i, \hat y_i^{(t-1)}) + g_i\,\delta + \tfrac{1}{2} h_i\,\delta^2$$

con gradiente y hessiano **respecto a la predicción previa**:

$$g_i = \frac{\partial\, \ell(y_i, \hat y)}{\partial \hat y}\bigg|_{\hat y = \hat y_i^{(t-1)}}, \qquad h_i = \frac{\partial^2 \ell(y_i, \hat y)}{\partial \hat y^2}\bigg|_{\hat y = \hat y_i^{(t-1)}}$$

El primer sumando no depende de $f_t$, así que se descarta. Queda el objetivo simplificado

$$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{n} \left[ g_i\, f_t(\mathbf{x}_i) + \tfrac{1}{2} h_i\, f_t(\mathbf{x}_i)^2 \right] + \Omega(f_t)$$

Los vectores $g$ y $h$ se calculan una sola vez por ronda y son lo **único** que el algoritmo necesita de la pérdida. Por eso cambiar de pérdida es solo cambiar cómo se calculan $g_i$ y $h_i$.

# Agrupar por hoja y solución cerrada

Sea $I_j=\{\,i : q(\mathbf{x}_i)=j\,\}$ el conjunto de observaciones en la hoja $j$, y sean

$$G_j = \sum_{i\in I_j} g_i, \qquad H_j = \sum_{i\in I_j} h_i$$

Como $f_t(\mathbf{x}_i)=w_j$ para todo $i\in I_j$, la suma sobre observaciones se reorganiza como suma sobre hojas:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T}\left[ G_j\, w_j + \tfrac{1}{2}\,(H_j + \lambda)\, w_j^2 \right] + \gamma\, T$$

Cada término depende de un solo $w_j$ y es una parábola, por lo que se minimiza por separado. Derivando e igualando a cero, $G_j + (H_j+\lambda)\,w_j = 0$, y con $H_j+\lambda>0$ el punto crítico es un mínimo:

$$w_j^{*} = -\frac{G_j}{H_j + \lambda}$$

Sustituyendo en la expresión anterior, el valor óptimo para una estructura fija es

$$\tilde{\mathcal{L}}^{*} = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma\, T$$

> **Detalle de la sustitución.** En $w_j^*$ se cumple $G_j w_j^* = -\frac{G_j^2}{H_j+\lambda}$ y $\tfrac12(H_j+\lambda)(w_j^*)^2=\tfrac12\frac{G_j^2}{H_j+\lambda}$. Sumando, cada hoja aporta $-\tfrac12\frac{G_j^2}{H_j+\lambda}$.

Este valor funciona como un **score de calidad** de la estructura $q$: menor es mejor. Es el equivalente de la impureza en CART, pero ya incluye la regularización.

**Interpretaciones del peso de hoja.**

- Es un **paso de Newton con ridge**. Sin regularización ($\lambda=0$) sería $-G_j/H_j$, el paso de Newton de la nota anterior.
- Reescrito como $w_j^*=\dfrac{\sum_{i\in I_j} h_i\,(-g_i/h_i)}{\sum_{i\in I_j} h_i + \lambda}$, es un **promedio ponderado por $h_i$** de los pseudo-objetivos $-g_i/h_i$, encogido hacia cero por $\lambda$.
- Con error cuadrático, $g_i=\hat y_i-y_i$ y $h_i=1$, así que $w_j^*=\dfrac{\sum_{i\in I_j}(y_i-\hat y_i)}{n_j+\lambda}$: **la media de los residuos de la hoja, encogida** por $\lambda$ (con $\lambda=0$ es exactamente la media).

# La ganancia de un corte

Sea $I$ el conjunto de observaciones de un nodo, que se parte en $I_L$ e $I_R$. Antes del corte el nodo es una hoja con score $\frac{G^2}{H+\lambda}$ (con $G=G_L+G_R$ y $H=H_L+H_R$) y un costo de hoja $\gamma$. Después hay dos hojas. La reducción del objetivo es

$$\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

El árbol se construye **codiciosamente**: se elige el corte de mayor ganancia, y si $\text{Gain}<0$ el corte no se hace. Esto da una poda natural controlada por $\gamma$.

**Observación sobre la implementación.** Al ejecutar `xgboost` 3.4.1 sobre el ejemplo numérico de más abajo, el valor `gain` que reporta el volcado del árbol es **el doble** de la fórmula anterior (1.0286 frente a 0.5143), es decir, no incluye el factor $\tfrac12$. Además observé que el corte se conserva con `gamma=1.0` y se poda con `gamma=1.03`, de modo que el umbral $\gamma$ se compara contra esa ganancia sin $\tfrac12$. En esa versión, para obtener el efecto del $\gamma$ de la fórmula del artículo hay que pasar `gamma` igual a $2\gamma$. No cambia qué corte gana, pero sí cómo interpretar el valor de `gamma`. Lo comprobé en un solo ejemplo y una sola versión.

# Equivalencia con la reducción de varianza

Tomemos error cuadrático ($g_i=-r_i$ con $r_i=y_i-\hat y_i$, y $h_i=1$), $\lambda=0$ y $\gamma=0$. Entonces $G_L=-\sum_{I_L} r_i = -n_L \bar r_L$ y $H_L=n_L$, y lo mismo para $R$. La ganancia es

$$\text{Gain}=\tfrac12\left[n_L\bar r_L^{\,2}+n_R\bar r_R^{\,2}-n\,\bar r^{\,2}\right]$$

Como $n\bar r = n_L \bar r_L + n_R \bar r_R$, el mismo cálculo de la nota anterior da

$$\text{Gain}=\frac{1}{2}\cdot\frac{n_L\,n_R}{n}\,\big(\bar r_L-\bar r_R\big)^2$$

Es decir, la mitad de la **reducción de suma de cuadrados de CART aplicada a los residuos**. Con error cuadrático, $\lambda=0$ y $\gamma=0$, XGBoost elige exactamente los mismos cortes que un árbol CART ajustado a los residuos.

Con $\lambda>0$ cada término $\frac{(\sum r_i)^2}{n_j+\lambda}=\frac{n_j^2\,\bar r_j^{\,2}}{n_j+\lambda}$ queda multiplicado por $\frac{n_j}{n_j+\lambda}<1$, y ese factor es **más pequeño cuanto menos observaciones tenga la hoja**. Así $\lambda$ desfavorece cortes que dejan hojas chicas.

# Búsqueda de cortes

## Algoritmo exacto

Para cada variable del nodo se ordenan las observaciones por su valor, se recorren de izquierda a derecha acumulando $G_L$ y $H_L$, y en cada posición se calcula $G_R=G-G_L$, $H_R=H-H_L$ y la ganancia en $O(1)$. El costo dominante es el ordenamiento, que XGBoost evita repetir guardando cada columna ya ordenada en bloques. Es el método `tree_method="exact"`.

## Por qué los pesos son $h_i$

El objetivo simplificado es, en realidad, un problema de **mínimos cuadrados ponderados**. Completando el cuadrado en cada término:

$$g_i f + \tfrac12 h_i f^2 = \tfrac12 h_i\left(f + \frac{g_i}{h_i}\right)^2 - \frac{g_i^2}{2h_i}$$

por lo que

$$\begin{aligned} \tilde{\mathcal{L}}^{(t)} &= \sum_{i=1}^{n} \tfrac12 h_i \Big( f_t(\mathbf{x}_i) - \big(-g_i/h_i\big) \Big)^2 \\ &\quad + \Omega(f_t) + \text{constante} \end{aligned}$$

El árbol ajusta los pseudo-objetivos $-g_i/h_i$ con peso $h_i$. Por eso, al elegir candidatos de corte por cuantiles, **los cuantiles deben ser ponderados por $h_i$**: dos observaciones con mucha curvatura pesan más que dos con poca.

## Candidatos aproximados con cuantiles ponderados

Para la variable $k$ y un valor $z$, el **rango ponderado** es

$$r_k(z) = \frac{\sum_{i:\,x_{ik}<z} h_i}{\sum_{i} h_i}$$

Se eligen candidatos $s_{k1}<s_{k2}<\dots$ tales que $\lvert r_k(s_{k,l}) - r_k(s_{k,l+1})\rvert<\varepsilon$, lo que da aproximadamente $1/\varepsilon$ candidatos por variable. Con error cuadrático $h_i=1$ y esto se reduce a cuantiles ordinarios.

En la documentación de XGBoost, `approx` calcula este resumen antes de construir cada árbol con todas las filas, usando el hessiano como peso. `hist` precalcula los intervalos una vez (`max_bin=256` por defecto) y, según la documentación, es un método usado por LightGBM con pequeñas diferencias de implementación. **Desde la versión 2.0, `hist` es el método por defecto** (`tree_method="auto"`).

## Histogramas

Con los intervalos fijos, en cada nodo se acumulan $(G,H)$ por intervalo recorriendo sus observaciones (costo lineal en el tamaño del nodo) y se evalúa la ganancia en los límites entre intervalos (costo proporcional al número de intervalos). Para ahorrar trabajo se usa la **resta de histogramas**: el histograma del hijo mayor es el del padre menos el del hijo menor.

## Valores faltantes

En cada corte se prueban **dos direcciones por defecto**. Primero se recorren los valores observados enviando los faltantes a la derecha y luego a la izquierda, y se queda la dirección con mayor ganancia. El resultado queda guardado en el árbol (en el volcado aparece como `missing=`). El costo es lineal en las entradas no faltantes.

# Regularización adicional

**Penalización L1** ($\alpha$, `reg_alpha`). Si se añade $\alpha\sum_j |w_j|$ al objetivo, cada hoja minimiza $G w+\tfrac12(H+\lambda)w^2+\alpha|w|$. Considerando los signos de $w$, la solución es una **umbralización suave**:

$$\begin{aligned} w^* &= -\frac{T_\alpha(G)}{H+\lambda} \\ T_\alpha(G) &= \operatorname{sign}(G)\,\max\big(\lvert G\rvert-\alpha,\,0\big) \end{aligned}$$

Si $\lvert G\rvert\le\alpha$ la hoja queda en cero, y en la ganancia $G^2$ se reemplaza por $T_\alpha(G)^2$.

**Otros mecanismos.**

- **Tasa de aprendizaje $\eta$** (`eta`): la predicción se actualiza con $\hat y^{(t)}=\hat y^{(t-1)}+\eta\, f_t$.
- **Submuestreo de filas** (`subsample`) **y de columnas** (`colsample_bytree`, `colsample_bylevel`, `colsample_bynode`).
- **`min_child_weight`**: mínimo de $H_j$ (suma de hessianos) por hoja. No es un conteo de observaciones: con error cuadrático $H_j=n_j$, pero con la logística $h_i=p_i(1-p_i)\le 1/4$, y una hoja necesita al menos $4\cdot$`min_child_weight` observaciones en el mejor caso.
- **`max_depth`** limita $T$ e implícitamente el orden de las interacciones.
- **`max_delta_step`**: acota $\lvert w_j\rvert$, útil en logística con clases muy desbalanceadas y en Poisson.

# Gradiente y hessiano por pérdida

| Pérdida | $\ell(y,\hat y)$ | $g_i$ | $h_i$ |
|:--|:--|:--|:--|
| Cuadrática | $\tfrac12(y-\hat y)^2$ | $\hat y - y$ | $1$ |
| Logística | $-yF+\ln(1+e^F)$ | $p-y$ | $p(1-p)$ |
| Poisson (log) | $e^{F}-yF$ | $e^{F}-y$ | $e^{F}$ |
| Softmax | $-\ln p_{y}$ | $p_k-\mathbf{1}\{y=k\}$ | $2\,p_k(1-p_k)$ |

En la logística $\hat y=F$ es el log-odds y $p=\sigma(F)$. Observaciones bien clasificadas ($p\approx 0$ o $1$) tienen $h\approx0$: casi no aportan a $H_j$ ni cuentan para `min_child_weight`.

En el caso **softmax**, el hessiano que usa XGBoost es una **cota superior diagonal**, no el hessiano verdadero: el hessiano exacto de la entropía cruzada multiclase tiene diagonal $p_k(1-p_k)$, y XGBoost usa el doble. Esto está documentado en su demo de objetivos personalizados.

# Ejemplo numérico: un corte con pérdida logística

Datos: $x=1,\dots,6$ e $y=(0,0,0,1,0,1)$. Margen inicial $0$ (es decir $p=0.5$), $\lambda=1$, $\gamma=0$, $\eta=1$, un solo árbol de profundidad 1 y `min_child_weight=0`.

**Gradientes y hessianos.** $g_i=p-y_i=(0.5,\,0.5,\,0.5,\,-0.5,\,0.5,\,-0.5)$ y $h_i=0.25$ para todos. Totales: $G=1.0$, $H=1.5$. El score de la raíz es $\frac{G^2}{H+\lambda}=\frac{1}{2.5}=0.4$.

**Ganancia de cada corte candidato.**

| Corte | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
|:--|:--|:--|:--|:--|:--|
| 1.5 | 0.5 | 0.25 | 0.5 | 1.25 | $-0.0444$ |
| 2.5 | 1.0 | 0.50 | 0.0 | 1.00 | 0.1333 |
| 3.5 | 1.5 | 0.75 | $-0.5$ | 0.75 | **0.5143** |
| 4.5 | 1.0 | 1.00 | 0.0 | 0.50 | 0.0500 |
| 5.5 | 1.5 | 1.25 | $-0.5$ | 0.25 | 0.4000 |

Verificación a mano del corte $3.5$:

$$\begin{aligned} \text{Gain} &= \tfrac12\left[\frac{1.5^2}{0.75+1}+\frac{(-0.5)^2}{0.75+1}-0.4\right] \\ &= \tfrac12\,[\,1.2857+0.1429-0.4\,] = 0.5143 \end{aligned}$$

**Pesos de hoja.**

$$\begin{aligned} w_L &= -\frac{1.5}{1.75} = -0.857 \\ w_R &= -\frac{-0.5}{1.75} = 0.286 \end{aligned}$$

**Nuevas probabilidades.** Para $x\le3$, $p=\sigma(-0.857)=0.298$. Para $x\ge4$, $p=\sigma(0.286)=0.571$. La log-loss media baja de $0.6931$ a $0.5047$.

**Contraste con XGBoost.** Entrenando ese árbol con `xgboost` 3.4.1 se obtienen el mismo corte ($x<3.5$), los mismos pesos de hoja ($-0.8571$ y $0.2857$) y `cover=1.5` en la raíz, que es $\sum h_i$. El `gain` reportado es $1.0286=2\times0.5143$, como se comentó arriba.

Dos observaciones sobre este ejemplo:

- El corte en $1.5$ tiene **ganancia negativa aun con $\gamma=0$**. Con $\lambda>0$ esto puede ocurrir: la regularización del score hace que partir una hoja pueda empeorar el objetivo.
- Con $\lambda=0$ la ganancia nunca es negativa (ver el primer ejercicio).

# Mapa de hiperparámetros

| Parámetro | Dónde entra | Efecto |
|:--|:--|:--|
| `eta` | $\hat y \leftarrow \hat y+\eta f_t$ | Contrae cada árbol; pide más rondas |
| `lambda` | $H_j+\lambda$ | Encoge $w_j$; desfavorece hojas chicas |
| `alpha` | $T_\alpha(G_j)$ | Deja en cero hojas con $\lvert G_j\rvert\le\alpha$ |
| `gamma` | $-\gamma$ en la ganancia | Umbral mínimo para aceptar un corte |
| `min_child_weight` | exige $H_j$ mínimo | Evita hojas con poca evidencia |
| `max_depth` | limita $T$ | Complejidad e interacciones |
| `subsample`, `colsample_*` | muestreo | Reduce varianza y correlación entre árboles |
| `max_delta_step` | acota $\lvert w_j\rvert$ | Estabiliza logística desbalanceada y Poisson |
| `tree_method`, `max_bin` | candidatos de corte | Exacto frente a aproximado; 256 intervalos por defecto en `hist` |

# Ejercicios

1. **Ganancia no negativa.** Muestra que con $\lambda=0$ y $h_i>0$ siempre se cumple $\text{Gain}\ge0$ (antes de restar $\gamma$). *Pista:* $f(G,H)=G^2/H$ es convexa y homogénea de grado 1, y por tanto subaditiva.
2. **Umbralización suave.** Deriva $w^*$ para el objetivo con penalización L1 considerando los casos $G>\alpha$, $G<-\alpha$ y $\lvert G\rvert\le\alpha$.
3. **Hessiano de softmax.** Calcula el hessiano verdadero de la entropía cruzada multiclase respecto a los logits y compara su diagonal con la que usa XGBoost.
4. **Efecto de $\lambda$.** Repite el ejemplo numérico con $\lambda=0$ y con $\lambda=5$. ¿Qué corte gana en cada caso y cómo cambian los pesos de hoja?

# Referencias

- Chen, T. y Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16. arXiv:1603.02754.
- Documentación de XGBoost: *Parameters*, *Tree Methods* y el demo *custom_softmax*.
- Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics, 29(5).
