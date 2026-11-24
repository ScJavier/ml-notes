---
title: "Artículo 3 · Hierarchical Shrinkage: encoger árboles sin tocar su estructura"
author: "Agarwal, Tan, Ronen, Singh y Yu · ICML 2022"
lang: es
---

*Resumen de lectura. Agarwal, A., Tan, Y. S., Ronen, O., Singh, C. y Yu, B. (2022). Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models. *ICML 2022*, PMLR 162:111–135. El archivo del proyecto es la versión preliminar de arXiv (2202.00858v1, febrero de 2022), anterior a la revisión de ICML. Implementación oficial: paquete `imodels` (`HSTreeRegressor`, `HSTreeClassifier`). La réplica reducida está en `ejemplos/hs_articulo.py`, con HS implementado a mano en unas 15 líneas; coincide exactamente con `imodels`.*

# La idea en un párrafo

Un árbol se regulariza casi siempre por su **estructura**: profundidad máxima, mínimo por hoja, poda por costo-complejidad. Pero la **predicción en cada hoja** se da por hecha: es el promedio de las respuestas que caen ahí. Hierarchical Shrinkage (HS) deja la estructura intacta y regulariza los **valores**: la predicción de una hoja se encoge hacia las medias de sus ancestros, más cuanto menos datos tiene cada ancestro, con un solo parámetro $\lambda$. Se aplica *a posteriori*, cuesta $O(\text{nodos})$, sirve para cualquier algoritmo de crecimiento y resulta ser **regresión ridge** sobre una base de “tocones” supervisados.

# La fórmula

Para un punto $x$, sea $t_L\subset t_{L-1}\subset\cdots\subset t_0$ el camino de su hoja a la raíz, $N(t)$ el número de muestras en el nodo $t$ y $\hat E_t\{y\}$ su respuesta media. La predicción del árbol es una suma telescópica:

$$\hat f(x)=\hat E_{t_0}\{y\}+\sum_{l=1}^L\big(\hat E_{t_l}\{y\}-\hat E_{t_{l-1}}\{y\}\big)$$

HS encoge cada incremento según el tamaño del **padre**:

$$\hat f_\lambda(x)=\hat E_{t_0}\{y\}+\sum_{l=1}^L\frac{\hat E_{t_l}\{y\}-\hat E_{t_{l-1}}\{y\}}{1+\lambda/N(t_{l-1})}$$

- Los cortes cerca de la raíz, con padres grandes, casi no se encogen.
- Los cortes profundos, con padres de pocas muestras, se encogen mucho: ahí es donde el promedio de la hoja es ruidoso.
- $\lambda=0$ devuelve el árbol original y $\lambda\to\infty$ la media global.
- No necesita los datos de entrenamiento, solo el árbol con medias y conteos por nodo.
- En un bosque o un boosting se aplica a cada árbol por separado.

**El contraste: encogimiento por hoja (LBS).** Encoger directamente la hoja hacia la raíz,

$$\hat f^{\,l}_\lambda(x)=\hat E_{t_0}\{y\}+\frac{\hat E_{t_L}\{y\}-\hat E_{t_0}\{y\}}{1+\lambda/N(t_L)}$$

aplica **el mismo** factor a todos los incrementos del camino. Es lo que hace XGBoost: el peso de hoja $w=-G/(H+\lambda)$, con pérdida cuadrática, es $\sum r_i/(n+\lambda)=\bar r\cdot\frac{1}{1+\lambda/n}$, el promedio del residuo encogido hacia cero según el tamaño de **la hoja** (nota 2). También aparece en BART como media posterior de una hoja con *prior* gaussiano. HS es la versión jerárquica: un factor distinto por nivel.

# Por qué funciona: HS es ridge (sección 3)

Klusowski (2021) mostró que un árbol es un modelo lineal sobre variables construidas con supervisión. A cada nodo interno $t$, con hijos $t_L$ y $t_R$, se le asocia el tocón

$$\psi_t(x)=\frac{N(t_R)\,\mathbf 1\{x\in t_L\}-N(t_L)\,\mathbf 1\{x\in t_R\}}{\sqrt{N(t_L)N(t_R)}}$$

que vale positivo en el hijo izquierdo, negativo en el derecho y cero fuera de $t$. Sobre la muestra, los vectores $\psi_t$ son **ortogonales** y $\lVert\psi_t\rVert^2=N(t)$. El árbol original es el ajuste por mínimos cuadrados de $y$ (centrada) sobre $\Psi=(\psi_{t_0},\dots,\psi_{t_{m-1}})$.

**Teorema 1.** Si $\hat\beta_\lambda$ resuelve el problema ridge $\min_\beta\sum_i(\beta^\top\Psi(x_i)-y_i)^2+\lambda\lVert\beta\rVert^2$, entonces $\hat f_\lambda(x)=\hat\beta_\lambda^\top\Psi(x)$.

> **Demostración (esbozo).** Por ortogonalidad, $\Psi^\top\Psi=\operatorname{diag}(N(t))$ y el problema se separa en $m$ problemas univariados: $\hat\beta_{\lambda,t}=\langle\psi_t,y\rangle/(N(t)+\lambda)$. Sustituyendo, $\langle\psi_t,y\rangle=\sqrt{N(t_L)N(t_R)}\,(\bar y_{t_L}-\bar y_{t_R})$. Para $x$ en el hijo izquierdo, la contribución del nodo es $\psi_t(x)\hat\beta_{\lambda,t}=\dfrac{N(t_R)(\bar y_{t_L}-\bar y_{t_R})}{N(t)+\lambda}$. Como $\bar y_{t_L}-\bar y_t=\dfrac{N(t_R)}{N(t)}(\bar y_{t_L}-\bar y_{t_R})$, la contribución es $\dfrac{\bar y_{t_L}-\bar y_t}{1+\lambda/N(t)}$, que es exactamente el término de HS. Igual para el hijo derecho. $\blacksquare$

LBS también es ridge, pero sobre la base de **indicadores de hoja** (one-hot de la hoja), que se obtiene de $\Psi$ por una transformación lineal invertible. Mismo espacio de funciones, distinta penalización.

**Por qué un solo $\lambda$ basta en HS y no en LBS** (argumento heurístico). En ridge univariado, el $\lambda$ óptimo es $\sigma^2/\beta^2$. Bajo “honestidad” (estructura independiente de las respuestas, por ejemplo con división de muestra):

- En la base de tocones, $\beta_*(t)$ es proporcional a una **diferencia** de medias entre hermanos. Esa diferencia decrece con la profundidad de forma suave ($\approx 2^{-\text{prof}(t)/p}$) y la profundidad es $O(\log n)$, así que el $\lambda$ óptimo varía poco entre nodos.
- En la base de hojas, $\beta_*(t)=E\{y\mid t\}$ es un **nivel**, que puede ser cualquier cosa, y el $\lambda$ óptimo varía mucho de hoja a hoja.

Este es el argumento que justifica el factor $1/(1+\lambda/N)$ con $N$ del padre. Es heurístico: en la práctica la estructura sí depende de $y$.

**Motivación de fondo.** Tan, Agarwal y Yu (2021) muestran cotas inferiores de generalización para árboles que predicen con el promedio de la hoja ante funciones aditivas: el problema no es la codicia del algoritmo, sino que promediar solo dentro de cada hoja desperdicia información. HS **agrupa información de varias hojas** a través de los ancestros.

# Resultados del artículo

**Datos.** 8 conjuntos de clasificación (de 270 a 6172 observaciones) y 8 de regresión (de 200 a 20 640), 2/3 para entrenar y 1/3 para probar, 10 particiones, $\lambda\in\{0.1,1,10,25,50,100\}$ por validación cruzada de 3 pliegues.

- **Árboles individuales.** HS **nunca empeora** y suele mejorar mucho. Con 15 hojas, la AUC sube en promedio 6.2 % (CART), 6.5 % (CART con poda) y 8 % (C4.5); el $R^2$ sube 9.8 % y 10.1 %. Las ganancias crecen con el número de hojas y en conjuntos pequeños. Que mejore incluso a CART **podado** muestra que regularizar valores no es lo mismo que regularizar estructura, y que se complementan: con HS conviene un árbol más grande.
- **Sesgo-varianza (figura 2).** Con datos de un modelo lineal, el error de CART tiene forma de U en el número de hojas; el de hsCART **baja de forma monótona**, porque HS controla la varianza. El $\lambda$ óptimo baja al aumentar las hojas.
- **HS frente a LBS.** HS es mejor de manera uniforme.
- **Bosques aleatorios.** hsRF mejora al RF y le gana a regularizarlo con profundidad máxima o con `mtry` (validados), en todos los conjuntos menos uno. Llega a su mejor desempeño con **unas cinco veces menos árboles**. Rinde como BART, pero se ajusta de 10 a 15 veces más rápido.

**Interpretabilidad** (sección 5, con RF):

- **Fronteras más simples.** En diabetes con dos variables, la AUC pasa de 0.733 a 0.787 y la frontera de decisión deja de estar fragmentada.
- **SHAP más estable.** La varianza de los valores SHAP entre 100 particiones distintas baja, incluso donde la exactitud no cambia.
- **SHAP más agrupado.** Los valores SHAP de cada variable forman cúmulos. Los autores lo leen como que HS elimina interacciones espurias y acerca el modelo a uno aditivo.

**Reproducción independiente.** Hay una reproducción publicada en el MLRC (NeurIPS 2023). Según su resumen, confirma que HS mejora en general a árboles y bosques, aunque “no siempre”, y que es muy rápido. HS supera a los otros regularizadores en árboles, pero no en bosques aleatorios. Además, los reproductores reportan que el código de experimentos de los autores no corría tal cual, y que el artículo no especifica cómo se ajustaron los hiperparámetros de los regularizadores del bosque.

# Réplica reducida

`ejemplos/hs_articulo.py`: HS a mano sobre árboles de scikit-learn, con $\lambda$ por validación cruzada de 3 pliegues en la rejilla del artículo. Como en la sección 2 del artículo, se ajusta **una vez** por pliegue y solo se recalculan los valores. La diferencia máxima con `imodels.HSTreeRegressor` es 0.0.

**A. Sesgo-varianza** ($y=\sum_{j\le10}x_j+\varepsilon$, $x\sim U[0,1]^{50}$, $\sigma^2=0.01$, $n=500$, 50 repeticiones; error medido contra $f$):

| hojas | CART: sesgo² / varianza / ECM | hsCART: sesgo² / varianza / ECM | $\lambda$ (mediana) |
|:--|:--|:--|:--|
| 2 | 0.693 / 0.100 / 0.793 | 0.716 / 0.072 / 0.787 | 100 |
| 8 | 0.437 / 0.256 / 0.693 | 0.514 / 0.156 / 0.670 | 50 |
| 16 | 0.351 / 0.325 / **0.676** | 0.454 / 0.174 / 0.627 | 50 |
| 32 | 0.292 / 0.405 / 0.697 | 0.425 / 0.177 / 0.603 | 50 |
| 128 | 0.258 / 0.574 / 0.832 | 0.415 / 0.179 / **0.594** | 50 |

Se ve la figura 2 del artículo. CART tiene su U clásica, con mínimo en 16 hojas. En hsCART la **varianza se estanca** en 0.18 aunque el árbol crezca, a cambio de algo de sesgo, y el ECM baja de forma monótona. Con 128 hojas, HS reduce el error 29 %.

**B. Datos del artículo** ($R^2$ en prueba, 10 particiones; CART con 20 hojas; RF con 50 árboles sin poda):

| | CART | hsCART | RF | hsRF |
|:--|:--|:--|:--|:--|
| diabetes ($n=442$) | 0.170 | **0.322** | 0.427 | 0.427 |
| Friedman1 ($n=200$) | 0.443 | **0.501** | 0.645 | 0.644 |

En árboles, la ganancia es grande y coincide con el artículo. En RF **no hay ganancia**, en línea con la reproducción independiente. Una posible razón: el promediado del bosque ya hace casi todo el trabajo de reducir varianza, y con $\lambda\le100$ el encogimiento de hojas de una muestra es limitado.

**C. Extensión: HS dentro de gradient boosting** (el artículo lo menciona pero no lo prueba). 200 árboles de 32 hojas, tasa 0.05, pérdida cuadrática:

| | GB | hsGB | $\lambda$ elegido |
|:--|:--|:--|:--|
| diabetes | 0.388 | **0.412** | 25 |
| Friedman1 | 0.793 | 0.793 | 0.1 (sin HS) |

En el conjunto pequeño y ruidoso ayuda; en Friedman1 la validación cruzada elige el $\lambda$ mínimo, es decir, prácticamente no aplicar HS. Esto tiene sentido: el boosting ya encoge (tasa de aprendizaje y, en XGBoost, el $\lambda$ de hoja, que es LBS). HS añade el encogimiento **jerárquico** dentro de cada árbol, que solo importa si los árboles son profundos y sus hojas tienen pocas muestras.

# Lectura crítica

- **La teoría es para regresión con honestidad.** La justificación de un $\lambda$ común es heurística y supone estructura independiente de $y$. En clasificación se aplica a las proporciones de clase igual que a medias.
- **Ganancias en árboles individuales: sólidas. En bosques: discutibles.** La reproducción independiente y la réplica de esta nota coinciden en que en RF la mejora no es robusta.
- **La rejilla de $\lambda$ es corta** (máximo 100). Con árboles profundos y muchos datos podría hacer falta más.
- **No se prueba en boosting**, que es el caso de tus notas. Según la réplica, el efecto ahí es pequeño y dependiente de los datos.
- **Interpretabilidad:** “SHAP más agrupado” es un indicio cualitativo de que el modelo es más aditivo, no una prueba. Sí es útil como diagnóstico: comparar SHAP con y sin HS muestra qué interacciones dependen de hojas pequeñas.

# Para conectar con tus notas

- **Nota 2** (XGBoost): el $\lambda$ de la regularización L2 de los pesos de hoja es exactamente LBS hacia cero. HS sería la versión que encoge también a lo largo del camino.
- **Notas 5 y 6** (explicabilidad): HS es una herramienta barata para estabilizar SHAP y reducir interacciones espurias, y se puede usar como diagnóstico.
- **Notas 9 y 10** (GPBoost, LaGaBoost): el mismo principio de encoger según el tamaño de muestra. Allí se encoge el promedio de un **grupo** hacia la media global con un factor $n_j/(n_j+\sigma^2/\sigma_1^2)$ estimado por verosimilitud; aquí se encoge el incremento de un **nodo** con $N/(N+\lambda)$ elegido por validación cruzada. La forma es la misma porque en ambos casos es ridge (un BLUP es una estimación ridge).
- **Nota 7** (teoría del aprendizaje): HS no cambia la clase de funciones (mismo árbol), solo cómo se ajustan los coeficientes. Reduce la complejidad efectiva sin reducir la capacidad.
