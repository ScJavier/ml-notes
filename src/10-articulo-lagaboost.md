---
title: "Artículo 2 · Latent Gaussian Model Boosting (LaGaBoost)"
author: "Fabio Sigrist · IEEE TPAMI 2023"
lang: es
---

*Resumen de lectura. Sigrist, F. (2023). Latent Gaussian Model Boosting. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(2), 1894–1905. arXiv:2105.08966 (v6, 2022). Implementado en la misma biblioteca `gpboost` que la nota 9. La réplica reducida está en `ejemplos/lagaboost_articulo.py` (`gpboost` 1.8.0, `lightgbm` 4.7.0).*

# La idea en un párrafo

GPBoost (nota 9) supone una respuesta gaussiana, lo que permite integrar el efecto aleatorio en forma cerrada. Para clasificación, conteos o datos positivos asimétricos hay que pasar a un **modelo gaussiano latente**: la respuesta sigue una distribución conocida (Bernoulli, Poisson, gamma…) cuyo parámetro depende de $F(X)+Zb$, con $b$ gaussiano. La verosimilitud marginal ya no tiene forma cerrada. LaGaBoost la reemplaza por la **aproximación de Laplace** y hace boosting funcional sobre esa aproximación, estimando los parámetros de covarianza al mismo tiempo. Es, literalmente, un GLMM con parte fija no paramétrica.

# El modelo

$$y_i\mid\mu_i\ \overset{\text{ind}}{\sim}\ p(y_i\mid\mu_i,\xi),\qquad \mu = F(X)+Zb,\qquad b\sim N\big(0,\Sigma(\theta)\big)$$

- $p(y\mid\mu,\xi)$ es la verosimilitud condicional; $\xi$ son parámetros auxiliares (por ejemplo, la forma de una gamma). Bernoulli y Poisson no tienen.
- $\mu\in\mathbb R^n$ está en la escala del predictor lineal (vía función liga): $P(y_i=1)=\Phi(\mu_i)$ en probit, $E[y_i]=e^{\mu_i}$ en Poisson.
- $Z$, $\Sigma$ y los tres tipos de predictores ($X$, $S$, $Z$) son exactamente como en GPBoost: efectos agrupados, procesos gaussianos o combinaciones.

La verosimilitud marginal es

$$p(y\mid F,\theta,\xi)=\int p(y\mid\mu,\xi)\,p(b\mid\theta)\,db$$

y lo ideal sería minimizar $-\log p(y\mid F,\theta,\xi)$ conjuntamente en $(F,\theta,\xi)$. No hay forma cerrada salvo en el caso gaussiano, así que se usa una aproximación $L^A$ que cumpla dos requisitos: que sea **barata de evaluar** (se evalúa muchas veces) y que su **gradiente respecto a $F$** también lo sea.

# La aproximación de Laplace

Se aproxima el integrando por una gaussiana centrada en su moda:

$$\tilde b=\arg\max_b\ \log p(y\mid F+Zb,\xi)-\tfrac12 b^\top\Sigma^{-1}b$$

(se encuentra con Newton), con $\tilde\mu=F+Z\tilde b$ y $\tilde W=\operatorname{diag}\big(-\partial^2_{\mu_i}\log p(y_i\mid\tilde\mu_i,\xi)\big)$. Entonces

$$L^{LA}(y,F,\theta,\xi)=-\log p(y\mid\tilde\mu,\xi)+\tfrac12\tilde b^\top\Sigma^{-1}\tilde b+\tfrac12\log\det\big(\Sigma Z^\top\tilde WZ+I_m\big)$$

Equivale a aproximar la posterior de los efectos por $b\mid y\approx N\big(\tilde b,\ (Z^\top\tilde WZ+\Sigma^{-1})^{-1}\big)$. Para usar Laplace, $\log p(y_i\mid\mu_i)$ debe ser tres veces derivable en $\mu_i$: la tercera derivada aparece en los gradientes.

**Alternativas** que el artículo menciona: cuadratura de Gauss-Hermite adaptativa (factible si la integral se factoriza en piezas de baja dimensión, como con efectos de un nivel) y *expectation propagation*, más precisa en clasificación binaria pero más cara.

## El gradiente respecto a $F$ (proposición 2.1)

$$\frac{\partial L^{LA}}{\partial F_i}=\underbrace{-\frac{\partial\log p(y_i\mid\tilde\mu_i,\xi)}{\partial\tilde\mu_i}}_{\text{(1)}}+\underbrace{\tfrac12\operatorname{tr}\!\Big[(Z^\top\tilde WZ+\Sigma^{-1})^{-1}Z^\top\frac{\partial\tilde W}{\partial F_i}Z\Big]}_{\text{(2)}}+\underbrace{\Big(\frac{\partial L^{LA}}{\partial\tilde b}\Big)^{\!\top}\frac{\partial\tilde b}{\partial F_i}}_{\text{(3)}}$$

con $\partial\tilde b/\partial F_i=-(Z^\top\tilde WZ+\Sigma^{-1})^{-1}Z^\top\tilde W_{\cdot i}$, que sale de derivar implícitamente la condición de primer orden de la moda.

Cómo leerlo:

- **(1)** es el gradiente de cualquier boosting de clasificación (*LogitBoost*), pero evaluado en $\tilde\mu=F+Z\tilde b$, es decir, **después de sumar el efecto aleatorio predicho**. Es el análogo exacto del caso gaussiano de la nota 9, donde el pseudo-residuo era $y-F-Z\hat b$ (escalado).
- **(2)** y **(3)** son correcciones porque $\tilde W$ y $\tilde b$ dependen de $F$. Usan la tercera derivada de la log-verosimilitud. En el caso gaussiano $\tilde W$ es constante y ambas desaparecen.

Los gradientes respecto a $\theta$ y $\xi$ tienen la misma estructura: término explícito más término implícito a través de $\tilde b$.

**Cómputo.** Con efectos agrupados, $Z^\top\tilde WZ+\Sigma^{-1}$ es rala y de dimensión $m<n$: Cholesky ralo. Con un GP se usa Woodbury y se factoriza $I_m+\tilde W^{1/2}Z\Sigma Z^\top\tilde W^{1/2}$, como en Rasmussen y Williams (cap. 5). En ambos casos, el cuello de botella es la factorización de Cholesky.

# El algoritmo

**LaGaBoost (algoritmo 1).** Igual que GPBoost: inicializar $F_0$ como constante; en cada iteración,

1. $(\theta_m,\xi_m)=\arg\min L^{A}(y,F_{m-1},\theta,\xi)$, arrancando en $(\theta_{m-1},\xi_{m-1})$.
2. Ajustar un árbol $f_m$ por mínimos cuadrados al gradiente negativo $-\partial L^A/\partial F$.
3. $F_m=F_{m-1}+\nu f_m$.

Aquí solo se usa el **paso de gradiente** (la nota 9 tenía también Newton e híbrido). En el artículo no aparece un paso de Newton sobre $L^{LA}$; requeriría segundas derivadas de una función que ya involucra terceras.

**LaGaBoostOOS (algoritmo 2).** La misma idea que GPBoostOOS: estimar $(\theta,\xi)$ con predicciones de $F$ fuera de muestra y después reentrenar con todo y los parámetros fijos. Motivo: si el boosting interpola los datos de entrenamiento, la estimación de la covarianza dentro de muestra se sesga.

**Predicción.** Para la parte latente,

$$\mu_p\mid y\approx N(\omega_p,\Omega_p),\qquad \omega_p=F(X_p)+Z_p(\Sigma,\Sigma_{op})^\top Z^\top\,\nabla_{\tilde\mu}\log p(y\mid\tilde\mu,\xi)$$

y la covarianza $\Omega_p$ sale de la posterior gaussiana. Para la respuesta hay que integrar $p(y_p\mid\mu_p)$ contra esa normal. Con Bernoulli-probit es analítico: $P(y_p=1\mid y)=\Phi\big(\omega_p/\sqrt{1+\Omega_p}\big)$. Con otras ligas se usa Gauss-Hermite adaptativa. Nótese que la media de la respuesta **no** es $\Phi(\omega_p)$: la incertidumbre del efecto aleatorio aplana la probabilidad hacia 0.5. Para un grupo nuevo, $\omega_p=F(X_p)$ y $\Omega_p$ incluye $\sigma_1^2$.

# Contra qué se compara

- **Modelos lineales gaussianos latentes** (GLMM o GP con $F$ lineal): falla en cuanto $F$ tiene no linealidades o interacciones.
- **LogitBoost / boosting de Poisson independiente** con el grupo o las coordenadas como variables más. Siguiendo la recomendación de LightGBM para categóricas de muchas categorías, el grupo entra como **numérico**, no categórico.
- **mboost** con *splines* espaciales y efectos de cresta.
- **Trabajo previo con árboles y efectos aleatorios no gaussianos** (GMERT, árboles GLMM de Fokkema et al., BiMM, GMERF): el artículo señala que son **heurísticos**, sin una función objetivo clara: no maximizan una verosimilitud marginal ni una aproximación a ella, ni por coordenadas ni con EM. No los incluye en la comparación.

# Resultados

**Simulación binaria agrupada** ($n=5000$, 500 grupos de 10, probit, $F$ de Hajjem con varianza 1, $\sigma_1^2=1$, 100 repeticiones). Las log-pérdidas del artículo son sumas sobre 5000 observaciones; aquí las divido entre $n$:

| | LaGaBoost | GLMM lineal | LogitBoost | mboost | LaGaBoostOOS |
|:--|:--|:--|:--|:--|:--|
| error de clasificación | **0.237** | 0.287 | 0.339 | 0.283 | 0.237 |
| log-pérdida por obs. | **0.484** | 0.557 | 0.606 | 0.583 | 0.484 |
| sesgo de $\hat\sigma_1^2$ | −0.195 | −0.354 | — | — | −0.194 |
| tiempo (s) | 0.66 | 0.04 | 0.08 | 17 | 3.3 |

En los grupos nuevos LaGaBoost también es el mejor (error 0.343).

**Simulación binaria espacial** ($n=500$, GP exponencial con $\rho=0.1$): error 0.309 frente a 0.331 del GP lineal, 0.350 de LogitBoost con coordenadas y 0.381 de mboost. Aquí la versión OOS sí mejora los parámetros: el RMSE de $\hat\sigma_1^2$ baja de 0.62 a 0.49.

**Poisson** (mismos diseños, varianzas 0.2): resultados cualitativamente iguales.

**¿Cuándo ayuda más? (sección 4.1).** Variando las observaciones por grupo (10 a 200) y el rango espacial ($\rho$ de 0.1 a 1), la mejora relativa sobre LogitBoost es mayor cuantas **menos observaciones por grupo** hay y cuanto **más rápido decae** la correlación espacial. Es decir, cuando la complejidad del efecto aleatorio es alta en relación con $n$, que es justo cuando la regularización apropiada importa más. El autor conjetura que parte de la ganancia viene de aprender mejor $F$, no solo los efectos.

**Datos reales** (validación cruzada anidada de 4 pliegues):

| | | LaGaBoost | lineal | LogitBoost | mboost |
|:--|:--|:--|:--|:--|:--|
| Pobreza (NLSY, 5755 obs., 1151 personas) | AUC | **0.732** | 0.731 | 0.700 | 0.703 |
| | log-pérdida | **0.579** | 0.588 | 0.601 | 0.608 |
| Especie vegetal (909 sitios, espacial) | AUC | **0.738** | 0.706 | 0.683 | 0.635 |
| | log-pérdida | **0.493** | 0.563 | 0.527 | 0.566 |

En pobreza, la ganancia sobre el GLMM lineal es pequeña. Mi lectura: hay solo 7 variables y un efecto por persona muy informativo, así que la no linealidad de $F$ aporta poco.

# Réplica reducida

`ejemplos/lagaboost_articulo.py`: mismo diseño binario agrupado, 10 repeticiones por configuración, hiperparámetros de árbol fijos (tasa 0.05, profundidad 5, mínimo 10 por hoja) y número de árboles por **paro temprano** en un conjunto de validación aparte. Para LaGaBoost, la validación usa el modelo completo ($F$ + efecto). LightGBM recibe el grupo como numérico, como en el artículo.

| obs. por grupo | LaGaBoost | LightGBM + grupo numérico | LightGBM sin grupo | $\hat\sigma_1^2$ (real 1) |
|:--|:--|:--|:--|:--|
| 10 (500 grupos) | **0.487** / 0.240 | 0.596 / 0.328 | 0.606 / 0.342 | 0.76 |
| 50 (100 grupos) | **0.465** / 0.224 | 0.519 / 0.259 | 0.607 / 0.346 | 0.86 |
| 200 (25 grupos) | **0.446** / 0.213 | 0.464 / 0.222 | 0.601 / 0.337 | 0.87 |

(Log-pérdida / error en grupos vistos; error estándar ≈ 0.002–0.005.)

- Con 10 por grupo se reproduce el artículo casi exactamente: 0.487 frente a 0.484, y 0.596 frente a 0.606 para el boosting independiente.
- La ventaja se reduce de 0.109 a 0.054 y a 0.019 de log-pérdida cuando crecen las observaciones por grupo, como en la figura 2 del artículo.
- **Grupos nuevos:** LaGaBoost da 0.605–0.607 en las tres configuraciones, esencialmente lo mismo que el modelo sin grupo (cae a $F$, como debe). LightGBM con el grupo numérico empeora (0.63, 0.76, 0.87): un ID nunca visto cae en particiones arbitrarias del eje del ID y recibe el efecto de grupos vecinos que no tienen nada que ver.
- $\hat\sigma_1^2$ sale **subestimado**, más cuanto menos datos por grupo hay, igual que en el artículo (−0.195).

# Lectura crítica

- **El sesgo de $\hat\sigma_1^2$ probablemente no es (solo) interpolación.** En el caso binario agrupado, LaGaBoostOOS no lo corrige (−0.194 frente a −0.195). Que la aproximación de Laplace subestima varianzas de efectos aleatorios en GLMM binarios con grupos pequeños es conocido en la literatura de GLMM. La réplica es consistente con eso: el sesgo baja al haber más datos por grupo. Atribuirlo a Laplace es interpretación mía; el artículo no lo discute. Si importan las probabilidades calibradas o la varianza del efecto, conviene contrastar con una integración más precisa (Gauss-Hermite adaptativa).
- **Solo gradiente.** Con pérdidas como la logística, el paso de Newton de XGBoost y LightGBM (nota 2) suele converger en menos árboles. Aquí no está disponible.
- **La comparación con boosting independiente usa el grupo como numérico.** Faltan LightGBM con el grupo categórico y CatBoost, que la nota 9 sí incluye en el caso gaussiano.
- **Los experimentos espaciales usan $n=500$** para evitar aproximaciones. No hay evaluación de Vecchia con respuestas no gaussianas en este artículo.
- **No hay incertidumbre sobre $\hat F$:** la varianza predictiva viene solo de $b$.

# Para conectar con tus notas

- **Nota 1:** LaGaBoost es boosting funcional sobre una pérdida que no es una suma de términos independientes y que, además, está aproximada.
- **Nota 6** (explicabilidad en clasificación): con efectos aleatorios, las atribuciones (SHAP, ALE) de $\hat F$ son atribuciones **condicionales al grupo**. El efecto del grupo queda fuera de $F$ y se reporta aparte como $\tilde b$.
- **Nota 8** (conformal): para clasificación por tipo de producto, Mondrian sobre las probabilidades de LaGaBoost da la garantía por tipo que el modelo, por sí solo, no da.
- **Nota 9:** todo lo de GPBoost aplica; lo nuevo es la aproximación de Laplace y su efecto sobre los parámetros de covarianza.
