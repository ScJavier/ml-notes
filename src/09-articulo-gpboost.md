---
title: "Artículo 1 · Gaussian Process Boosting (GPBoost)"
author: "Fabio Sigrist · JMLR 2022"
lang: es
---

*Resumen de lectura. Sigrist, F. (2022). Gaussian Process Boosting. *Journal of Machine Learning Research*, 23(232), 1–46. arXiv:2004.02653 (v7, 2024). Implementación: biblioteca `gpboost` (C++, con paquetes de Python y R; los árboles los construye LightGBM). La réplica reducida del final está en `ejemplos/gpboost_articulo.py` (`gpboost` 1.8.0, `lightgbm` 4.7.0).*

# La idea en un párrafo

Un modelo mixto clásico tiene una parte fija **lineal**, $X\beta$, y una parte aleatoria con estructura de covarianza (efectos por grupo, proceso gaussiano en el espacio o el tiempo). El boosting de árboles tiene una parte fija **no paramétrica** pero supone que, condicional a $F(X)$, las observaciones son **independientes**. GPBoost junta las dos cosas: la parte fija $F(\cdot)$ se aprende con boosting y la parte aleatoria $Zb$ se modela como en un modelo mixto, estimando conjuntamente $F$ y los parámetros de covarianza $\theta$ por máxima verosimilitud. De paso, da una forma de tratar **categóricas de alta cardinalidad** en boosting: como efectos aleatorios en lugar de variables de partición.

# El modelo

$$y = F(X) + Zb + \epsilon,\qquad b\sim N(0,\Sigma),\qquad \epsilon\sim N(0,\sigma^2 I_n)$$

- $F:\mathbb R^p\to\mathbb R$ vive en el espacio generado por los aprendices base (aquí, árboles de regresión). Es el **efecto fijo**, ahora no lineal.
- $b\in\mathbb R^m$ son los **efectos aleatorios**; $\Sigma=\Sigma(\theta)$ depende de pocos parámetros y, posiblemente, de variables $S$ (coordenadas, tiempo).
- $Z\in\mathbb R^{n\times m}$ conecta efectos con observaciones: matriz de incidencia (dummies del grupo), identidad (GP con un punto por ubicación) o con covariables (pendientes aleatorias).

Hay entonces **tres** conjuntos de predictores, que pueden traslaparse: $X$ (entra en $F$), $S$ (define la covarianza) y $Z$ (conecta $b$ con $y$). El boosting usual solo conoce $X$: o ignora $S$ y $Z$, o los mete en $X$.

Marginalmente $y\sim N\big(F(X),\Psi\big)$ con $\Psi = Z\Sigma Z^\top+\sigma^2 I_n$, y la pérdida es la log-verosimilitud negativa

$$L(y,F,\theta)=\tfrac12 (y-F)^\top\Psi^{-1}(y-F)+\tfrac12\log\det\Psi+\tfrac n2\log 2\pi$$

**Casos que cubre.**

| Estructura | $\Sigma$ | Uso típico |
|:--|:--|:--|
| Efectos aleatorios de un nivel | $\sigma_1^2 I_m$, $Z$ = dummies del grupo | panel, longitudinal, **categórica de alta cardinalidad** |
| Anidados, cruzados, pendientes aleatorias | bloques diagonales; $Z$ con covariables | varios niveles de agrupación |
| Proceso gaussiano | $\sigma_1^2\, r(\lVert s_j-s_k\rVert/\rho)$ | datos espaciales, temporales, o “ubicaciones” no espaciales |
| Combinaciones | suma de las anteriores | un GP global + efectos por grupo, o un GP por grupo |

Con $r(d)=e^{-d}$ (exponencial) o $e^{-d^2}$ (gaussiana); $\rho$ es el rango.

# El algoritmo

## Boosting con $\theta$ fijo

El objetivo es $\min_{F\in\mathcal H,\ \theta\in\Theta} L(y,F(X),\theta)$. Con $\theta$ fijo es boosting estándar sobre una pérdida **no separable**: ya no es una suma de $n$ términos independientes, sino un solo término multivariado. El gradiente funcional evaluado en los datos es

$$-\frac{\partial L}{\partial F}\Big|_{F_{m-1}} = \Psi^{-1}(y-F_{m-1})$$

Las tres variantes de paso (las mismas que en las notas 1 y 2):

- **Gradiente:** $f_m=\arg\min_f\lVert \Psi^{-1}(y-F_{m-1})-f\rVert^2$.
- **Newton:** $f_m=\arg\min_f (y-F_{m-1}-f)^\top\Psi^{-1}(y-F_{m-1}-f)$, mínimos cuadrados **generalizados**.
- **Híbrido:** la estructura del árbol (las particiones $\alpha$) sale del paso de gradiente y los valores de hoja $\gamma$ salen por MCG: $\gamma_m=(h^\top\Psi^{-1}h)^{-1}h^\top\Psi^{-1}(y-F_{m-1})$, con $h$ la matriz de pertenencia a hojas.

Detalle útil: Newton e híbrido son **invariantes a la escala** de la pérdida (no dependen de $\sigma^2$, porque $\Psi^{-1}$ y $(\Psi/\sigma^2)^{-1}$ dan el mismo minimizador); el gradiente no.

**Qué es el pseudo-residuo.** (Derivación mía, no está así en el artículo.) Con Woodbury, $\Psi^{-1}=\sigma^{-2}\big[I-Z(Z^\top Z+\sigma^2\Sigma^{-1})^{-1}Z^\top\big]$, y como el BLUP de $b$ es $\hat b=(Z^\top Z+\sigma^2\Sigma^{-1})^{-1}Z^\top(y-F)$,

$$\Psi^{-1}(y-F)=\frac{1}{\sigma^2}\big(y-F-Z\hat b\big)$$

El árbol se ajusta al **residuo al que ya se le quitó la predicción del efecto aleatorio**. En un solo nivel, $\hat b_j=\dfrac{n_j}{n_j+\sigma^2/\sigma_1^2}\,\bar r_j$, con $\bar r_j$ el residuo medio del grupo $j$: el promedio del grupo **encogido** hacia cero según cuántos datos tiene. Es exactamente el razonamiento de regresión con efectos aleatorios, ahora dentro de cada iteración de boosting.

## Estimación conjunta: algoritmo 1 (GPBoost)

Inicializar $F_0=\arg\min_c L(y,c\mathbf 1,\theta_0)$. Para $m=1,\dots,M$:

1. $\theta_m=\arg\min_\theta L(y,F_{m-1},\theta)$, optimizando **hasta converger** pero arrancando en $\theta_{m-1}$ (descenso por gradiente con Nesterov o *Fisher scoring*).
2. Paso de boosting (gradiente, Newton o híbrido) con $\Psi_m=\Psi(\theta_m)$.
3. $F_m=F_{m-1}+\nu f_m$.

**Por qué así y no de otra forma.**

- **Un paso en $F$ y un paso en $\theta$:** barato, pero el boosting se detiene temprano (es su regularización) y entonces nada garantiza que $\theta$ haya convergido.
- **Descenso por coordenadas completo en ambos** (reentrenar $F$ hasta el final, luego $\theta$, y repetir): caro, porque $F$ se reaprende muchas veces. Es lo que hacen, en esencia, MERT/MERF y RE-EM tree.
- **GPBoost:** optimización completa en $\theta$ (barata gracias al arranque en caliente) y un paso de boosting en $F$. Así $\theta$ siempre está en su óptimo dado el $F$ actual, y el paro temprano sobre $F$ no deja a $\theta$ a medias.

El artículo afirma convergencia **si** el riesgo es convexo en $(F,\theta)$. Ojo: la log-verosimilitud gaussiana en general no es convexa en los parámetros de covarianza, así que eso es una condición, no un resultado aplicable sin más.

El gradiente en $\theta$ es el de cualquier modelo mixto:

$$\frac{\partial L}{\partial\theta_k}=-\tfrac12 (y-F)^\top\Psi^{-1}\frac{\partial\Psi}{\partial\theta_k}\Psi^{-1}(y-F)+\tfrac12\operatorname{tr}\Big(\Psi^{-1}\frac{\partial\Psi}{\partial\theta_k}\Big)$$

y $\sigma^2$ tiene solución cerrada si se factoriza: $\hat\sigma^2=\frac1n (y-F)^\top(\Psi^\dagger)^{-1}(y-F)$ con $\Psi^\dagger=\Psi/\sigma^2$. No hay REML: no hay forma cerrada de $F$ dado $\theta$ para “perfilarla”.

## El problema de la interpolación: algoritmo 2 (GPBoostOOS)

Los árboles con boosting pueden interpolar los datos de entrenamiento y aun así generalizar bien. Pero si $F$ absorbe parte del ruido, el residuo dentro de muestra es demasiado pequeño y **$\hat\sigma^2$ sale sesgado hacia abajo**, incluso con paro temprano. La solución:

1. Partir los datos en $k$ pliegues.
2. En cada pliegue, entrenar GPBoost y predecir $\hat F$ **fuera de muestra**.
3. Estimar $\theta$ con $y-\hat F_{\text{oos}}$.
4. Reentrenar GPBoost con todos los datos y $\theta$ **fijo** (saltando el paso 1 del algoritmo 1).

Cuesta unas $k$ veces más. En los experimentos el error de predicción casi no cambia, pero los parámetros de covarianza (y por tanto las **predicciones probabilísticas**) mejoran.

## Datos grandes: aproximación de Vecchia

Con efectos agrupados, $\Psi$ es rala y todo se hace con álgebra rala y Woodbury. Con un GP, la factorización de Cholesky cuesta $O(m^3)$ en tiempo y $O(m^2)$ en memoria. La aproximación de Vecchia condiciona cada observación solo en sus $m_v$ vecinos más cercanos ya ordenados:

$$p(y\mid F,\theta)=\prod_i p(y_i\mid y_{1:i-1})\approx\prod_i p\big(y_i\mid y_{N(i)}\big)$$

Cada factor es gaussiano: $y_i\mid y_{N(i)}\sim N\big(F_i+A_i(y_{N(i)}-F_{N(i)}),\,D_i\big)$. Con $B$ triangular inferior unitaria ($B_{i,N(i)}=-A_i$) y $D=\operatorname{diag}(D_i)$, la precisión aproximada es

$$\tilde\Psi^{-1}=B^\top D^{-1}B$$

rala. Costo $O(n\,m_v^3)$ y memoria $O(n\,m_v)$: lineal en $n$. Suele bastar $m_v$ entre 10 y 15 (Datta et al., 2016).

**Aporte técnico del artículo** (proposiciones 3.1 y 3.2): fórmulas para el gradiente y la información de Fisher de la verosimilitud de Vecchia que solo requieren derivar $B$ y $D$, sin formar ni guardar $\Psi$ ni $\partial\Psi/\partial\theta_k$. Por observación se factoriza una matriz de $|N(i)|\times|N(i)|$, la mitad del trabajo del enfoque de Guinness (2021), que factoriza dos. Las proposiciones 3.3 y 3.4 dan la predicción con Vecchia según el orden (observados primero: reutiliza vecinos; predicciones primero: más precisa).

## Predicción

Es la condicional gaussiana estándar:

$$\mu_p = F(X_p)+Z_p\,\mathrm{Cov}(b_{\text{todo}},b)\,Z^\top\Psi^{-1}\big(y-F(X)\big)$$

y $\Xi_p$ la covarianza condicional correspondiente. Consecuencias prácticas:

- **Grupo visto:** $\hat F(x)+\hat b_j$, el BLUP encogido.
- **Grupo nuevo:** $\hat b_j=0$; la predicción es $\hat F(x)$ y la varianza incluye $\sigma_1^2$. Lo que se gana aquí viene **solo** de haber aprendido mejor $F$.
- **Sumas** (valor total de un portafolio, lluvia en una cuenca): $\mathbf 1^\top y_p\mid y\sim N(\mathbf 1^\top\mu_p,\ \mathbf 1^\top\Xi_p\mathbf 1)$. Aquí la correlación entre unidades importa mucho, y un boosting independiente subestima la varianza de la suma.

# Contra qué se compara, y por qué importa

- **Meter el grupo como categórica en el boosting** (LightGBM particiona las categorías en dos conjuntos; CatBoost usa estadísticas de target ordenadas, nota 4). Equivale a efectos **fijos**: mayor varianza con muchas categorías y pocos datos por categoría, y sin predicción probabilística.
- **MERT, MERF y RE-EM tree.** El apéndice A aclara qué optimizan: RE-EM tree es un descenso por coordenadas sobre la verosimilitud conjunta de $(F,\theta,b)$, no un EM. En MERT/MERF el “paso E” ajusta $F$ con una verosimilitud independiente tras restar $\hat b$, no con la verosimilitud multivariada, así que no queda claro a qué convergen. En las simulaciones, MERF a menudo no converge y es órdenes de magnitud más lento.
- **mboost** con *splines* espaciales y aprendices de cresta para grupos: continuo en el espacio, pero determinista y con la maldición de la dimensión en las ubicaciones.
- **Dos pasos** (boosting independiente y luego un GP sobre los residuos): no aprovecha la correlación para aprender $F$.

# Resultados

**Simulación, datos agrupados** ($n=5000$, 500 grupos de 10, $F$ de Hajjem, $\sigma_1^2=\sigma^2=1$, 100 repeticiones, hiperparámetros por validación cruzada):

| | GPBoost | MERF | LSBoost (cat.) | CatBoost | REEMtree | mboost | LinearME | GPBoostOOS |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| RMSE grupos vistos | **1.100** | 1.104 | 1.156 | 1.183 | 1.171 | 1.335 | 1.342 | 1.102 |
| RMSE grupos nuevos | **1.458** | 1.460 | 1.493 | 1.464 | 1.506 | 1.520 | 1.635 | 1.459 |
| RMSE de $F$ | **0.337** | 0.349 | — | — | 0.511 | 0.550 | 0.814 | 0.341 |
| sesgo de $\hat\sigma^2$ | −0.167 | 0.115 | −0.117 | 0.071 | 0.144 | 0.627 | 0.657 | 0.135 |
| tiempo (s) | 0.94 | 417 | 0.33 | 3.3 | 1.3 | 14 | 0.05 | 3.1 |

**Simulación, datos espaciales** ($n=500$, GP exponencial): GPBoost tiene RMSE 1.374 frente a 1.406 del enfoque en dos pasos, 1.466 del GP lineal y 1.474 de LSBoost con coordenadas. La diferencia más grande aparece en la **CRPS de sumas** de 20 puntos cercanos: 6.47 frente a 8.51 de LSBoost y 10.45 de mboost.

**Con $F$ lineal**, el modelo lineal gana, pero GPBoost queda muy cerca: pagar flexibilidad cuesta poco.

**Datos reales.**

- *Salarios* (panel NLSY, 28 534 observaciones de 4 711 mujeres; el ID es la categórica de alta cardinalidad): RMSE 0.296 frente a MERF 0.299, CatBoost 0.303, LinearME 0.305 y LightGBM 0.313–0.321. GPBoost gana en los 4 pliegues.
- *Casas* (25 357 ventas en Lucas County, Ohio; Vecchia con 50 vecinos): en predicción individual empata con LightGBM con coordenadas (RMSE 0.266 ambos). En **sumas** de 20 casas cercanas gana claramente: RMSE 1.72 frente a 1.94, y pérdida cuantílica al 5 % 0.185 frente a 0.234. En submuestras pequeñas (unas 400 obs.) gana también en predicción individual.
- *UCI* (9 conjuntos pequeños, el GP usa todas las variables como “ubicaciones”): rango promedio en RMSE de 1.33, frente a 2.22 de LightGBM y 2.44 del GP solo.

# Réplica reducida

`ejemplos/gpboost_articulo.py` repite el escenario agrupado con 20 repeticiones e **hiperparámetros fijos** (tasa 0.05, profundidad 5, 300 árboles, mínimo 10 por hoja; sin búsqueda ni paro temprano), así que es ilustrativo. GPBoostOOS está implementado a mano porque `gpb.cv(fit_GP_cov_pars_OOS=True)` todavía no existe en Python.

| | RMSE vistos | RMSE nuevos | RMSE de $F$ | $\hat\sigma^2$ (real 1) | $\hat\sigma_1^2$ (real 1) |
|:--|:--|:--|:--|:--|:--|
| GPBoost | 1.119 | 1.456 | 0.372 | 0.56 | 1.00 |
| GPBoostOOS | **1.111** | **1.451** | **0.351** | 1.17 | 1.00 |
| LightGBM, grupo categórico | 1.131 | 1.457 | — | — | — |
| LightGBM, sin grupo | 1.486 | 1.477 | — | — | — |

(Error estándar entre repeticiones ≈ 0.004–0.006 en los RMSE.)

Lo que se ve:

- Sin paro temprano, $F$ absorbe ruido y **$\hat\sigma^2$ se va a la mitad** (0.56). Es el fenómeno de la sección 3.3 en versión exagerada; en el artículo, con paro temprano por validación cruzada, el sesgo es −0.17.
- GPBoostOOS lo corrige y se pasa un poco hacia arriba (1.17), como en el artículo (+0.13).
- $\hat\sigma_1^2$ sale bien en ambos casos: con 10 observaciones por grupo, el promedio del grupo identifica bien el efecto.
- La ganancia en grupos nuevos es pequeña. Tiene sentido: ahí solo ayuda un mejor $F$.

Código mínimo:

```python
import gpboost as gpb
gp = gpb.GPModel(group_data=grupo, likelihood="gaussian")   # o gp_coords=coords, cov_function="exponential"
bst = gpb.train(params=params, train_set=gpb.Dataset(X, y), gp_model=gp, num_boost_round=300)
gp.get_cov_pars()                                           # sigma^2 (error) y sigma_1^2 (grupo)
pred = bst.predict(data=X_nuevo, group_data_pred=grupo_nuevo, pred_latent=False)
pred["response_mean"], pred["response_var"]
```

# Lectura crítica

- **Solo respuesta gaussiana.** Clasificación, conteos, etc. requieren integrar $b$ fuera. Es justo el siguiente artículo (LaGaBoost, aproximación de Laplace).
- **No hay incertidumbre sobre $\hat F$.** La varianza predictiva incluye $b$ y $\epsilon$ pero trata $\hat F$ como fijo. El autor lo deja como trabajo futuro. Para intervalos con garantía, conformal (nota 8) sigue siendo útil encima de GPBoost.
- **Sesgo de $\hat\sigma^2$.** Si importan las predicciones probabilísticas, hay que usar OOS o, al menos, paro temprano honesto.
- **Validación cruzada con datos dependientes.** Partir al azar mezcla grupos entre pliegues, y eso mide “grupos vistos”. Si en producción llegan grupos nuevos, hay que validar por grupo.
- **Dónde gana más** (lo muestra la sección 4.1 del artículo de LaGaBoost): con **pocas observaciones por grupo** y con correlación espacial que **decae rápido**, es decir, cuando la complejidad es alta en relación con $n$. Con muchos datos por grupo la ventaja sobre el boosting independiente se reduce.
- **Costo.** Con efectos agrupados es barato (0.9 s frente a 0.3 s de LightGBM); con un GP sin Vecchia, el paso en $\theta$ domina ($O(m^3)$ en cada iteración).

# Para conectar con tus notas

- **Nota 1** (boosting como gradiente funcional): aquí la pérdida no es separable y el gradiente se “blanquea” con $\Psi^{-1}$.
- **Nota 2** (paso de Newton de XGBoost): la variante Newton/híbrida es la versión con mínimos cuadrados generalizados de los valores de hoja.
- **Nota 4** (CatBoost): dos maneras de encoger categorías raras. CatBoost suaviza el promedio del target con un *prior* y usa orden para evitar fuga; GPBoost encoge con el BLUP, con la fuerza del encogimiento $\sigma^2/\sigma_1^2$ **estimada** de los datos.
- **Nota 8** (conformal por tipo de producto): el tipo de producto, o la versión o el modelo de un auto si son de alta cardinalidad, puede ser un efecto aleatorio en vez de una categórica, con calibración Mondrian encima.

**Siguientes artículos:** LaGaBoost (2105.08966) generaliza esto a verosimilitudes no gaussianas; MEGB (2406.04256) aplica la misma idea, con XGBoost, a la estimación en áreas pequeñas; Hierarchical Shrinkage (2202.00858) es otra forma de encogimiento jerárquico, pero dentro del árbol.
