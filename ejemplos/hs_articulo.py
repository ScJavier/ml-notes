"""Nota 11 (artículo Hierarchical Shrinkage, Agarwal et al. 2022): réplica reducida.

HS implementado a mano sobre árboles de scikit-learn (regresión): reemplaza el valor de cada nodo por
    f_lambda(t_L) = E_{t0} + sum_l (E_{t_l} - E_{t_{l-1}}) / (1 + lambda / N(t_{l-1}))
recorriendo el árbol de la raíz a las hojas. No cambia la estructura del árbol.

Experimentos:
  A. Sesgo-varianza (figura 2 del artículo): y = sum_{j<=10} x_j + eps, x ~ U[0,1]^50, sigma^2 = 0.01,
     n = 500, CART con k hojas frente a hsCART (lambda por VC de 3 pliegues).
  B. Datos del artículo disponibles sin red: diabetes (sklearn) y Friedman1 simulado (n = 200, p = 10).
     CART (20 hojas), hsCART, RF (50 árboles, sin poda), hsRF. R^2 en prueba, 10 particiones 2/3-1/3.
  C. Extensión que el artículo no prueba: HS sobre cada árbol de un gradient boosting (sklearn,
     pérdida cuadrática), comparado con el boosting sin HS.
Además verifica que la implementación coincide con imodels.HSTreeRegressor.
Uso: python3 ejemplos/hs_articulo.py
"""
import copy, json, os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error

LAMBDAS = [0.1, 1, 10, 25, 50, 100]


def hs_arbol(arbol, lam):
    """Devuelve una copia del árbol con HS aplicado (valores de todos los nodos reescritos)."""
    a = copy.deepcopy(arbol)
    t = a.tree_
    val = t.value[:, 0, 0].copy()          # media de la respuesta en cada nodo
    n = t.n_node_samples.astype(float)
    nuevo = np.empty_like(val)
    pila = [(0, val[0])]                   # (nodo, predicción encogida acumulada)
    while pila:
        nodo, acum = pila.pop()
        nuevo[nodo] = acum
        for hijo in (t.children_left[nodo], t.children_right[nodo]):
            if hijo != -1:
                pila.append((hijo, acum + (val[hijo] - val[nodo]) / (1 + lam / n[nodo])))
    t.value[:, 0, 0] = nuevo
    return a


def hs_bosque(bosque, lam):
    b = copy.deepcopy(bosque)
    b.estimators_ = [hs_arbol(e, lam) for e in b.estimators_]
    return b


def hs_boosting(gb, lam):
    g = copy.deepcopy(gb)
    for i in range(g.estimators_.shape[0]):
        g.estimators_[i, 0] = hs_arbol(g.estimators_[i, 0], lam)
    return g


def elige_lambda(ajusta, aplica, X, y, semilla=0):
    """lambda por VC de 3 pliegues: se ajusta una vez por pliegue y solo se recalculan los valores."""
    err = np.zeros(len(LAMBDAS))
    for tr, va in KFold(3, shuffle=True, random_state=semilla).split(X):
        m = ajusta(X[tr], y[tr])
        for k, lam in enumerate(LAMBDAS):
            err[k] += mean_squared_error(y[va], aplica(m, lam).predict(X[va]))
    return LAMBDAS[int(np.argmin(err))]


res = {}

# --- verificación contra imodels ---
try:
    from imodels import HSTreeRegressor
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(300, 5)); y = X[:, 0] + rng.normal(scale=0.3, size=300)
    base = DecisionTreeRegressor(max_leaf_nodes=20, random_state=0).fit(X, y)
    ref = HSTreeRegressor(estimator_=copy.deepcopy(base), reg_param=10, shrinkage_scheme_="node_based").fit(X, y)
    res["diferencia_max_vs_imodels"] = float(np.max(np.abs(ref.predict(X) - hs_arbol(base, 10).predict(X))))
except Exception as e:  # imodels es opcional
    res["diferencia_max_vs_imodels"] = f"no verificado: {e}"
print("verificación:", res["diferencia_max_vs_imodels"])

# --- A. sesgo-varianza ---
rng = np.random.default_rng(1)
Xt = rng.uniform(size=(500, 50)); ft = Xt[:, :10].sum(1)
HOJAS = [2, 4, 8, 16, 32, 64, 128]
pred = {(m, k): [] for m in ["cart", "hs"] for k in HOJAS}
lam_elegido = {k: [] for k in HOJAS}
for r in range(50):
    X = rng.uniform(size=(500, 50)); y = X[:, :10].sum(1) + rng.normal(scale=0.1, size=500)
    for k in HOJAS:
        aj = lambda A, b, k=k: DecisionTreeRegressor(max_leaf_nodes=k, random_state=0).fit(A, b)
        lam = elige_lambda(aj, hs_arbol, X, y, r)
        m = aj(X, y)
        pred[("cart", k)].append(m.predict(Xt)); pred[("hs", k)].append(hs_arbol(m, lam).predict(Xt))
        lam_elegido[k].append(lam)
A = {}
for k in HOJAS:
    for mtd in ["cart", "hs"]:
        P = np.array(pred[(mtd, k)])
        sesgo2 = float(np.mean((P.mean(0) - ft) ** 2)); var = float(np.mean(P.var(0)))
        A[f"{mtd}_{k}"] = {"sesgo2": sesgo2, "varianza": var, "mse": sesgo2 + var}
    A[f"lambda_mediana_{k}"] = float(np.median(lam_elegido[k]))
res["A_sesgo_varianza"] = A
for k in HOJAS:
    print("A", k, {m: round(A[f"{m}_{k}"]["mse"], 3) for m in ["cart", "hs"]}, "lambda", A[f"lambda_mediana_{k}"])

# --- B. datos del artículo ---
conjuntos = {"diabetes": load_diabetes(return_X_y=True),
             "friedman1": make_friedman1(n_samples=200, n_features=10, noise=1.0, random_state=0)}
B = {}
for nombre, (X, y) in conjuntos.items():
    r2 = {m: [] for m in ["cart", "hscart", "rf", "hsrf"]}
    for s in range(10):
        Xa, Xp, ya, yp = train_test_split(X, y, test_size=1 / 3, random_state=s)
        aj_c = lambda A_, b_: DecisionTreeRegressor(max_leaf_nodes=20, random_state=0).fit(A_, b_)
        aj_r = lambda A_, b_: RandomForestRegressor(n_estimators=50, max_features=1 / 3, random_state=0).fit(A_, b_)
        c = aj_c(Xa, ya); rf = aj_r(Xa, ya)
        r2["cart"].append(r2_score(yp, c.predict(Xp)))
        r2["hscart"].append(r2_score(yp, hs_arbol(c, elige_lambda(aj_c, hs_arbol, Xa, ya, s)).predict(Xp)))
        r2["rf"].append(r2_score(yp, rf.predict(Xp)))
        r2["hsrf"].append(r2_score(yp, hs_bosque(rf, elige_lambda(aj_r, hs_bosque, Xa, ya, s)).predict(Xp)))
    B[nombre] = {m: {"media": float(np.mean(v)), "ee": float(np.std(v, ddof=1) / np.sqrt(len(v)))} for m, v in r2.items()}
    print("B", nombre, {m: round(v["media"], 3) for m, v in B[nombre].items()})
res["B_datos"] = B

# --- C. HS dentro de gradient boosting ---
C = {}
for nombre, (X, y) in conjuntos.items():
    r2 = {m: [] for m in ["gb", "hsgb"]}; lams = []
    for s in range(10):
        Xa, Xp, ya, yp = train_test_split(X, y, test_size=1 / 3, random_state=s)
        aj = lambda A_, b_: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_leaf_nodes=32,
                                                      min_samples_leaf=1, random_state=0).fit(A_, b_)
        g = aj(Xa, ya); lam = elige_lambda(aj, hs_boosting, Xa, ya, s); lams.append(lam)
        r2["gb"].append(r2_score(yp, g.predict(Xp))); r2["hsgb"].append(r2_score(yp, hs_boosting(g, lam).predict(Xp)))
    C[nombre] = {m: {"media": float(np.mean(v)), "ee": float(np.std(v, ddof=1) / np.sqrt(len(v)))} for m, v in r2.items()}
    C[nombre]["lambda_mediana"] = float(np.median(lams))
    print("C", nombre, {m: round(C[nombre][m]["media"], 3) for m in r2}, "lambda", C[nombre]["lambda_mediana"])
res["C_boosting"] = C

json.dump(res, open(os.path.join(os.path.dirname(__file__), "resultados_hs.json"), "w"), indent=1)
