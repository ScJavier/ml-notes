"""Experimentos sintéticos para la nota de XGBoost. Uso: python xgboost_experimentos.py
(figuras en ../src/fig, números en resultados_xgboost.json)"""
import json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.datasets import make_friedman1
from sklearn.metrics import log_loss, roc_auc_score
from estilo import plt, guardar, AZUL, NARANJA, AQUA, VIOLETA, GRIS, TINTA, TINTA2
warnings.filterwarnings("ignore")
FIG = "../src/fig/"; R = {}
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
sig = lambda z: 1 / (1 + np.exp(-z))
BASE = dict(nthread=2, seed=0, tree_method="hist")

Xtr, ytr = make_friedman1(1000, 10, noise=1.0, random_state=1)
Xva, yva = make_friedman1(5000, 10, noise=1.0, random_state=2)
Xte, yte = make_friedman1(20000, 10, noise=1.0, random_state=3)
dtr, dva, dte = xgb.DMatrix(Xtr, ytr), xgb.DMatrix(Xva, yva), xgb.DMatrix(Xte, yte)

def entrenar(params, d=dtr, v=dva, rondas=5000, esperar=100):
    p = dict(BASE, objective="reg:squarederror"); p.update(params)
    return xgb.train(p, d, rondas, evals=[(v, "val")], early_stopping_rounds=esperar, verbose_eval=False)
def pred(b, d): return b.predict(d, iteration_range=(0, b.best_iteration + 1))
def hojas_por_arbol(b):
    t = b.trees_to_dataframe(); t = t[t.Tree <= b.best_iteration]
    return float((t.Feature == "Leaf").groupby(t.Tree).sum().mean())

# ---------------------------------------------------------------- A. lambda y eta
filas = []
for eta in (0.3, 0.05):
    for lam in (0, 1, 10, 100):
        b = entrenar(dict(eta=eta, reg_lambda=lam, max_depth=6, min_child_weight=1))
        t0 = b[0].trees_to_dataframe(); hojas0 = t0[t0.Feature == "Leaf"].Gain.abs()
        filas.append(dict(eta=eta, reg_lambda=lam, arboles=b.best_iteration + 1, hojas_por_arbol=round(hojas_por_arbol(b), 1),
                          peso_medio_hoja_arbol1=round(float(hojas0.mean()), 3), rmse=round(rmse(pred(b, dte), yte), 3)))
R["A_lambda_eta"] = filas

# ---------------------------------------------------------------- B. gamma: poda por ganancia mínima
# 300 rondas fijas (sin parada temprana) para ver la poda árbol por árbol
gammas = [0, 1, 5, 20, 100]
filas = []; curvas = {}
for g in gammas:
    b = xgb.train(dict(BASE, objective="reg:squarederror", eta=0.1, gamma=g, max_depth=6, reg_lambda=1), dtr, 300,
                  evals=[(dva, "val")], verbose_eval=False, evals_result=(hist := {}))
    t = b.trees_to_dataframe(); hojas = (t.Feature == "Leaf").groupby(t.Tree).sum().values
    curvas[g] = hojas; va = np.array(hist["val"]["rmse"]); mejor = int(va.argmin())
    filas.append(dict(gamma=g, hojas_arboles_1a10=round(float(hojas[:10].mean()), 1), hojas_arboles_100a300=round(float(hojas[99:].mean()), 1),
                      arboles_con_1_hoja=int((hojas == 1).sum()), mejor_M=mejor + 1,
                      rmse=round(rmse(b.predict(dte, iteration_range=(0, mejor + 1)), yte), 3)))
R["B_gamma"] = filas
b0 = xgb.train(dict(BASE, objective="reg:squarederror", eta=0.1, max_depth=6, reg_lambda=1), dtr, 300)
t0 = b0.trees_to_dataframe(); cort = t0[t0.Feature != "Leaf"]
R["B_ganancia_mediana_arbol"] = {i: round(float(cort[cort.Tree == i].Gain.median()), 2) for i in (0, 9, 49, 99, 299)}
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for g, col in zip(gammas, [GRIS, AZUL, AQUA, VIOLETA, NARANJA]):
    suav = np.convolve(curvas[g], np.ones(10) / 10, mode="valid")
    ax.plot(np.arange(10, 301), suav, color=col, label=f"gamma = {g}")
ax.set_xlabel("árbol"); ax.set_ylabel("hojas (media móvil de 10)"); ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("La poda se intensifica cuando los residuos se achican", loc="left")
guardar(fig, FIG + "xgb_gamma.png")

# ---------------------------------------------------------------- C. min_child_weight es suma de hessianos (logística)
r = np.random.default_rng(4)
def datos_c(n):
    X = r.normal(size=(n, 6)); eta_ = 2 * X[:, 0] - 1.5 * X[:, 1] + np.sin(2 * X[:, 2]) - 1.5
    return X, (r.uniform(size=n) < sig(eta_)).astype(int)
Xc, yc = datos_c(5000); Xcv, ycv = datos_c(5000); Xct, yct = datos_c(20000)
dc, dcv, dct = xgb.DMatrix(Xc, yc), xgb.DMatrix(Xcv, ycv), xgb.DMatrix(Xct, yct)
filas = []; curvas = {}
for mcw in (0, 1, 5, 20):
    b = entrenar(dict(objective="binary:logistic", eval_metric="logloss", eta=0.1, max_depth=6, min_child_weight=mcw), d=dc, v=dcv)
    hojas = b.predict(dc, pred_leaf=True, iteration_range=(0, b.best_iteration + 1))
    med = [float(np.median(np.unique(hojas[:, t], return_counts=True)[1])) for t in range(hojas.shape[1])]
    mn = [int(np.unique(hojas[:, t], return_counts=True)[1].min()) for t in range(hojas.shape[1])]
    curvas[mcw] = med
    p = pred(b, dct)
    filas.append(dict(min_child_weight=mcw, arboles=b.best_iteration + 1, hojas_por_arbol=round(hojas_por_arbol(b), 1),
                      min_obs_por_hoja=int(min(mn)), mediana_obs_por_hoja_arbol1=med[0], mediana_obs_por_hoja_ultimo=med[-1],
                      logloss=round(log_loss(yct, p), 4)))
R["C_min_child_weight"] = filas
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for mcw, col in [(1, AZUL), (5, NARANJA), (20, AQUA)]:
    ax.plot(np.arange(1, len(curvas[mcw]) + 1), curvas[mcw], color=col, label=f"min_child_weight = {mcw}")
ax.set_xlabel("árbol"); ax.set_ylabel("observaciones por hoja (mediana)"); ax.legend(loc="upper left")
ax.set_title("Logística: misma restricción, cada vez más datos por hoja", loc="left")
guardar(fig, FIG + "xgb_min_child_weight.png")

# ---------------------------------------------------------------- D. max_depth y eta
filas = []
for d in (2, 4, 6, 8):
    fila = dict(max_depth=d)
    for eta in (0.3, 0.1, 0.03):
        b = entrenar(dict(eta=eta, max_depth=d))
        fila[f"eta_{eta}"] = round(rmse(pred(b, dte), yte), 3); fila[f"M_{eta}"] = b.best_iteration + 1
    filas.append(fila)
R["D_depth_eta"] = filas

# ---------------------------------------------------------------- E. métodos de árbol: exact, approx, hist
Xb, yb = make_friedman1(200000, 10, noise=1.0, random_state=9)
db = xgb.DMatrix(Xb, yb)
filas = []
for tm, extra in [("exact", {}), ("approx", {}), ("hist", dict(max_bin=256)), ("hist", dict(max_bin=32)), ("hist", dict(max_bin=8))]:
    t0 = time.time()
    b = xgb.train(dict(BASE, objective="reg:squarederror", tree_method=tm, eta=0.1, max_depth=6, **extra), db, 200)
    filas.append(dict(tree_method=tm, **extra, segundos=round(time.time() - t0, 1), rmse=round(rmse(b.predict(dte), yte), 3)))
R["E_tree_method"] = filas

# ---------------------------------------------------------------- F. faltantes informativos
def con_faltantes(X, seed):
    X = X.copy(); rr = np.random.default_rng(seed)
    falta = (X[:, 3] > 0.6) & (rr.uniform(size=len(X)) < 0.8)   # x4 falta sobre todo cuando es grande
    X[falta, 3] = np.nan; return X, falta
Xf, fa = con_faltantes(Xtr, 1); Xfv, _ = con_faltantes(Xva, 2); Xft, fat = con_faltantes(Xte, 3)
media = np.nanmean(Xf[:, 3])
def imputar(X, indicador=False):
    Z = X.copy(); m = np.isnan(Z[:, 3]); Z[m, 3] = media
    return np.c_[Z, m.astype(float)] if indicador else Z
filas = []
for nombre, A, V, T in [("NaN nativo", Xf, Xfv, Xft), ("imputar con la media", imputar(Xf), imputar(Xfv), imputar(Xft)),
                        ("media + indicador", imputar(Xf, True), imputar(Xfv, True), imputar(Xft, True))]:
    b = entrenar(dict(eta=0.05, max_depth=4), d=xgb.DMatrix(A, ytr), v=xgb.DMatrix(V, yva))
    p = pred(b, xgb.DMatrix(T, yte))
    filas.append(dict(metodo=nombre, rmse=round(rmse(p, yte), 3), rmse_filas_con_faltante=round(rmse(p[fat], yte[fat]), 3)))
R["F_faltantes"] = filas
R["F_fraccion_faltante"] = round(float(fa.mean()), 3)
b = entrenar(dict(eta=0.05, max_depth=4), d=xgb.DMatrix(Xf, ytr), v=xgb.DMatrix(Xfv, yva))
t = b.trees_to_dataframe(); c4 = t[t.Feature == "f3"]
# dirección por defecto: ¿los faltantes van al lado "No" (x4 >= umbral)?
R["F_cortes_en_x4_faltante_a_lado_mayor"] = round(float((c4.Missing == c4.No).mean()), 3)
R["F_n_cortes_x4"] = int(len(c4))

# ---------------------------------------------------------------- G. max_delta_step con clase rara
r = np.random.default_rng(6)
def datos_g(n):
    X = r.normal(size=(n, 5)); e = 3 * X[:, 0] + 2 * X[:, 1] - 6.5
    return X, (r.uniform(size=n) < sig(e)).astype(int)
Xg, yg = datos_g(3000); Xgt, ygt = datos_g(100000)
filas = []
for mds in (0, 1):
    b = xgb.train(dict(BASE, objective="binary:logistic", eta=1.0, max_depth=6, min_child_weight=0, max_delta_step=mds, base_score=0.5), xgb.DMatrix(Xg, yg), 30)
    t0 = b[0].trees_to_dataframe(); h0 = t0[t0.Feature == "Leaf"].Gain
    p = b.predict(xgb.DMatrix(Xgt))
    filas.append(dict(max_delta_step=mds, max_abs_hoja_arbol1=round(float(h0.abs().max()), 2),
                      logloss=round(log_loss(ygt, p), 4), auc=round(roc_auc_score(ygt, p), 4)))
R["G_max_delta_step"] = filas
R["G_tasa_positivos"] = round(float(yg.mean()), 4)

# ---------------------------------------------------------------- H. restricción monótona
r = np.random.default_rng(8)
def datos_h(n):
    X = r.uniform(0, 1, size=(n, 3)); f = 2 * X[:, 0] + np.sin(6 * X[:, 1])
    return X, f + r.normal(0, 1.0, n), f
Xh, yh, _ = datos_h(400); Xhv, yhv, _ = datos_h(2000); Xht, yht, fht = datos_h(20000)
grid = np.linspace(0, 1, 101)
fig, ax = plt.subplots(figsize=(5.2, 3.7)); filas = []
for nombre, mc, col in [("sin restricción", "(0,0,0)", NARANJA), ("monótona en x1", "(1,0,0)", AZUL)]:
    b = entrenar(dict(eta=0.05, max_depth=3, monotone_constraints=mc), d=xgb.DMatrix(Xh, yh), v=xgb.DMatrix(Xhv, yhv))
    pdp = []
    for g in grid:
        Z = Xht[:3000].copy(); Z[:, 0] = g; pdp.append(pred(b, xgb.DMatrix(Z)).mean())
    pdp = np.array(pdp); ax.plot(grid, pdp - pdp.mean(), color=col, label=nombre)
    filas.append(dict(modelo=nombre, rmse_vs_verdad=round(rmse(pred(b, xgb.DMatrix(Xht)), fht), 3),
                      pdp_es_monotono=bool(np.all(np.diff(pdp) >= -1e-9))))
ax.plot(grid, 2 * grid - 1, color=TINTA, ls=":", lw=1.6, label="efecto real (2·x1)")
ax.set_xlabel("x1"); ax.set_ylabel("dependencia parcial (centrada)"); ax.legend(loc="upper left")
ax.set_title("400 observaciones con ruido alto", loc="left")
guardar(fig, FIG + "xgb_monotona.png")
R["H_monotona"] = filas

# ---------------------------------------------------------------- I. objetivo personalizado
def cuadratico(predt, d):
    y = d.get_label(); return predt - y, np.ones_like(predt)
b_int = xgb.train(dict(BASE, objective="reg:squarederror", eta=0.1, max_depth=4, base_score=0.0), dtr, 100)
b_pro = xgb.train(dict(BASE, eta=0.1, max_depth=4, base_score=0.0), dtr, 100, obj=cuadratico)
R["I_dif_max_personalizado_vs_interno"] = float(np.abs(b_int.predict(dte) - b_pro.predict(dte)).max())
def asimetrico(k):
    def obj(predt, d):
        y = d.get_label(); w = np.where(y > predt, k, 1.0)   # quedarse corto cuesta k veces más
        return -2 * w * (y - predt), 2 * w
    return obj
filas = []
for k in (1, 3, 9):
    b = xgb.train(dict(BASE, eta=0.1, max_depth=4, base_score=float(ytr.mean())), dtr, 300, obj=asimetrico(k))
    p = b.predict(dte)
    filas.append(dict(k=k, fraccion_subestimada=round(float(np.mean(yte > p)), 3), sesgo_medio=round(float(np.mean(p - yte)), 3)))
R["I_asimetrico"] = filas

# ---------------------------------------------------------------- J. inspección del modelo entrenado
b = entrenar(dict(eta=0.05, max_depth=4))
imp = pd.DataFrame({t: pd.Series(b.get_score(importance_type=t)) for t in ("weight", "gain", "total_gain", "cover", "total_cover")}).fillna(0)
imp = (imp / imp.max()).round(3)
R["J_importancias"] = imp.reset_index().rename(columns={"index": "variable"}).sort_values("total_gain", ascending=False).to_dict("records")
contrib = b.predict(dte, pred_contribs=True, iteration_range=(0, b.best_iteration + 1))
margen = b.predict(dte, output_margin=True, iteration_range=(0, b.best_iteration + 1))
R["J_contribs_suman_margen_dif_max"] = float(np.abs(contrib.sum(1) - margen).max())
inter = b.predict(xgb.DMatrix(Xte[:2000]), pred_interactions=True, iteration_range=(0, b.best_iteration + 1))
fuerza = np.abs(inter[:, :10, :10]).mean(0); np.fill_diagonal(fuerza, 0)
pares = sorted(((fuerza[i, j], f"f{i}×f{j}") for i in range(10) for j in range(i + 1, 10)), reverse=True)[:4]
R["J_interacciones_shap_top"] = [(n, round(float(2 * v), 3)) for v, n in pares]
R["J_forma_pred_interactions"] = list(inter.shape)

json.dump(R, open("resultados_xgboost.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
