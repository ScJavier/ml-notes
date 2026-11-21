"""Experimentos sintéticos para la nota de LightGBM. Uso: python lightgbm_experimentos.py"""
import time, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.datasets import make_friedman1
warnings.filterwarnings("ignore")
R = {}
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def fit(params, Xtr, ytr, Xva, yva, rounds=5000, cat="auto"):
    p = dict(objective="regression", verbose=-1, seed=0, num_threads=2, learning_rate=0.05); p.update(params)
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cat)
    return lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])

# A. Forma del árbol: hoja por hoja
X, y = make_friedman1(n_samples=20000, n_features=10, noise=1.0, random_state=0)
for nl in (31, 255):
    b = lgb.train(dict(objective="regression", verbose=-1, num_leaves=nl, num_threads=2), lgb.Dataset(X, y), 1)
    df = b.trees_to_dataframe(); leaves = df[df.left_child.isna()]
    R[f"A_profundidades_hojas_{nl}"] = {int(k): int(v) for k, v in leaves.node_depth.value_counts().sort_index().items()}

# B. num_leaves y min_data_in_leaf con pocos datos
Xtr, ytr = make_friedman1(2000, 10, noise=1.0, random_state=1)
Xva, yva = make_friedman1(2000, 10, noise=1.0, random_state=2)
Xte, yte = make_friedman1(20000, 10, noise=1.0, random_state=3)
rows = []
for nl in [4, 8, 16, 31, 63, 127, 255]:
    for mdl in [5, 20, 100]:
        m = fit(dict(num_leaves=nl, min_data_in_leaf=mdl), Xtr, ytr, Xva, yva)
        rows.append(dict(num_leaves=nl, min_data_in_leaf=mdl, iters=m.best_iteration, rmse=round(rmse(m.predict(Xte, num_iteration=m.best_iteration), yte), 3)))
R["B_num_leaves"] = rows

# C. GOSS vs bagging vs nada, datos grandes
Xb, yb = make_friedman1(300000, 10, noise=1.0, random_state=4)
Xbv, ybv = make_friedman1(20000, 10, noise=1.0, random_state=5)
rows = []
for name, prm in [("completo", {}), ("bagging 30%", dict(bagging_fraction=0.3, bagging_freq=1)),
                  ("GOSS 20%+10%", dict(data_sample_strategy="goss", top_rate=0.2, other_rate=0.1))]:
    t = time.time(); m = lgb.train(dict(objective="regression", verbose=-1, seed=0, num_threads=2, learning_rate=0.1, num_leaves=31, **prm), lgb.Dataset(Xb, yb), 300)
    rows.append(dict(metodo=name, segundos=round(time.time() - t, 1), rmse=round(rmse(m.predict(Xbv), ybv), 3)))
R["C_muestreo"] = rows

# D. Categórica de alta cardinalidad con niveles raros
rng = np.random.default_rng(7)
K = 300; freq = 1 / np.arange(1, K + 1) ** 1.1; freq /= freq.sum(); efecto = rng.normal(0, 1, K)
def datos_cat(n, seed):
    r = np.random.default_rng(seed); c = r.choice(K, n, p=freq); x = r.normal(size=(n, 3))
    yy = efecto[c] + x[:, 0] + 0.5 * x[:, 1] ** 2 + r.normal(0, 1, n)
    d = pd.DataFrame(x, columns=["x1", "x2", "x3"]); d["cat"] = pd.Categorical(c, categories=range(K)); return d, yy
dtr_, ytr_ = datos_cat(5000, 10); dva_, yva_ = datos_cat(5000, 11); dte_, yte_ = datos_cat(50000, 12)
R["D_niveles_con_menos_de_10_obs"] = int((dtr_.cat.value_counts() < 10).sum())
rows = []
for name, prm, enc in [
    ("nativa, por defecto", {}, "cat"),
    ("nativa, sin suavizar", dict(cat_smooth=0.1, cat_l2=0.0, min_data_per_group=1, max_cat_threshold=256), "cat"),
    ("nativa, más suavizado", dict(cat_smooth=50, cat_l2=10, min_data_per_group=200), "cat"),
    ("one-vs-rest", dict(max_cat_to_onehot=1000), "cat"),
    ("código entero como numérica", {}, "int")]:
    def enc_f(d):
        d = d.copy()
        if enc == "int": d["cat"] = d["cat"].cat.codes.astype(int)
        return d
    m = fit(prm, enc_f(dtr_), ytr_, enc_f(dva_), yva_)
    rows.append(dict(variante=name, iters=m.best_iteration, rmse=round(rmse(m.predict(enc_f(dte_), num_iteration=m.best_iteration), yte_), 3)))
R["D_categorica"] = rows
R["D_rmse_oraculo"] = 1.0

# E. max_bin y umbrales
r = np.random.default_rng(3); n = 20000
x = r.uniform(0, 1, n); yy = 2.0 * (x > 0.37) + r.normal(0, 0.5, n)
xt = r.uniform(0, 1, 50000); yt = 2.0 * (xt > 0.37) + r.normal(0, 0.5, 50000)
rows = []
for mb in [4, 8, 16, 64, 255]:
    m = lgb.train(dict(objective="regression", verbose=-1, max_bin=mb, num_leaves=2, learning_rate=1.0, num_threads=2), lgb.Dataset(x.reshape(-1, 1), yy), 1)
    thr = m.trees_to_dataframe().threshold.dropna().iloc[0]
    rows.append(dict(max_bin=mb, umbral=round(float(thr), 4), rmse=round(rmse(m.predict(xt.reshape(-1, 1)), yt), 4)))
R["E_max_bin"] = rows

# F. linear_tree con relación lineal por tramos
r = np.random.default_rng(4); n = 5000
X3 = r.uniform(-2, 2, (n, 2)); yy = np.where(X3[:, 0] > 0, 3 * X3[:, 0], -X3[:, 0]) + 0.5 * X3[:, 1] + r.normal(0, .3, n)
Xt3 = r.uniform(-2, 2, (20000, 2)); yt3 = np.where(Xt3[:, 0] > 0, 3 * Xt3[:, 0], -Xt3[:, 0]) + 0.5 * Xt3[:, 1] + r.normal(0, .3, 20000)
rows = []
for lt in [False, True]:
    for nl in [2, 4, 31]:
        for nr in (5, 50):
            m = lgb.train(dict(objective="regression", verbose=-1, linear_tree=lt, num_leaves=nl, learning_rate=0.3, num_threads=2, seed=0), lgb.Dataset(X3, yy), nr)
            rows.append(dict(linear_tree=lt, num_leaves=nl, arboles=nr, rmse=round(rmse(m.predict(Xt3), yt3), 3)))
R["F_linear_tree"] = rows

# G. EFB: muchas columnas dispersas mutuamente excluyentes
r = np.random.default_rng(5); n = 50000; K2 = 3000
c = r.integers(0, K2, n); onehot = np.zeros((n, K2), dtype=np.float32); onehot[np.arange(n), c] = 1
Xg = np.hstack([r.normal(size=(n, 5)).astype(np.float32), onehot]); yg = Xg[:, 0] + r.normal(0, 1, K2)[c] + r.normal(0, 1, n)
rows = []
for eb in [True, False]:
    t = time.time(); lgb.train(dict(objective="regression", verbose=-1, enable_bundle=eb, num_threads=2), lgb.Dataset(Xg, yg), 100)
    rows.append(dict(enable_bundle=eb, segundos=round(time.time() - t, 1)))
R["G_efb"] = rows

json.dump(R, open("resultados_lightgbm.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
