"""Experimentos sintéticos para la nota de CatBoost. Uso: python catboost_experimentos.py"""
import json, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb, catboost as cb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.datasets import make_friedman1
warnings.filterwarnings("ignore")
R = {}
sig = lambda z: 1 / (1 + np.exp(-z))

# 1. Fuga en la codificación por target: categórica de ruido puro con 1000 niveles
def datos(n, seed):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 2)); c = r.integers(0, 1000, n)
    y = (r.uniform(size=n) < sig(1.5 * x[:, 0] - x[:, 1])).astype(int)
    d = pd.DataFrame(x, columns=["x1", "x2"]); d["cat"] = c.astype(str); return d, y
dtr, ytr = datos(4000, 1); dte, yte = datos(20000, 2)
p0 = ytr.mean()
def ts_greedy(dtr, ytr, dte, a=1.0):
    s = pd.DataFrame({"c": dtr.cat, "y": ytr}).groupby("c").y.agg(["sum", "count"])
    m = (s["sum"] + a * p0) / (s["count"] + a)
    return dtr.cat.map(m).values, dte.cat.map(m).fillna(p0).values
def ts_loo(dtr, ytr, dte, a=1.0):
    s = pd.DataFrame({"c": dtr.cat, "y": ytr}).groupby("c").y.agg(["sum", "count"])
    S = dtr.cat.map(s["sum"]).values; N = dtr.cat.map(s["count"]).values
    tr = (S - ytr + a * p0) / (N - 1 + a)
    return tr, dte.cat.map((s["sum"] + a * p0) / (s["count"] + a)).fillna(p0).values
rows = []
def eval_lgb(name, Xtr, Xte):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0).fit(Xtr, ytr)
    ptr, pte = m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]
    imp = m.booster_.feature_importance("gain"); share = imp[-1] / imp.sum() if Xtr.shape[1] == 3 else 0.0
    rows.append(dict(metodo=name, auc_ent=round(roc_auc_score(ytr, ptr), 3), auc_prueba=round(roc_auc_score(yte, pte), 3),
                     logloss_prueba=round(log_loss(yte, pte), 4), peso_cat=round(float(share), 3)))
base_tr, base_te = dtr[["x1", "x2"]].values, dte[["x1", "x2"]].values
eval_lgb("sin la categórica", base_tr, base_te)
a, b = ts_greedy(dtr, ytr, dte); eval_lgb("TS codicioso (todo el entrenamiento)", np.c_[base_tr, a], np.c_[base_te, b])
a, b = ts_loo(dtr, ytr, dte); eval_lgb("TS leave-one-out", np.c_[base_tr, a], np.c_[base_te, b])
m = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4, verbose=0, random_seed=0, thread_count=2, cat_features=["cat"]).fit(dtr, ytr)
ptr, pte = m.predict_proba(dtr)[:, 1], m.predict_proba(dte)[:, 1]
imp = m.get_feature_importance(type="LossFunctionChange", data=cb.Pool(dte, yte, cat_features=["cat"]))
pvc = m.get_feature_importance()
rows.append(dict(metodo="CatBoost (TS ordenado)", auc_ent=round(roc_auc_score(ytr, ptr), 3), auc_prueba=round(roc_auc_score(yte, pte), 3),
                 logloss_prueba=round(log_loss(yte, pte), 4), peso_cat=round(float(pvc[2] / pvc.sum()), 3)))
R["1_fuga"] = rows
R["1_catboost_LossFunctionChange_en_prueba"] = dict(zip(["x1", "x2", "cat"], np.round(imp, 5).tolist()))

# 2. Boosting ordenado vs plano, datos pequeños
rows = []
for n in (200, 1000):
    res = {"Plain": [], "Ordered": []}
    for s in range(8):
        Xtr, ytr2 = make_friedman1(n, 10, noise=1.0, random_state=100 + s)
        Xte, yte2 = make_friedman1(10000, 10, noise=1.0, random_state=999)
        for bt in res:
            mm = cb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, boosting_type=bt, verbose=0, random_seed=s, thread_count=2).fit(Xtr, ytr2)
            res[bt].append(np.sqrt(np.mean((mm.predict(Xte) - yte2) ** 2)))
    for bt, v in res.items():
        rows.append(dict(n=n, boosting_type=bt, rmse_media=round(float(np.mean(v)), 3), rmse_de=round(float(np.std(v)), 3)))
    d = np.array(res["Ordered"]) - np.array(res["Plain"])
    rows.append(dict(n=n, boosting_type="Ordered − Plain (pareado)", rmse_media=round(float(d.mean()), 3), rmse_de=round(float(d.std()), 3)))
R["2_ordered_vs_plain"] = rows
# tiempo
Xtr, ytr2 = make_friedman1(20000, 10, noise=1.0, random_state=5)
for bt in ("Plain", "Ordered"):
    t = time.time(); cb.CatBoostRegressor(iterations=300, boosting_type=bt, verbose=0, thread_count=2).fit(Xtr, ytr2); R[f"2_segundos_{bt}"] = round(time.time() - t, 1)

# 3. Árboles simétricos: estructura de un modelo
mm = cb.CatBoostRegressor(iterations=3, depth=6, verbose=0, thread_count=2).fit(Xtr, ytr2)
mm.save_model("/tmp/cb.json", format="json"); js = json.load(open("/tmp/cb.json"))
t0 = js["oblivious_trees"][0]
R["3_arbol0"] = dict(n_cortes=len(t0["splits"]), n_hojas=len(t0["leaf_values"]),
                     cortes=[(s["float_feature_index"], round(s["border"], 3)) for s in t0["splits"]])

# 4. Combinaciones de categóricas: efecto solo por interacción (tipo XOR)
def datos_x(n, seed):
    r = np.random.default_rng(seed); c1 = r.integers(0, 6, n); c2 = r.integers(0, 6, n); x = r.normal(size=n)
    tab = np.random.default_rng(0).normal(0, 1.5, (6, 6)); tab -= tab.mean(0); tab -= tab.mean(1, keepdims=True)
    y = (r.uniform(size=n) < sig(tab[c1, c2] + 0.5 * x)).astype(int)
    return pd.DataFrame({"c1": c1.astype(str), "c2": c2.astype(str), "x": x}), y
dtr, ytr = datos_x(3000, 1); dte, yte = datos_x(30000, 2)
rows = []
for mcc in (1, 2, 4):
    for oh in (2, 10):
        mm = cb.CatBoostClassifier(iterations=500, learning_rate=0.05, depth=4, max_ctr_complexity=mcc, one_hot_max_size=oh, verbose=0, random_seed=0, thread_count=2, cat_features=["c1", "c2"]).fit(dtr, ytr)
        rows.append(dict(max_ctr_complexity=mcc, one_hot_max_size=oh, logloss=round(log_loss(yte, mm.predict_proba(dte)[:, 1]), 4)))
R["4_combinaciones"] = rows
R["4_entropia_base"] = round(log_loss(yte, np.full(len(yte), ytr.mean())), 4)

# 5. depth y l2_leaf_reg
Xtr, ytr2 = make_friedman1(2000, 10, noise=1.0, random_state=1); Xva, yva = make_friedman1(2000, 10, noise=1.0, random_state=2); Xte, yte2 = make_friedman1(20000, 10, noise=1.0, random_state=3)
rows = []
for dpt in (2, 4, 6, 8, 10):
    for l2 in (1, 3, 30):
        mm = cb.CatBoostRegressor(iterations=3000, learning_rate=0.05, depth=dpt, l2_leaf_reg=l2, verbose=0, random_seed=0, thread_count=2, od_type="Iter", od_wait=100).fit(Xtr, ytr2, eval_set=(Xva, yva), use_best_model=True)
        rows.append(dict(depth=dpt, l2_leaf_reg=l2, iters=mm.get_best_iteration(), rmse=round(float(np.sqrt(np.mean((mm.predict(Xte) - yte2) ** 2))), 3)))
R["5_depth_l2"] = rows
json.dump(R, open("resultados_catboost.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
