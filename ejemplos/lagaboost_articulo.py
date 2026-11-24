"""Nota 10 (artículo LaGaBoost, Sigrist): réplica reducida del experimento agrupado binario.

Datos como en la sección 4: y ~ Bernoulli(Phi(F(x) + b_grupo)), F = 'hajjem' reescalada a varianza 1,
b ~ N(0, 1), n = 5000. Se varía el número de observaciones por grupo (10, 50, 200) como en la
sección 4.1 (figura 2). Compara:
  - LaGaBoost (gpboost, likelihood='bernoulli_probit', aproximación de Laplace)
  - LightGBM binario con el grupo como variable numérica (como LogitBoost en el artículo)
  - LightGBM binario sin el grupo
El número de árboles de cada método se elige por paro temprano en un conjunto de validación
simulado aparte; el resto de hiperparámetros es fijo. Ilustrativo, no réplica exacta.
Uso: python3 ejemplos/lagaboost_articulo.py
"""
import json, os
import numpy as np
from scipy.stats import norm
import gpboost as gpb
import lightgbm as lgb

R, N = 10, 5000
POR_GRUPO = [10, 50, 200]


def F_hajjem(X):
    return 2 * X[:, 0] + X[:, 1] ** 2 + 4 * (X[:, 2] > 0) + 2 * np.log(np.abs(X[:, 0])) * X[:, 2]


_f = F_hajjem(np.random.default_rng(0).normal(size=(200_000, 9)))
MU, SD = _f.mean(), _f.std()


def simula(rng, b, g):
    X = rng.normal(size=(len(g), 9))
    F = (F_hajjem(X) - MU) / SD
    return X, (rng.uniform(size=len(g)) < norm.cdf(F + b[g])).astype(float)


def logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def error(p, y):
    return float(np.mean((p > 0.5) != y))


params = dict(learning_rate=0.05, max_depth=5, min_data_in_leaf=10, num_leaves=2**10, verbose=-1)
res = {}
for npg in POR_GRUPO:
    M = N // npg
    acc = {k: [] for k in ["laga_ll", "laga_err", "lgb_num_ll", "lgb_num_err", "lgb_sin_ll", "lgb_sin_err",
                           "laga_ll_new", "lgb_num_ll_new", "laga_sigma2_1", "rondas_laga", "rondas_lgb"]}
    for r in range(R):
        rng = np.random.default_rng(1000 * npg + r)
        b = rng.normal(size=M)
        g = np.repeat(np.arange(M), npg)
        X, y = simula(rng, b, g)
        Xv, yv = simula(rng, b, g)
        Xt, yt = simula(rng, b, g)
        Xn, yn = simula(rng, rng.normal(size=M), g)

        # LaGaBoost con paro temprano usando el modelo completo (F + b) en validación
        gp = gpb.GPModel(group_data=g, likelihood="bernoulli_probit")
        gp.set_prediction_data(group_data_pred=g)
        dtr = gpb.Dataset(X, y); dv = gpb.Dataset(Xv, yv, reference=dtr)
        bst = gpb.train(params=dict(params, metric="binary_logloss"), train_set=dtr, gp_model=gp,
                        num_boost_round=1000, valid_sets=dv, early_stopping_rounds=20,
                        use_gp_model_for_validation=True, verbose_eval=False)
        p = bst.predict(data=Xt, group_data_pred=g, pred_latent=False)["response_mean"]
        pn = bst.predict(data=Xn, group_data_pred=g + M, pred_latent=False)["response_mean"]
        acc["laga_ll"].append(logloss(p, yt)); acc["laga_err"].append(error(p, yt))
        acc["laga_ll_new"].append(logloss(pn, yn))
        acc["laga_sigma2_1"].append(float(np.asarray(gp.get_cov_pars()).ravel()[0]))
        acc["rondas_laga"].append(bst.best_iteration)

        # LightGBM con el grupo como numérica
        cg = lambda A, gg: np.column_stack([A, gg])
        m = lgb.train(dict(params, objective="binary"), lgb.Dataset(cg(X, g), y), num_boost_round=1000,
                      valid_sets=[lgb.Dataset(cg(Xv, g), yv)], callbacks=[lgb.early_stopping(20, verbose=False)])
        p = m.predict(cg(Xt, g))
        acc["lgb_num_ll"].append(logloss(p, yt)); acc["lgb_num_err"].append(error(p, yt))
        acc["lgb_num_ll_new"].append(logloss(m.predict(cg(Xn, g + M)), yn))
        acc["rondas_lgb"].append(m.best_iteration)

        # LightGBM sin grupo
        m = lgb.train(dict(params, objective="binary"), lgb.Dataset(X, y), num_boost_round=1000,
                      valid_sets=[lgb.Dataset(Xv, yv)], callbacks=[lgb.early_stopping(20, verbose=False)])
        p = m.predict(Xt)
        acc["lgb_sin_ll"].append(logloss(p, yt)); acc["lgb_sin_err"].append(error(p, yt))
        print(npg, r, {k: round(v[-1], 3) for k, v in acc.items()})
    res[str(npg)] = {k: {"media": float(np.mean(v)), "ee": float(np.std(v, ddof=1) / np.sqrt(len(v)))}
                     for k, v in acc.items()}

json.dump(res, open(os.path.join(os.path.dirname(__file__), "resultados_lagaboost.json"), "w"), indent=1)
for npg, d in res.items():
    print(npg, {k: round(v["media"], 4) for k, v in d.items()})
