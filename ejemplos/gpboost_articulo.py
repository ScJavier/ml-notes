"""Nota 9 (artículo GPBoost, Sigrist 2022): réplica reducida del experimento de datos agrupados.

Datos como en la sección 4.1 del artículo: F = 'hajjem', 500 grupos x 10 observaciones,
sigma2_1 = 1 (efecto aleatorio) y sigma2 = 1 (error). Compara:
  - GPBoost (árboles + efecto aleatorio por grupo)
  - LightGBM con el grupo como categórica
  - LightGBM sin el grupo
  - GPBoostOOS (algoritmo 2 del artículo), implementado a mano: en Python,
    gpb.cv(fit_GP_cov_pars_OOS=True) aún no existe (gpboost 1.8.0)
Hiperparámetros fijos (sin búsqueda), así que es ilustrativo, no una réplica exacta.
Uso: python3 ejemplos/gpboost_articulo.py
"""
import json, os
import numpy as np
import gpboost as gpb
import lightgbm as lgb

R, M, NPG = 20, 500, 10  # repeticiones, grupos, observaciones por grupo
N = M * NPG


def F_hajjem(X):
    f = 2 * X[:, 0] + X[:, 1] ** 2 + 4 * (X[:, 2] > 0) + 2 * np.log(np.abs(X[:, 0])) * X[:, 2]
    return f


# constantes para que Var(F) ~ 1
_x = np.random.default_rng(0).normal(size=(200_000, 9))
_f = F_hajjem(_x)
MU, SD = _f.mean(), _f.std()


def simula(rng, b, grupos):
    X = rng.normal(size=(len(grupos), 9))
    F = (F_hajjem(X) - MU) / SD
    return X, F, F + b[grupos] + rng.normal(size=len(grupos))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


params = dict(learning_rate=0.05, max_depth=5, min_data_in_leaf=10, num_leaves=2**10, verbose=-1)
RONDAS = 300
res = {k: [] for k in ["gpb", "lgb_cat", "lgb_sin", "gpb_new", "lgb_cat_new", "lgb_sin_new",
                       "gpb_F", "gpb_sigma2", "gpb_sigma2_1",
                       "oos", "oos_new", "oos_F", "oos_sigma2", "oos_sigma2_1"]}

for r in range(R):
    rng = np.random.default_rng(100 + r)
    b = rng.normal(size=M)
    g = np.repeat(np.arange(M), NPG)
    X, F, y = simula(rng, b, g)
    Xt, Ft, yt = simula(rng, b, g)                       # mismos grupos
    bn = rng.normal(size=M)
    Xn, Fn, yn = simula(rng, bn, g)                      # grupos nuevos
    gn = g + M                                           # etiquetas no vistas

    # GPBoost
    gp = gpb.GPModel(group_data=g, likelihood="gaussian")
    bst = gpb.train(params=params, train_set=gpb.Dataset(X, y), gp_model=gp, num_boost_round=RONDAS)
    p = bst.predict(data=Xt, group_data_pred=g, pred_latent=False)["response_mean"]
    pn = bst.predict(data=Xn, group_data_pred=gn, pred_latent=False)["response_mean"]
    pF = bst.predict(data=Xt, group_data_pred=g, pred_latent=True)["fixed_effect"]
    cov = gp.get_cov_pars()
    res["gpb"].append(rmse(p, yt)); res["gpb_new"].append(rmse(pn, yn)); res["gpb_F"].append(rmse(pF, Ft))
    res["gpb_sigma2"].append(float(np.asarray(cov).ravel()[0]))
    res["gpb_sigma2_1"].append(float(np.asarray(cov).ravel()[1]))

    # GPBoostOOS: (1) F fuera de muestra con 4 pliegues, (2) theta con y - F_oos, (3) GPBoost con theta fijo
    pl = rng.permutation(N) % 4; Foos = np.empty(N)
    for k in range(4):
        tr = pl != k
        gpk = gpb.GPModel(group_data=g[tr], likelihood="gaussian")
        bk = gpb.train(params=params, train_set=gpb.Dataset(X[tr], y[tr]), gp_model=gpk, num_boost_round=RONDAS)
        Foos[~tr] = bk.predict(data=X[~tr], group_data_pred=g[~tr], pred_latent=True)["fixed_effect"]
    gp0 = gpb.GPModel(group_data=g, likelihood="gaussian"); gp0.fit(y=y - Foos)
    th = np.asarray(gp0.get_cov_pars()).ravel()
    gp = gpb.GPModel(group_data=g, likelihood="gaussian"); gp.set_optim_params(params={"init_cov_pars": th})
    bst = gpb.train(params=params, train_set=gpb.Dataset(X, y), gp_model=gp, num_boost_round=RONDAS,
                    train_gp_model_cov_pars=False)
    res["oos"].append(rmse(bst.predict(data=Xt, group_data_pred=g, pred_latent=False)["response_mean"], yt))
    res["oos_new"].append(rmse(bst.predict(data=Xn, group_data_pred=gn, pred_latent=False)["response_mean"], yn))
    res["oos_F"].append(rmse(bst.predict(data=Xt, group_data_pred=g, pred_latent=True)["fixed_effect"], Ft))
    res["oos_sigma2"].append(float(th[0])); res["oos_sigma2_1"].append(float(th[1]))

    # LightGBM con grupo como categórica
    Xg, Xtg, Xng = (np.column_stack([A, gg]) for A, gg in [(X, g), (Xt, g), (Xn, gn)])
    m = lgb.train(params, lgb.Dataset(Xg, y, categorical_feature=[9]), num_boost_round=RONDAS)
    res["lgb_cat"].append(rmse(m.predict(Xtg), yt)); res["lgb_cat_new"].append(rmse(m.predict(Xng), yn))

    # LightGBM sin grupo
    m = lgb.train(params, lgb.Dataset(X, y), num_boost_round=RONDAS)
    res["lgb_sin"].append(rmse(m.predict(Xt), yt)); res["lgb_sin_new"].append(rmse(m.predict(Xn), yn))
    print(r, {k: round(v[-1], 3) for k, v in res.items()})

out = {k: {"media": float(np.mean(v)), "ee": float(np.std(v, ddof=1) / np.sqrt(len(v)))} for k, v in res.items()}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "resultados_gpboost.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
