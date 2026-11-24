"""Nota 12 (artículo MEGB, Messer y Schmid 2025): réplica reducida de la simulación basada en modelo.

Escenarios Linear-Normal y Complex-Normal de la tabla 1 del artículo: D = 50 áreas con N_d = 1000,
muestra estratificada con n_d entre 6 y 49 (n = 1375; el artículo usa 1410), x1, x2 ~ N(mu_d, .),
mu_d ~ U(-1, 1), efecto de área N(0, 500^2), error N(0, 1000^2). Se reporta el RRMSE (%) de las medias
de área, por área sobre las repeticiones, resumido con media y mediana sobre las 50 áreas.

Estimadores:
  - Directo: media muestral del área.
  - BHF: modelo lineal mixto con intercepto aleatorio (gpboost GPModel con X = [1, x1, x2]).
  - MEGB: algoritmo 1 del artículo (bucle tipo EM: XGBoost sobre y - b0 - Z v; LMM sobre y - f).
    Hiperparámetros de la sección 4.1 (eta 0.01, profundidad 3, min_child_weight 3, subsample 0.5,
    lambda 1, gamma 0.9); número de árboles por paro temprano en un 20 % de la muestra.
  - MERF: el mismo bucle con un bosque aleatorio (300 árboles, min_samples_leaf 5, max_features 1/3).
  - GPBoost: estimación conjunta (nota 9), mismos hiperparámetros de árbol salvo el submuestreo
    (GPBoost no admite bagging), paro temprano igual.
Uso: python3 ejemplos/megb_articulo.py
"""
import json, os, warnings
import numpy as np
import xgboost as xgb
import gpboost as gpb
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
R, D, ND = 20, 50, 1000
N_D = np.round(np.linspace(6, 49, D)).astype(int)
XGB = dict(eta=0.01, max_depth=3, min_child_weight=3, subsample=0.5, reg_lambda=1.0, gamma=0.9,
           objective="reg:squarederror", nthread=2)


def poblacion(rng, escenario):
    mu = rng.uniform(-1, 1, D)
    area = np.repeat(np.arange(D), ND)
    v = rng.normal(0, 500, D)
    e = rng.normal(0, 1000, D * ND)
    if escenario == "lineal":
        x1 = rng.normal(mu[area], 3); x2 = rng.normal(mu[area], 3)
        y = 5000 - 500 * x1 - 500 * x2 + v[area] + e
    else:
        x1 = rng.normal(mu[area], 4); x2 = rng.normal(mu[area], 2)
        y = 15000 - 500 * x1 * x2 - 250 * x2 ** 2 + v[area] + e
    return np.column_stack([x1, x2]), y, area


def muestra(rng, area):
    idx = [rng.choice(np.where(area == d)[0], N_D[d], replace=False) for d in rng.permutation(D)]
    return np.sort(np.concatenate(idx))


def lmm_intercepto(r, g):
    """LMM r = b0 + v_g + e. Devuelve b0, v (por área), sigma2_e, sigma2_v, -log-verosimilitud."""
    gp = gpb.GPModel(group_data=g, likelihood="gaussian")
    gp.fit(y=r, X=np.ones((len(r), 1)))
    b0 = float(np.asarray(gp.get_coef()).ravel()[0])
    re = gp.predict_training_data_random_effects()
    v = np.zeros(D); v[g] = np.asarray(re).ravel()
    s2e, s2v = np.asarray(gp.get_cov_pars()).ravel()[:2]
    return b0, v, s2e, s2v, gp.get_current_neg_log_likelihood()


def ajusta_xgb(X, y, rng):
    va = rng.uniform(size=len(y)) < 0.2
    dtr, dva = xgb.DMatrix(X[~va], y[~va]), xgb.DMatrix(X[va], y[va])
    m = xgb.train(XGB, dtr, num_boost_round=3000, evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=False)
    k = m.best_iteration + 1
    return xgb.train(XGB, xgb.DMatrix(X, y), num_boost_round=k), k


def em_mixto(X, y, g, ajusta, predice, rng, iter_max=25, tol=1e-6):
    """Algoritmo 1 del artículo (y el de MERF): alterna el modelo de árboles y el LMM."""
    v = np.zeros(D); b0 = 0.0; nll_ant = None
    for it in range(iter_max):
        m = ajusta(X, y - b0 - v[g], rng)
        f = predice(m, X)
        b0, v, s2e, s2v, nll = lmm_intercepto(y - f, g)
        if nll_ant is not None and abs(nll - nll_ant) / abs(nll_ant) < tol:
            break
        nll_ant = nll
    return m, b0, v, it + 1


def medias_area(f_pob, area, b0, v):
    return np.bincount(area, weights=f_pob) / ND + b0 + v


res = {}
for esc in ["lineal", "complejo"]:
    err = {m: [] for m in ["directo", "bhf", "megb", "merf", "gpboost"]}
    verdad = []; iters = {"megb": [], "merf": []}
    for r in range(R):
        rng = np.random.default_rng(10 * r + (esc == "lineal"))
        Xp, yp, area = poblacion(rng, esc)
        mu = np.bincount(area, weights=yp) / ND; verdad.append(mu)
        s = muestra(rng, area); X, y, g = Xp[s], yp[s], area[s]

        err["directo"].append(np.bincount(g, weights=y) / np.bincount(g) - mu)

        # BHF
        gp = gpb.GPModel(group_data=g, likelihood="gaussian")
        gp.fit(y=y, X=np.column_stack([np.ones(len(y)), X]))
        beta = np.asarray(gp.get_coef()).ravel()
        v = np.zeros(D); v[g] = np.asarray(gp.predict_training_data_random_effects()).ravel()
        err["bhf"].append(medias_area(np.column_stack([np.ones(len(yp)), Xp]) @ beta, area, 0.0, v) - mu)

        # MEGB
        m, b0, v, it = em_mixto(X, y, g, lambda A, b, rg: ajusta_xgb(A, b, rg)[0],
                                lambda m_, A: m_.predict(xgb.DMatrix(A)), rng)
        err["megb"].append(medias_area(m.predict(xgb.DMatrix(Xp)), area, b0, v) - mu); iters["megb"].append(it)

        # MERF (mismo bucle con bosque aleatorio)
        m, b0, v, it = em_mixto(X, y, g, lambda A, b, rg: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=1 / 3, n_jobs=2, random_state=0).fit(A, b),
                                lambda m_, A: m_.predict(A), rng)
        err["merf"].append(medias_area(m.predict(Xp), area, b0, v) - mu); iters["merf"].append(it)

        # GPBoost (estimación conjunta), rondas por paro temprano y reajuste con todos los datos
        par = dict(learning_rate=0.01, max_depth=3, min_data_in_leaf=3,
                   lambda_l2=1.0, num_leaves=8, verbose=-1)
        # Ojo: el paso de gradiente de GPBoost depende de la escala (gradiente = Psi^{-1}(y - F), del orden
        # de 1/sigma^2). Con y en miles y tasa 0.01 los árboles casi no se mueven, así que se estandariza y.
        sd = y.std(); ys = y / sd
        va = rng.uniform(size=len(y)) < 0.2
        gpv = gpb.GPModel(group_data=g[~va], likelihood="gaussian"); gpv.set_prediction_data(group_data_pred=g[va])
        dtr = gpb.Dataset(X[~va], ys[~va])
        b = gpb.train(params=dict(par, metric="l2"), train_set=dtr, gp_model=gpv, num_boost_round=3000,
                      valid_sets=gpb.Dataset(X[va], ys[va], reference=dtr), early_stopping_rounds=50,
                      use_gp_model_for_validation=True, verbose_eval=False)
        gpm = gpb.GPModel(group_data=g, likelihood="gaussian")
        b = gpb.train(params=par, train_set=gpb.Dataset(X, ys), gp_model=gpm, num_boost_round=b.best_iteration)
        fp = b.predict(data=Xp, group_data_pred=area, pred_latent=True)
        err["gpboost"].append(sd * np.bincount(area, weights=fp["fixed_effect"] + fp["random_effect_mean"]) / ND - mu)
        print(esc, r, {k: round(float(np.sqrt(np.mean(np.array(v_) ** 2))), 1) for k, v_ in err.items()}, flush=True)

    media_mu = np.mean(verdad, 0)
    out = {}
    for k, v_ in err.items():
        rrmse = 100 * np.sqrt(np.mean(np.array(v_) ** 2, 0)) / media_mu
        rb = 100 * np.mean(np.array(v_), 0) / media_mu
        out[k] = {"rrmse_media": float(rrmse.mean()), "rrmse_mediana": float(np.median(rrmse)),
                  "rb_media": float(rb.mean())}
    out["iter_em"] = {k: float(np.mean(v_)) for k, v_ in iters.items()}
    res[esc] = out
    print(esc, json.dumps(out, indent=1), flush=True)

json.dump(res, open(os.path.join(os.path.dirname(__file__), "resultados_megb.json"), "w"), indent=1)
