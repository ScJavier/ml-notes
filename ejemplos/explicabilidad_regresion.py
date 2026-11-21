"""Explicabilidad práctica de LightGBM: regresión con datos simulados de estructura conocida.
Uso: python explicabilidad_regresion.py   (genera figuras en ../src/fig y resultados_regresion.json)"""
import json, warnings
import numpy as np, pandas as pd, lightgbm as lgb, dalex as dx
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import r2_score
import simulacion as sim
from herramientas import h_friedman, ale_1d, centrar, coocurrencia_en_ramas, residuos_por_intervalos
from estilo import plt, guardar, AZUL, NARANJA, AQUA, GRIS, TINTA, TINTA2
warnings.filterwarnings("ignore")
FIG = "../src/fig/"; R = {}
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

# ------------------------------------------------------------------ datos y modelo
Xtr, ytr = sim.simular_regresion(6000, 1)
Xva, yva = sim.simular_regresion(2000, 2)
Xte, yte = sim.simular_regresion(4000, 3)
# sklearn no acepta columnas enteras en partial_dependence: las discretas se guardan como float
Xtr, Xva, Xte = [d.astype({"x4": float, "b": float}) for d in (Xtr, Xva, Xte)]

modelo = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
modelo.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
pred = modelo.predict(Xte)
R["arboles"] = int(modelo.best_iteration_)
R["rmse_modelo"] = rmse(pred, yte)
R["rmse_oraculo"] = rmse(sim.eta(Xte), yte)
R["r2_modelo"] = float(r2_score(yte, pred)); R["r2_oraculo"] = float(r2_score(yte, sim.eta(Xte)))
R["rmse_vs_verdad"] = rmse(pred, sim.eta(Xte))

# ------------------------------------------------------------------ 1. estructura del modelo
arb = modelo.booster_.trees_to_dataframe()
hojas = arb[arb.split_feature.isna()]
R["hojas_por_arbol"] = float(hojas.groupby("tree_index").size().mean())
R["profundidad_max"] = int(hojas.node_depth.max())
imp = pd.DataFrame({"split": modelo.booster_.feature_importance("split"), "gain": modelo.booster_.feature_importance("gain")},
                   index=modelo.booster_.feature_name())
imp = (imp / imp.sum()).round(3)

cortes = arb[arb.split_feature.notna()]
fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.6))
c2 = cortes[cortes.split_feature == "x2"]
axs[0].hist(c2.threshold.astype(float), bins=np.linspace(-2.5, 2.5, 51), weights=c2.split_gain, color=AZUL, rwidth=0.85)
axs[0].axvline(0.5, color=NARANJA, lw=2, ls="--"); axs[0].text(0.6, axs[0].get_ylim()[1] * 0.85, "umbral real 0.5", color=TINTA2)
axs[0].set_title("Umbrales de los cortes en x2 (ponderados por ganancia)", loc="left"); axs[0].set_xlabel("umbral")
c4 = cortes[cortes.split_feature == "x4"]
t4 = c4.threshold.astype(float).round(2)
vc = c4.groupby(t4).split_gain.sum()
axs[1].bar(vc.index, vc.values, width=0.5, color=AZUL)
axs[1].axvline(4, color=NARANJA, lw=2, ls="--"); axs[1].text(4.15, vc.max() * 0.85, "efecto satura en 4", color=TINTA2)
axs[1].set_title("Umbrales de los cortes en x4 (conteo)", loc="left"); axs[1].set_xlabel("umbral (corte: x4 ≤ umbral)")
fig.tight_layout(); guardar(fig, FIG + "reg_umbrales.png")
R["x4_ganancia_por_umbral"] = {str(k): round(float(v / vc.sum()), 3) for k, v in vc.items()}

cc = cortes[cortes.split_feature == "cat"]
mapa = dict(enumerate(sim.NIVELES))
izq = cc.threshold.astype(str).map(lambda s: "{" + ",".join(mapa[int(v)] for v in s.split("||")) + "}")
R["cat_conjuntos"] = (cc.groupby(izq).split_gain.sum() / cc.split_gain.sum()).sort_values(ascending=False).head(6).round(3).to_dict()
R["coocurrencia"] = coocurrencia_en_ramas(arb).head(6).round(3).to_dict()

# ------------------------------------------------------------------ 2. importancia por permutación
perm = permutation_importance(modelo, Xte, yte, scoring="neg_root_mean_squared_error", n_repeats=10, random_state=0)
imp["perm_media"] = perm.importances_mean.round(3); imp["perm_de"] = perm.importances_std.round(3)
exp = dx.Explainer(modelo, Xte, yte, label="LightGBM", verbose=False)
grupo = exp.model_parts(variable_groups={"x2+x3": ["x2", "x3"], "x2": ["x2"], "x3": ["x3"]}, B=10, random_state=0)
g = grupo.result.groupby("variable").dropout_loss.mean()
R["perm_grupo"] = {k: round(float(g[k] - g["_full_model_"]), 3) for k in ["x2", "x3", "x2+x3"]}
R["importancias"] = imp.reset_index().rename(columns={"index": "variable"}).to_dict("records")

orden = imp.sort_values("perm_media").index
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.barh(orden, imp.loc[orden, "perm_media"], xerr=imp.loc[orden, "perm_de"], color=AZUL, height=0.6, error_kw=dict(ecolor=TINTA2, lw=1))
ax.set_xlabel("aumento del RMSE al permutar"); ax.set_title("Importancia por permutación (prueba)", loc="left"); ax.grid(axis="y", visible=False)
guardar(fig, FIG + "reg_permutacion.png")

# ------------------------------------------------------------------ 3. PDP e ICE contra la verdad
verdad = {"x1": sim.f_x1, "x2": sim.f_x2, "x4": sim.f_x4, "x3": lambda x: 0 * np.asarray(x)}
fig, axs = plt.subplots(2, 2, figsize=(5.4, 4.8))
pdp_guardado = {}
for ax, v in zip(axs.ravel(), ["x1", "x2", "x4", "x3"]):
    res = partial_dependence(modelo, Xte, [v], kind="both", grid_resolution=60)
    grid = res["grid_values"][0]; pdv = res["average"][0]; ice = res["individual"][0]
    xo = Xte[v].values
    idx = np.random.default_rng(0).choice(len(ice), 60, replace=False)
    for i in idx:
        ax.plot(grid, centrar(grid, ice[i], xo), color=GRIS, lw=0.5, alpha=0.35)
    ax.plot(grid, centrar(grid, pdv, xo), color=AZUL, label="PDP del modelo")
    ax.plot(grid, centrar(grid, verdad[v](grid), xo), color=NARANJA, ls="--", label="efecto real")
    ax.set_title(v, loc="left"); pdp_guardado[v] = (grid, centrar(grid, pdv, xo))
axs[0, 0].legend(loc="lower right")
fig.suptitle("PDP (azul), ICE centradas (gris) y verdad (naranja)", x=0.02, ha="left", fontsize=10.5)
fig.tight_layout(); guardar(fig, FIG + "reg_pdp.png")
g2, p2 = pdp_guardado["x2"]; R["pdp_x2_salto"] = round(float(p2[g2 > 0.8].mean() - p2[g2 < 0.2].mean()), 3)
g3, p3 = pdp_guardado["x3"]; R["pdp_x3_rango"] = round(float(p3.max() - p3.min()), 3)

# PDP categórica
resc = partial_dependence(modelo, Xte, ["cat"], categorical_features=["cat"], kind="average")
niveles = list(resc["grid_values"][0]); pdc = resc["average"][0]
frec = Xte.cat.value_counts(normalize=True)
pdc_c = pdc - np.sum([pdc[i] * frec[l] for i, l in enumerate(niveles)])
real = np.array([sim.EFECTO_CAT[l] for l in niveles]); real_c = real - np.sum([real[i] * frec[l] for i, l in enumerate(niveles)])
R["pdp_cat"] = {l: dict(modelo=round(float(pdc_c[i]), 2), real=round(float(real_c[i]), 2), n=int(Xte.cat.value_counts()[l])) for i, l in enumerate(niveles)}

# ICE de x5 coloreadas por b
res = partial_dependence(modelo, Xte, ["x5"], kind="both", grid_resolution=50)
grid = res["grid_values"][0]; ice = res["individual"][0]; bb = Xte.b.values
fig, ax = plt.subplots(figsize=(5.2, 3.8))
rng = np.random.default_rng(1)
for bval, col in [(0, GRIS), (1, AZUL)]:
    ids = rng.choice(np.where(bb == bval)[0], 40, replace=False)
    for i in ids:
        ax.plot(grid, ice[i] - ice[i].mean(), color=col, lw=0.6, alpha=0.45)
ax.plot(grid, res["average"][0] - res["average"][0].mean(), color=TINTA, lw=2.2, label="PDP (promedio)")
ax.plot([], [], color=AZUL, label="ICE con b = 1"); ax.plot([], [], color=GRIS, label="ICE con b = 0")
ax.legend(loc="upper left"); ax.set_xlabel("x5"); ax.set_title("ICE de x5: dos poblaciones de curvas", loc="left")
guardar(fig, FIG + "reg_ice_x5.png")
pend = lambda y_: np.polyfit(grid[(grid > -1.5) & (grid < 1.5)], y_[(grid > -1.5) & (grid < 1.5)], 1)[0]
R["pendiente_pdp_x5"] = round(float(pend(res["average"][0])), 2)
R["pendiente_ice_x5_b1"] = round(float(np.median([pend(ice[i]) for i in np.where(bb == 1)[0][:300]])), 2)
R["pendiente_ice_x5_b0"] = round(float(np.median([pend(ice[i]) for i in np.where(bb == 0)[0][:300]])), 2)

# ------------------------------------------------------------------ 4. ALE frente a PDP con variables correlacionadas
# (a) en el modelo principal: rango de cada curva para x2 y x3
for v in ["x2", "x3"]:
    xo = Xte[v].values
    for tipo in ["accumulated", "partial"]:
        rr = exp.model_profile(type=tipo, variables=[v], N=2000, grid_points=60, random_state=0, verbose=False).result
        c = centrar(rr._x_.values, rr._yhat_.values, xo)
        R[f"{tipo}_{v}_rango"] = round(float(np.ptp(c[(rr._x_ > -2) & (rr._x_ < 2)])), 3)
# (b) ejemplo mínimo donde difieren: y = z1·z2 con z1, z2 normales correlacionadas
def ejemplo_z(rho, n=5000, seed=11):
    r = np.random.default_rng(seed)
    z1 = r.normal(size=n); z2 = rho * z1 + np.sqrt(1 - rho ** 2) * r.normal(size=n)
    Z = pd.DataFrame({"z1": z1, "z2": z2}); yz = z1 * z2 + 0.3 * r.normal(size=n)
    mz = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0).fit(Z, yz)
    return Z, yz, mz
tabla = []
puntos = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
for rho in (0.5, 0.7, 0.9):
    Z, yz, mz = ejemplo_z(rho); z1 = Z.z1.values
    res = partial_dependence(mz, Z, ["z1"], kind="average", grid_resolution=80, percentiles=(0.01, 0.99))
    gp, vp = res["grid_values"][0], centrar(res["grid_values"][0], res["average"][0], z1)
    ga, va = ale_1d(mz.predict, Z, "z1"); va = centrar(ga, va, z1)
    ez = dx.Explainer(mz, Z, yz, verbose=False)
    dd = ez.model_profile(type="accumulated", variables=["z1"], N=3000, grid_points=60, random_state=0, verbose=False).result
    vd = centrar(dd._x_.values, dd._yhat_.values, z1)
    ale_real = centrar(puntos, rho * puntos ** 2 / 2, z1)
    for nombre, g_, v_ in [("PDP modelo", gp, vp), ("ALE modelo", ga, va), ("dalex accumulated", dd._x_.values, vd)]:
        tabla.append(dict(rho=rho, curva=nombre, valores=np.round(np.interp(puntos, g_, v_), 2).tolist()))
    tabla.append(dict(rho=rho, curva="ALE real", valores=np.round(ale_real, 2).tolist()))
    if rho == 0.7:
        gg = np.linspace(-2.5, 2.5, 200)
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        ax.plot(gp, vp, color=AZUL, label="PDP del modelo")
        ax.plot(ga, va, color=AQUA, label="ALE del modelo")
        ax.plot(gg, centrar(gg, 0 * gg, z1), color=AZUL, ls=":", lw=1.6, label="PDP real (0)")
        ax.plot(gg, centrar(gg, rho * gg ** 2 / 2, z1), color=NARANJA, ls="--", label="ALE real (0.35·z1²)")
        ax.set_xlim(-2.5, 2.5); ax.set_xlabel("z1"); ax.legend(loc="upper center", fontsize=9)
        ax.set_title("y = z1·z2, corr(z1, z2) = 0.7", loc="left")
        guardar(fig, FIG + "reg_pdp_vs_ale.png")
        fuera = np.mean([np.mean(np.abs(Z.z2.values - rho * g) > 3 * np.sqrt(1 - rho ** 2)) for g in np.linspace(-2, 2, 41)])
        R["pdp_fraccion_fuera_de_soporte_rho07"] = round(float(fuera), 3)
R["pdp_vs_ale"] = tabla

# ------------------------------------------------------------------ 5. interacciones: H de Friedman
S = Xte.sample(200, random_state=0).reset_index(drop=True)
vars_ = sim.VARIABLES
H_mod = h_friedman(modelo.predict, S, vars_)
H_real = h_friedman(lambda X: np.asarray(sim.eta(X)), S, vars_).set_index("par")
H_mod["H2_real"] = H_mod.par.map(H_real.H2).round(3); H_mod["Hsin_real"] = H_mod.par.map(H_real.H_sin_normalizar).round(3)
R["H"] = H_mod.round(3).head(8).to_dict("records")
R["H_max_H2_par"] = H_mod.sort_values("H2", ascending=False).head(3).round(3).to_dict("records")

# ------------------------------------------------------------------ 6. explicaciones locales
obs = Xte[(Xte.b == 1) & (Xte.x5 > 1.2) & (Xte.cat == "E")].iloc[[0]]
R["obs_local"] = {k: (round(float(v), 2) if not isinstance(v, str) else v) for k, v in obs.iloc[0].astype(object).items()}
R["obs_pred"] = round(float(modelo.predict(obs)[0]), 2); R["obs_eta"] = round(float(sim.eta(obs).iloc[0]), 2)
bd = exp.predict_parts(obs, type="break_down", random_state=0).result[["variable", "contribution"]]
orden_alt = ["x5", "b", "x1", "x2", "x3", "x4", "cat", "ruido"]
bd2 = exp.predict_parts(obs, type="break_down", order=orden_alt, random_state=0).result[["variable", "contribution"]]
bdi = exp.predict_parts(obs, type="break_down_interactions", random_state=0).result[["variable", "contribution"]]
R["break_down"] = bd.round(3).values.tolist(); R["break_down_orden_alt"] = bd2.round(3).values.tolist(); R["break_down_inter"] = bdi.round(3).values.tolist()
cp = exp.predict_profile(obs, variables=["x5", "x1"], grid_points=60, verbose=False).result
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.8), sharey=True)
for ax, v in zip(axs, ["x5", "x1"]):
    c = cp[cp._vname_ == v]; ax.plot(c[v], c._yhat_, color=AZUL)
    ax.scatter([obs[v].iloc[0]], [R["obs_pred"]], color=NARANJA, zorder=3, s=40)
    ax.set_xlabel(v)
axs[0].set_ylabel("predicción"); fig.suptitle("Ceteris paribus de una observación", x=0.02, ha="left", fontsize=10.5)
fig.tight_layout(); guardar(fig, FIG + "reg_ceteris_paribus.png")

# ------------------------------------------------------------------ 7. residuos
res_te = yte - pred
fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.8))
axs[0].scatter(pred, res_te, s=3, alpha=0.25, color=GRIS, rasterized=True)
rb = residuos_por_intervalos(pred, res_te, q=15)
axs[0].plot(rb.x, rb.media, color=AZUL, label="media por intervalo"); axs[0].axhline(0, color=TINTA2, lw=1)
axs[0].set_xlabel("valor ajustado"); axs[0].set_ylabel("residuo"); axs[0].set_title("Residuos contra ajustados (prueba)", loc="left"); axs[0].legend(loc="upper left")
rb1 = residuos_por_intervalos(Xte.x1, res_te, q=15)
axs[1].plot(rb1.x, rb1.de, color=AZUL, marker="o", ms=4, label="DE de residuos por intervalo")
gg = np.linspace(-3, 3, 100); axs[1].plot(gg, 0.5 + 0.25 * np.abs(gg), color=NARANJA, ls="--", label="DE real del ruido")
axs[1].set_xlabel("x1"); axs[1].set_ylabel("desviación estándar"); axs[1].set_title("Dispersión de los residuos según x1", loc="left"); axs[1].legend(loc="upper center")
fig.tight_layout(); guardar(fig, FIG + "reg_residuos.png")
R["residuo_media_max_por_variable"] = {v: round(float(residuos_por_intervalos(Xte[v], res_te, q=10).media.abs().max()), 3) for v in ["x1", "x2", "x3", "x4", "x5", "ruido"]}

# modelo aditivo (sin interacciones) para ver qué detectan los residuos
aditivo = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            interaction_constraints=[[i] for i in range(Xtr.shape[1])], random_state=0, verbose=-1)
aditivo.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
pa = aditivo.predict(Xte); R["rmse_aditivo"] = rmse(pa, yte)
fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.4), sharex=True, sharey=True)
for ax, (nombre, p_) in zip(axs, [("modelo aditivo (sin interacciones)", pa), ("modelo completo", pred)]):
    rb = residuos_por_intervalos(Xte.x5, yte - p_, q=10, grupo=Xte.b)
    for bval, col in [(0, GRIS), (1, AZUL)]:
        s_ = rb[rb.g == bval]; ax.plot(s_.x, s_.media, color=col, marker="o", ms=4, label=f"b = {bval}")
    ax.axhline(0, color=TINTA2, lw=1); ax.set_title(nombre, loc="left"); ax.set_ylabel("residuo medio")
axs[0].legend(loc="upper left"); axs[1].set_xlabel("x5 (intervalos por deciles)")
fig.tight_layout(); guardar(fig, FIG + "reg_residuos_interaccion.png")

# ------------------------------------------------------------------ 8. árbol sustituto
sust = DecisionTreeRegressor(max_depth=3, min_samples_leaf=100, random_state=0).fit(Xtr.assign(cat=Xtr.cat.cat.codes), modelo.predict(Xtr))
R["sustituto_r2_fidelidad"] = round(float(r2_score(pred, sust.predict(Xte.assign(cat=Xte.cat.cat.codes)))), 3)
R["sustituto_texto"] = export_text(sust, feature_names=list(Xtr.columns), decimals=2)
for k in [k for k in R if isinstance(R[k], float)]:
    R[k] = round(R[k], 3)
json.dump(R, open("resultados_regresion.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
