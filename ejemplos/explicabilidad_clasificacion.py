"""Explicabilidad práctica de LightGBM: clasificación binaria con datos simulados de estructura conocida.
Uso: python explicabilidad_clasificacion.py   (genera figuras en ../src/fig y resultados_clasificacion.json)"""
import json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import simulacion as sim
from herramientas import h_friedman, centrar, residuos_por_intervalos, reemplazar
from estilo import plt, guardar, AZUL, NARANJA, AQUA, VIOLETA, GRIS, TINTA, TINTA2
warnings.filterwarnings("ignore")
FIG = "../src/fig/"; R = {}
logit = lambda p: np.log(p / (1 - p))

Xtr, ytr, _ = sim.simular_clasificacion(8000, 1)
Xva, yva, _ = sim.simular_clasificacion(3000, 2)
Xte, yte, pte_real = sim.simular_clasificacion(8000, 3)
Xtr, Xva, Xte = [d.astype({"x4": float, "b": float}) for d in (Xtr, Xva, Xte)]
R["tasa_positivos"] = round(float(ytr.mean()), 3)

def entrenar(**extra):
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=0, verbose=-1, **extra)
    return m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
modelo = entrenar()
p = modelo.predict_proba(Xte)[:, 1]
def metricas(pp): return dict(logloss=round(log_loss(yte, pp), 4), auc=round(roc_auc_score(yte, pp), 4), brier=round(brier_score_loss(yte, pp), 4), media_pred=round(float(pp.mean()), 3))
R["arboles"] = int(modelo.best_iteration_)
R["metricas"] = {"modelo": metricas(p), "oráculo (p real)": metricas(pte_real), "tasa constante": metricas(np.full(len(yte), ytr.mean()))}
R["tasa_prueba"] = round(float(yte.mean()), 3)

# ------------------------------------------------------------------ 1. calibración (y cómo se rompe al reponderar clases)
pesado = entrenar(scale_pos_weight=(1 - ytr.mean()) / ytr.mean())
pp = pesado.predict_proba(Xte)[:, 1]
R["metricas"]["con scale_pos_weight"] = metricas(pp)
R["scale_pos_weight"] = round(float((1 - ytr.mean()) / ytr.mean()), 2)
fig, ax = plt.subplots(figsize=(5.0, 4.4))
ax.plot([0, 1], [0, 1], color=TINTA2, lw=1)
for nombre, q, col in [("modelo", p, AZUL), ("con scale_pos_weight", pp, NARANJA)]:
    fo, mp = calibration_curve(yte, q, n_bins=15, strategy="quantile")
    ax.plot(mp, fo, color=col, marker="o", ms=4, label=nombre)
ax.set_xlabel("probabilidad predicha (media por intervalo)"); ax.set_ylabel("frecuencia observada")
ax.set_title("Curva de calibración (prueba)", loc="left"); ax.legend(loc="upper left"); ax.set_aspect("equal")
guardar(fig, FIG + "clf_calibracion.png")
# corrección analítica del desplazamiento: restar log(w) al log-odds
w = (1 - ytr.mean()) / ytr.mean()
pc = 1 / (1 + np.exp(-(logit(pp) - np.log(w))))
R["metricas"]["scale_pos_weight corregido"] = metricas(pc)

# ------------------------------------------------------------------ 2. escala: log-odds frente a probabilidad
raw = lambda X: modelo.predict(X, raw_score=True)
prob = lambda X: modelo.predict_proba(X)[:, 1]
S = Xte.sample(200, random_state=0).reset_index(drop=True)
vars_ = sim.VARIABLES
H_raw = h_friedman(raw, S, vars_)
H_prob = h_friedman(prob, S, vars_)
H_real_raw = h_friedman(lambda X: np.asarray(sim.INTERCEPTO_LOGIT + 0.8 * sim.eta(X)), S, vars_).set_index("par")
H_real_prob = h_friedman(lambda X: np.asarray(sim.prob_verdadera(X)), S, vars_).set_index("par")
tab = H_raw.set_index("par")[["H2"]].rename(columns={"H2": "H2_logodds_modelo"})
tab["H2_logodds_real"] = H_real_raw.H2; tab["H2_prob_modelo"] = H_prob.set_index("par").H2; tab["H2_prob_real"] = H_real_prob.H2
tab["H_prob_modelo_sin_norm"] = H_prob.set_index("par").H_sin_normalizar
tab = tab.sort_values("H2_prob_real", ascending=False)
R["H_escalas"] = tab.head(8).round(3).reset_index().to_dict("records")

# PDP en las dos escalas contra la verdad
verdad = {"x1": sim.f_x1, "x2": sim.f_x2, "x4": sim.f_x4}
fig, axs = plt.subplots(3, 2, figsize=(5.4, 6.6))
for fila, v in enumerate(["x1", "x2", "x4"]):
    grid = np.unique(np.quantile(Xte[v], np.linspace(0.02, 0.98, 50)))
    pd_raw = np.array([raw(reemplazar(Xte, v, np.full(len(Xte), g))).mean() for g in grid])
    pd_prob = np.array([prob(reemplazar(Xte, v, np.full(len(Xte), g))).mean() for g in grid])
    xo = Xte[v].values
    axs[fila, 0].plot(grid, centrar(grid, pd_raw, xo), color=AZUL)
    axs[fila, 0].plot(grid, centrar(grid, 0.8 * verdad[v](grid), xo), color=NARANJA, ls="--")
    # verdad en probabilidad: promedio de la probabilidad real fijando v
    pr = np.array([np.mean(sim.prob_verdadera(reemplazar(Xte, v, np.full(len(Xte), g)))) for g in grid])
    axs[fila, 1].plot(grid, pd_prob, color=AZUL, label="PDP del modelo")
    axs[fila, 1].plot(grid, pr, color=NARANJA, ls="--", label="PDP real")
    axs[fila, 0].set_ylabel(v)
axs[0, 0].set_title("log-odds (centrado)", loc="left"); axs[0, 1].set_title("probabilidad", loc="left")
axs[0, 1].legend(loc="upper left", fontsize=8.5)
fig.tight_layout(); guardar(fig, FIG + "clf_pdp_escalas.png")

# ICE en dos escalas para dos observaciones con distinta probabilidad base
alta = Xte[(prob(Xte) > 0.75)].iloc[[0]]; baja = Xte[(prob(Xte) < 0.08)].iloc[[0]]
grid = np.linspace(-3, 3, 80)
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.9))
for obs, col, nombre in [(alta, AZUL, "obs. con p alta"), (baja, VIOLETA, "obs. con p baja")]:
    rep = pd.concat([obs] * len(grid), ignore_index=True).assign(x1=grid)
    axs[0].plot(grid, raw(rep), color=col, label=nombre); axs[1].plot(grid, prob(rep), color=col)
axs[0].set_title("log-odds", loc="left"); axs[1].set_title("probabilidad", loc="left")
axs[0].set_xlabel("x1"); axs[1].set_xlabel("x1")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); guardar(fig, FIG + "clf_ice_escalas.png")
rep_a = pd.concat([alta] * len(grid), ignore_index=True).assign(x1=grid); rep_b = pd.concat([baja] * len(grid), ignore_index=True).assign(x1=grid)
R["ice_amplitud"] = {"alta_logodds": round(float(np.ptp(raw(rep_a))), 2), "baja_logodds": round(float(np.ptp(raw(rep_b))), 2),
                     "alta_prob": round(float(np.ptp(prob(rep_a))), 3), "baja_prob": round(float(np.ptp(prob(rep_b))), 3)}

# ------------------------------------------------------------------ 3. importancia por permutación con distintas métricas
imp = {}
for nombre, scoring in [("log-loss", "neg_log_loss"), ("AUC", "roc_auc"), ("Brier", "neg_brier_score")]:
    r_ = permutation_importance(modelo, Xte, yte, scoring=scoring, n_repeats=10, random_state=0)
    s_ = pd.Series(r_.importances_mean, index=Xte.columns)
    imp[nombre] = (s_ / s_.max()).round(3)
imp = pd.DataFrame(imp); imp["gain"] = pd.Series(modelo.booster_.feature_importance("gain"), index=Xte.columns)
imp["gain"] = (imp.gain / imp.gain.max()).round(3)
R["permutacion_metricas"] = imp.sort_values("log-loss", ascending=False).reset_index().rename(columns={"index": "variable"}).to_dict("records")

# ------------------------------------------------------------------ 4. residuos binarios: calibración por grupos y falta de interacción
aditivo = entrenar(interaction_constraints=[[i] for i in range(Xtr.shape[1])])
pa = aditivo.predict_proba(Xte)[:, 1]
R["metricas"]["aditivo (sin interacciones)"] = metricas(pa)
fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.4), sharex=True, sharey=True)
for ax, (nombre, q) in zip(axs, [("modelo aditivo (sin interacciones)", pa), ("modelo completo", p)]):
    rb = residuos_por_intervalos(Xte.x5, yte - q, q=10, grupo=Xte.b)
    for bval, col in [(0, GRIS), (1, AZUL)]:
        s_ = rb[rb.g == bval]
        ax.errorbar(s_.x, s_.media, yerr=1.96 * s_.de / np.sqrt(s_.n), color=col, marker="o", ms=4, capsize=2, lw=1.6, label=f"b = {bval}")
    ax.axhline(0, color=TINTA2, lw=1); ax.set_title(nombre, loc="left"); ax.set_ylabel("y − p̂ (media)")
axs[0].legend(loc="upper left"); axs[1].set_xlabel("x5 (intervalos por deciles)")
fig.tight_layout(); guardar(fig, FIG + "clf_residuos_interaccion.png")
cal_cat = pd.DataFrame({"cat": Xte.cat.astype(str), "y": yte, "p": p, "p_ad": pa}).groupby("cat").agg(n=("y", "size"), observada=("y", "mean"), predicha=("p", "mean"))
R["calibracion_por_cat"] = cal_cat.round(3).reset_index().to_dict("records")

json.dump(R, open("resultados_clasificacion.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
