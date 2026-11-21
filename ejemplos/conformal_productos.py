"""Conformal prediction por tipo de producto con un solo modelo LightGBM.
Uso: python conformal_productos.py   (figuras en ../src/fig, números en resultados_conformal.json)"""
import json, math, warnings, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import cross_val_predict, KFold
from scipy import stats
from estilo import plt, guardar, AZUL, NARANJA, AQUA, VIOLETA, GRIS, TINTA, TINTA2
warnings.filterwarnings("ignore")
FIG = "../src/fig/"; R = {}
ALFA = 0.10

# ---------------------------------------------------------------- datos: mezcla de tipos de producto
TIPOS = list("ABCDEFG")
PROB = np.array([0.34, 0.25, 0.15, 0.13, 0.08, 0.047, 0.003])
BASE = dict(A=10.0, B=12.0, C=9.0, D=11.0, E=13.0, F=10.5, G=12.0)
PEND = dict(A=-0.3, B=-0.5, C=-0.2, D=-0.4, E=-0.6, F=-0.3, G=-0.5)
def sd_ruido(tipo, x1):
    fija = dict(A=0.4, B=1.5, D=0.8, E=0.8, F=1.2, G=1.0)
    return np.where(tipo == "C", 0.2 + 0.2 * x1, np.vectorize(lambda t: fija.get(t, 0.0))(tipo))

def simular(n, seed):
    r = np.random.default_rng(seed)
    tipo = r.choice(TIPOS, n, p=PROB)
    x1 = r.uniform(0, 10, n); x2 = r.normal(size=n); x3 = r.uniform(size=n)
    media = np.vectorize(BASE.get)(tipo) + np.vectorize(PEND.get)(tipo) * x1 + 0.8 * np.sin(2 * x2) + 0.5 * x3
    y = media + sd_ruido(tipo, x1) * r.normal(size=n)
    X = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, tipo=pd.Categorical(tipo, categories=TIPOS)))
    return X, y

Xtr, ytr = simular(10000, 1)          # entrenamiento del modelo
Xpool, ypool = simular(400000, 2)      # de aquí salen calibración y prueba en cada repetición
R["n_entrenamiento_por_tipo"] = Xtr.tipo.value_counts().reindex(TIPOS).astype(int).to_dict()

# ---------------------------------------------------------------- modelos: media, dificultad y cuantiles
params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.9, random_state=0, verbose=-1)
idx = np.random.default_rng(0).permutation(len(Xtr)); a, b = idx[:8000], idx[8000:]
def ajustar(obj_extra, y_=ytr):
    m = lgb.LGBMRegressor(**params, **obj_extra)
    m.fit(Xtr.iloc[a], y_[a], eval_set=[(Xtr.iloc[b], y_[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m
mu = ajustar({})
# dificultad sigma(x): modelo para |residuo| con residuos fuera de muestra (validación cruzada)
oof = cross_val_predict(lgb.LGBMRegressor(**{**params, "n_estimators": mu.best_iteration_}), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=0))
sig = ajustar({}, y_=np.abs(ytr - oof))
q_lo = ajustar(dict(objective="quantile", alpha=ALFA / 2))
q_hi = ajustar(dict(objective="quantile", alpha=1 - ALFA / 2))

P = dict(mu=mu.predict(Xpool), sig=np.maximum(sig.predict(Xpool), 0.05), lo=q_lo.predict(Xpool), hi=q_hi.predict(Xpool))
tipo_pool = Xpool.tipo.astype(str).values
x1_pool = Xpool.x1.values

# ---------------------------------------------------------------- conformal: cuantil y métodos
def cuantil_conformal(s, alfa=ALFA):
    """k-ésimo menor puntaje con k = ceil((n+1)(1-alfa)); infinito si k > n."""
    n = len(s); k = math.ceil((n + 1) * (1 - alfa))
    return np.inf if k > n else float(np.sort(s)[k - 1])

def puntajes(metodo, i):
    y = ypool[i]
    if metodo == "abs":  return np.abs(y - P["mu"][i])
    if metodo == "norm": return np.abs(y - P["mu"][i]) / P["sig"][i]
    if metodo == "cqr":  return np.maximum(P["lo"][i] - y, y - P["hi"][i])

def intervalo(metodo, i, q):
    if metodo == "abs":  return P["mu"][i] - q, P["mu"][i] + q
    if metodo == "norm": return P["mu"][i] - q * P["sig"][i], P["mu"][i] + q * P["sig"][i]
    if metodo == "cqr":  return P["lo"][i] - q, P["hi"][i] + q

def evaluar(metodo, mondrian, ical, ite, grupos=None):
    """Cobertura y ancho por tipo en el conjunto de prueba. grupos: mapeo tipo -> grupo de calibración."""
    g_cal = tipo_pool[ical] if grupos is None else np.vectorize(grupos.get)(tipo_pool[ical])
    g_te = tipo_pool[ite] if grupos is None else np.vectorize(grupos.get)(tipo_pool[ite])
    s = puntajes(metodo, ical)
    if mondrian:
        qs = {g: cuantil_conformal(s[g_cal == g]) for g in np.unique(g_te)}
        q = np.vectorize(qs.get)(g_te)
    else:
        q = np.full(len(ite), cuantil_conformal(s))
    lo, hi = intervalo(metodo, ite, q)
    cubre = (ypool[ite] >= lo) & (ypool[ite] <= hi)
    out = {}
    for t in TIPOS:
        m = tipo_pool[ite] == t
        out[t] = (cubre[m].mean(), np.median(hi[m] - lo[m]), int((tipo_pool[ical] == t).sum()))
    out["total"] = (cubre.mean(), np.median(hi - lo), len(ical))
    return out, lo, hi, cubre

METODOS = [("marginal, |residuo|", "abs", False), ("Mondrian, |residuo|", "abs", True),
           ("marginal, normalizado", "norm", False), ("Mondrian, normalizado", "norm", True),
           ("marginal, CQR", "cqr", False), ("Mondrian, CQR", "cqr", True)]

# ---------------------------------------------------------------- repeticiones: calibración 5 000, prueba 50 000
REP = 300; N_CAL, N_TE = 5000, 50000
rng = np.random.default_rng(10)
acum = {nombre: {t: [] for t in TIPOS + ["total"]} for nombre, _, _ in METODOS}
for r in range(REP):
    perm = rng.choice(len(ypool), N_CAL + N_TE, replace=False)
    ical, ite = perm[:N_CAL], perm[N_CAL:]
    for nombre, met, mon in METODOS:
        res, *_ = evaluar(met, mon, ical, ite)
        for t, v in res.items(): acum[nombre][t].append(v)
tabla = {}
for nombre, _, _ in METODOS:
    tabla[nombre] = {t: dict(cobertura=round(float(np.mean([v[0] for v in acum[nombre][t]])), 3),
                             cobertura_de=round(float(np.std([v[0] for v in acum[nombre][t]])), 3),
                             ancho_mediano=round(float(np.median([v[1] for v in acum[nombre][t] if np.isfinite(v[1])])), 2),
                             frac_infinito=round(float(np.mean([not np.isfinite(v[1]) for v in acum[nombre][t]])), 3),
                             n_cal_medio=round(float(np.mean([v[2] for v in acum[nombre][t]])), 1))
                     for t in TIPOS + ["total"]}
R["tabla"] = tabla

# figura: cobertura por tipo, marginal vs Mondrian (|residuo|)
fig, ax = plt.subplots(figsize=(5.2, 3.8))
xs = np.arange(len(TIPOS)); w = 0.38
for j, (nombre, col) in enumerate([("marginal, |residuo|", NARANJA), ("Mondrian, |residuo|", AZUL)]):
    cov = [tabla[nombre][t]["cobertura"] for t in TIPOS]
    ax.bar(xs + (j - 0.5) * w, cov, width=w - 0.04, color=col, label=nombre)
ax.axhline(1 - ALFA, color=TINTA, lw=1.2, ls="--"); ax.text(-0.6, 0.905, "objetivo 90 %", color=TINTA2, fontsize=9, ha="left", va="bottom")
ax.set_xticks(xs); ax.set_xticklabels(TIPOS); ax.set_ylim(0.5, 1.02); ax.set_xlabel("tipo de producto"); ax.set_ylabel("cobertura en prueba (media)")
ax.set_title("Un solo cuantil para todos frente a uno por tipo", loc="left")
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.1))
guardar(fig, FIG + "conf_cobertura_tipo.png")

# ---------------------------------------------------------------- cobertura dentro del tipo C según x1
rng = np.random.default_rng(99)
perm = rng.choice(len(ypool), N_CAL + 200000, replace=False); ical, ite = perm[:N_CAL], perm[N_CAL:]
cond = {}
bandas = {}
for nombre, met, mon in [("Mondrian, |residuo|", "abs", True), ("Mondrian, normalizado", "norm", True), ("Mondrian, CQR", "cqr", True)]:
    res, lo, hi, cubre = evaluar(met, mon, ical, ite)
    mC = tipo_pool[ite] == "C"
    bins = np.digitize(x1_pool[ite][mC], [2, 4, 6, 8])
    cond[nombre] = [round(float(cubre[mC][bins == k].mean()), 3) for k in range(5)]
    bandas[nombre] = (lo[mC], hi[mC])
R["tipoC_cobertura_por_x1"] = dict(intervalos_x1=["0–2", "2–4", "4–6", "6–8", "8–10"], **cond)
# figura: bandas del tipo C (pocas observaciones de prueba para que se vea)
mC = np.where(tipo_pool[ite] == "C")[0][:600]
xx = x1_pool[ite][mC]; yy = ypool[ite][mC]; o = np.argsort(xx)
fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.6), sharex=True, sharey=True)
for ax, nombre, col in [(axs[0], "Mondrian, |residuo|", NARANJA), (axs[1], "Mondrian, normalizado", AZUL)]:
    lo, hi = bandas[nombre][0][:600][o], bandas[nombre][1][:600][o]
    dentro = (yy[o] >= lo) & (yy[o] <= hi)
    ax.vlines(xx[o], lo, hi, color=col, alpha=0.25, lw=1.2)
    ax.scatter(xx[o][dentro], yy[o][dentro], s=5, color=TINTA2, alpha=0.6, rasterized=True)
    ax.scatter(xx[o][~dentro], yy[o][~dentro], s=14, color=TINTA, marker="x", lw=0.9, label="fuera del intervalo")
    ax.set_title(nombre + f" — cobertura por tramo de x1: " + ", ".join(f"{c:.0%}" for c in cond[nombre]), loc="left", fontsize=9.5)
    ax.set_ylabel("y")
axs[0].legend(loc="lower left", fontsize=8.5); axs[1].set_xlabel("x1 (el ruido del tipo C crece con x1)")
fig.tight_layout(); guardar(fig, FIG + "conf_tipoC.png")

# ---------------------------------------------------------------- tipos pequeños: el tipo G
estrategias = {
    "Mondrian, solo G": dict(grupos=None),
    "G junto con E y F": dict(grupos={**{t: t for t in TIPOS}, "E": "EFG", "F": "EFG", "G": "EFG"}),
    "marginal (todos)": dict(grupos={t: "todos" for t in TIPOS}),
}
rng = np.random.default_rng(5)
peq = {k: [] for k in estrategias}
for n_cal in (2000, 5000, 20000):
    for k in estrategias: peq[k] = []
    for r in range(REP):
        perm = rng.choice(len(ypool), n_cal + N_TE, replace=False); ical, ite = perm[:n_cal], perm[n_cal:]
        for k, e in estrategias.items():
            res, *_ = evaluar("norm", True, ical, ite, grupos=e["grupos"])
            peq[k].append(res["G"])
    R[f"tipoG_ncal{n_cal}"] = {k: dict(n_cal_G=round(float(np.mean([v[2] for v in vals])), 1),
                                        cobertura=round(float(np.mean([v[0] for v in vals])), 3),
                                        p_cobertura_bajo_80=round(float(np.mean([v[0] < 0.8 for v in vals])), 3),
                                        ancho_mediano=round(float(np.median([v[1] for v in vals if np.isfinite(v[1])])), 2),
                                        frac_infinito=round(float(np.mean([not np.isfinite(v[1]) for v in vals])), 3))
                               for k, vals in peq.items()}

# ---------------------------------------------------------------- variabilidad de la cobertura: distribución Beta
rng = np.random.default_rng(3)
fig, axs = plt.subplots(3, 1, figsize=(5.2, 6.4), sharex=True)
beta_res = {}
for ax, n in zip(axs, (19, 100, 1000)):
    covs = []
    for _ in range(20000):
        s = rng.exponential(size=n); q = cuantil_conformal(s)
        covs.append(1 - np.exp(-q))                    # P(S <= q) exacta para S ~ Exp(1)
    covs = np.array(covs); l = math.floor((n + 1) * ALFA)
    a_, b_ = n + 1 - l, l
    ax.hist(covs, bins=np.linspace(0.6, 1, 81), density=True, color=AZUL, alpha=0.75)
    xg = np.linspace(0.6, 0.9999, 400); ax.plot(xg, stats.beta.pdf(xg, a_, b_), color=NARANJA, lw=2)
    ax.axvline(1 - ALFA, color=TINTA, ls="--", lw=1)
    ax.set_title(f"n de calibración = {n}: Beta({a_}, {b_})", loc="left", fontsize=10); ax.set_yticks([])
    beta_res[n] = dict(media_sim=round(float(covs.mean()), 4), media_teorica=round(a_ / (a_ + b_), 4),
                       de_sim=round(float(covs.std()), 4), p_bajo_85=round(float(np.mean(covs < 0.85)), 3),
                       p_bajo_85_teorica=round(float(stats.beta.cdf(0.85, a_, b_)), 3))
axs[-1].set_xlabel("cobertura condicional al conjunto de calibración")
fig.tight_layout(); guardar(fig, FIG + "conf_beta.png")
R["beta"] = beta_res

# ---------------------------------------------------------------- librerías: crepes y MAPIE contra la implementación propia
from crepes import ConformalRegressor
from mapie.regression import SplitConformalRegressor
from mapie.conditional_conformal_prediction import ConditionalSplitConformalRegressor
rng = np.random.default_rng(7)
perm = rng.choice(len(ypool), N_CAL + 3000, replace=False); ical, ite = perm[:N_CAL], perm[N_CAL:]
_, lo_m, hi_m, _ = evaluar("abs", True, ical, ite)
cr = ConformalRegressor().fit(residuals=ypool[ical] - P["mu"][ical], bins=tipo_pool[ical])
iv = cr.predict_int(y_hat=P["mu"][ite], bins=tipo_pool[ite], confidence=1 - ALFA)
R["crepes_mondrian_dif_max"] = float(np.max(np.abs(np.c_[lo_m, hi_m] - iv)))
_, lo_g, hi_g, _ = evaluar("abs", False, ical, ite)
mp = SplitConformalRegressor(estimator=mu, confidence_level=1 - ALFA, prefit=True).conformalize(Xpool.iloc[ical], ypool[ical])
_, ivm = mp.predict_interval(Xpool.iloc[ite])
R["mapie_marginal_dif_max"] = float(np.max(np.abs(np.c_[lo_g, hi_g] - ivm[:, :, 0])))
from sklearn.base import BaseEstimator, RegressorMixin

class ConCodigos(RegressorMixin, BaseEstimator):
    """Adaptador para MAPIE condicional: recibe una matriz numérica (tipo como código entero)
    y llama al modelo LightGBM con la columna categórica."""
    def __init__(self, modelo=None): self.modelo = modelo
    def fit(self, X, y): return self
    def predict(self, X):
        X = np.asarray(X, float)
        D = pd.DataFrame(dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], tipo=pd.Categorical.from_codes(X[:, 3].astype(int), TIPOS)))
        return self.modelo.predict(D)

Xnum = np.c_[Xpool[["x1", "x2", "x3"]].values, Xpool.tipo.cat.codes.values]
def solo_tipo(Z):
    Z = np.asarray(Z); return np.stack([(Z[:, 3] == k).astype(float) for k in range(len(TIPOS))], 1)
def tipo_y_x1_en_C(Z):
    Z = np.asarray(Z); C = (Z[:, 3] == TIPOS.index("C")).astype(float)
    return np.c_[solo_tipo(Z), C * Z[:, 0]]
ite2 = ite[:3000]
for nombre, fmap in [("solo_tipo", solo_tipo), ("tipo_y_x1_en_C", tipo_y_x1_en_C)]:
    t0 = time.time()
    cc = ConditionalSplitConformalRegressor(feature_map=fmap, estimator=ConCodigos(mu), confidence_level=1 - ALFA, prefit=True)
    cc.conformalize(Xnum[ical], ypool[ical])
    _, ivc = cc.predict_interval(Xnum[ite2])
    R[f"mapie_condicional_{nombre}_segundos_3000"] = round(time.time() - t0, 1)
    if nombre == "solo_tipo":
        R["mapie_condicional_vs_mondrian_dif_ancho_max"] = float(np.max(np.abs((ivc[:, 1, 0] - ivc[:, 0, 0]) - (hi_m[:3000] - lo_m[:3000]))))

# cobertura condicional a x1 dentro del tipo C con muchos puntos de prueba de C
rng = np.random.default_rng(99)
perm = rng.choice(len(ypool), N_CAL + 200000, replace=False); ical, ite = perm[:N_CAL], perm[N_CAL:]
iC = ite[tipo_pool[ite] == "C"][:8000]
filas = {}
for nombre, fmap in [("MAPIE condicional, solo tipo", solo_tipo), ("MAPIE condicional, tipo + x1 en C", tipo_y_x1_en_C)]:
    cc = ConditionalSplitConformalRegressor(feature_map=fmap, estimator=ConCodigos(mu), confidence_level=1 - ALFA, prefit=True)
    cc.conformalize(Xnum[ical], ypool[ical])
    _, ivc = cc.predict_interval(Xnum[iC])
    cubre = (ypool[iC] >= ivc[:, 0, 0]) & (ypool[iC] <= ivc[:, 1, 0])
    bins = np.digitize(x1_pool[iC], [2, 4, 6, 8])
    filas[nombre] = dict(por_x1=[round(float(cubre[bins == k].mean()), 3) for k in range(5)], total=round(float(cubre.mean()), 3),
                         ancho_mediano=round(float(np.median(ivc[:, 1, 0] - ivc[:, 0, 0])), 2))
R["tipoC_mapie_condicional"] = filas

json.dump(R, open("resultados_conformal.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
