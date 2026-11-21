"""Experimentos sintéticos para la nota de fundamentos de gradient boosting.
Uso: python fundamentos_experimentos.py   (figuras en ../src/fig, números en resultados_fundamentos.json)"""
import json, time, warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.datasets import make_friedman1
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from estilo import plt, guardar, AZUL, NARANJA, AQUA, VIOLETA, GRIS, TINTA, TINTA2
warnings.filterwarnings("ignore")
FIG = "../src/fig/"; R = {}
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------- implementación mínima (pérdida L2 y L1)
def gb(X, y, M, nu, depth=1, perdida="l2", hojas="treeboost", X_eval=None):
    """Gradient boosting mínimo. Devuelve predicciones de X_eval tras cada iteración.
    perdida: 'l2' o 'l1'. hojas: 'treeboost' (re-estima cada hoja con la pérdida verdadera)
    o 'media' (se queda con la media de los pseudo-residuos en la hoja)."""
    F0 = y.mean() if perdida == "l2" else np.median(y)
    F, Fe = np.full(len(y), F0), np.full(len(X_eval), F0)
    hist = []
    for m in range(M):
        r = y - F if perdida == "l2" else np.sign(y - F)
        t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X, r)
        if perdida == "l1" and hojas == "treeboost":
            hoja, hoja_e = t.apply(X), t.apply(X_eval)
            gam = {h: np.median((y - F)[hoja == h]) for h in np.unique(hoja)}
            paso, paso_e = np.vectorize(gam.get)(hoja), np.vectorize(gam.get)(hoja_e)
        else:
            paso, paso_e = t.predict(X), t.predict(X_eval)
        F = F + nu * paso; Fe = Fe + nu * paso_e
        hist.append(Fe.copy())
    return hist


# ---------------------------------------------------------------- A. boosting paso a paso en 1D
rng = np.random.default_rng(0)
f1d = lambda x: np.sin(1.3 * x) + 0.08 * (x - 5) ** 2
x = rng.uniform(0, 10, 300); y = f1d(x) + rng.normal(0, 0.3, 300)
xg = np.linspace(0, 10, 500)
hist = gb(x.reshape(-1, 1), y, 400, 0.3, depth=1, X_eval=xg.reshape(-1, 1))
fig, axs = plt.subplots(2, 2, figsize=(5.4, 4.6), sharex=True, sharey=True)
for ax, M in zip(axs.ravel(), [1, 5, 25, 400]):
    ax.scatter(x, y, s=4, color=GRIS, alpha=0.5, rasterized=True)
    ax.plot(xg, f1d(xg), color=NARANJA, ls="--", lw=1.6)
    ax.plot(xg, hist[M - 1], color=AZUL)
    ax.set_title(f"M = {M}", loc="left")
fig.suptitle(r"Stumps con $\nu$ = 0.3: función real (naranja) y modelo (azul)", x=0.02, ha="left", fontsize=10.5)
fig.tight_layout(); guardar(fig, FIG + "fund_pasos.png")
R["A_rmse_vs_verdad"] = {M: round(rmse(hist[M - 1], f1d(xg)), 3) for M in [1, 5, 25, 100, 400]}

# ---------------------------------------------------------------- B. shrinkage frente a número de iteraciones
Xtr, ytr = make_friedman1(1000, 10, noise=1.0, random_state=1)
Xva, yva = make_friedman1(5000, 10, noise=1.0, random_state=2)
Xte, yte = make_friedman1(20000, 10, noise=1.0, random_state=3)
fig, ax = plt.subplots(figsize=(5.2, 3.9))
filas = []
for nu, col in [(1.0, NARANJA), (0.3, VIOLETA), (0.1, AQUA), (0.03, AZUL)]:
    M = 3000
    m = GradientBoostingRegressor(n_estimators=M, learning_rate=nu, max_depth=3, random_state=0).fit(Xtr, ytr)
    va = np.array([rmse(p, yva) for p in m.staged_predict(Xva)])
    tr = np.array([rmse(p, ytr) for p in m.staged_predict(Xtr)])
    best = int(va.argmin())
    te = rmse(list(m.staged_predict(Xte))[best], yte)
    filas.append(dict(nu=nu, mejor_M=best + 1, rmse_val=round(float(va[best]), 3), rmse_prueba=round(te, 3),
                      rmse_ent_en_mejor=round(float(tr[best]), 3), rmse_val_final=round(float(va[-1]), 3)))
    ax.plot(np.arange(1, M + 1), va, color=col, label=rf"$\nu$ = {nu}")
    ax.scatter([best + 1], [va[best]], color=col, s=30, zorder=3)
ax.set_xscale("log"); ax.set_ylim(0.9, 3.0); ax.set_xlabel("iteraciones (escala log)"); ax.set_ylabel("RMSE de validación")
ax.set_title("Friedman #1, árboles de profundidad 3", loc="left"); ax.legend(loc="upper right")
guardar(fig, FIG + "fund_shrinkage.png")
R["B_shrinkage"] = filas

# ---------------------------------------------------------------- C. profundidad del árbol e interacciones
def f_aditiva(X): return 10 * np.sin(np.pi * X[:, 0]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
def f_friedman(X): return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
def f_triple(X): return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 15 * (X[:, 2] > 0.5) * (X[:, 3] > 0.5) * (X[:, 4] > 0.5) + 5 * X[:, 4]
filas = []
for nombre, f in [("aditiva", f_aditiva), ("Friedman #1 (orden 2)", f_friedman), ("con término de orden 3", f_triple)]:
    r = np.random.default_rng(5)
    Xa = r.uniform(size=(2000, 10)); ya = f(Xa) + r.normal(size=2000)
    Xv = r.uniform(size=(5000, 10)); yv = f(Xv) + r.normal(size=5000)
    Xt = r.uniform(size=(20000, 10)); yt = f(Xt) + r.normal(size=20000)
    fila = dict(datos=nombre)
    for d in (1, 2, 3, 6):
        m = GradientBoostingRegressor(n_estimators=4000, learning_rate=0.1, max_depth=d, random_state=0).fit(Xa, ya)
        va = np.array([rmse(p, yv) for p in m.staged_predict(Xv)]); best = int(va.argmin())
        fila[f"prof_{d}"] = round(rmse(list(m.staged_predict(Xt))[best], yt), 3); fila[f"M_{d}"] = best + 1
    filas.append(fila)
R["C_profundidad"] = filas

# ---------------------------------------------------------------- D. submuestreo (stochastic gradient boosting)
filas = []
for sub in (1.0, 0.8, 0.5, 0.2):
    errs, Ms, ts = [], [], []
    for s in range(5):
        Xa, ya = make_friedman1(1000, 10, noise=1.0, random_state=100 + s)
        t0 = time.time()
        m = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.05, max_depth=3, subsample=sub, random_state=s).fit(Xa, ya)
        ts.append(time.time() - t0)
        va = np.array([rmse(p, yva) for p in m.staged_predict(Xva)]); best = int(va.argmin())
        errs.append(rmse(list(m.staged_predict(Xte))[best], yte)); Ms.append(best + 1)
    filas.append(dict(subsample=sub, rmse_media=round(float(np.mean(errs)), 3), rmse_de=round(float(np.std(errs)), 3),
                      M_medio=int(np.mean(Ms)), segundos=round(float(np.mean(ts)), 1)))
R["D_submuestreo"] = filas

# ---------------------------------------------------------------- E. pérdidas robustas con atípicos (1D)
rng = np.random.default_rng(1)
x = rng.uniform(0, 10, 600); limpio = f1d(x) + rng.normal(0, 0.3, 600)
atip = rng.uniform(size=600) < 0.10
y = limpio + atip * rng.uniform(4, 8, 600)
fig, ax = plt.subplots(figsize=(5.2, 3.9))
ax.scatter(x[~atip], y[~atip], s=4, color=GRIS, alpha=0.5, rasterized=True)
ax.scatter(x[atip], y[atip], s=8, color=TINTA2, marker="x", lw=0.8, label="atípicos (10 %)")
ax.plot(xg, f1d(xg), color=TINTA, ls=":", lw=1.6, label="función real")
filas = []
for nombre, kw, col in [("cuadrática", dict(loss="squared_error"), NARANJA), ("absoluta", dict(loss="absolute_error"), AZUL),
                        ("Huber (α = 0.9)", dict(loss="huber", alpha=0.9), AQUA)]:
    m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=2, min_samples_leaf=25, random_state=0, **kw).fit(x.reshape(-1, 1), y)
    p = m.predict(xg.reshape(-1, 1))
    ax.plot(xg, p, color=col, label=nombre)
    filas.append(dict(perdida=nombre, rmse_vs_verdad=round(rmse(p, f1d(xg)), 3), sesgo_medio=round(float(np.mean(p - f1d(xg))), 3)))
ax.set_ylim(-1.5, 9); ax.set_title("Mismos datos, tres pérdidas", loc="left"); ax.set_xlabel("x")
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.17))
guardar(fig, FIG + "fund_perdidas_robustas.png")
R["E_robustas"] = filas
R["E_desplazamiento_esperado_media"] = round(float(0.10 * 6), 2)

# ---------------------------------------------------------------- F. TreeBoost: re-estimar las hojas con la pérdida verdadera (L1)
Xa, ya = make_friedman1(2000, 10, noise=1.0, random_state=7)
filas = []
for hojas in ("media", "treeboost"):
    h = gb(Xa, ya, 1000, 0.1, depth=3, perdida="l1", hojas=hojas, X_eval=Xte)
    filas.append(dict(hojas=hojas, **{f"MAE_M{M}": round(mae(h[M - 1], yte), 3) for M in (10, 50, 200, 1000)}))
R["F_treeboost_l1"] = filas
R["F_mae_oraculo"] = round(mae(f_friedman(np.c_[Xte]) if False else make_friedman1(20000, 10, noise=0.0, random_state=3)[1], yte), 3)

# ---------------------------------------------------------------- G. AdaBoost y la pérdida exponencial
r = np.random.default_rng(3)
def datos_g(n):
    X = r.uniform(-1, 1, size=(n, 4)); eta = 3 * X[:, 0] + 2 * np.sin(3 * X[:, 1]) - 2 * X[:, 2] ** 2
    p = 1 / (1 + np.exp(-eta)); return X, (r.uniform(size=n) < p).astype(int), eta
Xa, ya, _ = datos_g(4000); Xt, yt, eta_t = datos_g(20000)
filas = []
modelos = {
    "AdaBoost (SAMME, stumps)": AdaBoostClassifier(n_estimators=400, learning_rate=0.5, random_state=0),
    "GB pérdida exponencial": GradientBoostingClassifier(loss="exponential", n_estimators=400, learning_rate=0.1, max_depth=1, random_state=0),
    "GB log-loss": GradientBoostingClassifier(loss="log_loss", n_estimators=400, learning_rate=0.1, max_depth=1, random_state=0),
}
for nombre, m in modelos.items():
    m.fit(Xa, ya); F = m.decision_function(Xt)
    p = m.predict_proba(Xt)[:, 1]
    fila = dict(modelo=nombre, exactitud=round(accuracy_score(yt, p > 0.5), 3), auc=round(roc_auc_score(yt, F), 3))
    if nombre != "AdaBoost (SAMME, stumps)":
        b = np.polyfit(eta_t, F, 1)[0]
        fila["pendiente_F_vs_logodds_real"] = round(float(b), 3)
        fila["logloss"] = round(log_loss(yt, p), 4)
    filas.append(fila)
R["G_adaboost"] = filas

json.dump(R, open("resultados_fundamentos.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
