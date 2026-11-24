"""Nota 13 · Modelos mixtos 1: un intercepto aleatorio (escuelas y un examen).

Modelo: y_ij = mu + u_j + e_ij, u_j ~ N(0, tau^2), e_ij ~ N(0, sigma^2).
Valores: mu = 500, tau = 20 (escuela), sigma = 60 (alumno) -> ICC = 0.1, umbral natural k = sigma^2/tau^2 = 9.

Experimentos:
  1. Un conjunto de datos: promedios crudos vs BLUP vs verdad, según el tamaño de la escuela (figura).
  2. Monte Carlo: error de los tres estimadores (sin pooling, pooling completo, pooling parcial) por tamaño.
  3. Estimar tau^2 con pocas escuelas: dispersión de tau_hat y frecuencia de tau_hat = 0.
  4. La media global: MCO con error estándar ingenuo vs modelo mixto (cobertura del IC al 95 %).
Uso: python3 ejemplos/mixtos_01_escuelas.py   (statsmodels 0.14)
"""
import json, os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from estilo import plt, guardar, AZUL, NARANJA, AQUA, GRIS, TINTA, TINTA2

warnings.filterwarnings("ignore")
AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "..", "src", "fig")
MU, TAU, SIGMA = 500.0, 20.0, 60.0


def tamanos(rng, J):
    """Tamaños muy desiguales: log-uniforme entre 2 y 120 alumnos."""
    return np.round(np.exp(rng.uniform(np.log(2), np.log(120), J))).astype(int)


def simula(rng, J=40, tau=TAU, sigma=SIGMA):
    n = tamanos(rng, J)
    u = rng.normal(0, tau, J)
    esc = np.repeat(np.arange(J), n)
    y = MU + u[esc] + rng.normal(0, sigma, n.sum())
    return pd.DataFrame({"escuela": esc, "y": y}), u, n


def ajusta(df, reml=True):
    """REML (o ML) con statsmodels. Ojo: con method="lbfgs" el optimizador a veces se queda en tau = 0
    aunque el óptimo esté lejos de la frontera; se usa BFGS (el valor por defecto) y Powell de respaldo."""
    m = smf.mixedlm("y ~ 1", df, groups=df["escuela"]).fit(reml=reml, method=["bfgs", "powell"])
    tau2 = max(float(m.cov_re.iloc[0, 0]), 0.0); sig2 = float(m.scale)
    mu = float(m.fe_params.iloc[0]); se_mu = float(m.bse.iloc[0])
    g = df.groupby("escuela").y.agg(["mean", "size"])
    lam = tau2 / (tau2 + sig2 / g["size"].to_numpy())          # peso del dato propio
    blup = lam * (g["mean"].to_numpy() - mu)                   # BLUP = lambda_j (ybar_j - mu_hat)
    return mu, se_mu, tau2, sig2, blup


def tres_estimadores(df, n, mu_hat, blup):
    crudo = df.groupby("escuela").y.mean().to_numpy()
    completo = np.full(len(n), df.y.mean())
    parcial = mu_hat + blup
    return crudo, completo, parcial


res = {}

# ---------- 1. un conjunto de datos ----------
rng = np.random.default_rng(1)
df, u, n = simula(rng)
mu_hat, se_mu, tau2, sig2, blup = ajusta(df)
crudo, completo, parcial = tres_estimadores(df, n, mu_hat, blup)
verdad = MU + u
res["ej1"] = dict(mu_hat=mu_hat, tau_hat=tau2 ** .5, sigma_hat=sig2 ** .5, k_hat=sig2 / tau2,
                  ecm_crudo=float(np.mean((crudo - verdad) ** 2)), ecm_parcial=float(np.mean((parcial - verdad) ** 2)),
                  ecm_completo=float(np.mean((completo - verdad) ** 2)))
lam = tau2 / (tau2 + sig2 / n)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
orden = np.argsort(n)
for j in orden:
    ax.annotate("", xy=(n[j], parcial[j]), xytext=(n[j], crudo[j]),
                arrowprops=dict(arrowstyle="->", color=GRIS, lw=1))
ax.scatter(n, crudo, s=22, color=GRIS, label="promedio crudo de la escuela", zorder=3)
ax.scatter(n, parcial, s=22, color=AZUL, label="BLUP (pooling parcial)", zorder=4)
ax.scatter(n, verdad, s=34, marker="_", color=NARANJA, lw=2, label="verdad", zorder=5)
ax.axhline(mu_hat, color=TINTA2, lw=1, ls="--")
ax.text(118, mu_hat + 3, r"$\hat\mu$", color=TINTA2, ha="right")
ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels([2, 5, 10, 20, 50, 100])
ax.set_xlabel("alumnos evaluados en la escuela (escala log)"); ax.set_ylabel("puntaje promedio")
ax.legend(loc="upper right", fontsize=8.5)
guardar(fig, os.path.join(FIG, "mix1_embudo.png"))

fig, ax = plt.subplots(figsize=(6.4, 3.2))
nn = np.linspace(1, 120, 300)
for k_, c, lab in [(1, AQUA, "k = 1 (escuelas muy distintas)"), (9, AZUL, "k = 9 (este ejemplo)"),
                   (36, NARANJA, "k = 36 (escuelas muy parecidas)")]:
    ax.plot(nn, nn / (nn + k_), color=c, label=lab)
ax.set_xlabel("alumnos evaluados en la escuela, $n_j$"); ax.set_ylabel("peso del dato propio, $\\lambda_j$")
ax.set_ylim(0, 1.02); ax.legend(loc="lower right", fontsize=8.5)
guardar(fig, os.path.join(FIG, "mix1_peso.png"))

# ---------- 2. Monte Carlo: error por tamaño ----------
R = 1000
cortes = [(0, 5, "2–5"), (5, 15, "6–15"), (15, 40, "16–40"), (40, 200, "41–120")]
acum = {m: {c[2]: [] for c in cortes} for m in ("crudo", "completo", "parcial", "parcial_oraculo")}
mus, ses, tau2s, ses_ingenuo, mus_mco = [], [], [], [], []
for r in range(R):
    rng = np.random.default_rng(1000 + r)
    df, u, n = simula(rng)
    mu_hat, se_mu, tau2, sig2, blup = ajusta(df)
    crudo, completo, parcial = tres_estimadores(df, n, mu_hat, blup)
    # oráculo: mismo BLUP con tau y sigma verdaderos (para ver cuánto cuesta estimarlos)
    lam_o = TAU ** 2 / (TAU ** 2 + SIGMA ** 2 / n)
    oraculo = MU + lam_o * (crudo - MU)
    verdad = MU + u
    for m, est in [("crudo", crudo), ("completo", completo), ("parcial", parcial), ("parcial_oraculo", oraculo)]:
        for lo, hi, nom in cortes:
            sel = (n > lo) & (n <= hi)
            acum[m][nom].extend(list((est[sel] - verdad[sel]) ** 2))
    mus.append(mu_hat); ses.append(se_mu); tau2s.append(tau2)
    mus_mco.append(df.y.mean()); ses_ingenuo.append(df.y.std(ddof=1) / np.sqrt(len(df)))
res["ej2"] = {m: {k: float(np.sqrt(np.mean(v))) for k, v in d.items()} for m, d in acum.items()}
res["ej2_total"] = {m: float(np.sqrt(np.mean(sum(d.values(), [])))) for m, d in acum.items()}

# ---------- 4. media global: cobertura ----------
mus, ses, mus_mco, ses_ingenuo = map(np.array, (mus, ses, mus_mco, ses_ingenuo))
res["ej4"] = dict(
    sd_mu_mixto=float(mus.std()), sd_mu_mco=float(mus_mco.std()),
    se_mixto_prom=float(ses.mean()), se_ingenuo_prom=float(ses_ingenuo.mean()),
    cob_mixto=float(np.mean(np.abs(mus - MU) <= 1.96 * ses)),
    cob_ingenuo=float(np.mean(np.abs(mus_mco - MU) <= 1.96 * ses_ingenuo)),
    sesgo_mixto=float(mus.mean() - MU), sesgo_mco=float(mus_mco.mean() - MU))

# ---------- 3. estimar tau con pocas escuelas ----------
filas = []
for J in (5, 10, 20, 40, 100):
    for r in range(400):
        rng = np.random.default_rng(50_000 + 1000 * J + r)
        df, u, n = simula(rng, J=J)
        for reml in (True, False):
            _, _, tau2, _, _ = ajusta(df, reml=reml)
            filas.append(dict(J=J, reml=reml, tau=np.sqrt(max(tau2, 0))))
t3 = pd.DataFrame(filas)
res["ej3"] = {f"{J}_{'REML' if reml else 'ML'}": dict(
    mediana=float(g.tau.median()), p10=float(g.tau.quantile(.1)), p90=float(g.tau.quantile(.9)),
    frac_cero=float(np.mean(g.tau < 0.5)), media=float(g.tau.mean()))
    for (J, reml), g in t3.groupby(["J", "reml"])}

fig, ax = plt.subplots(figsize=(6.4, 3.4))
Js = [5, 10, 20, 40, 100]
for k_, (reml, c, lab) in enumerate([(True, AZUL, "REML"), (False, NARANJA, "ML")]):
    for i, J in enumerate(Js):
        v = t3[(t3.J == J) & (t3.reml == reml)].tau
        x = i + (k_ - 0.5) * 0.28
        ax.plot([x, x], [v.quantile(.1), v.quantile(.9)], color=c, lw=2.5, solid_capstyle="round")
        ax.scatter([x], [v.median()], color=c, s=28, zorder=3, label=lab if i == 0 else None)
ax.axhline(TAU, color=TINTA2, ls="--", lw=1); ax.text(-0.45, TAU + 1, r"$\tau$ verdadero", color=TINTA2, ha="left", fontsize=9)
ax.set_xticks(range(len(Js))); ax.set_xticklabels(Js)
ax.set_xlabel("número de escuelas, $J$"); ax.set_ylabel(r"$\hat\tau$ (mediana y percentiles 10–90)")
ax.legend(loc="upper right")
t3.to_csv(os.path.join(AQUI, "resultados_mixtos_01_tau.csv"), index=False)
guardar(fig, os.path.join(FIG, "mix1_tau.png"))

json.dump(res, open(os.path.join(AQUI, "resultados_mixtos_01.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
