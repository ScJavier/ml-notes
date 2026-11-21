"""Simulaciones y figuras para la nota de teoría del aprendizaje y dimensión VC.
Uso: python vc_simulaciones.py   (figuras en ../src/fig, números en resultados_vc.json)

Clase de hipótesis H_k: clasificadores en [0,1] que valen 1 en una unión de a lo sumo k intervalos
(VC = 2k), con extremos en una malla de G celdas. Con la malla, sup_h |R̂(h) - R(h)| y el ERM se
calculan exactamente por programación dinámica."""
import json, math, warnings
import numpy as np
from sklearn.svm import LinearSVC
from estilo import plt, guardar, AZUL, NARANJA, AQUA, VIOLETA, GRIS, TINTA, TINTA2
warnings.filterwarnings("ignore")
FIG = "../src/fig/"; R = {}

G = 400
CENTROS = (np.arange(G) + 0.5) / G
OBJETIVO = [(0.10, 0.25), (0.40, 0.60), (0.75, 0.85)]          # unión de 3 intervalos
RUIDO = 0.10                                                     # prob. de invertir la etiqueta
f_obj = np.zeros(G, bool)
for a, b in OBJETIVO:
    f_obj |= (CENTROS > a) & (CENTROS < b)
p1 = np.where(f_obj, 1 - RUIDO, RUIDO)                           # P(y=1 | celda)
# masa de error verdadera por celda según la etiqueta que ponga h
t0, t1 = p1 / G, (1 - p1) / G


def muestra(n, rng):
    c = rng.integers(0, G, n)
    y = (rng.uniform(size=n) < p1[c]).astype(int)
    return c, y


def costos_empiricos(c, y, n):
    """e_c(l): fracción de la muestra en la celda c con etiqueta distinta de l."""
    unos = np.bincount(c[y == 1], minlength=G); ceros = np.bincount(c[y == 0], minlength=G)
    return unos / n, ceros / n                                     # (error si l=0, error si l=1)


def optimizar(v0, v1, k, signo=1.0):
    """max (signo=1) o min (signo=-1) de sum_c v_{l_c}(c) sobre etiquetados con <= k rachas de unos.
    Devuelve el valor y el etiquetado."""
    v0, v1 = signo * v0, signo * v1
    NEG = -1e18
    A0 = np.full(k + 1, NEG); A1 = np.full(k + 1, NEG); A0[0] = 0.0
    back = np.zeros((G, 2, k + 1, 2), dtype=np.int16)              # (celda, etiqueta, rachas) -> (etiqueta previa, rachas previas)
    for g in range(G):
        n0 = np.maximum(A0, A1) + v0[g]
        back[g, 0, :, 0] = np.where(A0 >= A1, 0, 1); back[g, 0, :, 1] = np.arange(k + 1)
        cont = A1                                                   # seguir en la misma racha
        nueva = np.full(k + 1, NEG); nueva[1:] = A0[:-1]           # abrir una racha nueva
        n1 = np.maximum(cont, nueva) + v1[g]
        back[g, 1, :, 0] = np.where(cont >= nueva, 1, 0)
        back[g, 1, :, 1] = np.where(cont >= nueva, np.arange(k + 1), np.arange(k + 1) - 1)
        A0, A1 = n0, n1
    l = 0 if A0.max() >= A1.max() else 1
    j = int(np.argmax(A0 if l == 0 else A1)); val = max(A0.max(), A1.max())
    etiq = np.zeros(G, int)
    for g in range(G - 1, -1, -1):
        etiq[g] = l
        l, j = back[g, l, j, 0], back[g, l, j, 1]
    return signo * val, etiq


def riesgo_verdadero(etiq):
    return float(np.sum(np.where(etiq == 1, t1, t0)))


# ---------------------------------------------------------------- 1. convergencia uniforme
rng = np.random.default_rng(0)
ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
ks = [1, 2, 5, 10]
REP = 40
sup = {k: [] for k in ks}
for n in ns:
    vals = {k: [] for k in ks}
    for r in range(REP):
        c, y = muestra(n, rng)
        e0, e1 = costos_empiricos(c, y, n)
        d0, d1 = e0 - t0, e1 - t1                                  # aporte de cada celda a R̂ - R
        for k in ks:
            mx, _ = optimizar(d0, d1, k, 1.0); mn, _ = optimizar(d0, d1, k, -1.0)
            vals[k].append(max(mx, -mn))
    for k in ks:
        sup[k].append(float(np.mean(vals[k])))
R["sup_desvio"] = {f"k={k} (VC={2*k})": dict(zip(ns, np.round(sup[k], 4).tolist())) for k in ks}

# cota VC (Abu-Mostafa et al.): eps = sqrt(8/n * ln(4 m_H(2n) / delta)), con m_H(2n) <= (e*2n/d)^d
def cota_vc(n, d, delta=0.05):
    ln_m = d * math.log(math.e * 2 * n / d) if 2 * n > d else 2 * n * math.log(2)
    return math.sqrt(8 / n * (ln_m + math.log(4 / delta)))
R["cota_vc"] = {f"d={d}": {n: round(cota_vc(n, d), 3) for n in [100, 1000, 10000, 100000, 1000000]} for d in (2, 10, 20, 100)}

fig, ax = plt.subplots(figsize=(5.2, 4.0))
for k, col in zip(ks, [AZUL, AQUA, VIOLETA, NARANJA]):
    ax.plot(ns, sup[k], color=col, marker="o", ms=4, label=f"k = {k}  (VC = {2*k})")
nn = np.array(ns, float)
ax.plot(nn, [cota_vc(n, 10) for n in nn], color=TINTA2, ls="--", lw=1.4, label="cota VC con d = 10, δ = 0.05")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n (escala log)"); ax.set_ylabel(r"$\sup_h |\hat R(h)-R(h)|$ (media)")
ax.set_title("Convergencia uniforme: uniones de k intervalos", loc="left"); ax.legend(loc="lower left", fontsize=8.5)
guardar(fig, FIG + "vc_convergencia.png")
# pendiente en log-log (tasa empírica)
R["pendiente_loglog"] = {k: round(float(np.polyfit(np.log(ns[2:]), np.log(sup[k][2:]), 1)[0]), 3) for k in ks}
R["razon_sup_k10_k1"] = round(float(np.mean(np.array(sup[10]) / np.array(sup[1]))), 2)

# ---------------------------------------------------------------- 2. ERM y el compromiso sesgo-complejidad
ks2 = [1, 2, 3, 4, 6, 10, 20, 40]
erm = {}
for n in (100, 500, 5000):
    tr, te = {k: [] for k in ks2}, {k: [] for k in ks2}
    for r in range(REP):
        c, y = muestra(n, rng); e0, e1 = costos_empiricos(c, y, n)
        for k in ks2:
            val, etiq = optimizar(e0, e1, k, -1.0)
            tr[k].append(val); te[k].append(riesgo_verdadero(etiq))
    erm[n] = dict(entrenamiento=[round(float(np.mean(tr[k])), 4) for k in ks2], verdadero=[round(float(np.mean(te[k])), 4) for k in ks2])
R["erm"] = {str(n): dict(k=ks2, **v) for n, v in erm.items()}
mejor = {k: None for k in ()}
R["mejor_k"] = {str(n): ks2[int(np.argmin(v["verdadero"]))] for n, v in erm.items()}
R["riesgo_bayes"] = RUIDO
R["mejor_en_clase"] = {k: round(riesgo_verdadero(optimizar(t0, t1, k, -1.0)[1]), 4) for k in ks2}

fig, ax = plt.subplots(figsize=(5.2, 4.0))
x_ = np.arange(len(ks2))
for n, col in zip((100, 500, 5000), [NARANJA, VIOLETA, AZUL]):
    ax.plot(x_, erm[n]["verdadero"], color=col, marker="o", ms=4, label=f"riesgo verdadero, n = {n}")
    ax.plot(x_, erm[n]["entrenamiento"], color=col, ls=":", lw=1.4)
ax.plot(x_, [R["mejor_en_clase"][k] for k in ks2], color=TINTA2, ls="--", lw=1.4, label="mejor de la clase")
ax.axhline(RUIDO, color=GRIS, lw=1); ax.text(len(ks2) - 1, RUIDO - 0.012, "riesgo de Bayes", ha="right", color=TINTA2, fontsize=9)
ax.set_xticks(x_); ax.set_xticklabels([str(k) for k in ks2]); ax.set_xlabel("k (VC = 2k); la verdad tiene k = 3")
ax.set_ylabel("riesgo (error de clasificación)"); ax.set_ylim(0, 0.36)
ax.set_title("ERM: punteado = entrenamiento, sólido = verdadero", loc="left"); ax.legend(loc="upper right", fontsize=8.5)
guardar(fig, FIG + "vc_erm.png")

# ---------------------------------------------------------------- 3. fragmentación con semiplanos en R^2
P = np.array([[0.2, 0.25], [0.8, 0.3], [0.45, 0.85]])
fig, axs = plt.subplots(3, 3, figsize=(5.4, 5.4))
etiquetados = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
for ax, lab in zip(axs.ravel()[:8], etiquetados):
    lab = np.array(lab)
    if lab.min() != lab.max():
        s = LinearSVC(C=1e4, max_iter=100000).fit(P, lab)
        w, b0 = s.coef_[0], s.intercept_[0]
        xs = np.linspace(-0.1, 1.1, 50)
        if abs(w[1]) > 1e-9:
            ax.plot(xs, -(w[0] * xs + b0) / w[1], color=TINTA2, lw=1.4)
        else:
            ax.axvline(-b0 / w[0], color=TINTA2, lw=1.4)
    ax.scatter(P[:, 0], P[:, 1], c=[AZUL if v else NARANJA for v in lab], s=60, edgecolor=TINTA, lw=0.6, zorder=3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax = axs.ravel()[8]
Q = np.array([[0.2, 0.2], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]]); lq = [1, 1, 0, 0]
ax.scatter(Q[:, 0], Q[:, 1], c=[AZUL if v else NARANJA for v in lq], s=60, edgecolor=TINTA, lw=0.6)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax.set_title("4 puntos: imposible", fontsize=9, color=TINTA2)
fig.suptitle("Los 8 etiquetados de 3 puntos se separan con una recta", x=0.02, ha="left", fontsize=10.5)
fig.tight_layout(); guardar(fig, FIG + "vc_fragmentacion.png")

# ---------------------------------------------------------------- 4. lema de Sauer–Shelah
def sauer(n, d): return sum(math.comb(n, i) for i in range(0, min(n, d) + 1))
nsa = np.arange(1, 61)
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(nsa, nsa * 1.0, color=GRIS, ls="--", label=r"$2^n$ (todas las dicotomías)")
for d, col in [(3, AZUL), (10, VIOLETA), (20, NARANJA)]:
    ax.plot(nsa, [math.log2(sauer(int(n), d)) for n in nsa], color=col, label=fr"$\sum_{{i\leq {d}}}\binom{{n}}{{i}}$, d = {d}")
ax.set_xlabel("n"); ax.set_ylabel("log₂ (número de dicotomías)"); ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("Crecimiento exponencial hasta n = d, polinomial después", loc="left")
guardar(fig, FIG + "vc_sauer.png")
R["sauer_ejemplos"] = {f"n={n},d={d}": dict(sauer=sauer(n, d), dos_n=2 ** n) for n, d in [(20, 3), (40, 3), (40, 10)]}

json.dump(R, open("resultados_vc.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
