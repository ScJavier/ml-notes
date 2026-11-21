"""Datos simulados con estructura conocida para los ejemplos de explicabilidad.

Variables (las mismas para regresión y clasificación):
  x1    continua, U(-3, 3)            efecto 2·sin(x1)                    (no lineal suave)
  x2    continua, N(0, 1)             efecto 1.5·1{x2 > 0.5}              (umbral)
  x3    continua, 0.9·x2 + ruido      sin efecto                           (correlacionada con x2, r ≈ 0.9)
  x4    discreta, Poisson(3)          efecto 0.6·min(x4, 4)               (conteo que satura)
  cat   categórica, 6 niveles A–F     efectos fijos por nivel
  b     binaria, Bernoulli(0.4)       sin efecto principal
  x5    continua, N(0, 1)             efecto 2·x5·b                        (solo interacción con b)
  ruido continua, N(0, 1)             sin efecto
"""
import numpy as np, pandas as pd

NIVELES = list("ABCDEF")
PROB_NIVELES = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
EFECTO_CAT = {"A": 0.0, "B": 0.8, "C": -0.5, "D": 0.0, "E": 1.5, "F": -1.0}
VARIABLES = ["x1", "x2", "x3", "x4", "cat", "b", "x5", "ruido"]

def simular_X(n, seed):
    r = np.random.default_rng(seed)
    x2 = r.normal(size=n)
    X = pd.DataFrame({
        "x1": r.uniform(-3, 3, n),
        "x2": x2,
        "x3": 0.9 * x2 + np.sqrt(1 - 0.81) * r.normal(size=n),
        "x4": r.poisson(3, n),
        "cat": pd.Categorical(r.choice(NIVELES, n, p=PROB_NIVELES), categories=NIVELES),
        "b": r.binomial(1, 0.4, n),
        "x5": r.normal(size=n),
        "ruido": r.normal(size=n),
    })
    return X

# Componentes verdaderos (para comparar con lo que recupera el modelo)
def f_x1(x): return 2 * np.sin(x)
def f_x2(x): return 1.5 * (np.asarray(x) > 0.5)
def f_x4(x): return 0.6 * np.minimum(x, 4)
def f_cat(c): return np.array([EFECTO_CAT[v] for v in np.asarray(c, dtype=object)])
def f_int(x5, b): return 2 * np.asarray(x5) * np.asarray(b)

def eta(X):
    """Predictor verdadero (sin ruido)."""
    return f_x1(X.x1) + f_x2(X.x2) + f_x4(X.x4) + f_cat(X.cat) + f_int(X.x5, X.b)

def sd_ruido(X):
    """Desviación estándar del ruido en regresión: crece con |x1| (heterocedasticidad)."""
    return 0.5 + 0.25 * np.abs(X.x1.values)

def simular_regresion(n, seed):
    X = simular_X(n, seed)
    r = np.random.default_rng(seed + 10_000)
    y = eta(X) + sd_ruido(X) * r.normal(size=n)
    return X, y.values if hasattr(y, "values") else y

INTERCEPTO_LOGIT = -3.2
def prob_verdadera(X):
    return 1 / (1 + np.exp(-(INTERCEPTO_LOGIT + 0.8 * eta(X))))

def simular_clasificacion(n, seed):
    X = simular_X(n, seed)
    r = np.random.default_rng(seed + 20_000)
    p = np.asarray(prob_verdadera(X))
    return X, (r.uniform(size=n) < p).astype(int), p
