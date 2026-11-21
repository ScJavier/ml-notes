"""Funciones auxiliares de explicabilidad que no vienen en las librerías (o conviene ver por dentro)."""
import itertools
import numpy as np, pandas as pd


def reemplazar(df, col, valores):
    """Asigna valores a una columna respetando el tipo categórico."""
    df = df.copy()
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        df[col] = pd.Categorical(valores, categories=df[col].cat.categories)
    else:
        df[col] = valores
    return df


def dependencia_parcial_en_muestra(predecir, S, cols):
    """PD_cols evaluada en los valores observados de S (fuerza bruta: n^2 predicciones).

    Para cada fila i de S fija las columnas `cols` en los valores de la fila i,
    predice sobre todo S y promedia. Devuelve un vector de longitud n, centrado."""
    n = len(S)
    grande = pd.concat([S] * n, ignore_index=True)
    for c in cols:
        grande = reemplazar(grande, c, np.repeat(np.asarray(S[c].values, dtype=object if isinstance(S[c].dtype, pd.CategoricalDtype) else None), n))
    pd_ = predecir(grande).reshape(n, n).mean(axis=1)
    return pd_ - pd_.mean()


def h_friedman(predecir, S, variables, pares=None):
    """Estadístico H de Friedman y Popescu (2008) por pares.

    H2   = sum (PD_jk - PD_j - PD_k)^2 / sum PD_jk^2       (proporción, adimensional)
    Hsin = sqrt(mean (PD_jk - PD_j - PD_k)^2)              (sin normalizar, en unidades de la predicción)
    """
    uno = {v: dependencia_parcial_en_muestra(predecir, S, [v]) for v in variables}
    filas = []
    for j, k in (pares or itertools.combinations(variables, 2)):
        jk = dependencia_parcial_en_muestra(predecir, S, [j, k])
        resto = jk - uno[j] - uno[k]
        filas.append(dict(par=f"{j}×{k}", H2=float((resto ** 2).sum() / (jk ** 2).sum()),
                          H_sin_normalizar=float(np.sqrt((resto ** 2).mean()))))
    return pd.DataFrame(filas).sort_values("H_sin_normalizar", ascending=False).reset_index(drop=True)


def centrar(grid, valores, x_obs):
    """Centra una curva para que su promedio sobre la distribución observada de la variable sea 0."""
    return valores - np.mean(np.interp(x_obs, grid, valores))


def coocurrencia_en_ramas(arboles):
    """Pares (padre, hijo) de variables distintas en cortes consecutivos de una rama, ponderados por la ganancia del hijo.

    `arboles` es la salida de Booster.trees_to_dataframe(). Es una pista estructural de interacción,
    no una medida de efecto."""
    idx = arboles.set_index("node_index")
    hijos = arboles[arboles.split_feature.notna() & arboles.parent_index.notna()]
    cuenta = {}
    for _, nodo in hijos.iterrows():
        padre = idx.loc[nodo.parent_index, "split_feature"]
        if padre != nodo.split_feature:
            par = " × ".join(sorted([padre, nodo.split_feature]))
            cuenta[par] = cuenta.get(par, 0.0) + nodo.split_gain
    s = pd.Series(cuenta).sort_values(ascending=False)
    return (s / s.sum()).rename("fraccion_ganancia")


def residuos_por_intervalos(x, r, q=10, grupo=None):
    """Media y desviación estándar de los residuos por intervalos (cuantiles) de x, opcionalmente por grupo."""
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "r": np.asarray(r)})
    if grupo is not None:
        df["g"] = np.asarray(grupo)
    df["bin"] = pd.qcut(df.x, q, duplicates="drop")
    claves = ["g", "bin"] if grupo is not None else ["bin"]
    out = df.groupby(claves, observed=True).agg(x=("x", "mean"), media=("r", "mean"), de=("r", "std"), n=("r", "size")).reset_index()
    return out


def ale_1d(predecir, X, col, k=40):
    """ALE de primer orden (Apley y Zhu, 2020) para una variable numérica.

    Intervalos por cuantiles; en cada intervalo se promedia f(borde superior) - f(borde inferior)
    sobre las observaciones que caen en él; se acumula y se centra con pesos por frecuencia."""
    x = X[col].values.astype(float)
    bordes = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)))
    idx = np.clip(np.searchsorted(bordes, x, side="left") - 1, 0, len(bordes) - 2)
    Xa, Xb = X.copy(), X.copy()
    Xa[col] = bordes[idx]; Xb[col] = bordes[idx + 1]
    d = predecir(Xb) - predecir(Xa)
    nb = len(bordes) - 1
    media = np.array([d[idx == j].mean() if np.any(idx == j) else 0.0 for j in range(nb)])
    cuenta = np.bincount(idx, minlength=nb)
    acum = np.concatenate([[0.0], np.cumsum(media)])
    centro_intervalo = (acum[:-1] + acum[1:]) / 2
    acum = acum - np.sum(centro_intervalo * cuenta) / cuenta.sum()
    return bordes, acum
