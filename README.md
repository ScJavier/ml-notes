# Notas ML

PWA para leer notas de estudio de ML con fórmulas (KaTeX), barra lateral, búsqueda y progreso de lectura guardado en el dispositivo.

- `src/*.md`: notas en Markdown con `$...$` y `$$...$$` (el prefijo numérico solo define el orden).
- `tools/build.py`: convierte las notas a `notes/*.html` (KaTeX ya renderizado) y genera `notes/index.json`.
  Requiere `pandoc` y Node: `cd tools && npm install && python3 build.py`.
- `app.js`, `style.css`, `sw.js`, `manifest.webmanifest`: la app. Funciona sin conexión.
- `vendor/katex`: CSS y fuentes de KaTeX 0.19.1 (deben coincidir con la versión de `tools/package.json`).
- `ejemplos/`: scripts que generan los experimentos y figuras de las notas (`pip install -r ejemplos/requirements.txt`).
  - `fundamentos_experimentos.py`, `xgboost_experimentos.py`: experimentos sintéticos de las notas 1 y 2.
  - `lightgbm_experimentos.py`, `catboost_experimentos.py`: experimentos sintéticos de las notas 3 y 4.
  - `simulacion.py`: datos con estructura conocida; `herramientas.py`: H de Friedman, ALE, residuos agrupados, etc.
  - `explicabilidad_regresion.py`, `explicabilidad_clasificacion.py`: notas 5 y 6 (escriben figuras en `src/fig/`).
  - `vc_simulaciones.py`: nota 7 (convergencia uniforme y ERM exactos por programación dinámica).
  - `conformal_productos.py`: nota 8 (conformal marginal, Mondrian por tipo, normalizado, CQR, crepes y MAPIE).
- Las figuras de `src/fig/` se incrustan como WebP dentro de cada nota al construir, para que funcionen sin conexión.
