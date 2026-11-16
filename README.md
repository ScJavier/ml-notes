# Notas ML

PWA para leer notas de estudio de ML con fórmulas (KaTeX), barra lateral, búsqueda y progreso de lectura guardado en el dispositivo.

- `src/*.md`: notas en Markdown con `$...$` y `$$...$$` (el prefijo numérico solo define el orden).
- `tools/build.py`: convierte las notas a `notes/*.html` (KaTeX ya renderizado) y genera `notes/index.json`.
  Requiere `pandoc` y Node: `cd tools && npm install && python3 build.py`.
- `app.js`, `style.css`, `sw.js`, `manifest.webmanifest`: la app. Funciona sin conexión.
- `vendor/katex`: CSS y fuentes de KaTeX 0.19.1 (deben coincidir con la versión de `tools/package.json`).
