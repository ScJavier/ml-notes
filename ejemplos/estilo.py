"""Estilo de figuras: legibles en pantalla de móvil (ancho ~400 px)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, NARANJA, AQUA, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRIS, TINTA, TINTA2, FONDO, REJILLA = "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "text.color": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9.5,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

def guardar(fig, ruta):
    fig.savefig(ruta)
    plt.close(fig)
