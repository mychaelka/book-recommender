"""Shared matplotlib style for the exploratory notebooks."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2a78d6"      # primary series / single-series charts
ORANGE = "#eb6834"    # second series
AQUA = "#1baf7a"      # third series
CATEGORICAL = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Sequential single-hue ramp (light -> dark) for magnitude, e.g. hexbin density
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "ytick.left": False,
        "legend.frameon": False,
        "patch.edgecolor": SURFACE,   # thin surface gap between adjacent bars
        "patch.linewidth": 1,
    })


def hbar(ax, counts, title, color=BLUE, label_fmt="{:,.0f}"):
    """Horizontal bar chart of a value_counts() Series, largest at the top, values labelled."""
    counts = counts.iloc[::-1]
    ax.barh(counts.index.astype(str), counts.values, color=color, height=0.7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    for y, value in enumerate(counts.values):
        ax.text(value, y, " " + label_fmt.format(value), va="center",
                fontsize=8, color=INK_SECONDARY)
    ax.margins(x=0.12)
    return ax
