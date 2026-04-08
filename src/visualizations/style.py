from __future__ import annotations

import matplotlib.pyplot as plt

# Baby pink theme
PRIMARY_COLOR = "#F8BBD0"
SECONDARY_COLOR = "#F48FB1"
ACCENT_COLOR = "#EC407A"
DARK_PINK = "#D81B60"
LIGHT_PINK = "#FCE4EC"
SOFT_PURPLE = "#CE93D8"
GRID_COLOR = "#F9DDE7"
TEXT_COLOR = "#4A4A4A"
WHITE = "#FFFFFF"

SERIES_COLORS = [
    "#F8BBD0",
    "#F48FB1",
    "#EC407A",
    "#CE93D8",
    "#D81B60",
    "#F06292",
]


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "axes.edgecolor": LIGHT_PINK,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "text.color": TEXT_COLOR,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })


def finish_plot(output_path, dpi: int = 220) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()