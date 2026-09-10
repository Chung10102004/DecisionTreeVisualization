"""Shared visual tokens.

The palette is the validated reference instance from the data-viz method: the
eight categorical hues pass the CVD, chroma and lightness checks in fixed order,
and the first three additionally clear the all-pairs floors — which covers every
bundled dataset, none of which has more than three classes.

Colours are assigned to a class **by its position in the sorted class list**, so
they never move when a filter changes which classes are on screen.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

# Categorical — fixed order, never cycled through generated hues.
CATEGORICAL: List[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

OTHER = "#898781"  # a 9th class folds into "Other" rather than inventing a hue

# Sequential (single hue, light -> dark) for magnitude: heatmaps, regression leaves.
SEQUENTIAL: List[str] = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

# Diverging: warm/cool poles with a neutral (not a hue) in the middle.
DIVERGING_LOW = "#2a78d6"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#e34948"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
AXIS = "#c3c2b7"
GRID = "#e8e7e2"

# Semantic accents used by the tree diagram (kept out of the categorical slots).
HIGHLIGHT = "#e34948"
POSITIVE = "#1baf7a"

FONT_STACK = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]

MAX_CATEGORICAL = len(CATEGORICAL)


def class_colors(classes: Sequence) -> Dict[str, str]:
    """Map class labels to hues in fixed order; anything past slot 8 is 'Other'."""
    colors: Dict[str, str] = {}
    for index, label in enumerate(classes):
        colors[str(label)] = CATEGORICAL[index] if index < MAX_CATEGORICAL else OTHER
    return colors


def sequential_color(fraction: float) -> str:
    """Pick a step from the blue ramp for a value already scaled to 0..1."""
    if fraction != fraction:  # NaN
        return SEQUENTIAL[0]
    fraction = max(0.0, min(1.0, float(fraction)))
    index = int(round(fraction * (len(SEQUENTIAL) - 1)))
    return SEQUENTIAL[index]


def blend_to_surface(hex_color: str, strength: float) -> str:
    """Mix a hue toward the chart surface; ``strength`` 1.0 keeps it fully saturated."""
    strength = max(0.0, min(1.0, float(strength)))
    src = [int(hex_color[i: i + 2], 16) for i in (1, 3, 5)]
    dst = [int(SURFACE[i: i + 2], 16) for i in (1, 3, 5)]
    mixed = [int(round(d + (s - d) * strength)) for s, d in zip(src, dst)]
    return "#%02X%02X%02X" % tuple(mixed)


def apply_matplotlib_style() -> None:
    """Recessive chrome, no top/right spines, text in ink tokens rather than hues."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 10,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelcolor": TEXT_SECONDARY,
            "ytick.labelcolor": TEXT_SECONDARY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "figure.autolayout": False,
        }
    )
