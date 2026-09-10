"""Decision regions in two dimensions.

The regions come from the tree's own ``predict`` on a grid, so what you see is
literally what the model does — no re-derivation of the rules.  Features other
than the two on the axes are held at a representative value (median for numbers,
mode for categories), which the caption states explicitly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.node import Tree
from . import theme

theme.apply_matplotlib_style()


def representative_values(frame: pd.DataFrame, feature_types: Dict[str, str],
                          exclude: Sequence[str] = ()) -> Dict[str, Any]:
    """Median for numeric columns, most common value for categorical ones."""
    held: Dict[str, Any] = {}
    for column in frame.columns:
        if column in exclude:
            continue
        series = frame[column].dropna()
        if series.empty:
            held[column] = None
        elif feature_types.get(column) == "numeric" and pd.api.types.is_numeric_dtype(series):
            held[column] = float(series.median())
        else:
            modes = series.mode()
            held[column] = modes.iloc[0] if len(modes) else None
    return held


def plot_decision_boundary_2d(
    tree: Tree,
    frame: pd.DataFrame,
    y: Sequence,
    feature_x: str,
    feature_y: str,
    feature_types: Dict[str, str],
    resolution: int = 220,
    max_points: int = 500,
    title: Optional[str] = None,
) -> plt.Figure:
    """Grid-predict over two numeric features and scatter the real data on top."""
    x_values = pd.to_numeric(frame[feature_x], errors="coerce")
    y_values = pd.to_numeric(frame[feature_y], errors="coerce")
    valid = x_values.notna() & y_values.notna()
    x_values, y_values = x_values[valid], y_values[valid]
    labels = np.asarray(list(y), dtype=object)[valid.to_numpy()]

    figure, axes = plt.subplots(figsize=(6.2, 5.0))
    if x_values.empty:
        axes.text(0.5, 0.5, "Pick two numeric features", ha="center", va="center",
                  color=theme.TEXT_MUTED)
        axes.axis("off")
        return figure

    pad_x = (x_values.max() - x_values.min()) * 0.06 or 0.5
    pad_y = (y_values.max() - y_values.min()) * 0.06 or 0.5
    grid_x = np.linspace(x_values.min() - pad_x, x_values.max() + pad_x, resolution)
    grid_y = np.linspace(y_values.min() - pad_y, y_values.max() + pad_y, resolution)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

    held = representative_values(frame, feature_types, exclude=(feature_x, feature_y))
    rows: List[Dict[str, Any]] = []
    for gx, gy in zip(mesh_x.ravel(), mesh_y.ravel()):
        row = dict(held)
        row[feature_x] = float(gx)
        row[feature_y] = float(gy)
        rows.append(row)

    predictions = np.array([tree.predict_one(row) for row in rows], dtype=object)
    classes = [str(c) for c in (tree.class_names or sorted({str(p) for p in predictions}))]
    colors = theme.class_colors(classes)
    index_of = {name: i for i, name in enumerate(classes)}
    coded = np.array([index_of.get(str(p), -1) for p in predictions]).reshape(mesh_x.shape)

    cmap = matplotlib.colors.ListedColormap(
        [theme.blend_to_surface(colors[name], 0.22) for name in classes]
    )
    axes.imshow(
        coded, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
        vmin=0, vmax=max(len(classes) - 1, 1),
        extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]),
    )

    order = np.arange(len(x_values))
    if len(order) > max_points:
        order = np.random.default_rng(0).choice(order, size=max_points, replace=False)
    for name in classes:
        mask = np.array([str(v) == name for v in labels])[order]
        if not mask.any():
            continue
        axes.scatter(
            x_values.to_numpy()[order][mask], y_values.to_numpy()[order][mask],
            s=34, color=colors[name], edgecolor=theme.SURFACE, linewidth=1.4,
            label=name, zorder=3,
        )

    axes.set_xlim(grid_x[0], grid_x[-1])
    axes.set_ylim(grid_y[0], grid_y[-1])
    axes.set_xlabel(feature_x)
    axes.set_ylabel(feature_y)
    axes.set_title(title or f"Decision regions of the {tree.algorithm} tree", loc="left", pad=10)
    axes.legend(loc="best", title=None)

    others = [f"{k} = {v}" for k, v in held.items() if v is not None]
    if others:
        axes.text(0.0, -0.16, "Other features held at: " + ", ".join(others[:4])
                  + ("…" if len(others) > 4 else ""),
                  transform=axes.transAxes, fontsize=8.5, color=theme.TEXT_MUTED)
    figure.tight_layout()
    return figure
