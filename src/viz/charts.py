"""Matplotlib figures for the app.

Each function picks the form from the data's job — magnitude comparisons become
sorted horizontal bars with direct labels, a scan over a continuous parameter
becomes a line, a matrix becomes a sequential heatmap — and returns a Figure so
the UI layer stays free of plotting code.

No dual axes anywhere: when two measures have different scales they get stacked
panels that share the x axis instead.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..core.splitter import SplitCandidate
from ..core.trace import BuildStep
from . import theme

theme.apply_matplotlib_style()

BAR_HEIGHT = 0.62


def _new_figure(width: float, height: float) -> Tuple[plt.Figure, plt.Axes]:
    figure, axes = plt.subplots(figsize=(width, height))
    return figure, axes


def _label_bars(axes: plt.Axes, bars, values: Sequence[float], fmt: str = "{:.4f}",
                pad: float = 0.01) -> None:
    """Direct labels on every bar — the relief the contrast WARN requires."""
    span = max([abs(v) for v in values] + [1e-9])
    for bar, value in zip(bars, values):
        axes.text(
            bar.get_width() + span * pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center", ha="left", fontsize=9, color=theme.TEXT_SECONDARY,
        )


def _corner_note(axes: plt.Axes, text: str, marker_x_fraction: float, color: str) -> None:
    """Put a callout in the corner furthest from the marked point.

    Offset annotations collide with the curve whenever the optimum sits near an
    edge, which is exactly where optima like to sit.
    """
    on_left = marker_x_fraction < 0.5
    axes.text(
        0.97 if on_left else 0.03,
        0.95,
        text,
        transform=axes.transAxes,
        ha="right" if on_left else "left",
        va="top",
        fontsize=9,
        color=color,
        linespacing=1.35,
    )


def _tidy(axes: plt.Axes, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    if title:
        axes.set_title(title, loc="left", pad=10)
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)
    axes.grid(axis="x", alpha=0.9)
    axes.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Split selection
# --------------------------------------------------------------------------- #
def plot_candidate_gains(step: BuildStep, score_name: str = "Information Gain") -> plt.Figure:
    """Rank every candidate split. The winner is the only saturated bar."""
    candidates = step.ranked_candidates
    if not candidates:
        figure, axes = _new_figure(6, 2)
        axes.text(0.5, 0.5, "No candidate splits at this node", ha="center", va="center",
                  color=theme.TEXT_MUTED)
        axes.axis("off")
        return figure

    labels = [c.feature_name for c in candidates]
    values = [c.score for c in candidates]
    winner = step.chosen.feature_name if step.chosen else None

    height = max(2.0, 0.45 * len(candidates) + 1.2)
    figure, axes = _new_figure(6.4, height)
    positions = np.arange(len(candidates))[::-1]

    colors, hatches = [], []
    for candidate in candidates:
        if not candidate.is_valid:
            colors.append(theme.blend_to_surface(theme.TEXT_MUTED, 0.35))
            hatches.append("//")
        elif candidate.feature_name == winner:
            colors.append(theme.CATEGORICAL[0])
            hatches.append("")
        else:
            colors.append(theme.blend_to_surface(theme.CATEGORICAL[0], 0.32))
            hatches.append("")

    bars = axes.barh(positions, values, height=BAR_HEIGHT, color=colors,
                     edgecolor=theme.SURFACE, linewidth=1.5)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    axes.set_yticks(positions)
    axes.set_yticklabels(labels)
    _label_bars(axes, bars, values)
    axes.set_xlim(0, max(max(values), 1e-6) * 1.25)
    _tidy(axes, xlabel=score_name, title=f"{score_name} of every candidate split")

    unusable = [c for c in candidates if not c.is_valid]
    if unusable:
        axes.text(0.0, -0.16, "Hatched bars are unusable splits (rejected by the size rules).",
                  transform=axes.transAxes, fontsize=8.5, color=theme.TEXT_MUTED)
    figure.tight_layout()
    return figure


def plot_impurity_breakdown(step: BuildStep, impurity_display: str = "Entropy") -> plt.Figure:
    """Impurity before the split versus the weighted impurity after it."""
    if step.chosen is None:
        figure, axes = _new_figure(6, 1.6)
        axes.text(0.5, 0.5, "This node was not split", ha="center", va="center",
                  color=theme.TEXT_MUTED)
        axes.axis("off")
        return figure

    before = step.impurity_value
    after = step.chosen.weighted_impurity
    figure, axes = _new_figure(6.4, 2.4)
    bars = axes.barh(
        [1, 0], [before, after], height=BAR_HEIGHT,
        color=[theme.blend_to_surface(theme.CATEGORICAL[0], 0.32), theme.CATEGORICAL[0]],
        edgecolor=theme.SURFACE, linewidth=1.5,
    )
    axes.set_yticks([1, 0])
    axes.set_yticklabels([f"Parent {impurity_display}", "Weighted children"])
    _label_bars(axes, bars, [before, after], fmt="{:.4f}")

    axes.annotate(
        f"gain = {step.chosen.gain:.4f}",
        xy=(after, 0.5), xytext=(before, 0.5), textcoords="data",
        arrowprops=dict(arrowstyle="<->", color=theme.HIGHLIGHT, lw=1.6),
        color=theme.HIGHLIGHT, fontsize=9.5, ha="center", va="bottom",
    )
    axes.set_xlim(0, max(before, after, 1e-6) * 1.3)
    _tidy(axes, xlabel=impurity_display,
          title=f"The split on {step.chosen.feature_name} removes this much impurity")
    figure.tight_layout()
    return figure


def plot_threshold_scan(candidate: SplitCandidate, score_name: str = "Gain") -> plt.Figure:
    """Score as a function of the cut point, with the chosen threshold marked."""
    scan = [(t, s) for t, s in candidate.threshold_scan if s == s]
    figure, axes = _new_figure(6.4, 2.8)
    if not scan:
        axes.text(0.5, 0.5, "No candidate thresholds", ha="center", va="center",
                  color=theme.TEXT_MUTED)
        axes.axis("off")
        return figure

    thresholds = [t for t, _s in scan]
    scores = [s for _t, s in scan]
    axes.plot(thresholds, scores, color=theme.CATEGORICAL[0], marker="o",
              markersize=4, markerfacecolor=theme.SURFACE, markeredgewidth=1.4)
    if candidate.threshold is not None:
        axes.axvline(candidate.threshold, color=theme.HIGHLIGHT, lw=1.6, ls="--")
        axes.plot([candidate.threshold], [candidate.score], marker="o", markersize=9,
                  color=theme.HIGHLIGHT, markeredgecolor=theme.SURFACE, markeredgewidth=2, zorder=5)
        span = max(thresholds) - min(thresholds)
        position = (candidate.threshold - min(thresholds)) / span if span > 0 else 0.5
        _corner_note(
            axes,
            f"best t = {candidate.threshold:.4g}\n{score_name} = {candidate.score:.4f}",
            position, theme.HIGHLIGHT,
        )
    axes.grid(axis="y", alpha=0.9)
    axes.set_axisbelow(True)
    _tidy(axes, xlabel=f"threshold on {candidate.feature_name}", ylabel=score_name,
          title=f"Every candidate threshold for {candidate.feature_name}")
    axes.grid(axis="x", alpha=0.0)
    figure.tight_layout()
    return figure


def plot_class_distribution(class_counts: Dict[Any, float], classes: Optional[Sequence] = None,
                            title: str = "Class distribution at this node") -> plt.Figure:
    """A bar, not a donut: bars are what let you compare two counts accurately."""
    labels = [str(c) for c in (classes or class_counts.keys())]
    values = [float(class_counts.get(label, class_counts.get(_coerce(label), 0.0))) for label in labels]
    colors = theme.class_colors(labels)

    figure, axes = _new_figure(4.6, max(1.6, 0.42 * len(labels) + 1.0))
    positions = np.arange(len(labels))[::-1]
    bars = axes.barh(positions, values, height=BAR_HEIGHT,
                     color=[colors[label] for label in labels],
                     edgecolor=theme.SURFACE, linewidth=1.5)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels)
    _label_bars(axes, bars, values, fmt="{:g}")
    axes.set_xlim(0, max(values + [1]) * 1.2)
    _tidy(axes, xlabel="samples", title=title)
    figure.tight_layout()
    return figure


def _coerce(label: str) -> Any:
    try:
        return float(label)
    except (TypeError, ValueError):
        return label


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(matrix: np.ndarray, labels: Sequence[str],
                          title: str = "Confusion matrix") -> plt.Figure:
    """Sequential single-hue heatmap; every cell carries its count as text."""
    matrix = np.asarray(matrix)
    size = max(3.4, 0.7 * len(labels) + 2.2)
    figure, axes = _new_figure(size, size * 0.92)
    peak = matrix.max() if matrix.size and matrix.max() > 0 else 1

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            axes.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                              facecolor=theme.sequential_color(value / peak),
                              edgecolor=theme.SURFACE, linewidth=2)
            )
            axes.text(j, i, f"{value:g}", ha="center", va="center", fontsize=10,
                      color=theme.SURFACE if value / peak > 0.55 else theme.TEXT_PRIMARY,
                      fontweight="bold" if i == j else "normal")

    axes.set_xlim(-0.5, matrix.shape[1] - 0.5)
    axes.set_ylim(matrix.shape[0] - 0.5, -0.5)
    axes.set_xticks(range(len(labels)))
    axes.set_yticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=30, ha="right")
    axes.set_yticklabels(labels)
    axes.set_xlabel("predicted")
    axes.set_ylabel("actual")
    axes.set_title(title, loc="left", pad=10)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(length=0)
    figure.tight_layout()
    return figure


def plot_learning_curve_depth(rows: Sequence[Dict[str, float]],
                              metric: str = "Accuracy") -> plt.Figure:
    """Train versus test as the tree is allowed to grow — where overfitting shows."""
    depths = [r["depth"] for r in rows]
    figure, axes = _new_figure(6.4, 3.2)
    series = [("train", theme.CATEGORICAL[0]), ("test", theme.CATEGORICAL[1])]
    for name, color in series:
        values = [r[name] for r in rows]
        axes.plot(depths, values, color=color, marker="o", markersize=5,
                  markerfacecolor=theme.SURFACE, markeredgewidth=1.6, label=name)
        axes.annotate(name, xy=(depths[-1], values[-1]), xytext=(6, 0),
                      textcoords="offset points", color=color, fontsize=9.5, va="center")

    best = max(rows, key=lambda r: r["test"])
    axes.axvline(best["depth"], color=theme.TEXT_MUTED, lw=1, ls=":")
    axes.annotate(f"best test {metric.lower()}\nat depth {best['depth']:g}",
                  xy=(best["depth"], best["test"]), xytext=(0, -14),
                  textcoords="offset points", fontsize=9, color=theme.TEXT_SECONDARY,
                  ha="center", va="top")

    axes.legend(loc="upper left")
    axes.grid(axis="y", alpha=0.9)
    axes.set_axisbelow(True)
    axes.set_xlabel("max_depth")
    axes.set_ylabel(metric)
    axes.set_title(f"{metric} as the tree is allowed to grow deeper", loc="left", pad=10)
    axes.set_xlim(min(depths) - 0.4, max(depths) + (max(depths) - min(depths)) * 0.28 + 0.6)
    figure.tight_layout()
    return figure


def plot_alpha_curve(rows: Sequence[Dict[str, float]], metric: str = "accuracy") -> plt.Figure:
    """Two measures, two stacked panels sharing x — never a second y axis."""
    alphas = [r["alpha"] for r in rows]
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 4.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    top.plot(alphas, [r["score"] for r in rows], color=theme.CATEGORICAL[0],
             marker="o", markersize=5, markerfacecolor=theme.SURFACE, markeredgewidth=1.6)
    best = max(rows, key=lambda r: r["score"])
    top.plot([best["alpha"]], [best["score"]], marker="o", markersize=10,
             color=theme.HIGHLIGHT, markeredgecolor=theme.SURFACE, markeredgewidth=2, zorder=5)
    span = max(alphas) - min(alphas)
    position = (best["alpha"] - min(alphas)) / span if span > 0 else 0.5
    _corner_note(top, f"best: alpha = {best['alpha']:.4g}\n{best['leaves']:g} leaves",
                 position, theme.HIGHLIGHT)
    top.set_ylabel(f"validation {metric}")
    top.set_title("Cost-complexity pruning: what each alpha buys", loc="left", pad=10)
    top.grid(axis="y", alpha=0.9)
    top.set_axisbelow(True)

    bottom.plot(alphas, [r["leaves"] for r in rows], color=theme.CATEGORICAL[1],
                marker="o", markersize=5, markerfacecolor=theme.SURFACE, markeredgewidth=1.6)
    bottom.set_ylabel("leaves")
    bottom.set_xlabel("alpha")
    bottom.grid(axis="y", alpha=0.9)
    bottom.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_feature_importance(importances: Dict[str, float]) -> plt.Figure:
    """Weighted impurity decrease per feature, normalised — sorted bars."""
    items = [(k, v) for k, v in importances.items() if v > 0]
    if not items:
        figure, axes = _new_figure(6, 1.6)
        axes.text(0.5, 0.5, "The tree is a single leaf — no feature was used.",
                  ha="center", va="center", color=theme.TEXT_MUTED)
        axes.axis("off")
        return figure

    items.sort(key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _v in items]
    values = [v for _k, v in items]
    figure, axes = _new_figure(6.4, max(2.0, 0.42 * len(items) + 1.2))
    positions = np.arange(len(items))[::-1]
    bars = axes.barh(positions, values, height=BAR_HEIGHT, color=theme.CATEGORICAL[0],
                     edgecolor=theme.SURFACE, linewidth=1.5)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels)
    _label_bars(axes, bars, values, fmt="{:.1%}")
    axes.set_xlim(0, max(values) * 1.25)
    _tidy(axes, xlabel="share of total impurity decrease", title="Which features the tree relies on")
    figure.tight_layout()
    return figure


def plot_entropy_curve(mark: Optional[float] = None) -> plt.Figure:
    """H(p) for a two-class node — the intuition behind 'impurity'."""
    p = np.linspace(0.001, 0.999, 400)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    figure, axes = _new_figure(5.6, 3.0)
    axes.plot(p, h, color=theme.CATEGORICAL[0])
    axes.fill_between(p, 0, h, color=theme.blend_to_surface(theme.CATEGORICAL[0], 0.12))
    if mark is not None and 0 < mark < 1:
        value = -(mark * np.log2(mark) + (1 - mark) * np.log2(1 - mark))
        axes.plot([mark], [value], marker="o", markersize=9, color=theme.HIGHLIGHT,
                  markeredgecolor=theme.SURFACE, markeredgewidth=2, zorder=5)
        axes.annotate(f"p = {mark:.2f}\nH = {value:.3f}", xy=(mark, value), xytext=(8, -6),
                      textcoords="offset points", fontsize=9, color=theme.HIGHLIGHT, va="top")
    axes.set_xlabel("proportion of the positive class, p")
    axes.set_ylabel("entropy H(p)")
    axes.set_title("Entropy is highest at a 50/50 split and zero when a node is pure",
                   loc="left", pad=10)
    axes.grid(axis="y", alpha=0.9)
    axes.set_axisbelow(True)
    axes.set_ylim(0, 1.08)
    figure.tight_layout()
    return figure


def plot_impurity_comparison() -> plt.Figure:
    """Entropy, Gini and error side by side — same shape, different steepness."""
    p = np.linspace(0.001, 0.999, 400)
    curves = {
        "Entropy": -(p * np.log2(p) + (1 - p) * np.log2(1 - p)),
        "Gini": 1 - (p ** 2 + (1 - p) ** 2),
        "Misclassification": np.minimum(p, 1 - p),
    }
    figure, axes = _new_figure(6.0, 3.2)
    label_at = {"Entropy": 0.30, "Gini": 0.70, "Misclassification": 0.82}
    for index, (name, values) in enumerate(curves.items()):
        color = theme.CATEGORICAL[index]
        axes.plot(p, values, color=color, label=name)
        anchor = int(np.argmin(np.abs(p - label_at[name])))
        axes.annotate(name, xy=(p[anchor], values[anchor]), xytext=(0, 8),
                      textcoords="offset points", ha="center", fontsize=9.5, color=color)
    axes.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    axes.set_xlabel("proportion of the positive class, p")
    axes.set_ylabel("impurity")
    axes.set_ylim(0, 1.22)
    axes.set_title("All three impurities peak at a 50/50 node", loc="left", pad=10)
    axes.grid(axis="y", alpha=0.9)
    axes.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_cv_folds(rows: Sequence[Dict[str, Any]], metric: str = "Accuracy") -> plt.Figure:
    """Per-fold scores with the mean drawn as a reference line."""
    figure, axes = _new_figure(6.0, 2.8)
    labels = [str(r["Fold"]) for r in rows]
    values = [float(r[metric]) for r in rows]
    positions = np.arange(len(rows))
    bars = axes.bar(positions, values, width=0.6, color=theme.CATEGORICAL[0],
                    edgecolor=theme.SURFACE, linewidth=1.5)
    mean = float(np.mean(values)) if values else 0.0
    axes.axhline(mean, color=theme.HIGHLIGHT, lw=1.4, ls="--")
    axes.annotate(f"mean = {mean:.3f}", xy=(positions[-1], mean), xytext=(4, 4),
                  textcoords="offset points", color=theme.HIGHLIGHT, fontsize=9)
    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                  ha="center", va="bottom", fontsize=9, color=theme.TEXT_SECONDARY)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_xlabel("fold")
    axes.set_ylabel(metric)
    axes.set_ylim(0, max(values + [1.0]) * 1.15)
    axes.set_title(f"{metric} on each cross-validation fold", loc="left", pad=10)
    axes.grid(axis="y", alpha=0.9)
    axes.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_algorithm_comparison(rows: Sequence[Dict[str, Any]], metric: str = "Test accuracy"
                              ) -> plt.Figure:
    """One bar per algorithm — identity by label, colour only as reinforcement."""
    labels = [str(r["Algorithm"]) for r in rows]
    values = [float(r[metric]) for r in rows]
    colors = [theme.CATEGORICAL[i % theme.MAX_CATEGORICAL] for i in range(len(labels))]
    figure, axes = _new_figure(6.0, max(2.0, 0.5 * len(labels) + 1.2))
    positions = np.arange(len(labels))[::-1]
    bars = axes.barh(positions, values, height=BAR_HEIGHT, color=colors,
                     edgecolor=theme.SURFACE, linewidth=1.5)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels)
    _label_bars(axes, bars, values, fmt="{:.3f}")
    axes.set_xlim(0, max(values + [1.0]) * 1.2)
    _tidy(axes, xlabel=metric, title=f"{metric} by algorithm")
    figure.tight_layout()
    return figure
