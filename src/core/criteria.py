"""Impurity measures and split-quality scores.

Every measure comes in two flavours:

* a plain ``float`` version used inside hot loops, and
* a ``*_c`` version returning a :class:`Computation` that carries the symbolic
  formula, the same formula with the node's real numbers substituted in, and the
  final result.

The ``Computation`` objects are what let the UI show *"formula -> numbers ->
result"* for every quantity it displays, including the split candidates that
lost.  Nothing here imports streamlit or pandas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import latex as tex

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Computation container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Term:
    """One addend of a formula, kept separately so the UI can highlight it."""

    latex: str
    value: float
    label: str = ""


@dataclass
class Computation:
    """A single evaluated formula, ready to be shown as three lines."""

    name: str
    symbolic: str
    substituted: str
    result: float
    terms: List[Term] = field(default_factory=list)
    operands: Dict[str, Any] = field(default_factory=dict)
    note: Optional[str] = None
    children: Dict[str, "Computation"] = field(default_factory=dict)

    @property
    def lhs(self) -> str:
        """Left-hand side of the symbolic form, e.g. ``H(S)``."""
        return self.symbolic.split("=", 1)[0].strip() if "=" in self.symbolic else self.name

    def full_latex(self, precision: int = 3) -> str:
        """The whole chain on one line: ``H(S) = -9/14 log ... = 0.940``."""
        return f"{self.symbolic} = {self.substituted} = {tex.num(self.result, precision)}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbolic": self.symbolic,
            "substituted": self.substituted,
            "result": self.result,
            "note": self.note,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


# --------------------------------------------------------------------------- #
# Basic counting
# --------------------------------------------------------------------------- #
def class_counts(y: Sequence, weights: Optional[np.ndarray] = None) -> Dict[Any, float]:
    """Weighted count per class, ordered by first appearance for stable display."""
    y = np.asarray(y, dtype=object)
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    counts: Dict[Any, float] = {}
    for label, w in zip(y, np.asarray(weights, dtype=float)):
        counts[label] = counts.get(label, 0.0) + float(w)
    return counts


def total_weight(y: Sequence, weights: Optional[np.ndarray] = None) -> float:
    if weights is None:
        return float(len(y))
    return float(np.sum(weights))


def class_probabilities(y: Sequence, weights: Optional[np.ndarray] = None) -> Dict[Any, float]:
    counts = class_counts(y, weights)
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def majority_class(y: Sequence, weights: Optional[np.ndarray] = None) -> Any:
    counts = class_counts(y, weights)
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], str(kv[0])))[0]


def is_pure(y: Sequence) -> bool:
    y = np.asarray(y, dtype=object)
    return len(y) > 0 and len(set(y.tolist())) == 1


# --------------------------------------------------------------------------- #
# Classification impurities
# --------------------------------------------------------------------------- #
def entropy(y: Sequence, weights: Optional[np.ndarray] = None) -> float:
    counts = class_counts(y, weights)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    acc = 0.0
    for c in counts.values():
        p = c / total
        if p > EPS:
            acc -= p * np.log2(p)
    return float(max(acc, 0.0))


def entropy_c(y: Sequence, weights: Optional[np.ndarray] = None, symbol: str = "S") -> Computation:
    counts = class_counts(y, weights)
    total = sum(counts.values())
    terms: List[Term] = []
    pieces: List[str] = []
    for label, c in counts.items():
        p = (c / total) if total > 0 else 0.0
        f = tex.frac(c, total)
        if p <= EPS:
            pieces.append("0")
            terms.append(Term(latex="0", value=0.0, label=str(label)))
            continue
        pieces.append(r"-%s\log_2 %s" % (f, f))
        terms.append(Term(latex=r"-%s\log_2 %s" % (f, f), value=-p * float(np.log2(p)), label=str(label)))
    value = entropy(y, weights)
    return Computation(
        name="entropy",
        symbolic=r"H(%s) = -\sum_{i} p_i \log_2 p_i" % symbol,
        substituted=tex.joined(pieces) if pieces else "0",
        result=value,
        terms=terms,
        operands={"counts": counts, "total": total},
        note="Node is pure, so its entropy is 0." if len([c for c in counts.values() if c > 0]) <= 1 else None,
    )


def gini(y: Sequence, weights: Optional[np.ndarray] = None) -> float:
    counts = class_counts(y, weights)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(max(1.0 - sum((c / total) ** 2 for c in counts.values()), 0.0))


def gini_c(y: Sequence, weights: Optional[np.ndarray] = None, symbol: str = "S") -> Computation:
    counts = class_counts(y, weights)
    total = sum(counts.values())
    terms: List[Term] = []
    pieces: List[str] = []
    for label, c in counts.items():
        p = (c / total) if total > 0 else 0.0
        pieces.append(r"\left(%s\right)^2" % tex.frac(c, total))
        terms.append(Term(latex=pieces[-1], value=p ** 2, label=str(label)))
    body = " + ".join(pieces) if pieces else "0"
    return Computation(
        name="gini",
        symbolic=r"Gini(%s) = 1 - \sum_{i} p_i^{2}" % symbol,
        substituted=r"1 - \left[%s\right]" % body,
        result=gini(y, weights),
        terms=terms,
        operands={"counts": counts, "total": total},
        note="Node is pure, so its Gini index is 0." if len([c for c in counts.values() if c > 0]) <= 1 else None,
    )


def misclassification_error(y: Sequence, weights: Optional[np.ndarray] = None) -> float:
    probs = class_probabilities(y, weights)
    if not probs:
        return 0.0
    return float(max(1.0 - max(probs.values()), 0.0))


def misclassification_error_c(y: Sequence, weights: Optional[np.ndarray] = None, symbol: str = "S") -> Computation:
    counts = class_counts(y, weights)
    total = sum(counts.values())
    best_label, best_count = (None, 0.0)
    if counts:
        best_label, best_count = max(counts.items(), key=lambda kv: kv[1])
    return Computation(
        name="error",
        symbolic=r"Err(%s) = 1 - \max_i p_i" % symbol,
        substituted=r"1 - %s" % tex.frac(best_count, total),
        result=misclassification_error(y, weights),
        terms=[Term(latex=tex.frac(best_count, total), value=(best_count / total) if total else 0.0,
                    label=str(best_label))],
        operands={"counts": counts, "total": total, "majority": best_label},
    )


# --------------------------------------------------------------------------- #
# Regression impurities
# --------------------------------------------------------------------------- #
def _as_float_array(y: Sequence) -> np.ndarray:
    return np.asarray(list(y), dtype=float)


def mse(y: Sequence, weights: Optional[np.ndarray] = None) -> float:
    values = _as_float_array(y)
    if values.size == 0:
        return 0.0
    w = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0:
        return 0.0
    mean = float(np.sum(values * w) / total)
    return float(np.sum(w * (values - mean) ** 2) / total)


def mse_c(y: Sequence, weights: Optional[np.ndarray] = None, symbol: str = "S") -> Computation:
    values = _as_float_array(y)
    w = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    total = float(np.sum(w)) if values.size else 0.0
    mean = float(np.sum(values * w) / total) if total > 0 else 0.0
    return Computation(
        name="mse",
        symbolic=r"MSE(%s) = \frac{1}{|%s|}\sum_{i}\left(y_i - \bar{y}\right)^{2}"
        % (symbol, symbol),
        substituted=r"\frac{1}{%s}\sum_{i}\left(y_i - %s\right)^{2}" % (tex.num(total), tex.num(mean)),
        result=mse(y, weights),
        terms=[Term(latex=r"\bar{y} = %s" % tex.num(mean), value=mean, label="mean")],
        operands={"n": total, "mean": mean},
        note="The mean of the %s target values in this node is %s." % (tex.num(total), tex.num(mean)),
    )


def mae(y: Sequence, weights: Optional[np.ndarray] = None) -> float:
    values = _as_float_array(y)
    if values.size == 0:
        return 0.0
    w = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0:
        return 0.0
    median = float(np.median(values))
    return float(np.sum(w * np.abs(values - median)) / total)


def mae_c(y: Sequence, weights: Optional[np.ndarray] = None, symbol: str = "S") -> Computation:
    values = _as_float_array(y)
    median = float(np.median(values)) if values.size else 0.0
    n = float(len(values)) if weights is None else float(np.sum(weights))
    return Computation(
        name="mae",
        symbolic=r"MAE(%s) = \frac{1}{|%s|}\sum_{i}\left|y_i - \tilde{y}\right|"
        % (symbol, symbol),
        substituted=r"\frac{1}{%s}\sum_{i}\left|y_i - %s\right|" % (tex.num(n), tex.num(median)),
        result=mae(y, weights),
        terms=[Term(latex=r"\tilde{y} = %s" % tex.num(median), value=median, label="median")],
        operands={"n": n, "median": median},
    )


# --------------------------------------------------------------------------- #
# Impurity registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Impurity:
    """Bundles the scalar and explained versions of one impurity measure."""

    key: str
    display: str
    symbol: str
    task: str
    value_fn: Callable[..., float]
    compute_fn: Callable[..., Computation]
    gain_name: str
    gain_symbol: str

    def value(self, y, weights=None) -> float:
        return self.value_fn(y, weights)

    def explain(self, y, weights=None, symbol: str = "S") -> Computation:
        return self.compute_fn(y, weights, symbol)


IMPURITY_REGISTRY: Dict[str, Impurity] = {
    "entropy": Impurity("entropy", "Entropy", "H", "classification", entropy, entropy_c,
                        "Information Gain", "IG"),
    "gini": Impurity("gini", "Gini index", "Gini", "classification", gini, gini_c,
                     "Gini Gain", "\\Delta Gini"),
    "error": Impurity("error", "Misclassification error", "Err", "classification",
                      misclassification_error, misclassification_error_c, "Error Reduction", "\\Delta Err"),
    "mse": Impurity("mse", "Mean squared error", "MSE", "regression", mse, mse_c,
                    "Variance Reduction", "\\Delta MSE"),
    "mae": Impurity("mae", "Mean absolute error", "MAE", "regression", mae, mae_c,
                    "MAE Reduction", "\\Delta MAE"),
}


def get_impurity(key: str) -> Impurity:
    try:
        return IMPURITY_REGISTRY[key]
    except KeyError as exc:  # pragma: no cover - guarded by the UI
        raise KeyError(f"Unknown impurity '{key}'. Available: {sorted(IMPURITY_REGISTRY)}") from exc


# --------------------------------------------------------------------------- #
# Split-quality scores
# --------------------------------------------------------------------------- #
def weighted_impurity_c(
    children: Sequence[tuple],
    impurity: Impurity,
    parent_symbol: str = "S",
) -> Computation:
    """``sum_v (|S_v|/|S|) * I(S_v)`` with every child's own formula attached.

    ``children`` is a sequence of ``(branch_label, y_child, weights_child)``.
    """
    child_comps: Dict[str, Computation] = {}
    sizes: List[float] = []
    pieces: List[str] = []
    terms: List[Term] = []

    for branch_label, y_child, w_child in children:
        size = total_weight(y_child, w_child)
        sizes.append(size)
        child_comps[str(branch_label)] = impurity.explain(y_child, w_child, symbol=f"S_{{{branch_label}}}")

    total = float(sum(sizes))
    acc = 0.0
    for (branch_label, _y, _w), size in zip(children, sizes):
        comp = child_comps[str(branch_label)]
        share = (size / total) if total > 0 else 0.0
        acc += share * comp.result
        pieces.append(r"%s\left(%s\right)" % (tex.frac(size, total), tex.num(comp.result)))
        terms.append(Term(latex=pieces[-1], value=share * comp.result, label=str(branch_label)))

    return Computation(
        name="weighted_impurity",
        symbolic=r"\sum_{v} \frac{|%s_v|}{|%s|}\,%s(%s_v)"
        % (parent_symbol, parent_symbol, impurity.symbol, parent_symbol),
        substituted=" + ".join(pieces) if pieces else "0",
        result=float(acc),
        terms=terms,
        operands={"sizes": sizes, "total": total},
        children=child_comps,
    )


def impurity_gain_c(
    parent_comp: Computation,
    weighted_comp: Computation,
    feature_label: str,
    impurity: Impurity,
    parent_symbol: str = "S",
) -> Computation:
    """Impurity of the parent minus the weighted impurity of the children."""
    result = float(parent_comp.result - weighted_comp.result)
    sym = parent_symbol
    return Computation(
        name="gain",
        symbolic=r"%s(%s, %s) = %s(%s) - \sum_{v} \frac{|%s_v|}{|%s|}\,%s(%s_v)"
        % (impurity.gain_symbol, sym, tex.text(feature_label), impurity.symbol, sym,
           sym, sym, impurity.symbol, sym),
        substituted=r"%s - \left[%s\right]" % (tex.num(parent_comp.result), weighted_comp.substituted),
        result=result,
        terms=weighted_comp.terms,
        operands={"parent": parent_comp.result, "weighted": weighted_comp.result, "feature": feature_label},
        children={"parent_impurity": parent_comp, "weighted_impurity": weighted_comp},
    )


def split_info_c(children: Sequence[tuple], feature_label: str, parent_symbol: str = "S") -> Computation:
    """C4.5 intrinsic value: the entropy of the *split shape* itself.

    Splits that shatter the data into many tiny branches get a large SplitInfo,
    which is exactly what stops Gain Ratio from falling for ID-like columns.
    """
    sizes = [total_weight(y_child, w_child) for _label, y_child, w_child in children]
    total = float(sum(sizes))
    pieces: List[str] = []
    terms: List[Term] = []
    acc = 0.0
    for (branch_label, _y, _w), size in zip(children, sizes):
        share = (size / total) if total > 0 else 0.0
        f = tex.frac(size, total)
        if share <= EPS:
            pieces.append("0")
            terms.append(Term(latex="0", value=0.0, label=str(branch_label)))
            continue
        acc -= share * float(np.log2(share))
        pieces.append(r"-%s\log_2 %s" % (f, f))
        terms.append(Term(latex=pieces[-1], value=-share * float(np.log2(share)), label=str(branch_label)))

    sym = parent_symbol
    return Computation(
        name="split_info",
        symbolic=r"SplitInfo(%s, %s) = -\sum_{v} \frac{|%s_v|}{|%s|}\log_2 \frac{|%s_v|}{|%s|}"
        % (sym, tex.text(feature_label), sym, sym, sym, sym),
        substituted=tex.joined(pieces) if pieces else "0",
        result=float(max(acc, 0.0)),
        terms=terms,
        operands={"sizes": sizes, "total": total, "n_branches": len(sizes)},
        note="%d branches: the more branches, the larger SplitInfo, and the harder it is to score a high Gain Ratio."
        % len(sizes),
    )


def gain_ratio_c(
    gain_comp: Computation,
    split_info_comp: Computation,
    feature_label: str,
    parent_symbol: str = "S",
) -> Computation:
    """Gain Ratio = Information Gain / SplitInfo, guarding a ~zero denominator."""
    si = split_info_comp.result
    note = None
    if si <= EPS:
        ratio = 0.0
        note = ("SplitInfo is 0 (the split puts every sample in one branch), so Gain Ratio "
                "is undefined; it is treated as 0 and the candidate is rejected.")
    else:
        ratio = float(gain_comp.result / si)
    sym = parent_symbol
    return Computation(
        name="gain_ratio",
        symbolic=r"GainRatio(%s, %s) = \frac{IG(%s, %s)}{SplitInfo(%s, %s)}"
        % (sym, tex.text(feature_label), sym, tex.text(feature_label), sym, tex.text(feature_label)),
        substituted=tex.frac(gain_comp.result, si) if si > EPS else r"\frac{%s}{0}" % tex.num(gain_comp.result),
        result=ratio,
        operands={"gain": gain_comp.result, "split_info": si, "feature": feature_label},
        note=note,
        children={"gain": gain_comp, "split_info": split_info_comp},
    )


def comparison_c(feature_label: str, value: Any, operator: str, threshold: Any, outcome: bool) -> Computation:
    """The single test performed at one node while routing a sample (Predict page)."""
    op_tex = {"<=": r"\leq", "<": "<", ">": ">", ">=": r"\geq", "==": "=", "in": r"\in"}.get(operator, operator)
    lhs = r"x\left[%s\right]" % tex.text(feature_label)
    rhs = tex.num(threshold) if isinstance(threshold, (int, float, np.floating)) else tex.text(threshold)
    shown = tex.num(value) if isinstance(value, (int, float, np.floating)) else tex.text(value)
    return Computation(
        name="comparison",
        symbolic=r"%s %s %s" % (lhs, op_tex, rhs),
        substituted=r"%s %s %s" % (shown, op_tex, rhs),
        result=1.0 if outcome else 0.0,
        operands={"feature": feature_label, "value": value, "operator": operator, "threshold": threshold},
        note=r"\text{TRUE}" if outcome else r"\text{FALSE}",
    )


def probability_c(counts: Dict[Any, float], label: Any, where: str = "leaf") -> Computation:
    """``P(class | node)`` written out with the actual training counts."""
    total = float(sum(counts.values()))
    hit = float(counts.get(label, 0.0))
    return Computation(
        name="probability",
        symbolic=r"P\left(%s \mid %s\right) = \frac{\text{count}(%s)}{\text{count(all)}}"
        % (tex.text(label), tex.text(where), tex.text(label)),
        substituted=tex.frac(hit, total),
        result=(hit / total) if total > 0 else 0.0,
        operands={"counts": counts, "label": label, "total": total},
    )


def threshold_c(feature_label: str, low: float, high: float) -> Computation:
    """The midpoint rule C4.5/CART use to turn a numeric column into a test."""
    mid = (float(low) + float(high)) / 2.0
    return Computation(
        name="threshold",
        symbolic=r"t = \frac{v_i + v_{i+1}}{2}",
        substituted=r"\frac{%s + %s}{2}" % (tex.num(low), tex.num(high)),
        result=mid,
        operands={"feature": feature_label, "low": low, "high": high},
    )


def rule_check_c(name: str, lhs_label: str, lhs: float, operator: str, rhs_label: str, rhs: float) -> Computation:
    """A stopping rule written as an inequality with both sides substituted."""
    op_tex = {"<=": r"\leq", "<": "<", ">": ">", ">=": r"\geq", "==": "="}.get(operator, operator)
    return Computation(
        name=name,
        symbolic=r"%s %s %s" % (tex.text(lhs_label), op_tex, tex.text(rhs_label)),
        substituted=r"%s %s %s" % (tex.num(lhs), op_tex, tex.num(rhs)),
        result=float(lhs),
        operands={"lhs": lhs, "rhs": rhs, "operator": operator},
    )
