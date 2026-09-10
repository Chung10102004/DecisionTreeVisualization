"""Candidate split generation and scoring.

A *candidate* is one concrete way of partitioning the samples that reached a
node.  Every candidate keeps the full ``Computation`` chain that produced its
score, so the UI can open the losing candidates too and show exactly why they
lost.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import criteria as crit
from . import latex as tex
from .node import (
    IN,
    LEFT,
    OUT,
    RIGHT,
    SPLIT_CATEGORICAL_BINARY,
    SPLIT_CATEGORICAL_MULTIWAY,
    SPLIT_NUMERIC_BINARY,
    is_missing,
)

MAX_EXHAUSTIVE_CATEGORIES = 8


@dataclass
class SplitCandidate:
    """One scored way of splitting a node."""

    feature_index: int
    feature_name: str
    split_type: str
    partitions: Dict[str, np.ndarray]
    branch_labels: Dict[str, str]
    threshold: Optional[float] = None
    categories: Optional[Tuple[Any, ...]] = None

    child_class_counts: Dict[str, Dict[Any, float]] = field(default_factory=dict)
    child_impurities: Dict[str, float] = field(default_factory=dict)
    child_sizes: Dict[str, float] = field(default_factory=dict)

    weighted_impurity: float = 0.0
    gain: float = 0.0
    split_info: float = 0.0
    gain_ratio: float = 0.0
    score: float = 0.0
    score_name: str = "Information Gain"

    is_valid: bool = True
    reject_reason: Optional[str] = None
    threshold_scan: List[Tuple[float, float]] = field(default_factory=list)
    computations: Dict[str, crit.Computation] = field(default_factory=dict)
    missing_count: float = 0.0

    @property
    def branch_keys(self) -> List[str]:
        return list(self.partitions.keys())

    @property
    def n_branches(self) -> int:
        return len(self.partitions)

    def describe(self) -> str:
        """Short label of the test itself, shown in the candidates table."""
        if self.split_type == SPLIT_NUMERIC_BINARY:
            return f"{self.feature_name} <= {self.threshold:.4g}"
        if self.split_type == SPLIT_CATEGORICAL_BINARY:
            return f"{self.feature_name} in {{{', '.join(str(c) for c in (self.categories or ()))}}}"
        return f"{self.feature_name} (multiway, {self.n_branches} branches)"

    def branch_summary(self) -> str:
        return " | ".join(
            f"{self.branch_labels.get(k, k)}: n={self.child_sizes.get(k, 0):g}"
            for k in self.partitions
        )


class Splitter:
    """Base splitter: owns the data, subclasses decide which candidates exist."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
        feature_types: Dict[str, str],
        impurity: crit.Impurity,
        weights: Optional[np.ndarray] = None,
        min_samples_leaf: int = 1,
        max_thresholds: Optional[int] = None,
        score_key: str = "gain",
    ) -> None:
        self.X = X
        self.y = y
        self.feature_names = list(feature_names)
        self.feature_types = dict(feature_types)
        self.impurity = impurity
        self.weights = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_thresholds = max_thresholds
        self.score_key = score_key

    # ------------------------------------------------------------- utilities
    def column(self, indices: np.ndarray, feature_index: int) -> np.ndarray:
        return self.X[indices, feature_index]

    def is_numeric(self, feature_name: str) -> bool:
        return self.feature_types.get(feature_name) == "numeric"

    def _score_of(self, candidate: SplitCandidate) -> float:
        return getattr(candidate, self.score_key, candidate.gain)

    # ------------------------------------------------------------ evaluation
    def evaluate_partition(
        self,
        feature_index: int,
        split_type: str,
        partitions: Dict[str, np.ndarray],
        branch_labels: Dict[str, str],
        parent_comp: crit.Computation,
        threshold: Optional[float] = None,
        categories: Optional[Tuple[Any, ...]] = None,
        missing_count: float = 0.0,
        with_computations: bool = True,
    ) -> SplitCandidate:
        """Turn a partition into a fully scored (and fully explained) candidate."""
        feature_name = self.feature_names[feature_index]
        candidate = SplitCandidate(
            feature_index=feature_index,
            feature_name=feature_name,
            split_type=split_type,
            partitions=partitions,
            branch_labels=branch_labels,
            threshold=threshold,
            categories=categories,
            missing_count=missing_count,
        )

        children = [(key, self.y[idx], self.weights[idx]) for key, idx in partitions.items()]
        for key, idx in partitions.items():
            candidate.child_sizes[key] = float(np.sum(self.weights[idx]))
            candidate.child_class_counts[key] = crit.class_counts(self.y[idx], self.weights[idx])

        weighted = crit.weighted_impurity_c(children, self.impurity)
        gain = crit.impurity_gain_c(parent_comp, weighted, feature_name, self.impurity)
        info = crit.split_info_c(children, feature_name)
        ratio = crit.gain_ratio_c(gain, info, feature_name)

        candidate.child_impurities = {key: comp.result for key, comp in weighted.children.items()}
        candidate.weighted_impurity = weighted.result
        candidate.gain = gain.result
        candidate.split_info = info.result
        candidate.gain_ratio = ratio.result
        candidate.score_name = self.impurity.gain_name if self.score_key == "gain" else "Gain Ratio"
        candidate.score = self._score_of(candidate)

        if with_computations:
            candidate.computations = {
                "parent_impurity": parent_comp,
                "weighted_impurity": weighted,
                "gain": gain,
                "split_info": info,
                "gain_ratio": ratio,
            }

        self._validate(candidate)
        return candidate

    def _validate(self, candidate: SplitCandidate) -> None:
        non_empty = [k for k, idx in candidate.partitions.items() if len(idx) > 0]
        if len(non_empty) < 2:
            candidate.is_valid = False
            candidate.reject_reason = "The split puts every sample in a single branch, so it separates nothing."
            return
        too_small = [
            candidate.branch_labels.get(k, k)
            for k, idx in candidate.partitions.items()
            if 0 < len(idx) < self.min_samples_leaf
        ]
        if too_small:
            candidate.is_valid = False
            candidate.reject_reason = (
                f"Branch(es) {', '.join(too_small)} would hold fewer than min_samples_leaf="
                f"{self.min_samples_leaf} samples."
            )

    # -------------------------------------------------------------- generate
    def generate(
        self,
        indices: np.ndarray,
        available_features: Sequence[int],
        parent_comp: crit.Computation,
    ) -> List[SplitCandidate]:
        candidates: List[SplitCandidate] = []
        for feature_index in available_features:
            candidate = self.candidate_for_feature(indices, feature_index, parent_comp)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def candidate_for_feature(
        self, indices: np.ndarray, feature_index: int, parent_comp: crit.Computation
    ) -> Optional[SplitCandidate]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------- shared building blocks
    def _multiway_partition(
        self, indices: np.ndarray, feature_index: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str], float]:
        """One branch per distinct value; missing values are set aside."""
        values = self.column(indices, feature_index)
        partitions: Dict[str, np.ndarray] = {}
        labels: Dict[str, str] = {}
        missing = 0.0
        buckets: Dict[str, List[int]] = {}
        for position, value in enumerate(values):
            if is_missing(value):
                missing += float(self.weights[indices[position]])
                continue
            buckets.setdefault(str(value), []).append(indices[position])
        for key in sorted(buckets):
            partitions[key] = np.array(buckets[key], dtype=int)
            labels[key] = f"= {key}"
        return partitions, labels, missing

    def _numeric_thresholds(self, indices: np.ndarray, feature_index: int) -> List[float]:
        """Midpoints between adjacent distinct values whose labels differ.

        Restricting to class boundaries is the classic C4.5/CART shortcut: any
        optimal threshold always sits on one of them.
        """
        values = self.column(indices, feature_index)
        keep = [i for i, v in enumerate(values) if not is_missing(v)]
        if len(keep) < 2:
            return []
        numeric = np.array([float(values[i]) for i in keep], dtype=float)
        labels = np.array([self.y[indices[i]] for i in keep], dtype=object)
        order = np.argsort(numeric, kind="mergesort")
        numeric, labels = numeric[order], labels[order]

        thresholds: List[float] = []
        for i in range(len(numeric) - 1):
            if numeric[i] == numeric[i + 1]:
                continue
            if labels[i] == labels[i + 1] and self.impurity.task == "classification":
                continue
            thresholds.append((numeric[i] + numeric[i + 1]) / 2.0)

        if not thresholds:  # regression, or every neighbour shares a label
            thresholds = [
                (numeric[i] + numeric[i + 1]) / 2.0
                for i in range(len(numeric) - 1)
                if numeric[i] != numeric[i + 1]
            ]
        if self.max_thresholds and len(thresholds) > self.max_thresholds:
            step = len(thresholds) / float(self.max_thresholds)
            thresholds = [thresholds[int(i * step)] for i in range(self.max_thresholds)]
        return thresholds

    def _numeric_partition(
        self, indices: np.ndarray, feature_index: int, threshold: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str], float]:
        values = self.column(indices, feature_index)
        left, right, missing = [], [], 0.0
        for position, value in enumerate(values):
            if is_missing(value):
                missing += float(self.weights[indices[position]])
                continue
            (left if float(value) <= threshold else right).append(indices[position])
        name = self.feature_names[feature_index]
        partitions = {LEFT: np.array(left, dtype=int), RIGHT: np.array(right, dtype=int)}
        labels = {LEFT: f"<= {threshold:.4g}", RIGHT: f"> {threshold:.4g}"}
        return partitions, labels, missing

    def _best_numeric_candidate(
        self, indices: np.ndarray, feature_index: int, parent_comp: crit.Computation
    ) -> Optional[SplitCandidate]:
        """Scan every candidate threshold, keep the best, remember the whole scan."""
        thresholds = self._numeric_thresholds(indices, feature_index)
        if not thresholds:
            return None
        best: Optional[SplitCandidate] = None
        scan: List[Tuple[float, float]] = []
        for threshold in thresholds:
            partitions, labels, missing = self._numeric_partition(indices, feature_index, threshold)
            candidate = self.evaluate_partition(
                feature_index, SPLIT_NUMERIC_BINARY, partitions, labels, parent_comp,
                threshold=threshold, missing_count=missing, with_computations=False,
            )
            scan.append((threshold, candidate.score if candidate.is_valid else float("nan")))
            if candidate.is_valid and (best is None or candidate.score > best.score):
                best = candidate
        if best is None:
            # Keep the last one so the UI can still explain why nothing was usable.
            partitions, labels, missing = self._numeric_partition(indices, feature_index, thresholds[-1])
            best = self.evaluate_partition(
                feature_index, SPLIT_NUMERIC_BINARY, partitions, labels, parent_comp,
                threshold=thresholds[-1], missing_count=missing,
            )
        else:
            best = self.evaluate_partition(
                feature_index, SPLIT_NUMERIC_BINARY, best.partitions, best.branch_labels, parent_comp,
                threshold=best.threshold, missing_count=best.missing_count,
            )
        best.threshold_scan = scan
        best.computations["threshold"] = crit.Computation(
            name="threshold_choice",
            symbolic=r"t^{*} = \arg\max_{t} %s(S, %s \leq t)"
            % (self.impurity.gain_symbol, tex.text(self.feature_names[feature_index])),
            substituted=r"\arg\max \text{ over %d candidate thresholds}" % len(thresholds),
            result=float(best.threshold),
            operands={"thresholds": thresholds, "scan": scan},
            note="Thresholds are midpoints between adjacent values whose class labels differ.",
        )
        return best


class MultiwaySplitter(Splitter):
    """ID3: one branch per distinct value, categorical features only.

    Numeric columns are expected to have been discretised upstream; if one slips
    through it is still treated as a set of labels, which is what ID3 does.
    """

    def candidate_for_feature(
        self, indices: np.ndarray, feature_index: int, parent_comp: crit.Computation
    ) -> Optional[SplitCandidate]:
        partitions, labels, missing = self._multiway_partition(indices, feature_index)
        if not partitions:
            return None
        return self.evaluate_partition(
            feature_index, SPLIT_CATEGORICAL_MULTIWAY, partitions, labels, parent_comp,
            missing_count=missing,
        )


class HybridSplitter(Splitter):
    """C4.5: multiway on categorical features, binary threshold on numeric ones."""

    def candidate_for_feature(
        self, indices: np.ndarray, feature_index: int, parent_comp: crit.Computation
    ) -> Optional[SplitCandidate]:
        name = self.feature_names[feature_index]
        if self.is_numeric(name):
            return self._best_numeric_candidate(indices, feature_index, parent_comp)
        partitions, labels, missing = self._multiway_partition(indices, feature_index)
        if not partitions:
            return None
        return self.evaluate_partition(
            feature_index, SPLIT_CATEGORICAL_MULTIWAY, partitions, labels, parent_comp,
            missing_count=missing,
        )


class BinarySplitter(Splitter):
    """CART: always two branches — a numeric threshold or a category subset."""

    def candidate_for_feature(
        self, indices: np.ndarray, feature_index: int, parent_comp: crit.Computation
    ) -> Optional[SplitCandidate]:
        name = self.feature_names[feature_index]
        if self.is_numeric(name):
            return self._best_numeric_candidate(indices, feature_index, parent_comp)
        return self._best_subset_candidate(indices, feature_index, parent_comp)

    # ------------------------------------------------------------------ subsets
    def _categories(self, indices: np.ndarray, feature_index: int) -> List[Any]:
        values = self.column(indices, feature_index)
        seen: List[Any] = []
        for value in values:
            if is_missing(value):
                continue
            if value not in seen:
                seen.append(value)
        return sorted(seen, key=str)

    def _subset_partition(
        self, indices: np.ndarray, feature_index: int, subset: Tuple[Any, ...]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str], float]:
        values = self.column(indices, feature_index)
        inside, outside, missing = [], [], 0.0
        members = set(subset)
        for position, value in enumerate(values):
            if is_missing(value):
                missing += float(self.weights[indices[position]])
                continue
            (inside if value in members else outside).append(indices[position])
        pretty = ", ".join(str(c) for c in subset)
        partitions = {IN: np.array(inside, dtype=int), OUT: np.array(outside, dtype=int)}
        labels = {IN: f"in {{{pretty}}}", OUT: f"not in {{{pretty}}}"}
        return partitions, labels, missing

    def _subset_proposals(self, categories: List[Any], indices: np.ndarray, feature_index: int) -> List[Tuple]:
        """Exhaustive for few categories, Breiman's ordering trick for many."""
        k = len(categories)
        if k <= 1:
            return []
        if k <= MAX_EXHAUSTIVE_CATEGORIES:
            proposals = []
            for size in range(1, k // 2 + 1):
                for subset in itertools.combinations(categories, size):
                    if size * 2 == k and categories[0] not in subset:
                        continue  # complement duplicates
                    proposals.append(subset)
            return proposals
        ordered = self._order_categories(categories, indices, feature_index)
        return [tuple(ordered[: i + 1]) for i in range(k - 1)]

    def _order_categories(self, categories: List[Any], indices: np.ndarray, feature_index: int) -> List[Any]:
        """Rank categories by their mean target, which makes contiguous cuts optimal."""
        values = self.column(indices, feature_index)
        if self.impurity.task == "regression":
            key_of = {c: [] for c in categories}
            for position, value in enumerate(values):
                if not is_missing(value) and value in key_of:
                    key_of[value].append(float(self.y[indices[position]]))
            return sorted(categories, key=lambda c: (np.mean(key_of[c]) if key_of[c] else 0.0))
        first_class = sorted({str(v) for v in self.y[indices]})[0]
        ratio: Dict[Any, float] = {}
        for category in categories:
            mask = [p for p, v in enumerate(values) if v == category]
            if not mask:
                ratio[category] = 0.0
                continue
            hits = sum(1 for p in mask if str(self.y[indices[p]]) == first_class)
            ratio[category] = hits / len(mask)
        return sorted(categories, key=lambda c: ratio[c])

    def _best_subset_candidate(
        self, indices: np.ndarray, feature_index: int, parent_comp: crit.Computation
    ) -> Optional[SplitCandidate]:
        categories = self._categories(indices, feature_index)
        proposals = self._subset_proposals(categories, indices, feature_index)
        if not proposals:
            return None
        best: Optional[SplitCandidate] = None
        for subset in proposals:
            partitions, labels, missing = self._subset_partition(indices, feature_index, subset)
            candidate = self.evaluate_partition(
                feature_index, SPLIT_CATEGORICAL_BINARY, partitions, labels, parent_comp,
                categories=tuple(subset), missing_count=missing, with_computations=False,
            )
            if candidate.is_valid and (best is None or candidate.score > best.score):
                best = candidate
        if best is None:
            return None
        rebuilt = self.evaluate_partition(
            feature_index, SPLIT_CATEGORICAL_BINARY, best.partitions, best.branch_labels, parent_comp,
            categories=best.categories, missing_count=best.missing_count,
        )
        rebuilt.computations["subset"] = crit.Computation(
            name="subset_choice",
            symbolic=r"A^{*} = \arg\max_{A \subset \text{values}} %s(S, %s \in A)"
            % (self.impurity.gain_symbol, tex.text(self.feature_names[feature_index])),
            substituted=r"\arg\max \text{ over %d candidate subsets}" % len(proposals),
            result=rebuilt.score,
            operands={"n_proposals": len(proposals), "chosen": rebuilt.categories},
            note=("All %d non-trivial subsets were tried." % len(proposals))
            if len(categories) <= MAX_EXHAUSTIVE_CATEGORIES
            else ("Categories were ordered by class ratio and the %d contiguous cuts were tried "
                  "(Breiman's shortcut, avoids 2^k enumeration)." % len(proposals)),
        )
        return rebuilt


def rank_candidates(candidates: Sequence[SplitCandidate], only_valid: bool = False) -> List[SplitCandidate]:
    """Best first; invalid candidates always sink to the bottom but are kept."""
    pool = [c for c in candidates if c.is_valid] if only_valid else list(candidates)
    return sorted(pool, key=lambda c: (c.is_valid, c.score, c.gain), reverse=True)


def argmax_computation(
    candidates: Sequence[SplitCandidate], chosen: Optional[SplitCandidate], score_name: str
) -> crit.Computation:
    """The selection step itself, written out with every feature's score."""
    ranked = rank_candidates(candidates)
    pieces = [
        r"%s{:}\ %s" % (tex.text(c.feature_name), tex.num(c.score))
        for c in ranked
        if c.is_valid
    ]
    body = r",\quad ".join(pieces) if pieces else r"\text{no valid candidate}"
    return crit.Computation(
        name="argmax",
        symbolic=r"A^{*} = \arg\max_{A}\ %s(S, A)" % tex.escape(score_name),
        substituted=r"\arg\max\left\{%s\right\}" % body,
        result=chosen.score if chosen is not None else float("nan"),
        operands={"scores": {c.feature_name: c.score for c in ranked}, "criterion": score_name},
        note=(f"{chosen.feature_name} wins with {score_name} = {chosen.score:.4f}."
              if chosen is not None else "No candidate passed the validity checks."),
    )
