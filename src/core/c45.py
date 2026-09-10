"""C4.5 (Quinlan, 1993) — ID3 plus the three fixes that made it usable.

1. **Gain Ratio** instead of raw Information Gain, so a near-unique column such
   as an ID can no longer win by shattering the data.
2. **Numeric thresholds**, chosen as midpoints between adjacent values whose
   labels differ.
3. **Fractional instances** for missing values: a sample with an unknown value
   is sent down *every* branch, weighted by how big that branch is.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import criteria as crit
from .base_tree import BaseDecisionTree
from .node import Node, SPLIT_CATEGORICAL_MULTIWAY
from .splitter import HybridSplitter, SplitCandidate, Splitter, rank_candidates


class C45Tree(BaseDecisionTree):
    algorithm = "C4.5"
    default_impurity = "entropy"
    score_key = "gain_ratio"
    consume_features = False  # handled per split type in _child_features

    def __init__(self, use_gain_filter: bool = True, **kwargs) -> None:
        kwargs.setdefault("impurity", "entropy")
        kwargs.setdefault("task", "classification")
        kwargs.setdefault("missing_strategy", "fractional")
        # C4.5's real default. It is also the mechanism that keeps ID-like columns
        # out of the tree: a column of unique values cannot give any branch 2 samples.
        kwargs.setdefault("min_samples_leaf", 2)
        if kwargs.get("task") == "regression":
            raise ValueError("C4.5 is a classification algorithm; use CART for regression targets.")
        self.use_gain_filter = use_gain_filter
        super().__init__(**kwargs)

    def _make_splitter(self) -> Splitter:
        return HybridSplitter(
            X=self._X,
            y=self._y,
            feature_names=self._feature_names,
            feature_types=self._feature_types,
            impurity=self.impurity,
            weights=self._w,
            min_samples_leaf=self.min_samples_leaf,
            max_thresholds=self.max_thresholds,
            score_key="gain_ratio",
        )

    # --------------------------------------------------------------- selection
    def _select(self, candidates: Sequence[SplitCandidate]) -> Optional[SplitCandidate]:
        """Quinlan's rule: keep only candidates with at least average gain, then
        take the best Gain Ratio among those.

        Without the filter, a split with a tiny gain but an even tinier SplitInfo
        can post a huge ratio and win on noise.
        """
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return None
        if not self.use_gain_filter or len(valid) == 1:
            return max(valid, key=lambda c: (c.gain_ratio, c.gain))

        mean_gain = float(np.mean([c.gain for c in valid]))
        shortlist = [c for c in valid if c.gain >= mean_gain - 1e-12]
        if not shortlist:
            shortlist = valid
        self._last_gain_filter = crit.rule_check_c(
            "gain_filter", "mean gain of candidates", mean_gain, "<=", "gain of shortlisted candidates",
            min(c.gain for c in shortlist),
        )
        return max(shortlist, key=lambda c: (c.gain_ratio, c.gain))

    def _child_features(self, available: Sequence[int], chosen: SplitCandidate) -> List[int]:
        """Categorical tests exhaust their column; numeric thresholds do not."""
        if chosen.split_type == SPLIT_CATEGORICAL_MULTIWAY:
            return [f for f in available if f != chosen.feature_index]
        return list(available)

    # ----------------------------------------------------------- missing data
    def _handle_missing(self, node: Node, chosen: SplitCandidate) -> Dict[str, np.ndarray]:
        partitions = {k: np.asarray(v, dtype=int) for k, v in chosen.partitions.items()}
        placed = {int(i) for idx in partitions.values() for i in idx}
        missing = [int(i) for i in node.sample_indices if int(i) not in placed]
        if not missing or self.missing_strategy != "fractional":
            return super()._handle_missing(node, chosen)

        known_weight = {k: float(np.sum(self._w[idx])) for k, idx in partitions.items()}
        total_known = sum(known_weight.values())
        if total_known <= 0:
            return super()._handle_missing(node, chosen)

        extra: Dict[str, List[int]] = {k: [] for k in partitions}
        for source in missing:
            for key, weight in known_weight.items():
                share = weight / total_known
                if share <= 0:
                    continue
                extra[key].append(self._append_row(source, float(self._w[source]) * share))

        for key, added in extra.items():
            if added:
                partitions[key] = np.concatenate([partitions[key], np.array(added, dtype=int)])

        shares = ", ".join(f"{k}: {known_weight[k] / total_known:.2f}" for k in partitions)
        self.trace_.notes.append(
            f"Node #{node.node_id}: {len(missing)} sample(s) have no value for "
            f"{chosen.feature_name}. Each was split into fractional copies and sent down every "
            f"branch with weights proportional to the known data ({shares})."
        )
        return partitions
