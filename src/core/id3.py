"""ID3 (Quinlan, 1986).

Entropy + Information Gain, one branch per distinct value, and each feature is
used at most once along any path.  ID3 has no notion of a numeric threshold, so
numeric columns must be discretised before they get here — the trace says so
loudly when one slips through.
"""
from __future__ import annotations

from typing import List, Sequence

from .base_tree import BaseDecisionTree
from .splitter import MultiwaySplitter, Splitter


class ID3Tree(BaseDecisionTree):
    algorithm = "ID3"
    default_impurity = "entropy"
    score_key = "gain"
    consume_features = True

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("impurity", "entropy")
        kwargs.setdefault("task", "classification")
        if kwargs.get("task") == "regression":
            raise ValueError("ID3 is a classification algorithm; use CART for regression targets.")
        super().__init__(**kwargs)

    def _make_splitter(self) -> Splitter:
        numeric = [name for name, kind in self._feature_types.items()
                   if kind == "numeric" and name in self._feature_names]
        if numeric:
            self.trace_.notes.append(
                "ID3 cannot compare numbers, only match values. Column(s) "
                + ", ".join(numeric)
                + " are numeric and were treated as plain labels — discretise them on the Data page "
                  "for a meaningful ID3 tree."
            )
        return MultiwaySplitter(
            X=self._X,
            y=self._y,
            feature_names=self._feature_names,
            feature_types=self._feature_types,
            impurity=self.impurity,
            weights=self._w,
            min_samples_leaf=self.min_samples_leaf,
            score_key="gain",
        )
