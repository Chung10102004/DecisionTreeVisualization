"""CART (Breiman et al., 1984).

Always a binary question — ``x <= t`` for numbers, ``x in {A, B}`` for
categories — scored with the Gini index for classification or squared error for
regression.  Unlike ID3, a feature may be tested again further down, which is
how CART carves out non-rectangular regions from a single column.
"""
from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

from .base_tree import BaseDecisionTree
from .splitter import BinarySplitter, Splitter


class CARTTree(BaseDecisionTree):
    algorithm = "CART"
    default_impurity = "gini"
    score_key = "gain"
    consume_features = False

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("impurity", "gini")
        kwargs.setdefault("task", "classification")
        super().__init__(**kwargs)

    def _make_splitter(self) -> Splitter:
        return BinarySplitter(
            X=self._X,
            y=self._y,
            feature_names=self._feature_names,
            feature_types=self._feature_types,
            impurity=self.impurity,
            weights=self._w,
            min_samples_leaf=self.min_samples_leaf,
            max_thresholds=self.max_thresholds,
            score_key="gain",
        )


class CARTRegressor(CARTTree):
    """CART for a numeric target: leaves predict the mean, splits cut variance."""

    algorithm = "CART (regression)"
    default_impurity = "mse"

    def __init__(self, **kwargs) -> None:
        kwargs["task"] = "regression"
        kwargs.setdefault("impurity", "mse")
        super().__init__(**kwargs)
