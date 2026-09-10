"""Session state, in one place.

Streamlit reruns the whole script on every interaction, so anything that must
survive a click lives in ``st.session_state``.  Wrapping it in one object keeps
string keys out of the page modules and makes the invalidation rules explicit:
changing the data invalidates the model, changing the model invalidates the
prediction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

DEFAULTS: Dict[str, Any] = {
    "dataset": None,
    "preprocessor": None,
    "prepared": None,
    "selected_features": None,
    "train_index": None,
    "test_index": None,
    "split_config": {"test_size": 0.3, "stratify": True, "random_state": 42},
    "algorithm": "ID3",
    "hyperparams": {},
    "model": None,
    "trace": None,
    "current_step": 0,
    "pruned_tree": None,
    "prune_steps": None,
    "prune_method": None,
    "use_pruned": False,
    "prediction_trace": None,
    "prediction_sample": None,
    "comparison": None,
    "render_options": {},
    "selected_node": None,
    "selected_candidate": None,
}


class AppState:
    """Typed-ish accessor over ``st.session_state``."""

    def __init__(self) -> None:
        for key, value in DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = value.copy() if isinstance(value, dict) else value

    # ------------------------------------------------------------ generic get
    def __getattr__(self, name: str) -> Any:
        if name in DEFAULTS:
            return st.session_state.get(name, DEFAULTS[name])
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in DEFAULTS:
            st.session_state[name] = value
        else:  # pragma: no cover - defensive
            super().__setattr__(name, value)

    # --------------------------------------------------------------- readiness
    def is_data_ready(self) -> bool:
        return self.dataset is not None and self.prepared is not None

    def is_model_ready(self) -> bool:
        return self.model is not None and self.trace is not None

    def has_prediction(self) -> bool:
        return self.prediction_trace is not None

    # ------------------------------------------------------------ invalidation
    def set_dataset(self, dataset, preprocessor=None, prepared=None,
                    selected_features: Optional[List[str]] = None) -> None:
        """A new dataset invalidates everything downstream of it."""
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.prepared = prepared
        self.selected_features = selected_features
        self.train_index = None
        self.test_index = None
        self.reset_model()

    def reset_model(self) -> None:
        self.model = None
        self.trace = None
        self.current_step = 0
        self.comparison = None
        self.selected_node = None
        self.selected_candidate = None
        self.reset_pruning()
        self.reset_prediction()

    def reset_pruning(self) -> None:
        self.pruned_tree = None
        self.prune_steps = None
        self.prune_method = None
        self.use_pruned = False

    def reset_prediction(self) -> None:
        self.prediction_trace = None
        self.prediction_sample = None

    def set_model(self, model, trace) -> None:
        self.model = model
        self.trace = trace
        self.current_step = max(len(trace) - 1, 0)
        self.reset_pruning()
        self.reset_prediction()

    # --------------------------------------------------------------- accessors
    @property
    def tree(self):
        """The tree the app should display: the pruned one when it is in use."""
        if self.use_pruned and self.pruned_tree is not None:
            return self.pruned_tree
        return self.model.tree if self.model is not None else None

    @property
    def original_tree(self):
        return self.model.tree if self.model is not None else None

    def split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Train/test indices, computed on first use from the split config."""
        if self.train_index is None or self.test_index is None:
            from ..core.metrics import train_test_split

            prepared = self.prepared
            config = self.split_config
            train, test = train_test_split(
                len(prepared.y), prepared.y,
                test_size=config["test_size"], stratify=config["stratify"],
                random_state=config["random_state"],
            )
            self.train_index, self.test_index = train, test
        return np.asarray(self.train_index), np.asarray(self.test_index)

    def training_data(self):
        prepared = self.prepared
        train, _test = self.split_indices()
        if len(train) == 0:
            return prepared.X, prepared.y
        return prepared.X[train], prepared.y[train]

    def test_data(self):
        prepared = self.prepared
        _train, test = self.split_indices()
        return prepared.X[test], prepared.y[test]

    def update_render_options(self, **kwargs) -> None:
        options = dict(self.render_options)
        options.update(kwargs)
        self.render_options = options


def get_state() -> AppState:
    """One shared instance per rerun."""
    return AppState()
