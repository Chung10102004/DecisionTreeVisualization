"""Turning a Dataset into arrays the learners can consume.

Two jobs matter here beyond the mechanical conversion:

* **Discretisation.** ID3 cannot compare numbers, so numeric columns have to
  become interval labels first. The bin edges are kept so the same transform can
  be replayed on a brand-new sample at prediction time.
* **Missing values.** Whether to drop, impute, or keep them is a real modelling
  choice — C4.5 wants them kept, ID3 cannot cope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .loaders import CATEGORICAL, NUMERIC, Dataset

EQUAL_WIDTH = "equal_width"
EQUAL_FREQUENCY = "equal_frequency"

MISSING_KEEP = "keep"
MISSING_DROP = "drop"
MISSING_IMPUTE = "impute"


@dataclass
class BinningInfo:
    """The bin edges used for one column, replayable on new values."""

    column: str
    edges: List[float]
    labels: List[str]
    strategy: str

    def apply(self, value: Any) -> Any:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        for i in range(len(self.edges) - 1):
            upper = self.edges[i + 1]
            if number <= upper or i == len(self.edges) - 2:
                return self.labels[i]
        return self.labels[-1]

    def describe(self) -> str:
        return f"{self.column}: {len(self.labels)} bins ({self.strategy}) -> {', '.join(self.labels)}"


@dataclass
class PreparedData:
    """Everything ``fit`` needs, plus the frame the UI shows next to it."""

    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    feature_types: Dict[str, str]
    class_names: List[Any]
    frame: pd.DataFrame
    target: str
    task: str = "classification"
    binnings: Dict[str, BinningInfo] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(list(indices), dtype=int)
        return self.X[idx], self.y[idx]

    def frame_for(self, indices: Sequence[int]) -> pd.DataFrame:
        return self.frame.iloc[list(indices)]


def make_bins(values: pd.Series, n_bins: int, strategy: str) -> BinningInfo:
    """Compute bin edges, then name each bin after the interval it covers."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return BinningInfo(str(values.name), [0.0, 1.0], ["(all)"], strategy)

    low, high = float(clean.min()), float(clean.max())
    if low == high:
        return BinningInfo(str(values.name), [low, high], [f"= {low:.4g}"], strategy)

    if strategy == EQUAL_FREQUENCY:
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = sorted({float(clean.quantile(q)) for q in quantiles})
        if len(edges) < 2:
            edges = [low, high]
    else:
        edges = list(np.linspace(low, high, n_bins + 1))

    edges[0], edges[-1] = low, high
    labels = []
    for i in range(len(edges) - 1):
        left = "[" if i == 0 else "("
        labels.append(f"{left}{edges[i]:.4g}, {edges[i + 1]:.4g}]")
    return BinningInfo(str(values.name), [float(e) for e in edges], labels, strategy)


class Preprocessor:
    """Applies missing-value policy and optional discretisation, reusably."""

    def __init__(
        self,
        missing_strategy: str = MISSING_KEEP,
        discretize: Sequence[str] = (),
        n_bins: int = 3,
        bin_strategy: str = EQUAL_WIDTH,
    ) -> None:
        self.missing_strategy = missing_strategy
        self.discretize = list(discretize)
        self.n_bins = int(n_bins)
        self.bin_strategy = bin_strategy
        self.binnings: Dict[str, BinningInfo] = {}
        self.impute_values: Dict[str, Any] = {}
        self.notes: List[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, dataset: Dataset) -> "Preprocessor":
        df = dataset.df
        self.binnings = {}
        self.impute_values = {}
        self.notes = []

        for column in self.discretize:
            if column in df.columns and column != dataset.target:
                self.binnings[column] = make_bins(df[column], self.n_bins, self.bin_strategy)

        if self.missing_strategy == MISSING_IMPUTE:
            for column in dataset.features:
                series = df[column]
                if not series.isna().any():
                    continue
                if dataset.type_of(column) == NUMERIC and pd.api.types.is_numeric_dtype(series):
                    self.impute_values[column] = float(series.mean())
                else:
                    modes = series.mode(dropna=True)
                    self.impute_values[column] = modes.iloc[0] if len(modes) else None
        return self

    # ------------------------------------------------------------- transform
    def transform_frame(self, df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        out = df.copy()
        for column, value in self.impute_values.items():
            if column in out.columns and value is not None:
                out[column] = out[column].fillna(value)
        if self.missing_strategy == MISSING_DROP:
            feature_columns = [c for c in out.columns if c != target]
            before = len(out)
            out = out.dropna(subset=feature_columns)
            if before != len(out):
                self.notes.append(f"Dropped {before - len(out)} row(s) containing a missing value.")
        for column, binning in self.binnings.items():
            if column in out.columns:
                out[column] = out[column].map(binning.apply)
        return out

    def transform_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the very same transform to one new sample from the Predict page."""
        out = dict(sample)
        for column, value in self.impute_values.items():
            if column in out and (out[column] is None or (isinstance(out[column], float) and np.isnan(out[column]))):
                out[column] = value
        for column, binning in self.binnings.items():
            if column in out:
                out[column] = binning.apply(out[column])
        return out

    def resulting_types(self, dataset: Dataset) -> Dict[str, str]:
        types = dict(dataset.feature_types)
        for column in self.binnings:
            types[column] = CATEGORICAL
        return types

    # --------------------------------------------------------------- prepare
    def prepare(self, dataset: Dataset, features: Optional[Sequence[str]] = None) -> PreparedData:
        self.fit(dataset)
        frame = self.transform_frame(dataset.df, dataset.target)
        selected = list(features) if features else dataset.features
        selected = [f for f in selected if f in frame.columns and f != dataset.target]
        if not selected:
            raise ValueError("No feature columns are selected.")

        types = {name: self.resulting_types(dataset).get(name, CATEGORICAL) for name in selected}
        X = frame[selected].to_numpy(dtype=object)
        X = np.where(pd.isna(X), None, X)
        y = frame[dataset.target].to_numpy(dtype=object)

        notes = list(dataset.notes) + list(self.notes)
        for binning in self.binnings.values():
            if binning.column in selected:
                notes.append("Discretised " + binning.describe())

        class_names: List[Any] = [] if dataset.task == "regression" else sorted(
            {str(v) for v in y if v is not None}
        )
        return PreparedData(
            X=X, y=y, feature_names=selected, feature_types=types, class_names=class_names,
            frame=frame[selected + [dataset.target]], target=dataset.target, task=dataset.task,
            binnings=dict(self.binnings), notes=notes,
        )


def suggest_discretization(dataset: Dataset, algorithm: str) -> List[str]:
    """ID3 is the only algorithm here that *needs* numeric columns binned."""
    if algorithm != "ID3":
        return []
    return dataset.numeric_features()
