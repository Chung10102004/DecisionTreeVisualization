"""Evaluation metrics and data splitting, implemented from scratch.

No sklearn anywhere in this project, including here: the numbers on the Evaluate
page are computed by the same code the user can read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _as_object(values: Any) -> np.ndarray:
    if hasattr(values, "values"):
        values = values.values
    return np.asarray(list(values), dtype=object)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def accuracy(y_true: Any, y_pred: Any) -> float:
    a, b = _as_object(y_true), _as_object(y_pred)
    if len(a) == 0:
        return 0.0
    return float(np.mean([str(x) == str(y) for x, y in zip(a, b)]))


def confusion_matrix(y_true: Any, y_pred: Any, labels: Optional[Sequence] = None) -> Tuple[np.ndarray, List]:
    """Rows are the true class, columns the predicted one."""
    a, b = _as_object(y_true), _as_object(y_pred)
    if labels is None:
        labels = sorted({str(v) for v in np.concatenate([a, b])} if len(a) else set())
    labels = list(labels)
    index = {str(label): i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, guess in zip(a, b):
        i, j = index.get(str(truth)), index.get(str(guess))
        if i is not None and j is not None:
            matrix[i, j] += 1
    return matrix, labels


def per_class_scores(y_true: Any, y_pred: Any, labels: Optional[Sequence] = None) -> Dict[str, Dict[str, float]]:
    """Precision / recall / F1 / support for every class."""
    matrix, labels = confusion_matrix(y_true, y_pred, labels)
    scores: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[i, :].sum()),
        }
    return scores


def precision_recall_f1(y_true: Any, y_pred: Any, average: str = "macro",
                        labels: Optional[Sequence] = None) -> Dict[str, float]:
    scores = per_class_scores(y_true, y_pred, labels)
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if average == "macro":
        weights = {k: 1.0 for k in scores}
    elif average == "weighted":
        weights = {k: float(v["support"]) for k, v in scores.items()}
    else:
        raise ValueError("average must be 'macro' or 'weighted'.")
    total = sum(weights.values()) or 1.0
    return {
        metric: sum(scores[k][metric] * weights[k] for k in scores) / total
        for metric in ("precision", "recall", "f1")
    }


def classification_report_rows(y_true: Any, y_pred: Any, labels: Optional[Sequence] = None) -> List[Dict[str, Any]]:
    scores = per_class_scores(y_true, y_pred, labels)
    rows = [
        {
            "Class": label,
            "Precision": round(values["precision"], 4),
            "Recall": round(values["recall"], 4),
            "F1": round(values["f1"], 4),
            "Support": values["support"],
        }
        for label, values in scores.items()
    ]
    for average in ("macro", "weighted"):
        agg = precision_recall_f1(y_true, y_pred, average=average, labels=labels)
        rows.append(
            {
                "Class": f"{average} avg",
                "Precision": round(agg["precision"], 4),
                "Recall": round(agg["recall"], 4),
                "F1": round(agg["f1"], 4),
                "Support": int(len(_as_object(y_true))),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def _as_float(values: Any) -> np.ndarray:
    if hasattr(values, "values"):
        values = values.values
    return np.asarray(list(values), dtype=float)


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    a, b = _as_float(y_true), _as_float(y_pred)
    return float(np.mean((a - b) ** 2)) if a.size else 0.0


def root_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    a, b = _as_float(y_true), _as_float(y_pred)
    return float(np.mean(np.abs(a - b))) if a.size else 0.0


def r2_score(y_true: Any, y_pred: Any) -> float:
    a, b = _as_float(y_true), _as_float(y_pred)
    if a.size == 0:
        return 0.0
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def score_model(y_true: Any, y_pred: Any, task: str = "classification") -> Dict[str, float]:
    """One dict of headline numbers, whichever task we are on."""
    if task == "regression":
        return {
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }
    agg = precision_recall_f1(y_true, y_pred, average="macro")
    return {
        "Accuracy": accuracy(y_true, y_pred),
        "Precision (macro)": agg["precision"],
        "Recall (macro)": agg["recall"],
        "F1 (macro)": agg["f1"],
    }


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _half_up(value: float) -> int:
    """Round .5 away from zero — Python's banker's rounding surprises students."""
    return int(math.floor(float(value) + 0.5))


def _largest_remainder(exact: Dict[str, float], total: int) -> Dict[str, int]:
    """Turn fractional per-class quotas into integers that sum to ``total`` exactly."""
    base = {k: int(math.floor(v)) for k, v in exact.items()}
    leftover = total - sum(base.values())
    order = sorted(exact, key=lambda k: (exact[k] - base[k], exact[k]), reverse=True)
    for key in order:
        if leftover <= 0:
            break
        base[key] += 1
        leftover -= 1
    return base


def train_test_split(
    n_samples: int,
    y: Optional[Any] = None,
    test_size: float = 0.3,
    stratify: bool = True,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return index arrays, so callers can slice a DataFrame or an array alike."""
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    if test_size <= 0:
        return indices, np.array([], dtype=int)
    if test_size >= 1:
        return np.array([], dtype=int), indices

    wanted = _half_up(n_samples * test_size)

    if stratify and y is not None:
        labels = _as_object(y)
        classes = sorted({str(v) for v in labels})
        pools = {c: rng.permutation(indices[np.array([str(v) == c for v in labels])]) for c in classes}
        quotas = _largest_remainder({c: len(pools[c]) * test_size for c in classes}, wanted)
        parts = [pools[c][: quotas[c]] for c in classes if quotas[c] > 0]
        test_idx = np.sort(np.concatenate(parts)) if parts else np.array([], dtype=int)
    else:
        shuffled = rng.permutation(indices)
        test_idx = np.sort(shuffled[:wanted])

    mask = np.ones(n_samples, dtype=bool)
    mask[test_idx] = False
    return indices[mask], test_idx


def k_fold_indices(
    n_samples: int, k: int = 5, shuffle: bool = True, random_state: Optional[int] = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Plain k-fold: returns ``(train_idx, valid_idx)`` for each fold."""
    if k < 2:
        raise ValueError("k must be at least 2.")
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    if shuffle:
        indices = rng.permutation(indices)
    folds = np.array_split(indices, k)
    output: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        valid = np.sort(folds[i])
        train = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        output.append((train, valid))
    return output
