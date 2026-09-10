"""Metrics and splitting, checked against hand-computed numbers."""
from __future__ import annotations

import numpy as np
import pytest

from src.core import metrics as mx

TRUE = ["a", "a", "a", "b", "b", "b", "b"]
PRED = ["a", "a", "b", "b", "b", "b", "a"]


def test_accuracy_counts_matches():
    assert mx.accuracy(TRUE, PRED) == pytest.approx(5 / 7)
    assert mx.accuracy([], []) == 0.0


def test_confusion_matrix_rows_are_the_truth():
    matrix, labels = mx.confusion_matrix(TRUE, PRED)
    assert labels == ["a", "b"]
    assert matrix.tolist() == [[2, 1], [1, 3]]
    assert matrix.sum() == len(TRUE)


def test_per_class_scores_by_hand():
    scores = mx.per_class_scores(TRUE, PRED)
    assert scores["a"]["precision"] == pytest.approx(2 / 3)
    assert scores["a"]["recall"] == pytest.approx(2 / 3)
    assert scores["b"]["precision"] == pytest.approx(3 / 4)
    assert scores["b"]["recall"] == pytest.approx(3 / 4)
    assert scores["b"]["support"] == 4


def test_macro_and_weighted_averages_differ_when_classes_are_unbalanced():
    macro = mx.precision_recall_f1(TRUE, PRED, average="macro")
    weighted = mx.precision_recall_f1(TRUE, PRED, average="weighted")
    assert macro["f1"] == pytest.approx((2 / 3 + 3 / 4) / 2)
    assert weighted["f1"] == pytest.approx((3 * (2 / 3) + 4 * (3 / 4)) / 7)


def test_report_has_a_row_per_class_plus_two_averages():
    rows = mx.classification_report_rows(TRUE, PRED)
    assert [r["Class"] for r in rows] == ["a", "b", "macro avg", "weighted avg"]


def test_regression_metrics():
    truth, guess = [1.0, 2.0, 3.0], [1.0, 2.0, 5.0]
    assert mx.mean_squared_error(truth, guess) == pytest.approx(4 / 3)
    assert mx.root_mean_squared_error(truth, guess) == pytest.approx(np.sqrt(4 / 3))
    assert mx.mean_absolute_error(truth, guess) == pytest.approx(2 / 3)
    assert mx.r2_score(truth, truth) == pytest.approx(1.0)


def test_train_test_split_is_a_partition():
    train, test = mx.train_test_split(100, ["a"] * 50 + ["b"] * 50, test_size=0.25)
    assert len(train) + len(test) == 100
    assert not set(train) & set(test)
    assert len(test) == 25


def test_stratified_split_keeps_class_proportions():
    labels = np.array(["a"] * 80 + ["b"] * 20)
    _train, test = mx.train_test_split(100, labels, test_size=0.5, stratify=True)
    counts = {label: int((labels[test] == label).sum()) for label in ("a", "b")}
    assert counts["a"] == 40 and counts["b"] == 10


def test_split_is_reproducible_for_a_fixed_seed():
    first = mx.train_test_split(50, None, 0.3, stratify=False, random_state=7)
    second = mx.train_test_split(50, None, 0.3, stratify=False, random_state=7)
    assert (first[1] == second[1]).all()


def test_k_fold_covers_every_sample_exactly_once():
    folds = mx.k_fold_indices(23, k=5)
    assert len(folds) == 5
    seen = np.concatenate([valid for _train, valid in folds])
    assert sorted(seen.tolist()) == list(range(23))
    for train, valid in folds:
        assert not set(train) & set(valid)


def test_k_fold_rejects_k_below_two():
    with pytest.raises(ValueError):
        mx.k_fold_indices(10, k=1)
