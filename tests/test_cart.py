"""CART: binary splits, Gini, category subsets, and regression."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.cart import CARTRegressor, CARTTree
from src.core.node import SPLIT_CATEGORICAL_BINARY, SPLIT_NUMERIC_BINARY


def test_iris_root_is_the_classic_petal_length_cut(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=3).fit(df[features], df[target], feature_types=types)
    root = model.tree.root
    assert root.feature_name == "PetalLength"
    assert root.threshold == pytest.approx(2.45, abs=0.05)
    assert root.split_type == SPLIT_NUMERIC_BINARY


def test_every_split_is_binary(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=4).fit(df[features], df[target], feature_types=types)
    for node in model.tree.root.iter_nodes():
        if not node.is_leaf:
            assert len(node.children) == 2


def test_impurity_never_increases_from_parent_to_weighted_children(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=4).fit(df[features], df[target], feature_types=types)
    for step in model.get_trace().steps:
        if step.chosen is not None:
            assert step.chosen.gain >= -1e-12
            assert step.chosen.weighted_impurity <= step.impurity_value + 1e-12


def test_accuracy_on_iris_is_high(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=3).fit(df[features], df[target], feature_types=types)
    assert (model.tree.predict(df[features]) == df[target].values).mean() > 0.95


def test_categorical_columns_are_split_into_subsets(play_tennis):
    df, features, types, target = play_tennis
    model = CARTTree().fit(df[features], df[target], feature_types=types)
    root = model.tree.root
    assert root.split_type == SPLIT_CATEGORICAL_BINARY
    assert root.categories is not None and len(root.categories) >= 1
    assert (model.tree.predict(df[features]) == df[target].values).all()


def test_entropy_may_be_used_instead_of_gini(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=2, impurity="entropy").fit(df[features], df[target], feature_types=types)
    assert model.tree.impurity_name == "entropy"
    assert model.tree.root.feature_name == "PetalLength"


def test_regression_leaves_predict_the_mean(auto_mpg):
    df, features, types, target = auto_mpg
    model = CARTRegressor(max_depth=3, min_samples_leaf=5).fit(
        df[features], df[target], feature_types=types
    )
    assert model.tree.task == "regression"
    predictions = np.asarray(model.tree.predict(df[features]), dtype=float)
    rmse = float(np.sqrt(np.mean((predictions - df[target].values) ** 2)))
    baseline = float(np.std(df[target].values))
    assert rmse < baseline  # the tree must beat predicting the global mean

    leaf = model.tree.root.leaves()[0]
    members = df[target].values[leaf.sample_indices]
    assert float(leaf.prediction) == pytest.approx(float(np.mean(members)), abs=1e-6)


def test_min_samples_leaf_is_respected(iris):
    df, features, types, target = iris
    model = CARTTree(min_samples_leaf=10).fit(df[features], df[target], feature_types=types)
    assert all(leaf.n_samples >= 10 for leaf in model.tree.root.leaves())
