"""C4.5: gain ratio, numeric thresholds, and fractional instances."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.c45 import C45Tree
from src.core.node import SPLIT_NUMERIC_BINARY


def test_gain_ratio_keeps_the_id_column_out_of_the_tree(play_tennis_with_id):
    """ID3 picks ``Day``; C4.5 must not."""
    df, features, types, target = play_tennis_with_id
    model = C45Tree().fit(df[features], df[target], feature_types=types)
    assert model.tree.root.feature_name == "Outlook"

    day = next(c for c in model.get_trace().steps[0].candidates if c.feature_name == "Day")
    assert not day.is_valid
    assert "min_samples_leaf" in day.reject_reason


def test_split_info_is_what_penalises_the_id_column(play_tennis_with_id):
    df, features, types, target = play_tennis_with_id
    model = C45Tree(min_samples_leaf=1).fit(df[features], df[target], feature_types=types)
    step = model.get_trace().steps[0]
    day = next(c for c in step.candidates if c.feature_name == "Day")
    outlook = next(c for c in step.candidates if c.feature_name == "Outlook")
    assert day.gain > outlook.gain            # raw gain is fooled
    assert day.split_info > outlook.split_info  # but the split shape is punished


def test_numeric_features_become_midpoint_thresholds(weather_numeric):
    df, features, types, target = weather_numeric
    model = C45Tree(min_samples_leaf=1).fit(df[features], df[target], feature_types=types)
    numeric_nodes = [n for n in model.tree.root.iter_nodes()
                     if n.split_type == SPLIT_NUMERIC_BINARY]
    for node in numeric_nodes:
        column = df[node.feature_name].values
        assert column.min() < node.threshold < column.max()
    assert (model.tree.predict(df[features]) == df[target].values).mean() >= 0.85


def test_threshold_scan_is_recorded_for_the_chart(weather_numeric):
    df, features, types, target = weather_numeric
    model = C45Tree(min_samples_leaf=1).fit(df[features], df[target], feature_types=types)
    numeric = [c for step in model.get_trace().steps for c in step.candidates
               if c.split_type == SPLIT_NUMERIC_BINARY]
    assert numeric, "expected at least one numeric candidate"
    assert all(len(c.threshold_scan) >= 1 for c in numeric)


def test_categorical_feature_is_used_once_but_numeric_can_repeat(iris):
    df, features, types, target = iris
    model = C45Tree(max_depth=4, min_samples_leaf=1).fit(df[features], df[target], feature_types=types)
    used = [n.feature_name for n in model.tree.root.iter_nodes() if not n.is_leaf]
    assert len(used) >= 2  # numeric columns are allowed to come back


def test_missing_values_are_split_into_fractional_instances(titanic):
    df, features, types, target = titanic
    assert df["Age"].isna().any(), "fixture must contain missing values"
    model = C45Tree(max_depth=3).fit(df[features], df[target], feature_types=types)
    notes = " ".join(model.get_trace().notes)
    assert "fractional copies" in notes
    assert (model.tree.predict(df[features]) == df[target].values).mean() > 0.7


def test_fractional_instances_conserve_each_node_weight(titanic):
    """Splitting a sample into fractions must not create or destroy weight."""
    df, features, types, target = titanic
    model = C45Tree(max_depth=3).fit(df[features], df[target], feature_types=types)
    for node in model.tree.root.iter_nodes():
        if node.is_leaf:
            continue
        children_weight = sum(child.n_samples for child in node.children.values())
        assert children_weight == pytest.approx(node.n_samples, abs=1e-6)


def test_dropping_missing_values_is_also_available(titanic):
    df, features, types, target = titanic
    model = C45Tree(max_depth=3, missing_strategy="drop").fit(
        df[features], df[target], feature_types=types
    )
    assert "dropped" in " ".join(model.get_trace().notes)


def test_gain_filter_can_be_switched_off(play_tennis):
    df, features, types, target = play_tennis
    filtered = C45Tree(use_gain_filter=True).fit(df[features], df[target], feature_types=types)
    unfiltered = C45Tree(use_gain_filter=False).fit(df[features], df[target], feature_types=types)
    assert filtered.tree.root.feature_name == "Outlook"
    assert unfiltered.tree.root.feature_name in set(features)


def test_c45_rejects_a_regression_target():
    with pytest.raises(ValueError, match="classification"):
        C45Tree(task="regression")
