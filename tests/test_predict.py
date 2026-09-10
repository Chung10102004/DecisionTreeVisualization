"""Prediction and its step-by-step explanation."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.base_tree import explain_sample
from src.core.cart import CARTTree
from src.core.id3 import ID3Tree


@pytest.fixture
def tennis_model(play_tennis):
    df, features, types, target = play_tennis
    model = ID3Tree().fit(df[features], df[target], feature_types=types)
    return model, df, features, target


def test_explanation_follows_the_expected_path(tennis_model):
    model, *_ = tennis_model
    sample = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Weak"}
    trace = model.explain(sample)
    assert [s.feature for s in trace.steps if s.feature] == ["Outlook", "Humidity"]
    assert trace.predicted_class == "No"
    assert trace.confidence == pytest.approx(1.0)
    assert trace.rule_text == "IF Outlook = Sunny AND Humidity = High THEN No"


def test_every_hop_states_the_comparison_with_real_numbers(tennis_model):
    model, *_ = tennis_model
    trace = model.explain({"Outlook": "Rain", "Temperature": "Mild",
                           "Humidity": "High", "Wind": "Strong"})
    for step in trace.steps:
        assert step.narrative.strip()
        if not step.is_leaf:
            assert step.comparison is not None
            assert step.comparison.substituted.strip()
    assert trace.probability_computation is not None
    assert trace.probability_computation.substituted.strip()


def test_explanation_agrees_with_predict(tennis_model):
    model, df, features, target = tennis_model
    for _index, row in df.iterrows():
        sample = {f: row[f] for f in features}
        assert model.explain(sample).predicted_class == model.tree.predict_one(sample)


def test_numeric_comparison_is_substituted(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=3).fit(df[features], df[target], feature_types=types)
    trace = model.explain({"SepalLength": 5.1, "SepalWidth": 3.5,
                           "PetalLength": 1.4, "PetalWidth": 0.2})
    first = trace.steps[0]
    assert "1.4" in first.comparison.substituted
    assert "2.45" in first.comparison.substituted
    assert trace.predicted_class == "setosa"


def test_an_unseen_category_falls_back_and_says_so(tennis_model):
    model, *_ = tennis_model
    trace = model.explain({"Outlook": "Snowy", "Temperature": "Cool",
                           "Humidity": "High", "Wind": "Weak"})
    assert trace.warnings
    assert any(step.unseen_value for step in trace.steps)
    assert trace.predicted_class is not None


def test_a_missing_value_falls_back_and_says_so(tennis_model):
    model, *_ = tennis_model
    trace = model.explain({"Outlook": None, "Temperature": "Cool",
                           "Humidity": "High", "Wind": "Weak"})
    assert any("missing" in warning for warning in trace.warnings)


def test_sample_can_be_given_positionally_or_as_a_dict(tennis_model):
    model, df, features, target = tennis_model
    row = df.iloc[0]
    positional = model.tree.predict_one([row[f] for f in features])
    as_dict = model.tree.predict_one({f: row[f] for f in features})
    assert positional == as_dict


def test_wrong_number_of_values_is_rejected(tennis_model):
    model, *_ = tennis_model
    with pytest.raises(ValueError, match="expects"):
        model.tree.predict_one(["Sunny", "Cool"])


def test_probabilities_sum_to_one(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=3).fit(df[features], df[target], feature_types=types)
    for probabilities in model.predict_proba(df[features].head(10)):
        assert sum(probabilities.values()) == pytest.approx(1.0)


def test_path_node_ids_form_a_real_root_to_leaf_chain(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=3).fit(df[features], df[target], feature_types=types)
    trace = explain_sample(model.tree, df[features].iloc[7])
    nodes = model.tree.nodes_by_id
    assert trace.path_node_ids[0] == model.tree.root.node_id
    for parent_id, child_id in zip(trace.path_node_ids, trace.path_node_ids[1:]):
        assert nodes[child_id].parent_id == parent_id
    assert trace.final_leaf_id == trace.path_node_ids[-1]
