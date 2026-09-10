"""ID3 on PlayTennis — the tree every textbook prints, node for node."""
from __future__ import annotations

import pytest

from src.core.id3 import ID3Tree


@pytest.fixture
def fitted(play_tennis):
    df, features, types, target = play_tennis
    model = ID3Tree().fit(df[features], df[target], feature_types=types)
    return model, df, features, target


def test_root_splits_on_outlook_with_the_expected_gain(fitted):
    model, *_ = fitted
    root = model.tree.root
    assert root.feature_name == "Outlook"
    assert root.gain == pytest.approx(0.247, abs=1e-3)
    assert root.n_samples == 14
    assert root.impurity == pytest.approx(0.940, abs=1e-3)


def test_tree_shape_matches_the_textbook(fitted):
    model, *_ = fitted
    stats = model.tree.stats()
    assert stats["leaves"] == 5
    assert stats["depth"] == 2
    assert stats["nodes"] == 8


def test_overcast_branch_is_a_pure_yes_leaf(fitted):
    model, *_ = fitted
    overcast = model.tree.root.children["Overcast"]
    assert overcast.is_leaf
    assert overcast.prediction == "Yes"
    assert overcast.stop_reason == "pure"
    assert overcast.n_samples == 4
    assert overcast.impurity == pytest.approx(0.0)


def test_sunny_splits_on_humidity_and_rain_on_wind(fitted):
    model, *_ = fitted
    assert model.tree.root.children["Sunny"].feature_name == "Humidity"
    assert model.tree.root.children["Rain"].feature_name == "Wind"


def test_it_reproduces_the_training_data_perfectly(fitted):
    model, df, features, target = fitted
    assert (model.tree.predict(df[features]) == df[target].values).all()


def test_every_leaf_is_pure_so_the_five_rules_are_certain(fitted):
    model, *_ = fitted
    rules = model.tree.to_rules()
    assert len(rules) == 5
    assert all(rule.confidence == pytest.approx(1.0) for rule in rules)


def test_id3_consumes_a_feature_once_it_has_been_used(fitted):
    model, *_ = fitted
    for node in model.tree.root.iter_nodes():
        if node.is_leaf:
            continue
        conditions = model.tree.path_conditions(node.node_id)
        assert all(not condition.startswith(node.feature_name) for condition in conditions)


def test_id3_is_fooled_by_an_id_column(play_tennis_with_id):
    """Information Gain alone prefers a column of unique values — this is the bug
    that Gain Ratio exists to fix, so we pin the behaviour down."""
    df, features, types, target = play_tennis_with_id
    model = ID3Tree().fit(df[features], df[target], feature_types=types)
    assert model.tree.root.feature_name == "Day"
    assert model.tree.root.gain == pytest.approx(0.940, abs=1e-3)


def test_max_depth_turns_deeper_nodes_into_leaves(play_tennis):
    df, features, types, target = play_tennis
    model = ID3Tree(max_depth=1).fit(df[features], df[target], feature_types=types)
    assert model.tree.depth() == 1
    reasons = {n.stop_reason for n in model.tree.root.leaves()}
    assert "max_depth" in reasons


def test_lenses_dataset_trains_and_generalises(tmp_path):
    import pandas as pd
    from tests.conftest import DATA

    df = pd.read_csv(DATA / "lenses.csv")
    features = ["Age", "SpectaclePrescrip", "Astigmatism", "TearProdRate"]
    model = ID3Tree().fit(df[features], df["Lenses"], feature_types={f: "categorical" for f in features})
    assert model.tree.root.feature_name == "TearProdRate"
    assert (model.tree.predict(df[features]) == df["Lenses"].values).mean() >= 0.9


def test_a_numeric_column_is_flagged_in_the_trace(iris):
    """ID3 cannot compare numbers; a freshly built model must say so, not crash.

    The warning is recorded while the splitter is constructed, which happens
    before any node is processed — the trace has to be ready by then.
    """
    df, features, types, target = iris
    model = ID3Tree(max_depth=2).fit(df[features], df[target], feature_types=types)
    notes = " ".join(model.get_trace().notes)
    assert "ID3 cannot compare numbers" in notes
    assert "PetalLength" in notes
