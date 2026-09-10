"""Loading, type inference, discretisation, and the external-CSV path."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src.datasets.loaders import (
    CATEGORICAL,
    NUMERIC,
    DatasetError,
    dataset_from_frame,
    infer_feature_types,
    infer_task,
    list_builtin,
    load_builtin,
    load_uploaded,
    read_table,
)
from src.datasets.preprocess import (
    EQUAL_FREQUENCY,
    MISSING_DROP,
    MISSING_IMPUTE,
    Preprocessor,
    make_bins,
    suggest_discretization,
)


def test_every_bundled_dataset_loads_and_is_consistent():
    catalogue = list_builtin()
    assert len(catalogue) >= 8
    for entry in catalogue:
        dataset = load_builtin(entry["name"])
        assert len(dataset.df) == entry["rows"]
        assert dataset.target in dataset.df.columns
        assert dataset.target not in dataset.features
        assert set(dataset.feature_types) >= set(dataset.features)


def test_unknown_dataset_name_is_rejected():
    with pytest.raises(KeyError):
        load_builtin("Definitely Not A Dataset")


def test_play_tennis_variants_differ_only_by_the_id_column():
    plain = load_builtin("PlayTennis")
    with_id = load_builtin("PlayTennis + Day (ID column)")
    assert "Day" not in plain.features
    assert "Day" in with_id.features


def test_type_inference_calls_a_low_cardinality_number_categorical():
    frame = pd.DataFrame({"code": [1, 2, 1, 2], "value": np.arange(4) * 1.5, "y": list("aabb")})
    types = infer_feature_types(frame, "y")
    assert types["code"] == CATEGORICAL
    assert types["value"] == NUMERIC


def test_task_inference():
    classification = pd.DataFrame({"x": range(30), "y": ["a", "b"] * 15})
    regression = pd.DataFrame({"x": range(30), "y": np.linspace(0, 5, 30)})
    assert infer_task(classification, "y") == "classification"
    assert infer_task(regression, "y") == "regression"


def test_semicolon_separated_file_is_detected():
    text = "a;b;target\n1;2;yes\n3;4;no\n"
    frame = read_table(io.BytesIO(text.encode()))
    assert list(frame.columns) == ["a", "b", "target"]
    assert len(frame) == 2


def test_helpful_errors_for_broken_files():
    with pytest.raises(DatasetError, match="empty"):
        read_table(io.BytesIO(b"   "))
    with pytest.raises(DatasetError, match="[Oo]nly one column"):
        read_table(io.BytesIO(b"single\n1\n2\n"))


def test_rows_with_no_target_are_dropped_and_reported():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": ["a", None, "b"]})
    dataset = dataset_from_frame(frame, "y")
    assert len(dataset.df) == 2
    assert any("Dropped" in note for note in dataset.notes)


def test_external_csv_round_trip(tmp_path):
    """The upload path the app uses, on a file it has never seen."""
    source = pd.read_csv("data/iris.csv")
    source.columns = ["a", "b", "c", "d", "kind"]
    path = tmp_path / "renamed.csv"
    source.to_csv(path, index=False)

    dataset = load_uploaded(str(path), target="kind")
    assert dataset.target == "kind"
    assert len(dataset.features) == 4
    assert all(dataset.type_of(f) == NUMERIC for f in dataset.features)
    prepared = Preprocessor().prepare(dataset)
    assert prepared.X.shape == (150, 4)
    assert sorted(prepared.class_names) == ["setosa", "versicolor", "virginica"]


def test_binning_covers_the_range_and_is_replayable():
    values = pd.Series([1.0, 2.0, 3.0, 4.0], name="v")
    binning = make_bins(values, 2, "equal_width")
    assert len(binning.labels) == 2
    assert binning.apply(1.0) == binning.labels[0]
    assert binning.apply(4.0) == binning.labels[-1]
    assert binning.apply(None) is None


def test_equal_frequency_bins_split_the_counts_evenly():
    values = pd.Series(list(range(100)), name="v")
    binning = make_bins(values, 4, EQUAL_FREQUENCY)
    assigned = [binning.apply(v) for v in values]
    sizes = pd.Series(assigned).value_counts()
    assert sizes.max() - sizes.min() <= 2


def test_discretisation_turns_numeric_columns_categorical_for_id3():
    dataset = load_builtin("Iris")
    assert suggest_discretization(dataset, "ID3") == dataset.numeric_features()
    assert suggest_discretization(dataset, "CART") == []

    preprocessor = Preprocessor(discretize=dataset.numeric_features(), n_bins=3)
    prepared = preprocessor.prepare(dataset)
    assert all(kind == CATEGORICAL for kind in prepared.feature_types.values())
    assert any("Discretised" in note for note in prepared.notes)


def test_the_same_binning_is_applied_to_a_new_sample():
    dataset = load_builtin("Iris")
    preprocessor = Preprocessor(discretize=["PetalLength"], n_bins=3)
    preprocessor.prepare(dataset)
    transformed = preprocessor.transform_sample({"PetalLength": 1.4, "PetalWidth": 0.2})
    assert isinstance(transformed["PetalLength"], str)
    assert transformed["PetalWidth"] == 0.2


def test_missing_value_strategies():
    dataset = load_builtin("Titanic (mini)")
    assert dataset.missing_counts()

    kept = Preprocessor().prepare(dataset)
    dropped = Preprocessor(missing_strategy=MISSING_DROP).prepare(dataset)
    imputed = Preprocessor(missing_strategy=MISSING_IMPUTE).prepare(dataset)

    assert len(dropped) < len(kept)
    assert len(imputed) == len(kept)
    age = imputed.feature_names.index("Age")
    assert not any(v is None for v in imputed.X[:, age])


def test_selecting_a_feature_subset():
    dataset = load_builtin("Iris")
    prepared = Preprocessor().prepare(dataset, features=["PetalLength", "PetalWidth"])
    assert prepared.feature_names == ["PetalLength", "PetalWidth"]
    assert prepared.X.shape[1] == 2

    with pytest.raises(ValueError, match="No feature columns"):
        Preprocessor().prepare(dataset, features=["nope"])
