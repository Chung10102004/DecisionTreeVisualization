"""Impurity measures, checked against numbers a textbook prints."""
from __future__ import annotations

import math

import pytest

from src.core import criteria as crit


def test_entropy_of_play_tennis_is_the_textbook_value(play_tennis):
    df, _features, _types, target = play_tennis
    assert crit.entropy(df[target]) == pytest.approx(0.940, abs=5e-4)


def test_entropy_formula_shows_the_real_fractions(play_tennis):
    df, _features, _types, target = play_tennis
    comp = crit.entropy_c(df[target].values)
    assert r"\frac{9}{14}" in comp.substituted
    assert r"\frac{5}{14}" in comp.substituted
    assert comp.result == pytest.approx(0.940, abs=5e-4)


def test_gini_of_play_tennis():
    counts = ["Yes"] * 9 + ["No"] * 5
    expected = 1 - (9 / 14) ** 2 - (5 / 14) ** 2
    assert crit.gini(counts) == pytest.approx(expected)
    assert crit.gini_c(counts).result == pytest.approx(expected)


@pytest.mark.parametrize("measure", ["entropy", "gini", "error"])
def test_pure_and_empty_nodes_have_zero_impurity(measure):
    impurity = crit.get_impurity(measure)
    assert impurity.value(["a"] * 7) == pytest.approx(0.0)
    assert impurity.value([]) == pytest.approx(0.0)
    assert impurity.value(["a"]) == pytest.approx(0.0)


def test_binary_entropy_peaks_at_one():
    assert crit.entropy(["a", "b"]) == pytest.approx(1.0)


def test_information_gain_matches_the_classic_play_tennis_numbers(play_tennis):
    df, features, _types, target = play_tennis
    impurity = crit.get_impurity("entropy")
    parent = impurity.explain(df[target].values)
    expected = {"Outlook": 0.247, "Humidity": 0.151, "Wind": 0.048, "Temperature": 0.029}

    for feature, want in expected.items():
        children = [(v, df.loc[df[feature] == v, target].values, None) for v in sorted(df[feature].unique())]
        weighted = crit.weighted_impurity_c(children, impurity)
        gain = crit.impurity_gain_c(parent, weighted, feature, impurity)
        assert gain.result == pytest.approx(want, abs=1e-3), feature
        assert "0.940" in gain.substituted


def test_split_info_grows_with_the_number_of_branches():
    two = crit.split_info_c([("a", ["x"] * 7, None), ("b", ["y"] * 7, None)], "F")
    fourteen = crit.split_info_c([(str(i), ["x"], None) for i in range(14)], "F")
    assert two.result == pytest.approx(1.0)
    assert fourteen.result == pytest.approx(math.log2(14))
    assert fourteen.result > two.result


def test_gain_ratio_divides_gain_by_split_info(play_tennis):
    df, _features, _types, target = play_tennis
    impurity = crit.get_impurity("entropy")
    parent = impurity.explain(df[target].values)
    children = [(v, df.loc[df["Outlook"] == v, target].values, None) for v in sorted(df["Outlook"].unique())]
    weighted = crit.weighted_impurity_c(children, impurity)
    gain = crit.impurity_gain_c(parent, weighted, "Outlook", impurity)
    info = crit.split_info_c(children, "Outlook")
    ratio = crit.gain_ratio_c(gain, info, "Outlook")
    assert ratio.result == pytest.approx(gain.result / info.result)
    assert ratio.result == pytest.approx(0.156, abs=1e-3)


def test_gain_ratio_with_zero_split_info_is_reported_not_crashed():
    impurity = crit.get_impurity("entropy")
    children = [("only", ["a", "b"], None)]
    parent = impurity.explain(["a", "b"])
    weighted = crit.weighted_impurity_c(children, impurity)
    gain = crit.impurity_gain_c(parent, weighted, "F", impurity)
    info = crit.split_info_c(children, "F")
    ratio = crit.gain_ratio_c(gain, info, "F")
    assert ratio.result == 0.0
    assert ratio.note and "undefined" in ratio.note


def test_weighted_counts_are_supported_for_c45_fractional_instances():
    y = ["a", "b"]
    weights = [0.75, 0.25]
    counts = crit.class_counts(y, weights)
    assert counts == {"a": 0.75, "b": 0.25}
    assert crit.entropy(y, weights) == pytest.approx(-0.75 * math.log2(0.75) - 0.25 * math.log2(0.25))


def test_regression_impurities():
    assert crit.mse([1, 2, 3, 4]) == pytest.approx(1.25)
    assert crit.mae([1, 2, 3, 4]) == pytest.approx(1.0)
    assert crit.mse_c([1, 2, 3, 4]).result == pytest.approx(1.25)
