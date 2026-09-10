"""The formulas as the user sees them: symbolic, substituted, result."""
from __future__ import annotations

import re

import pytest

from src.core import criteria as crit
from src.core.cart import CARTTree
from src.core.id3 import ID3Tree
from src.viz.mathfmt import MathFormatter, markdown_table

FORMATTER = MathFormatter()


@pytest.fixture
def fitted(play_tennis):
    df, features, types, target = play_tennis
    model = ID3Tree().fit(df[features], df[target], feature_types=types)
    return model, model.get_trace()


def test_root_entropy_shows_the_real_fractions_and_result(fitted):
    _model, trace = fitted
    block = FORMATTER.render(trace.steps[0].impurity_computation)
    assert r"\frac{9}{14}" in block.substituted
    assert r"\frac{5}{14}" in block.substituted
    assert r"\log_2" in block.symbolic
    assert block.result == "0.940"


def test_information_gain_of_the_winner_renders_as_0_247(fitted):
    _model, trace = fitted
    gain = trace.steps[0].chosen.computations["gain"]
    block = FORMATTER.render(gain)
    assert block.result == "0.247"
    assert "0.940" in block.substituted        # the parent entropy is visible
    assert r"\frac{5}{14}" in block.substituted  # so are the branch weights
    assert "Outlook" in block.symbolic


def test_losing_candidates_also_carry_a_full_substituted_chain(fitted):
    _model, trace = fitted
    losers = [c for c in trace.steps[0].candidates if c.feature_name != "Outlook"]
    assert losers
    for candidate in losers:
        blocks = FORMATTER.candidate_chain(candidate)
        assert blocks
        for block in blocks:
            assert block.symbolic.strip()
            assert block.substituted.strip()
            assert block.result.strip()


def test_rendered_result_always_matches_the_value_the_algorithm_used(fitted):
    """A display rounding must never disagree with the number that drove the split."""
    _model, trace = fitted
    for step in trace.steps:
        for candidate in step.candidates:
            for computation in candidate.computations.values():
                rendered = FORMATTER.render(computation).result
                assert float(rendered) == pytest.approx(computation.result, abs=5e-4)


def test_every_computation_in_a_whole_fit_renders_without_blanks(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=3).fit(df[features], df[target], feature_types=types)
    trace = model.get_trace()
    seen = 0
    for step in trace.steps:
        for computation in [step.impurity_computation, step.selection_computation,
                            step.stop_computation]:
            if computation is None:
                continue
            block = FORMATTER.render(computation)
            assert block.symbolic.strip() and block.substituted.strip() and block.result.strip()
            seen += 1
    assert seen > 5


def test_no_latex_escape_leaks_into_a_math_symbol(fitted):
    """``S_{3}`` must stay a subscript, not become ``S\\_\\{3\\}``."""
    _model, trace = fitted
    for step in trace.steps:
        assert r"\_" not in step.impurity_computation.symbolic
        assert r"\{" not in step.impurity_computation.symbolic.replace(r"\{}", "")


def test_feature_names_with_awkward_characters_are_escaped():
    computation = crit.comparison_c("Size (cm) & weight_kg", 3.0, "<=", 5.0, True)
    assert r"\&" in computation.symbolic
    assert r"\_" in computation.symbolic


def test_argmax_block_lists_every_feature_and_names_the_winner(fitted):
    _model, trace = fitted
    block = FORMATTER.selection(trace.steps[0])
    for feature in ("Outlook", "Humidity", "Wind", "Temperature"):
        assert feature in block.substituted
    assert "Outlook" in (block.note or "")


def test_candidates_table_keeps_the_losers_and_ranks_them(fitted):
    _model, trace = fitted
    frame = FORMATTER.candidates_dataframe(trace.steps[0])
    assert len(frame) == 4
    assert frame.iloc[0]["Feature"] == "Outlook"
    assert list(frame["Gain"]) == sorted(frame["Gain"], reverse=True)


def test_stop_rule_is_rendered_as_a_substituted_inequality(play_tennis):
    df, features, types, target = play_tennis
    model = ID3Tree(max_depth=1).fit(df[features], df[target], feature_types=types)
    stopped = next(s for s in model.get_trace().steps if s.stop_reason == "max_depth")
    block = FORMATTER.render(stopped.stop_computation)
    assert r"\geq" in block.symbolic
    assert "1" in block.substituted


def test_worked_example_contains_the_three_lines_for_the_root(fitted):
    _model, trace = fitted
    text = FORMATTER.worked_example(trace.steps[0], "Entropy", "Information Gain")
    assert "Impurity before splitting" in text
    assert r"\frac{9}{14}" in text
    assert "0.940" in text
    assert "0.247" in text
    assert "Score every candidate split" in text


def test_worked_example_document_covers_every_step(fitted):
    _model, trace = fitted
    document = FORMATTER.worked_example_document(trace)
    for step in trace.steps:
        assert f"Node #{step.node_id}" in document
    assert document.startswith("# ")


def test_markdown_table_needs_no_tabulate(fitted):
    _model, trace = fitted
    table = markdown_table(FORMATTER.candidates_dataframe(trace.steps[0]))
    assert table.count("\n") >= 5
    assert table.startswith("| Rank")
