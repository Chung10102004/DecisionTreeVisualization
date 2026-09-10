"""The build trace: one step per node, and a valid partial tree at every step."""
from __future__ import annotations

import pytest

from src.core.cart import CARTTree
from src.core.id3 import ID3Tree
from src.core.trace import ACTION_LEAF, ACTION_SPLIT


@pytest.fixture
def traced(play_tennis):
    df, features, types, target = play_tennis
    model = ID3Tree().fit(df[features], df[target], feature_types=types)
    return model.get_trace()


def test_there_is_exactly_one_step_per_node(traced):
    assert len(traced.steps) == traced.tree.count_nodes()
    assert {s.node_id for s in traced.steps} == set(traced.tree.nodes_by_id)


def test_steps_are_recorded_in_breadth_first_order(traced):
    depths = [step.depth for step in traced.steps]
    assert depths == sorted(depths)


def test_every_step_explains_itself(traced):
    for step in traced.steps:
        assert step.narrative.strip()
        assert step.impurity_computation.substituted.strip()
        assert step.action in (ACTION_SPLIT, ACTION_LEAF)
        if step.action == ACTION_LEAF:
            assert step.stop_reason
        else:
            assert step.chosen is not None
            assert step.selection_computation is not None


def test_every_candidate_carries_a_full_computation_chain(traced):
    keys = {"parent_impurity", "weighted_impurity", "gain", "split_info", "gain_ratio"}
    for step in traced.steps:
        for candidate in step.candidates:
            assert keys <= set(candidate.computations)
            for computation in candidate.computations.values():
                assert computation.substituted.strip()


def test_losing_candidates_are_kept_not_discarded(traced):
    root_step = traced.steps[0]
    assert len(root_step.candidates) == 4
    assert {c.feature_name for c in root_step.candidates} == {
        "Outlook", "Temperature", "Humidity", "Wind"
    }


def test_partial_tree_grows_monotonically(traced):
    sizes = [traced.tree_at_step(k).count_nodes() for k in range(len(traced))]
    assert sizes == sorted(sizes)
    assert sizes[-1] == traced.tree.count_nodes()


def test_pending_nodes_shrink_to_zero(traced):
    pending = [len(traced.frontier_at_step(k)) for k in range(len(traced))]
    assert pending[0] > 0
    assert pending[-1] == 0


def test_partial_tree_never_shows_grandchildren_of_an_unprocessed_node(traced):
    for k in range(len(traced)):
        for node in traced.tree_at_step(k).root.iter_nodes():
            if node.is_pending:
                assert not node.children


def test_tree_at_step_is_clamped_to_valid_range(traced):
    assert traced.tree_at_step(-5).count_nodes() == 1
    assert traced.tree_at_step(10_000).count_nodes() == traced.tree.count_nodes()


def test_summary_table_has_one_row_per_step(traced):
    frame = traced.summary_dataframe()
    assert len(frame) == len(traced.steps)
    assert "Action" in frame.columns


def test_stop_conditions_record_the_inequality_that_fired(play_tennis):
    df, features, types, target = play_tennis
    model = ID3Tree(max_depth=1).fit(df[features], df[target], feature_types=types)
    stopped = [s for s in model.get_trace().steps if s.stop_reason == "max_depth"]
    assert stopped
    for step in stopped:
        assert step.stop_computation is not None
        assert "1" in step.stop_computation.substituted


def test_trace_works_for_cart_too(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=3).fit(df[features], df[target], feature_types=types)
    trace = model.get_trace()
    assert len(trace.steps) == trace.tree.count_nodes()
    assert all(step.narrative for step in trace.steps)
