"""Pruning: fewer leaves, no worse validation score, every step explained."""
from __future__ import annotations

import pytest

from src.core.cart import CARTTree
from src.core.metrics import accuracy, train_test_split
from src.core.pruning import (
    CostComplexityPruner,
    PessimisticPruner,
    ReducedErrorPruner,
    upper_error_bound,
)


@pytest.fixture
def overgrown(iris):
    df, features, types, target = iris
    train, valid = train_test_split(len(df), df[target], test_size=0.4, random_state=0)
    model = CARTTree(min_samples_leaf=1).fit(
        df.iloc[train][features], df.iloc[train][target], feature_types=types
    )
    return model.tree, df.iloc[valid][features], df.iloc[valid][target]


def test_reduced_error_pruning_shrinks_without_hurting(overgrown):
    tree, x_valid, y_valid = overgrown
    before = accuracy(y_valid, tree.predict(x_valid))
    pruned, steps = ReducedErrorPruner().prune(tree, x_valid, y_valid)
    after = accuracy(y_valid, pruned.predict(x_valid))
    assert pruned.count_leaves() <= tree.count_leaves()
    assert after >= before - 1e-9
    assert any(step.action == "prune" for step in steps)


def test_pruning_leaves_the_original_tree_untouched(overgrown):
    tree, x_valid, y_valid = overgrown
    leaves_before = tree.count_leaves()
    ReducedErrorPruner().prune(tree, x_valid, y_valid)
    assert tree.count_leaves() == leaves_before


def test_every_prune_step_explains_itself(overgrown):
    tree, x_valid, y_valid = overgrown
    _pruned, steps = ReducedErrorPruner().prune(tree, x_valid, y_valid)
    for step in steps:
        assert step.reason.strip()
        assert step.action in ("prune", "keep", "candidate")


def test_cost_complexity_alphas_increase_and_leaves_decrease(overgrown):
    tree, _x, _y = overgrown
    sequence = CostComplexityPruner().alpha_sequence(tree)
    alphas = [alpha for alpha, _tree, _leaves in sequence]
    leaves = [count for _alpha, _tree, count in sequence]
    assert leaves[0] == tree.count_leaves()
    assert leaves[-1] == 1
    assert leaves == sorted(leaves, reverse=True)
    assert alphas == sorted(alphas)


def test_cost_complexity_picks_a_subtree_from_its_own_sequence(overgrown):
    tree, x_valid, y_valid = overgrown
    pruned, steps = CostComplexityPruner().prune(tree, x_valid, y_valid)
    assert pruned.count_leaves() <= tree.count_leaves()
    assert all(step.alpha is not None for step in steps)
    assert accuracy(y_valid, pruned.predict(x_valid)) >= accuracy(y_valid, tree.predict(x_valid)) - 1e-9


def test_pessimistic_pruning_needs_no_validation_data(overgrown):
    tree, _x, _y = overgrown
    pruned, steps = PessimisticPruner().prune(tree)
    assert pruned.count_leaves() <= tree.count_leaves()
    assert steps


def test_upper_error_bound_is_pessimistic_and_shrinks_with_more_data():
    assert upper_error_bound(0, 2) > 0.0                       # 0/2 is not really 0% error
    assert upper_error_bound(0, 2) > upper_error_bound(0, 100)  # more evidence, less inflation
    assert upper_error_bound(1, 10) >= 1 / 10
