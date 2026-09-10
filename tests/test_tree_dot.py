"""The DOT renderer: valid syntax, right node count, correct highlighting."""
from __future__ import annotations

import re

import pytest

from src.core.cart import CARTTree
from src.core.id3 import ID3Tree
from src.viz.tree_dot import RenderOptions, TreeDotRenderer, render_text_tree


@pytest.fixture
def tree(play_tennis):
    df, features, types, target = play_tennis
    return ID3Tree().fit(df[features], df[target], feature_types=types)


def _node_ids(dot: str) -> set:
    return {m.group(1) for m in re.finditer(r"^\s*(n\d+) \[label=<", dot, re.MULTILINE)}


def test_dot_is_syntactically_plausible(tree):
    dot = TreeDotRenderer().render(tree.tree)
    assert dot.startswith("digraph DecisionTree {")
    assert dot.rstrip().endswith("}")
    assert dot.count("{") == dot.count("}")
    assert dot.count("<TABLE") == dot.count("</TABLE>")


def test_every_node_appears_exactly_once(tree):
    dot = TreeDotRenderer().render(tree.tree)
    assert len(_node_ids(dot)) == tree.tree.count_nodes()


def test_every_edge_connects_two_declared_nodes(tree):
    dot = TreeDotRenderer().render(tree.tree)
    declared = _node_ids(dot)
    edges = re.findall(r"^\s*(n\d+) -> (n\d+)", dot, re.MULTILINE)
    assert edges
    for source, target in edges:
        assert source in declared and target in declared


def test_awkward_feature_names_do_not_break_the_label(play_tennis):
    df, features, types, target = play_tennis
    renamed = df.rename(columns={"Outlook": "Sky & <cloud> \"state\""})
    columns = ["Sky & <cloud> \"state\"", "Temperature", "Humidity", "Wind"]
    model = ID3Tree().fit(renamed[columns], renamed[target],
                          feature_types={c: "categorical" for c in columns})
    dot = TreeDotRenderer().render(model.tree)
    assert "&amp;" in dot and "&lt;cloud&gt;" in dot
    assert dot.count("{") == dot.count("}")
    # The raw characters must not survive into the label.
    assert " & " not in dot.replace("&amp;", "").replace("&lt;", "").replace("&gt;", "")


def test_max_display_depth_truncates_and_says_so(iris):
    df, features, types, target = iris
    model = CARTTree(max_depth=4).fit(df[features], df[target], feature_types=types)
    full = TreeDotRenderer().render(model.tree)
    clipped = TreeDotRenderer().render(model.tree, max_display_depth=1)
    assert len(_node_ids(clipped)) < len(_node_ids(full))
    assert "more node(s) hidden" in clipped


def test_focus_node_renders_only_that_subtree(tree):
    node = tree.tree.root.children["Sunny"]
    dot = TreeDotRenderer().render(tree.tree, focus_node_id=node.node_id)
    assert len(_node_ids(dot)) == node.count_nodes()
    assert f"n{tree.tree.root.node_id} [" not in dot


def test_highlight_path_marks_exactly_the_predicted_route(tree):
    trace = tree.explain({"Outlook": "Sunny", "Temperature": "Cool",
                          "Humidity": "High", "Wind": "Weak"})
    dot = TreeDotRenderer().render(tree.tree, highlight_path=trace.path_node_ids,
                                   highlight_edges=trace.path_edges())
    for node_id in trace.path_node_ids:
        line = next(l for l in dot.splitlines() if l.strip().startswith(f"n{node_id} [label"))
        assert "#e34948" in line.lower()
    off_path = set(tree.tree.nodes_by_id) - set(trace.path_node_ids)
    for node_id in off_path:
        line = next(l for l in dot.splitlines() if l.strip().startswith(f"n{node_id} [label"))
        assert "#e34948" not in line.lower()


def test_pending_nodes_are_drawn_dashed(tree):
    trace = tree.get_trace()
    partial = trace.tree_at_step(0)
    dot = TreeDotRenderer().render(partial)
    assert "dashed" in dot
    assert "not expanded yet" in dot


def test_display_toggles_actually_remove_content(tree):
    with_all = TreeDotRenderer(RenderOptions()).render(tree.tree)
    minimal = TreeDotRenderer(RenderOptions(
        show_impurity=False, show_samples=False, show_counts=False,
        show_distribution_bar=False, show_node_ids=False)).render(tree.tree)
    assert "entropy =" in with_all and "entropy =" not in minimal
    assert "samples =" in with_all and "samples =" not in minimal
    assert len(minimal) < len(with_all)


def test_text_tree_lists_every_node(tree):
    text = render_text_tree(tree.tree)
    assert len([line for line in text.splitlines() if line.strip()]) == tree.tree.count_nodes()
    assert "Outlook = Overcast" in text


def test_legend_covers_every_class(tree):
    legend = TreeDotRenderer().legend_dot(tree.tree)
    for label in tree.tree.class_names:
        assert str(label) in legend
