"""3 · Tree — the finished tree: diagram, rules, structure, pruning."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from ...core.metrics import accuracy, root_mean_squared_error
from ...core.pruning import (
    CostComplexityPruner,
    PessimisticPruner,
    ReducedErrorPruner,
)
from ...core.trace import STOP_REASONS
from ...viz import charts
from ...viz.mathfmt import MathFormatter
from ...viz.tree_dot import RenderOptions, TreeDotRenderer, render_text_tree
from ..components import (
    render_class_legend,
    render_formula,
    render_node_card,
    render_sidebar_status,
    show_figure,
    render_tree_view,
    require_model,
    show_dataframe,
)
from ..state import get_state

FORMATTER = MathFormatter()


# --------------------------------------------------------------------------- #
def _display_controls(state) -> RenderOptions:
    with st.expander("Display options", expanded=False):
        left, middle, right = st.columns(3)
        options = RenderOptions(
            show_impurity=left.checkbox("Show impurity", True, key="tv_imp"),
            show_samples=left.checkbox("Show sample counts", True, key="tv_samp"),
            show_counts=middle.checkbox("Show class counts", True, key="tv_counts"),
            show_percentages=middle.checkbox("As percentages", False, key="tv_pct"),
            show_distribution_bar=right.checkbox("Show distribution bar", True, key="tv_bar"),
            show_node_ids=right.checkbox("Show node ids", True, key="tv_ids"),
        )
        options.orientation = st.radio(
            "Layout", ["TB", "LR"], horizontal=True, key="tv_orient",
            help="TB draws the tree downwards, LR from left to right (better for deep trees).",
        )
        depth = state.tree.depth()
        if depth <= 1:
            # A stump has nothing to collapse, and a slider whose min equals its max is an error.
            st.caption("The tree is only one level deep, so there is nothing to collapse.")
            options.max_display_depth = None
        else:
            limit = st.slider(
                "Show only the first N levels", 1, depth, depth, key="tv_depth",
                help="Hidden subtrees are collapsed into a '… n more nodes' note so a large tree "
                     "stays readable.",
            )
            options.max_display_depth = None if limit >= depth else int(limit)
    state.update_render_options(**{
        "show_impurity": options.show_impurity, "show_samples": options.show_samples,
        "show_counts": options.show_counts, "show_percentages": options.show_percentages,
        "show_distribution_bar": options.show_distribution_bar,
        "show_node_ids": options.show_node_ids, "orientation": options.orientation,
    })
    return options


def _node_options(tree) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for node in sorted(tree.root.iter_nodes(), key=lambda n: n.node_id):
        kind = f"leaf → {node.prediction}" if node.is_leaf else node.test_description()
        labels[f"#{node.node_id} (depth {node.depth}) · {kind}"] = node.node_id
    return labels


def _tab_diagram(state) -> None:
    tree = state.tree
    options = _display_controls(state)

    node_labels = _node_options(tree)
    left, right = st.columns([3, 2], gap="large")

    with right:
        st.markdown("**Node inspector**")
        keys = list(node_labels)
        selected_label = st.selectbox("Node", keys, index=0, key="tv_node")
        node_id = node_labels[selected_label]
        state.selected_node = node_id
        node = tree.node(node_id)
        render_node_card(node, tree)

        step = state.trace.step_for_node(node_id) if state.trace else None
        if step is not None:
            render_formula(step.impurity_computation, caption="Computed from the counts above.")
            if step.chosen is not None:
                render_formula(step.chosen.computations.get("gain"),
                               caption="Why this split was chosen over the alternatives.")
            elif step.stop_computation is not None:
                render_formula(step.stop_computation)
        else:
            st.caption("This node was created by pruning, so it has no build step of its own.")

        focus = st.checkbox("Draw only the subtree rooted here", value=False, key="tv_focus")

    with left:
        render_class_legend(tree)
        renderer = TreeDotRenderer(options)
        dot = renderer.render(
            tree,
            highlight_node=node_id,
            max_display_depth=options.max_display_depth,
            focus_node_id=node_id if focus else None,
        )
        st.graphviz_chart(dot, use_container_width=False)
        st.caption("Fill colour = majority class · colour intensity = purity · "
                   "the selected node has a red outline.")

        downloads = st.columns(3)
        downloads[0].download_button("Download .dot", dot, file_name="tree.dot",
                                     mime="text/vnd.graphviz", width="stretch")
        from ...viz.tree_dot import dot_to_svg, has_graphviz_binary

        svg = dot_to_svg(dot) if has_graphviz_binary() else None
        if svg:
            downloads[1].download_button("Download .svg", svg, file_name="tree.svg",
                                         mime="image/svg+xml", width="stretch")
        else:
            downloads[1].caption("`sudo apt install graphviz` enables SVG/PNG export.")
        downloads[2].download_button("Download .json", json.dumps(tree.to_dict(), indent=2, default=str),
                                     file_name="tree.json", mime="application/json", width="stretch")


def _tab_rules(state) -> None:
    tree = state.tree
    st.markdown("**Indented text tree** — always legible, however large the tree gets.")
    text = render_text_tree(tree)
    st.code(text, language=None)
    st.download_button("Download the text tree (.txt)", text, file_name="tree.txt",
                       mime="text/plain")

    st.divider()
    st.markdown("**Decision rules** — one per leaf, in IF … THEN form.")
    rules = tree.to_rules()
    frame = pd.DataFrame(
        [
            {
                "Leaf": f"#{rule.leaf_id}",
                "Rule": rule.as_text(),
                "Samples": rule.n_samples,
                "Confidence": f"{rule.confidence:.1%}",
            }
            for rule in rules
        ]
    )
    show_dataframe(frame)
    st.download_button("Download the rules (.txt)", "\n".join(r.as_text() for r in rules),
                       file_name="rules.txt", mime="text/plain")
    weak = [r for r in rules if r.confidence < 0.7]
    if weak:
        st.warning(f"{len(weak)} rule(s) are backed by a mixed leaf (confidence below 70%). "
                   "Those are the predictions to distrust first.")


def _tab_structure(state) -> None:
    tree = state.tree
    stats = tree.stats()
    metrics = st.columns(5)
    metrics[0].metric("Nodes", stats["nodes"])
    metrics[1].metric("Leaves", stats["leaves"])
    metrics[2].metric("Depth", stats["depth"])
    metrics[3].metric("Avg samples / leaf", f"{stats['avg_samples_per_leaf']:.1f}")
    metrics[4].metric("Training rows", stats["n_train"])

    left, right = st.columns([1, 1], gap="large")
    with left:
        show_figure(charts.plot_feature_importance(tree.feature_importances()))
        st.caption("Importance = the impurity each feature removed, weighted by how many samples "
                   "passed through the node, normalised to 100%.")
    with right:
        depths = [leaf.depth for leaf in tree.root.leaves()]
        counts = pd.Series(depths).value_counts().sort_index()
        show_figure(charts.plot_class_distribution(
            {f"depth {d}": int(c) for d, c in counts.items()},
            [f"depth {d}" for d in counts.index],
            title="How deep the leaves sit"))

    st.markdown("**Every node**")
    rows = []
    for node in sorted(tree.root.iter_nodes(), key=lambda n: n.node_id):
        rows.append(
            {
                "Node": node.node_id,
                "Depth": node.depth,
                "Parent": node.parent_id if node.parent_id is not None else "-",
                "Branch from parent": node.branch_label or "-",
                "Test": node.test_description() or "(leaf)",
                "Samples": node.n_samples,
                "Impurity": round(node.impurity, 4),
                "Class counts": ", ".join(f"{k}={v:g}" for k, v in node.class_counts.items()) or "-",
                "Prediction": node.prediction,
                "Stop reason": node.stop_reason or "-",
            }
        )
    show_dataframe(pd.DataFrame(rows))


def _tab_pruning(state) -> None:
    tree = state.original_tree
    prepared = state.prepared
    X_test, y_test = state.test_data()

    st.markdown("Pruning trades a little training fit for a smaller, more honest tree.")
    if len(y_test) == 0 and prepared.task == "classification":
        st.warning("Reduced-error and cost-complexity pruning need a validation set. "
                   "Give the test set a non-zero share on the Data page, or use pessimistic "
                   "pruning, which needs no held-out data.")

    method = st.radio(
        "Method",
        ["Reduced error (needs validation data)",
         "Cost-complexity (CART, alpha sequence)",
         "Pessimistic / error-based (C4.5, no validation data)"],
        key="prune_method_choice",
    )

    allow_ties = True
    if method.startswith("Reduced"):
        allow_ties = st.checkbox(
            "Prune when accuracy merely ties (textbook behaviour)", value=True,
            key="prune_ties",
            help="On a small validation set a tie is easy to reach, so the textbook rule can "
                 "collapse the tree all the way to a single leaf. Untick to prune only when "
                 "accuracy strictly improves.",
        )

    if st.button("Run pruning", type="primary", key="prune_run"):
        if method.startswith("Reduced"):
            pruner = ReducedErrorPruner(allow_ties=allow_ties)
            pruned, steps = pruner.prune(tree, X_test, y_test)
        elif method.startswith("Cost"):
            pruner = CostComplexityPruner(task=prepared.task)
            pruned, steps = pruner.prune(tree, X_test, y_test)
        else:
            pruner = PessimisticPruner()
            pruned, steps = pruner.prune(tree)
        state.pruned_tree = pruned
        state.prune_steps = steps
        state.prune_method = pruner.name
        state.use_pruned = True
        st.rerun()

    if state.prune_steps is None:
        return

    pruned = state.pruned_tree
    st.success(f"{state.prune_method}: {tree.count_leaves()} leaves → {pruned.count_leaves()} leaves.")
    if pruned.count_leaves() <= 1:
        st.warning(
            "Pruning collapsed the tree to a single leaf. That is a real result, not a bug: on a "
            "validation set this small, always predicting the majority class scores as well as "
            "any subtree, so every split looks unnecessary. Enlarge the validation set on the "
            "Data page, or require a strict improvement rather than a tie."
        )
    state.use_pruned = st.toggle(
        "Use the pruned tree everywhere in the app", value=bool(state.use_pruned),
        key="prune_use",
        help="Turn this on and the Predict and Evaluate pages will use the pruned tree.",
    )

    if prepared.task == "classification" and len(y_test):
        columns = st.columns(2)
        columns[0].metric("Test accuracy before", f"{accuracy(y_test, tree.predict(X_test)):.1%}")
        columns[1].metric("Test accuracy after", f"{accuracy(y_test, pruned.predict(X_test)):.1%}")

    candidates = [s for s in state.prune_steps if s.action == "candidate"]
    if candidates:
        rows = [{"alpha": s.alpha, "leaves": s.leaves_after, "score": s.metric_after}
                for s in candidates]
        show_figure(charts.plot_alpha_curve(rows))

    st.markdown("**Before and after**")
    left, right = st.columns(2, gap="large")
    with left:
        st.caption(f"Original — {tree.count_leaves()} leaves")
        render_tree_view(tree, state, key="prune_before", show_downloads=False,
                         options=RenderOptions(show_distribution_bar=False, show_counts=False))
    with right:
        st.caption(f"Pruned — {pruned.count_leaves()} leaves")
        render_tree_view(pruned, state, key="prune_after", show_downloads=False,
                         options=RenderOptions(show_distribution_bar=False, show_counts=False))

    st.markdown("**Every decision the pruner made**")
    for step in state.prune_steps:
        icon = {"prune": "✂️", "keep": "🌱", "candidate": "•"}.get(step.action, "•")
        with st.expander(f"{icon} step {step.step_id + 1} · {step.action} · {step.reason[:90]}…",
                         expanded=False):
            st.write(step.reason)
            if step.computation is not None:
                render_formula(step.computation, show_title=False)


def render() -> None:
    state = get_state()
    render_sidebar_status(state)
    require_model(state)

    st.title("3 · The finished tree")
    if state.use_pruned and state.pruned_tree is not None:
        st.caption(f"Showing the **pruned** tree ({state.prune_method}). "
                   "Switch it off in the Pruning tab to go back to the original.")
    else:
        st.caption("The tree exactly as the algorithm left it. Select any node on the right to "
                   "see the numbers behind it.")

    tabs = st.tabs(["Tree diagram", "Text & rules", "Structure & importance", "Pruning"])
    with tabs[0]:
        _tab_diagram(state)
    with tabs[1]:
        _tab_rules(state)
    with tabs[2]:
        _tab_structure(state)
    with tabs[3]:
        _tab_pruning(state)
