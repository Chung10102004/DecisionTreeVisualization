"""2 · Build — watch the tree being constructed, one node at a time.

The left column shows the tree as it looked at the selected step; the right
column shows every number that produced that step, each with its formula and the
node's real values substituted in.
"""
from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import streamlit as st

from ...core.criteria import get_impurity
from ...core.metrics import accuracy, root_mean_squared_error, score_model
from ...core.node import SPLIT_NUMERIC_BINARY
from ...core.trace import ACTION_LEAF, STOP_REASONS, BuildStep
from ...viz import charts
from ...viz.mathfmt import MathFormatter
from ...viz.tree_dot import RenderOptions
from ..components import (
    build_model,
    render_algorithm_config,
    render_blocks,
    render_class_legend,
    render_formula,
    render_sidebar_status,
    show_figure,
    render_step_navigator,
    render_tree_view,
    require_data,
    show_dataframe,
)
from ..state import get_state

FORMATTER = MathFormatter()


# --------------------------------------------------------------------------- #
def _train(state, config) -> None:
    prepared = state.prepared
    X_train, y_train = state.training_data()
    algorithm = config["algorithm"]
    model = build_model(algorithm, config, task=prepared.task)
    with st.spinner(f"Growing the {algorithm} tree…"):
        started = time.perf_counter()
        model.fit(X_train, y_train, prepared.feature_names, prepared.feature_types)
        elapsed = time.perf_counter() - started
    state.set_model(model, model.get_trace())
    state.hyperparams = config
    st.session_state["_train_seconds"] = elapsed


def _final_tree_block(state) -> None:
    tree = state.original_tree
    prepared = state.prepared
    X_train, y_train = state.training_data()
    X_test, y_test = state.test_data()
    stats = tree.stats()

    st.subheader("Final tree")
    metrics = st.columns(5)
    metrics[0].metric("Leaves", stats["leaves"])
    metrics[1].metric("Depth", stats["depth"])
    metrics[2].metric("Nodes", stats["nodes"])
    if prepared.task == "regression":
        metrics[3].metric("Train RMSE",
                          f"{root_mean_squared_error(y_train, [float(v) for v in tree.predict(X_train)]):.3f}")
        metrics[4].metric("Test RMSE",
                          f"{root_mean_squared_error(y_test, [float(v) for v in tree.predict(X_test)]):.3f}"
                          if len(y_test) else "-")
    else:
        metrics[3].metric("Train accuracy", f"{accuracy(y_train, tree.predict(X_train)):.1%}")
        metrics[4].metric("Test accuracy",
                          f"{accuracy(y_test, tree.predict(X_test)):.1%}" if len(y_test) else "-")

    render_class_legend(tree)
    render_tree_view(tree, state, key="final_tree", show_downloads=False,
                     options=RenderOptions(orientation="TB", show_node_ids=True))
    st.caption(f"Trained in {st.session_state.get('_train_seconds', 0):.3f} s. "
               "Open **3 · Tree** for the full-size diagram, the rules, and pruning.")

    for note in state.trace.notes[:6]:
        st.caption(f"ℹ️ {note}")


# --------------------------------------------------------------------------- #
def _samples_table(state, step: BuildStep) -> None:
    prepared = state.prepared
    train_index, _test = state.split_indices()
    positions = [int(i) for i in step.sample_indices if int(i) < len(train_index)]
    if not positions:
        st.caption("These samples are fractional copies created for missing values.")
        return
    original = train_index[positions]
    frame = prepared.frame.iloc[original]
    with st.expander(f"The {len(frame)} training sample(s) sitting in this node", expanded=False):
        show_dataframe(frame)


def _candidate_detail(step: BuildStep, candidate) -> None:
    st.markdown(f"**Full calculation for `{candidate.describe()}`**")
    if not candidate.is_valid:
        st.error(f"This split is not usable: {candidate.reject_reason}")
    show_dataframe(FORMATTER.child_counts_dataframe(candidate))
    render_blocks(FORMATTER.candidate_chain(candidate))
    for key in ("threshold", "subset"):
        if key in candidate.computations:
            render_formula(candidate.computations[key])


def _explanation(state, step: BuildStep) -> None:
    trace = state.trace
    impurity = get_impurity(trace.impurity_name)

    st.info(step.narrative)

    st.markdown("**Where we are**")
    st.code(step.path_text(), language=None)
    _samples_table(state, step)

    st.divider()
    st.markdown("#### 1 · How impure is this node?")
    if step.class_counts:
        st.caption("Class counts here: " + step.counts_text())
    render_formula(step.impurity_computation, show_title=False)

    if step.action == ACTION_LEAF:
        st.divider()
        st.markdown("#### 2 · Why the node stops here")
        reason = STOP_REASONS.get(step.stop_reason or "", step.stop_reason or "no split was possible")
        st.warning(f"This node becomes a **leaf predicting `{step.prediction}`** because "
                   f"{reason}.")
        if step.stop_computation is not None:
            render_formula(step.stop_computation, show_title=False,
                           caption="The rule that fired, with the node's own numbers.")
        if step.candidates:
            with st.expander("Candidates that were scored before stopping", expanded=False):
                show_dataframe(FORMATTER.candidates_dataframe(step))
        return

    st.divider()
    st.markdown("#### 2 · Score every possible split")
    st.caption("Losing candidates are kept — select any row below to see its full working.")
    show_dataframe(FORMATTER.candidates_dataframe(step))
    show_figure(charts.plot_candidate_gains(step, trace.score_name))

    names = [c.feature_name for c in step.ranked_candidates]
    default = names.index(step.chosen.feature_name) if step.chosen and step.chosen.feature_name in names else 0
    chosen_name = st.selectbox("Inspect a candidate", names, index=default,
                               key=f"cand_{step.step_id}")
    candidate = next(c for c in step.ranked_candidates if c.feature_name == chosen_name)

    st.divider()
    st.markdown("#### 3 · The working, term by term")
    _candidate_detail(step, candidate)

    if candidate.split_type == SPLIT_NUMERIC_BINARY and candidate.threshold_scan:
        show_figure(charts.plot_threshold_scan(candidate, trace.score_name))

    st.divider()
    st.markdown("#### 4 · Making the choice")
    render_formula(step.selection_computation, show_title=False)
    show_figure(charts.plot_impurity_breakdown(step, impurity.display))


# --------------------------------------------------------------------------- #
def render() -> None:
    state = get_state()
    render_sidebar_status(state)
    require_data(state)

    st.title("2 · Build the tree, step by step")
    st.caption("Every number below is shown as *formula → the node's own values → result*, so the "
               "tree is never asking you to take a figure on trust.")

    with st.expander("Algorithm and hyper-parameters", expanded=not state.is_model_ready()):
        config = render_algorithm_config(state, key_prefix="build")
        if st.button("Train", type="primary", key="build_train"):
            _train(state, config)
            st.rerun()

    if not state.is_model_ready():
        st.info("Set the options above and press **Train** to grow a tree.")
        return

    _final_tree_block(state)
    st.divider()

    st.subheader("Replay the construction")
    trace = state.trace
    current = render_step_navigator(trace, state, key="build_nav")
    step = trace.steps[current]

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**The tree at this step**")
        st.caption("Solid = already decided · dashed = queued, not yet processed · "
                   "red outline = the node being processed now.")
        partial = trace.tree_at_step(current)
        render_tree_view(
            partial, state, key=f"step_{current}", show_downloads=False,
            highlight_node=step.node_id,
            options=RenderOptions(orientation="TB", show_node_ids=True, show_distribution_bar=True),
        )
        pending = trace.frontier_at_step(current)
        st.caption(f"{len(pending)} node(s) still queued."
                   if pending else "The queue is empty — the tree is complete.")

    with right:
        _explanation(state, step)

    st.divider()
    with st.expander("Full build log", expanded=False):
        show_dataframe(trace.summary_dataframe())
        for note in trace.notes:
            st.caption(f"ℹ️ {note}")

    st.subheader("Take the working away with you")
    columns = st.columns(2)
    impurity = get_impurity(trace.impurity_name)
    columns[0].download_button(
        "Download this step's worked calculation (.md)",
        FORMATTER.worked_example(step, impurity.display, trace.score_name),
        file_name=f"step_{step.step_id + 1}_node_{step.node_id}.md",
        mime="text/markdown", width="stretch",
    )
    columns[1].download_button(
        "Download every step as one document (.md)",
        FORMATTER.worked_example_document(
            trace, title=f"{trace.algorithm} on {state.dataset.name} — worked example"),
        file_name=f"{trace.algorithm.lower().replace('.', '')}_worked_example.md",
        mime="text/markdown", width="stretch", type="primary",
    )
    with st.expander("Preview the worked calculation for this step", expanded=False):
        st.markdown(FORMATTER.worked_example(step, impurity.display, trace.score_name))
