"""2 · Build — watch the tree being constructed, like a video.

After **Train** the construction plays in a full-width player (``src/viz/theater``):
the tree fills the stage and grows as nodes are decided, and a subtitle strip
under it shows the one formula the current phase computes.  Play, pause, step,
scrub, replay and full-screen all happen in the browser — every frame is prepared
up front, so nothing here re-runs while the video plays.

The finished tree with its metrics, and the complete working for any node, sit
below the player in collapsed sections so they do not spoil the ending.
"""
from __future__ import annotations

import time
import uuid

import streamlit as st

from ...core.criteria import get_impurity
from ...core.metrics import accuracy, root_mean_squared_error
from ...core.node import SPLIT_NUMERIC_BINARY
from ...core.trace import ACTION_LEAF, STOP_REASONS, BuildStep
from ...viz import charts
from ...viz.mathfmt import MathFormatter
from ...viz.theater import theater_html
from ...viz.tree_dot import RenderOptions, has_graphviz_binary
from .. import narration
from ..components import (
    build_model,
    render_algorithm_config,
    render_blocks,
    render_class_legend,
    render_formula,
    render_sidebar_status,
    render_tree_view,
    require_data,
    show_dataframe,
    show_figure,
)
from ..state import AppState, get_state

FORMATTER = MathFormatter()

# The player is an iframe whose height Streamlit fixes in pixels; this stretches
# it to the viewport instead so the tree gets the room a video would.
_PLAYER_CSS = """
<style>
iframe[data-testid="stIFrame"] { height: calc(100vh - 210px) !important; min-height: 560px; }
</style>
"""
_PLAYER_FALLBACK_HEIGHT = 720


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _train(state: AppState, config) -> None:
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


def _embed(html: str) -> None:
    """``st.iframe`` (Streamlit ≥ 1.6x) or the older components API — same iframe either way."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=_PLAYER_FALLBACK_HEIGHT)
    else:  # pragma: no cover - older Streamlit
        import streamlit.components.v1 as components

        components.html(html, height=_PLAYER_FALLBACK_HEIGHT, scrolling=False)


def _theater(state: AppState) -> dict:
    """Build the player once per trained model and keep it in the session."""
    cached = state.build_theater
    if cached is None:
        with st.spinner("Preparing the animation…"):
            key = uuid.uuid4().hex[:12]
            cached = {"key": key, "html": theater_html(state.trace, key)}
        state.build_theater = cached
    return cached


# --------------------------------------------------------------------------- #
# Below the player: the finished tree, and the full working on demand
# --------------------------------------------------------------------------- #
def _final_tree_block(state: AppState) -> None:
    tree = state.original_tree
    prepared = state.prepared
    X_train, y_train = state.training_data()
    X_test, y_test = state.test_data()
    stats = tree.stats()

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


def _samples_table(state: AppState, step: BuildStep) -> None:
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


def _why(text: str) -> None:
    if text:
        st.markdown(f"💡 *Why this formula, here:* {text}")


def _full_working(state: AppState, step: BuildStep) -> None:
    """Everything the player summarised for one node, with the charts and the why."""
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
    _why(narration.why_impurity(trace))

    if step.action == ACTION_LEAF:
        st.divider()
        st.markdown("#### 2 · Why the node stops here")
        reason = STOP_REASONS.get(step.stop_reason or "", step.stop_reason or "no split was possible")
        st.warning(f"This node becomes a **leaf predicting `{step.prediction}`** because "
                   f"{reason}.")
        if step.stop_computation is not None:
            render_formula(step.stop_computation, show_title=False,
                           caption="The rule that fired, with the node's own numbers.")
        _why(narration.why_stop(step, trace))
        if step.candidates:
            with st.expander("Candidates that were scored before stopping", expanded=False):
                show_dataframe(FORMATTER.candidates_dataframe(step))
        return

    st.divider()
    st.markdown("#### 2 · Score every possible split")
    st.caption("Losing candidates are kept — select any row below to see its full working.")
    show_dataframe(FORMATTER.candidates_dataframe(step))
    show_figure(charts.plot_candidate_gains(step, trace.score_name))
    _why(narration.why_scoring(trace))

    names = [c.feature_name for c in step.ranked_candidates]
    default = names.index(step.chosen.feature_name) if step.chosen and step.chosen.feature_name in names else 0
    st.divider()
    st.markdown("#### 3 · The working, term by term")
    chosen_name = st.selectbox("Inspect a candidate", names, index=default,
                               key=f"cand_{step.step_id}")
    candidate = next(c for c in step.ranked_candidates if c.feature_name == chosen_name)
    _candidate_detail(step, candidate)
    if candidate.split_type == SPLIT_NUMERIC_BINARY and candidate.threshold_scan:
        show_figure(charts.plot_threshold_scan(candidate, trace.score_name))
    _why(narration.why_working(trace, candidate))

    st.divider()
    st.markdown("#### 4 · Making the choice")
    render_formula(step.selection_computation, show_title=False)
    show_figure(charts.plot_impurity_breakdown(step, impurity.display))
    _why(narration.why_choice(trace))


# --------------------------------------------------------------------------- #
def render() -> None:
    state = get_state()
    render_sidebar_status(state)
    require_data(state)

    st.title("2 · Build the tree, step by step")
    st.caption("The construction plays like a video: the tree grows on the stage while the "
               "strip below shows the one formula each moment computes.")

    with st.expander("Algorithm and hyper-parameters", expanded=not state.is_model_ready()):
        config = render_algorithm_config(state, key_prefix="build")
        if st.button("Train", type="primary", key="build_train"):
            _train(state, config)
            st.rerun()

    if not state.is_model_ready():
        st.info("Set the options above and press **Train** — the construction then plays back "
                "one formula at a time.")
        return

    trace = state.trace
    theater = _theater(state)
    st.markdown(_PLAYER_CSS, unsafe_allow_html=True)
    _embed(theater["html"])
    hints = ["**Space** play/pause · **←/→** step · **F** full screen · **R** replay"]
    if not has_graphviz_binary():
        hints.append("Trees are drawn in the browser (needs access to cdn.jsdelivr.net); "
                     "`sudo apt install graphviz` renders them locally instead.")
    st.caption(" — ".join(hints))

    with st.expander("The finished tree and its metrics", expanded=False):
        _final_tree_block(state)

    with st.expander("Full working for one node — tables, charts, and why each formula", expanded=False):
        labels = [f"Step {s.step_id + 1} · node #{s.node_id} (depth {s.depth}, {s.action.lower()})"
                  for s in trace.steps]
        picked = st.selectbox("Node", labels, index=min(state.current_step, len(labels) - 1),
                              key="build_detail_node")
        step = trace.steps[labels.index(picked)]
        state.current_step = step.step_id
        _full_working(state, step)
        impurity = get_impurity(trace.impurity_name)
        st.download_button(
            "Download this step's worked calculation (.md)",
            FORMATTER.worked_example(step, impurity.display, trace.score_name),
            file_name=f"step_{step.step_id + 1}_node_{step.node_id}.md",
            mime="text/markdown", key=f"dl_step_{step.step_id}",
        )

    with st.expander("Full build log", expanded=False):
        show_dataframe(trace.summary_dataframe())
        for note in trace.notes:
            st.caption(f"ℹ️ {note}")

    st.download_button(
        "Download every step as one document (.md)",
        FORMATTER.worked_example_document(
            trace, title=f"{trace.algorithm} on {state.dataset.name} — worked example"),
        file_name=f"{trace.algorithm.lower().replace('.', '')}_worked_example.md",
        mime="text/markdown", type="primary",
    )
