"""2 · Build — watch the tree being constructed, one node at a time.

After **Train** the construction plays back on its own.  Each node is revealed in
phases — measure the impurity, score the candidate splits, work through the best
one, make the choice — and the tree on the left grows when the phase that decides
the node completes.  The finished tree is only shown once the playback has
reached the end (or the user skips to it).

The animation lives in a ``st.fragment`` so a tick re-renders this section only;
``run_every`` is the timer, switched on while playing and off while paused.
"""
from __future__ import annotations

import time
from typing import Optional

import streamlit as st

from ...core.criteria import get_impurity
from ...core.metrics import accuracy, root_mean_squared_error
from ...core.node import SPLIT_NUMERIC_BINARY
from ...core.trace import ACTION_LEAF, STOP_REASONS, BuildStep, TrainingTrace
from ...viz import charts
from ...viz.mathfmt import MathFormatter
from ...viz.tree_dot import RenderOptions
from .. import narration, playback
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
from ..playback import Cursor
from ..state import AppState, get_state

FORMATTER = MathFormatter()

SCRUB_KEY = "build_scrub"
SPEED_KEY = "build_speed_widget"


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
    state.set_model(model, model.get_trace())  # also rewinds and starts the playback
    state.hyperparams = config
    st.session_state["_train_seconds"] = elapsed


# --------------------------------------------------------------------------- #
# Playback cursor <-> session state
# --------------------------------------------------------------------------- #
def _cursor(state: AppState, trace: TrainingTrace) -> Cursor:
    return Cursor(int(state.current_step), int(state.build_phase)).clamp(trace)


def _store(state: AppState, cursor: Cursor, trace: TrainingTrace) -> None:
    state.current_step = cursor.step
    state.build_phase = cursor.phase
    if cursor == playback.last(trace):
        state.build_revealed = True


def _pause(state: AppState) -> None:
    state.build_playing = False


def _play(state: AppState) -> None:
    state.build_playing = True
    state.build_last_tick = time.monotonic()


def _on_scrub(state: AppState, trace: TrainingTrace) -> None:
    """Slider callback: jump to a step with everything in it revealed, and pause."""
    _store(state, playback.at_step(trace, int(st.session_state[SCRUB_KEY]) - 1), trace)
    _pause(state)


def _on_speed(state: AppState) -> None:
    state.build_speed = float(st.session_state[SPEED_KEY])
    state.build_last_tick = time.monotonic()


def _handle_controls(state: AppState, trace: TrainingTrace) -> None:
    """The transport buttons. Each one pauses except Play and Replay."""
    cursor = _cursor(state, trace)
    playing = bool(state.build_playing)

    top = st.columns([1.2, 1.2, 1.6, 3], gap="small")
    if playing:
        if top[0].button("⏸ Pause", key="build_pause", width="stretch", type="primary"):
            _pause(state)
    else:
        if top[0].button("▶ Play", key="build_play", width="stretch", type="primary"):
            if cursor == playback.last(trace):  # pressing Play at the end restarts
                _store(state, playback.first(), trace)
            _play(state)
    if top[1].button("🔁 Replay", key="build_replay", width="stretch",
                     help="Rewind to the first node and play the construction again"):
        state.start_playback()
    if top[2].button("⏩ Skip to the end", key="build_skip", width="stretch",
                     help="Reveal the finished tree without waiting"):
        _store(state, playback.last(trace), trace)
        _pause(state)
    st.session_state[SPEED_KEY] = float(state.build_speed)
    top[3].select_slider(
        "Seconds per phase", options=playback.SPEED_OPTIONS, key=SPEED_KEY,
        on_change=_on_speed, args=(state,),
        help="How long each formula stays on screen before the next one appears",
    )

    nav = st.columns([1, 1, 1, 1, 6], gap="small")
    if nav[0].button("⏮", key="build_first", width="stretch", help="First node"):
        _store(state, playback.first(), trace)
        _pause(state)
    if nav[1].button("‹ Back", key="build_back", width="stretch", help="Previous phase"):
        _store(state, playback.retreat(cursor, trace), trace)
        _pause(state)
    if nav[2].button("Next ›", key="build_next", width="stretch", help="Next phase"):
        _store(state, playback.advance(cursor, trace)[0], trace)
        _pause(state)
    if nav[3].button("⏭", key="build_last", width="stretch", help="Last node"):
        _store(state, playback.last(trace), trace)
        _pause(state)

    cursor = _cursor(state, trace)  # re-read: a button may have moved it
    st.session_state[SCRUB_KEY] = cursor.step + 1
    with nav[4]:
        if len(trace) > 1:
            st.slider("Node being processed", 1, len(trace), key=SCRUB_KEY,
                      on_change=_on_scrub, args=(state, trace),
                      help="Scrub to any node; the whole step is shown at once.")
        else:
            st.caption("The tree has a single node.")


def _tick_if_due(state: AppState, trace: TrainingTrace) -> None:
    """Advance one phase when the timer is running and an interval has elapsed.

    The elapsed-time check is what stops a click (which also reruns the fragment)
    from counting as a tick, and stops the first run after a full rerun from
    jumping ahead early.
    """
    if not state.build_playing:
        return
    now = time.monotonic()
    if now - float(state.build_last_tick) < float(state.build_speed) * 0.5:
        return
    cursor, finished = playback.advance(_cursor(state, trace), trace)
    _store(state, cursor, trace)
    state.build_last_tick = now
    if finished:
        _pause(state)
        state.build_revealed = True


# --------------------------------------------------------------------------- #
# The explanation, revealed phase by phase
# --------------------------------------------------------------------------- #
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
    """The motivation for a formula, set apart from the arithmetic itself."""
    if text:
        st.markdown(f"💡 *Why this formula, here:* {text}")


def _explanation(state: AppState, step: BuildStep, phase: int) -> None:
    trace = state.trace
    impurity = get_impurity(trace.impurity_name)
    revealed_all = phase >= playback.max_phase(step)

    if revealed_all:
        st.info(step.narrative)
    else:
        st.info(f"{step.node_label} (depth {step.depth}) comes off the queue with "
                f"{step.n_samples:g} samples — {step.counts_text() or 'no class counts'}.")

    st.markdown("**Where we are**")
    st.code(step.path_text(), language=None)
    _samples_table(state, step)

    # Phase 0 — impurity ------------------------------------------------------
    st.divider()
    st.markdown("#### 1 · How impure is this node?")
    if step.class_counts:
        st.caption("Class counts here: " + step.counts_text())
    render_formula(step.impurity_computation, show_title=False)
    _why(narration.why_impurity(trace))

    if step.action == ACTION_LEAF:
        if phase < playback.PHASE_STOP:
            return
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

    # Phase 1 — score every candidate ----------------------------------------
    if phase < playback.PHASE_SCORE:
        return
    st.divider()
    st.markdown("#### 2 · Score every possible split")
    st.caption("Losing candidates are kept — select any row below to see its full working.")
    show_dataframe(FORMATTER.candidates_dataframe(step))
    show_figure(charts.plot_candidate_gains(step, trace.score_name))
    _why(narration.why_scoring(trace))

    # Phase 2 — the working for one candidate --------------------------------
    if phase < playback.PHASE_WORKING:
        return
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

    # Phase 3 — the choice ----------------------------------------------------
    if phase < playback.PHASE_CHOICE:
        return
    st.divider()
    st.markdown("#### 4 · Making the choice")
    render_formula(step.selection_computation, show_title=False)
    show_figure(charts.plot_impurity_breakdown(step, impurity.display))
    _why(narration.why_choice(trace))


# --------------------------------------------------------------------------- #
# The finished tree — shown only once the playback has got there
# --------------------------------------------------------------------------- #
def _final_tree_block(state: AppState) -> None:
    tree = state.original_tree
    prepared = state.prepared
    X_train, y_train = state.training_data()
    X_test, y_test = state.test_data()
    stats = tree.stats()

    st.success("🌳 Every node has been processed — this is the finished tree.")
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
# The animated section (one fragment)
# --------------------------------------------------------------------------- #
def _animation(state: AppState, registered_interval: Optional[float]) -> None:
    trace = state.trace

    def wanted_interval() -> Optional[float]:
        return float(state.build_speed) if state.build_playing else None

    # ``run_every`` is fixed when the fragment is created, so whenever the timer
    # should change state (play, pause, new speed) the whole page has to rerun.
    if wanted_interval() != registered_interval:
        st.rerun(scope="app")

    st.subheader("Watch the tree being built")
    _handle_controls(state, trace)
    _tick_if_due(state, trace)
    if wanted_interval() != registered_interval:
        st.rerun(scope="app")

    cursor = _cursor(state, trace)
    step = trace.steps[cursor.step]
    st.progress(
        playback.progress(cursor, trace),
        text=f"Step {cursor.step + 1} of {len(trace)} · node #{step.node_id} at depth "
             f"{step.depth} · {playback.phase_label(step, cursor.phase)}",
    )

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**The tree at this moment**")
        st.caption("Solid = already decided · dashed = queued, not yet processed · "
                   "red outline = the node being processed now.")
        # Until the deciding phase the node is still open: show the tree as it was
        # before this step, with the node outlined, so the split appears on the tick
        # that chooses it.
        shown_step = cursor.step if cursor.is_complete(step) else cursor.step - 1
        partial = trace.tree_at_step(shown_step)
        render_tree_view(
            partial, state, key=f"step_{cursor.step}_{cursor.phase}", show_downloads=False,
            highlight_node=step.node_id,
            options=RenderOptions(orientation="TB", show_node_ids=True, show_distribution_bar=True),
        )
        pending = trace.frontier_at_step(shown_step)
        st.caption(f"{len(pending)} node(s) still queued."
                   if pending else "The queue is empty — the tree is complete.")
        if state.build_playing:
            st.caption("⏳ " + narration.next_up(step, cursor.phase, trace))

    with right:
        _explanation(state, step, cursor.phase)
        impurity = get_impurity(trace.impurity_name)
        st.download_button(
            "Download this step's worked calculation (.md)",
            FORMATTER.worked_example(step, impurity.display, trace.score_name),
            file_name=f"step_{step.step_id + 1}_node_{step.node_id}.md",
            mime="text/markdown", key=f"dl_step_{step.step_id}",
        )

    if state.build_revealed:
        st.divider()
        _final_tree_block(state)
    elif not state.build_playing:
        st.caption("The finished tree appears here once the playback reaches the last node — "
                   "press **Play** or **Skip to the end**.")


# --------------------------------------------------------------------------- #
def render() -> None:
    state = get_state()
    render_sidebar_status(state)
    require_data(state)

    st.title("2 · Build the tree, step by step")
    st.caption("Every number below is shown as *formula → the node's own values → result*, "
               "with a note on why that formula is needed at that moment.")

    with st.expander("Algorithm and hyper-parameters", expanded=not state.is_model_ready()):
        config = render_algorithm_config(state, key_prefix="build")
        if st.button("Train", type="primary", key="build_train"):
            _train(state, config)
            st.rerun()

    if not state.is_model_ready():
        st.info("Set the options above and press **Train** — the construction will then play "
                "back one formula at a time.")
        return

    interval = float(state.build_speed) if state.build_playing else None

    @st.fragment(run_every=interval)
    def animated_section() -> None:
        _animation(state, interval)

    animated_section()

    trace = state.trace
    st.divider()
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
