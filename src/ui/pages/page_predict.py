"""4 · Predict — replay a single prediction, hop by hop.

The point of this page is the *why*: at every node it prints the test, the
sample's own value substituted into it, the branch that follows, and how the
class distribution narrows as the sample descends.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ...core.base_tree import explain_sample
from ...core.criteria import probability_c
from ...datasets.loaders import NUMERIC
from ...viz import charts
from ...viz.tree_dot import RenderOptions
from ..components import (
    render_class_legend,
    render_formula,
    render_formula_inline,
    render_sidebar_status,
    show_figure,
    render_tree_view,
    require_model,
    safe_frame,
    show_dataframe,
)
from ..state import get_state


# --------------------------------------------------------------------------- #
def _widget_form(state, defaults: Dict[str, Any], key_prefix: str) -> Dict[str, Any]:
    """One control per feature, typed to match the column."""
    prepared = state.prepared
    frame = prepared.frame
    sample: Dict[str, Any] = {}
    columns = st.columns(min(3, max(1, len(prepared.feature_names))))

    for index, name in enumerate(prepared.feature_names):
        target_column = columns[index % len(columns)]
        series = frame[name].dropna()
        kind = prepared.feature_types.get(name)
        with target_column:
            if kind == NUMERIC and pd.api.types.is_numeric_dtype(pd.to_numeric(series, errors="coerce")):
                values = pd.to_numeric(series, errors="coerce").dropna()
                low, high = float(values.min()), float(values.max())
                default = float(defaults.get(name, values.median()))
                default = min(max(default, low), high)
                span = high - low
                sample[name] = st.slider(
                    name, low, high, default,
                    step=float(span / 100) if span > 0 else 0.1,
                    key=f"{key_prefix}_{name}",
                )
            else:
                options = sorted({str(v) for v in series.unique()})
                default_value = str(defaults.get(name, options[0] if options else ""))
                index_of = options.index(default_value) if default_value in options else 0
                sample[name] = st.selectbox(name, options, index=index_of,
                                            key=f"{key_prefix}_{name}")
    return sample


def _pick_row(state) -> Optional[Dict[str, Any]]:
    prepared = state.prepared
    _train, test = state.split_indices()
    pool = test if len(test) else np.arange(len(prepared))
    label = "test set" if len(test) else "dataset"

    if len(pool) == 0:
        st.info("There are no rows to pick from.")
        return None
    if len(pool) == 1:
        st.caption(f"The {label} holds a single row.")
        position = 0
    else:
        position = st.number_input(
            f"Row number in the {label}", 0, len(pool) - 1, 0, key="pred_rowidx"
        )
    original = int(pool[int(position)])
    row = prepared.frame.iloc[original]
    show_dataframe(row.to_frame("value").T)
    st.caption(f"True label: **{row[prepared.target]}**")
    return {name: row[name] for name in prepared.feature_names}


def _paste_row(state) -> Optional[Dict[str, Any]]:
    prepared = state.prepared
    st.caption("Paste one comma-separated row, values in this order: "
               + ", ".join(prepared.feature_names))
    text = st.text_input("CSV row", key="pred_paste")
    if not text.strip():
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != len(prepared.feature_names):
        st.error(f"Expected {len(prepared.feature_names)} values, got {len(parts)}.")
        return None
    sample: Dict[str, Any] = {}
    for name, value in zip(prepared.feature_names, parts):
        if prepared.feature_types.get(name) == NUMERIC:
            try:
                sample[name] = float(value)
            except ValueError:
                sample[name] = value
        else:
            sample[name] = value
    return sample


# --------------------------------------------------------------------------- #
def _render_timeline(state, trace) -> None:
    for step in trace.steps:
        if step.is_leaf:
            st.success(f"**Step {step.index + 1} — leaf #{step.node_id}.** {step.narrative}")
            continue

        header = (f"**Step {step.index + 1} — node #{step.node_id}** · test "
                  f"`{step.test}`")
        with st.container(border=True):
            st.markdown(header)
            if step.comparison is not None:
                st.latex(step.comparison.symbolic)
                st.latex(r"\Rightarrow\ " + step.comparison.substituted
                         + r"\ \Rightarrow\ " + (step.comparison.note or ""))
            st.markdown(f"→ {step.outcome_text}")
            if step.unseen_value:
                st.warning("This value was never seen at this node during training.")
            counts = ", ".join(f"{k}={v:g}" for k, v in step.node_class_counts.items()) or "-"
            st.caption(f"{step.n_samples:g} training sample(s) reach this node · {counts}")


def _render_result(state, trace) -> None:
    tree = state.tree
    st.subheader("Prediction")
    columns = st.columns([1, 1, 2])
    columns[0].metric("Predicted", str(trace.predicted_class))
    columns[1].metric("Confidence", f"{trace.confidence:.1%}")
    columns[2].metric("Training samples supporting it", f"{trace.support:g}")

    if trace.probabilities and tree.task == "classification":
        show_figure(
            charts.plot_class_distribution(
                trace.probabilities, list(trace.probabilities.keys()),
                title=f"Class probabilities at leaf #{trace.final_leaf_id}"),
            width="stretch",
        )
        st.markdown("**Where that confidence comes from**")
        render_formula(trace.probability_computation, show_title=False)
        with st.expander("Probability of every class at this leaf", expanded=False):
            leaf = tree.node(trace.final_leaf_id)
            for label in trace.probabilities:
                render_formula(probability_c(leaf.class_counts, label,
                                             where=f"leaf #{trace.final_leaf_id}"),
                               show_title=False)

    st.markdown("**The rule that fired**")
    st.code(trace.rule_text, language=None)
    for warning in trace.warnings:
        st.warning(warning)


def _what_if(state, base_sample: Dict[str, Any]) -> None:
    prepared = state.prepared
    tree = state.tree
    st.caption("Change one value and see whether the path — and the answer — moves.")
    feature = st.selectbox("Feature to vary", prepared.feature_names, key="whatif_feature")
    series = prepared.frame[feature].dropna()

    if prepared.feature_types.get(feature) == NUMERIC:
        values = pd.to_numeric(series, errors="coerce").dropna()
        grid = np.linspace(float(values.min()), float(values.max()), 25)
    else:
        grid = sorted({str(v) for v in series.unique()})

    rows = []
    for value in grid:
        candidate = dict(base_sample)
        candidate[feature] = value
        transformed = state.preprocessor.transform_sample(candidate) if state.preprocessor else candidate
        result = explain_sample(tree, transformed)
        rows.append(
            {
                feature: value if not isinstance(value, float) else round(float(value), 4),
                "Prediction": str(result.predicted_class),
                "Confidence": f"{result.confidence:.1%}",
                "Leaf": f"#{result.final_leaf_id}",
                "Path": " → ".join(f"#{n}" for n in result.path_node_ids),
            }
        )
    frame = pd.DataFrame(rows)
    show_dataframe(frame)
    flips = frame["Prediction"].nunique()
    if flips > 1:
        st.info(f"Changing **{feature}** alone flips the prediction between "
                f"{flips} different classes — this feature genuinely matters for this sample.")
    else:
        st.info(f"Changing **{feature}** alone never changes the prediction for this sample; "
                "the other features already decide it.")


# --------------------------------------------------------------------------- #
def _pruned_banner(state) -> None:
    """Say so when the pruned tree, not the trained one, is answering."""
    if state.use_pruned and state.pruned_tree is not None:
        st.info(f"Using the **pruned** tree ({state.prune_method}, "
                f"{state.pruned_tree.count_leaves()} leaves). Turn pruning off in the Pruning tab "
                "of the Tree page to go back to the original.")


def render() -> None:
    state = get_state()
    render_sidebar_status(state)
    require_model(state)

    prepared = state.prepared
    st.title("4 · Predict, and see why")
    _pruned_banner(state)
    st.caption("Give the tree a new sample. It will walk it from the root to a leaf and show the "
               "test, the value, and the branch taken at every hop.")

    defaults = dict(state.prediction_sample or {})
    tabs = st.tabs(["Enter values", "Pick a row from the data", "Paste a CSV row"])
    sample: Optional[Dict[str, Any]] = None
    with tabs[0]:
        sample = _widget_form(state, defaults, "pred")
    with tabs[1]:
        picked = _pick_row(state)
    with tabs[2]:
        pasted = _paste_row(state)

    source = st.radio("Use the sample from", ["Enter values", "Pick a row", "Paste a CSV row"],
                      horizontal=True, key="pred_source")
    if source == "Pick a row":
        sample = picked
    elif source == "Paste a CSV row":
        sample = pasted

    if st.button("Explain this prediction", type="primary", key="pred_go"):
        if not sample:
            st.error("No sample to explain — fill the form or pick a row first.")
        else:
            transformed = state.preprocessor.transform_sample(sample) if state.preprocessor else sample
            state.prediction_sample = sample
            state.prediction_trace = explain_sample(state.tree, transformed)

    if not state.has_prediction():
        st.info("Choose a sample above and press **Explain this prediction**.")
        return

    trace = state.prediction_trace
    st.divider()

    st.markdown("**The sample being explained**")
    show_dataframe(pd.DataFrame([trace.sample]))
    if state.preprocessor and state.preprocessor.binnings:
        st.caption("Numeric values were mapped to the same bins the tree was trained on.")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**The path through the tree**")
        render_class_legend(state.tree)
        render_tree_view(
            state.tree, state, key="predict_tree", show_downloads=False,
            highlight_path=trace.path_node_ids,
            highlight_edges=trace.path_edges(),
            highlight_node=trace.final_leaf_id,
            options=RenderOptions(orientation="TB", show_distribution_bar=True),
        )
        st.caption("The red outline and red edges are the route this sample took.")
    with right:
        st.markdown("**Step by step**")
        _render_timeline(state, trace)

    st.divider()
    _render_result(state, trace)

    st.divider()
    st.subheader("What if?")
    _what_if(state, state.prediction_sample or {})

    st.download_button("Download this explanation (.txt)", trace.as_text(),
                       file_name="explanation.txt", mime="text/plain")
