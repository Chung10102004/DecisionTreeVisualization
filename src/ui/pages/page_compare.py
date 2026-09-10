"""6 · Compare — the three algorithms on the same data, side by side."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ...core import metrics as mx
from ...datasets.loaders import NUMERIC
from ...viz import charts
from ...viz.boundary import plot_decision_boundary_2d
from ...viz.tree_dot import RenderOptions
from ..components import build_model, render_sidebar_status, render_tree_view, require_data, show_dataframe, show_figure
from ..state import get_state

ALGORITHMS = ["ID3", "C4.5", "CART"]


def _run_comparison(state, shared: Dict[str, Any]) -> List[Dict[str, Any]]:
    prepared = state.prepared
    X_train, y_train = state.training_data()
    X_test, y_test = state.test_data()
    results: List[Dict[str, Any]] = []

    for algorithm in ALGORITHMS:
        params = dict(shared)
        params["impurity"] = "entropy" if algorithm in ("ID3", "C4.5") else "gini"
        if algorithm == "C4.5":
            params.setdefault("min_samples_leaf", max(shared.get("min_samples_leaf", 1), 2))
        model = build_model(algorithm, params, task=prepared.task)
        started = time.perf_counter()
        model.fit(X_train, y_train, prepared.feature_names, prepared.feature_types)
        elapsed = time.perf_counter() - started
        tree = model.tree
        stats = tree.stats()

        row: Dict[str, Any] = {
            "Algorithm": algorithm,
            "Root split": tree.root.test_description() or "(single leaf)",
            "Criterion": model._score_display(),
            "Nodes": stats["nodes"],
            "Leaves": stats["leaves"],
            "Depth": stats["depth"],
            "Train seconds": round(elapsed, 4),
        }
        if prepared.task == "regression":
            row["Train RMSE"] = round(mx.root_mean_squared_error(
                y_train, [float(v) for v in tree.predict(X_train)]), 4)
            row["Test RMSE"] = round(mx.root_mean_squared_error(
                y_test, [float(v) for v in tree.predict(X_test)]), 4) if len(y_test) else None
        else:
            row["Train accuracy"] = round(mx.accuracy(y_train, tree.predict(X_train)), 4)
            row["Test accuracy"] = round(mx.accuracy(y_test, tree.predict(X_test)), 4) if len(y_test) else None
        results.append({"row": row, "model": model})
    return results


def _narrate_differences(results: List[Dict[str, Any]]) -> List[str]:
    """Turn the numeric differences into sentences that name the mechanism."""
    lines: List[str] = []
    by_name = {r["row"]["Algorithm"]: r for r in results}

    id3, c45, cart = by_name.get("ID3"), by_name.get("C4.5"), by_name.get("CART")
    if id3 and c45:
        id3_root = id3["model"].tree.root
        c45_root = c45["model"].tree.root
        if id3_root.feature_name != c45_root.feature_name:
            step = c45["model"].get_trace().steps[0]
            candidate = next((c for c in step.candidates
                              if c.feature_name == id3_root.feature_name), None)
            extra = ""
            if candidate is not None and not candidate.is_valid:
                extra = f" C4.5 rejected it outright: {candidate.reject_reason}"
            elif candidate is not None:
                extra = (f" C4.5 saw the same gain ({candidate.gain:.4f}) but divided it by a "
                         f"SplitInfo of {candidate.split_info:.4f}, giving a Gain Ratio of only "
                         f"{candidate.gain_ratio:.4f}.")
            lines.append(
                f"**Different roots.** ID3 opens on `{id3_root.feature_name}`, C4.5 on "
                f"`{c45_root.feature_name}`. Information Gain rewards a column for splitting the "
                f"data into many small piles; Gain Ratio divides that reward by how fragmented the "
                f"split is.{extra}"
            )
        else:
            lines.append(f"**Same root.** Both ID3 and C4.5 open on `{id3_root.feature_name}` — "
                         "on this data the many-values bias does not bite.")

    if cart:
        cart_stats = cart["row"]
        lines.append(
            f"**CART is binary.** It asks yes/no questions, so it needs {cart_stats['Depth']} "
            f"levels where a multiway split would need fewer, but it can revisit the same feature "
            "further down and carve out finer regions."
        )
    if id3 and cart:
        if id3["row"]["Leaves"] and cart["row"]["Leaves"]:
            lines.append(
                f"**Size.** ID3 ends with {id3['row']['Leaves']} leaves, CART with "
                f"{cart['row']['Leaves']}. More leaves means more specific rules — and more chance "
                "the tree has memorised noise."
            )
    return lines


def _combined_importance(results: List[Dict[str, Any]], candidates: List[str]) -> List[str]:
    """Numeric features ranked by how much the three trees leaned on them."""
    totals: Dict[str, float] = {name: 0.0 for name in candidates}
    for result in results:
        for name, score in result["model"].tree.feature_importances().items():
            if name in totals:
                totals[name] += float(score)
    return [name for name, score in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            if score > 0] or list(candidates)


def render() -> None:
    state = get_state()
    render_sidebar_status(state)
    require_data(state)

    prepared = state.prepared
    st.title("6 · Compare the three algorithms")
    st.caption("Same data, same split, same limits — the only thing that changes is how a split is "
               "chosen. That is what makes the differences legible.")

    if prepared.task == "regression":
        st.warning("ID3 and C4.5 are classification algorithms. On a numeric target only CART can "
                   "run, so there is nothing to compare — switch to a classification dataset.")
        return

    with st.expander("Shared settings", expanded=True):
        left, right = st.columns(2)
        shared = {
            "max_depth": left.slider("max_depth", 1, 12, 4, key="cmp_depth"),
            "min_samples_split": left.slider("min_samples_split", 2, 30, 2, key="cmp_mss"),
            "min_samples_leaf": right.slider("min_samples_leaf", 1, 20, 1, key="cmp_msl"),
            "min_impurity_decrease": right.slider("min_impurity_decrease", 0.0, 0.2, 0.0,
                                                  step=0.005, key="cmp_mid"),
        }
        st.caption("C4.5 keeps a floor of min_samples_leaf = 2 — that floor is precisely what stops "
                   "it choosing an identifier column.")

    if st.button("Train all three", type="primary", key="cmp_go"):
        state.comparison = _run_comparison(state, shared)

    if not state.comparison:
        st.info("Press **Train all three** to build an ID3, a C4.5 and a CART tree on this data.")
        return

    results = state.comparison
    frame = pd.DataFrame([r["row"] for r in results])
    st.subheader("Side by side")
    show_dataframe(frame)

    metric = "Test accuracy" if "Test accuracy" in frame.columns and frame["Test accuracy"].notna().all() \
        else "Train accuracy"
    show_figure(charts.plot_algorithm_comparison(frame.to_dict("records"), metric))

    st.subheader("What the differences mean")
    for line in _narrate_differences(results):
        st.markdown(f"- {line}")

    st.subheader("The three trees")
    columns = st.columns(3, gap="medium")
    for column, result in zip(columns, results):
        with column:
            st.markdown(f"**{result['row']['Algorithm']}** · {result['row']['Leaves']} leaves")
            render_tree_view(
                result["model"].tree, state, key=f"cmp_{result['row']['Algorithm']}",
                show_downloads=False,
                options=RenderOptions(show_counts=False, show_impurity=False,
                                      show_distribution_bar=True, show_node_ids=False,
                                      font_size=9),
            )

    numeric_features = [f for f in prepared.feature_names
                        if prepared.feature_types.get(f) == NUMERIC]
    st.subheader("Decision regions")
    if len(numeric_features) < 2:
        st.info("A 2-D decision boundary needs two numeric features. This dataset has "
                f"{len(numeric_features)}.")
        return

    # Default to the two features the trees actually rely on. Picking the first two
    # columns tends to produce three identically-coloured plots, because the features
    # that decide the prediction are then pinned at their median.
    ranked = _combined_importance(results, numeric_features)
    default_x = numeric_features.index(ranked[0]) if ranked else 0
    default_y = numeric_features.index(ranked[1]) if len(ranked) > 1 else min(1, len(numeric_features) - 1)

    left, right = st.columns(2)
    feature_x = left.selectbox("X axis", numeric_features, index=default_x, key="cmp_x")
    feature_y = right.selectbox("Y axis", numeric_features, index=default_y, key="cmp_y")
    if feature_x == feature_y:
        st.warning("Pick two different features.")
        return

    resolution = st.select_slider("Grid resolution", [80, 120, 160, 220], value=120,
                                  key="cmp_res",
                                  help="Higher is smoother but slower — every grid point is a real "
                                       "prediction made by the tree.")
    with st.spinner("Predicting over the grid…"):
        boundary_columns = st.columns(3, gap="medium")
        for column, result in zip(boundary_columns, results):
            with column:
                figure = plot_decision_boundary_2d(
                    result["model"].tree, prepared.frame[prepared.feature_names], prepared.y,
                    feature_x, feature_y, prepared.feature_types, resolution=resolution,
                    title=result["row"]["Algorithm"],
                )
                show_figure(figure)
    st.caption("Every region is drawn by asking the tree itself for a prediction at that point. "
               "The staircase edges are the tree's axis-aligned splits.")
    st.caption(f"Only {feature_x} and {feature_y} vary here; the remaining features are pinned at "
               "a representative value, printed under each plot. If a plot comes out a single "
               "colour, those pinned features are the ones deciding the outcome — pick them for "
               "the axes instead.")
