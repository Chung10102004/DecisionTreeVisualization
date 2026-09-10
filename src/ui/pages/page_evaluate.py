"""5 · Evaluate — how well does the tree actually do, and where does it fail."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from ...core import metrics as mx
from ...viz import charts
from ..components import (
    build_model,
    render_sidebar_status,
    show_figure,
    require_model,
    show_dataframe,
)
from ..state import get_state


def _headline(state) -> None:
    prepared = state.prepared
    tree = state.tree
    X_train, y_train = state.training_data()
    X_test, y_test = state.test_data()

    st.subheader("Headline numbers")
    if prepared.task == "regression":
        train_pred = [float(v) for v in tree.predict(X_train)]
        rows = [{"Split": "train", **mx.score_model(y_train, train_pred, "regression")}]
        if len(y_test):
            test_pred = [float(v) for v in tree.predict(X_test)]
            rows.append({"Split": "test", **mx.score_model(y_test, test_pred, "regression")})
        show_dataframe(pd.DataFrame(rows).round(4))
        return

    train_pred = tree.predict(X_train)
    rows = [{"Split": "train", **mx.score_model(y_train, train_pred)}]
    if len(y_test):
        rows.append({"Split": "test", **mx.score_model(y_test, tree.predict(X_test))})
    frame = pd.DataFrame(rows).round(4)
    show_dataframe(frame)

    if len(y_test):
        gap = float(frame.loc[0, "Accuracy"] - frame.loc[1, "Accuracy"])
        if gap > 0.15:
            st.warning(f"Training accuracy is {gap:.0%} higher than test accuracy — the tree has "
                       "memorised the training set. Try a smaller max_depth, a larger "
                       "min_samples_leaf, or pruning on the Tree page.")


def _confusion(state) -> None:
    tree = state.tree
    X_test, y_test = state.test_data()
    if not len(y_test):
        st.info("No test set — give it a non-zero share on the Data page to see these numbers.")
        return
    predictions = tree.predict(X_test)
    labels = [str(c) for c in (tree.class_names or [])]
    matrix, labels = mx.confusion_matrix(y_test, predictions, labels or None)

    left, right = st.columns([1, 1], gap="large")
    with left:
        show_figure(charts.plot_confusion_matrix(matrix, labels, "Confusion matrix (test set)"))
    with right:
        st.markdown("**Per-class scores (test set)**")
        show_dataframe(pd.DataFrame(mx.classification_report_rows(y_test, predictions, labels)))
        st.caption("Precision: of the samples predicted as this class, how many were right. "
                   "Recall: of the samples truly in this class, how many were found.")


def _depth_curve(state) -> None:
    prepared = state.prepared
    config = dict(state.hyperparams or {})
    algorithm = config.get("algorithm", state.algorithm)
    X_train, y_train = state.training_data()
    X_test, y_test = state.test_data()
    if not len(y_test):
        st.info("A depth curve needs a test set.")
        return

    max_depth = st.slider("Depths to try", 2, 12, 8, key="eval_depths")
    if not st.button("Compute the depth curve", key="eval_depth_go"):
        return

    rows: List[Dict[str, float]] = []
    progress = st.progress(0.0)
    for index, depth in enumerate(range(1, max_depth + 1), start=1):
        params = {**config, "max_depth": depth}
        model = build_model(algorithm, params, task=prepared.task)
        model.fit(X_train, y_train, prepared.feature_names, prepared.feature_types)
        if prepared.task == "regression":
            train_score = -mx.root_mean_squared_error(y_train, [float(v) for v in model.predict(X_train)])
            test_score = -mx.root_mean_squared_error(y_test, [float(v) for v in model.predict(X_test)])
        else:
            train_score = mx.accuracy(y_train, model.predict(X_train))
            test_score = mx.accuracy(y_test, model.predict(X_test))
        rows.append({"depth": depth, "train": train_score, "test": test_score})
        progress.progress(index / max_depth)
    progress.empty()

    metric = "Accuracy" if prepared.task == "classification" else "negative RMSE"
    show_figure(charts.plot_learning_curve_depth(rows, metric))
    st.caption("The gap between the two lines is overfitting made visible: the training line keeps "
               "climbing while the test line flattens or falls.")


def _cross_validation(state) -> None:
    prepared = state.prepared
    config = dict(state.hyperparams or {})
    algorithm = config.get("algorithm", state.algorithm)

    folds = st.slider("Number of folds (k)", 2, 10, 5, key="eval_k")
    if not st.button("Run cross-validation", key="eval_cv_go"):
        return

    rows: List[Dict[str, Any]] = []
    progress = st.progress(0.0)
    splits = mx.k_fold_indices(len(prepared), k=folds, random_state=state.split_config["random_state"])
    for index, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = build_model(algorithm, config, task=prepared.task)
        model.fit(prepared.X[train_idx], prepared.y[train_idx],
                  prepared.feature_names, prepared.feature_types)
        if prepared.task == "regression":
            predicted = [float(v) for v in model.predict(prepared.X[valid_idx])]
            rows.append({"Fold": index, "RMSE": round(mx.root_mean_squared_error(
                prepared.y[valid_idx], predicted), 4), "Leaves": model.tree.count_leaves()})
        else:
            predicted = model.predict(prepared.X[valid_idx])
            rows.append({"Fold": index,
                         "Accuracy": round(mx.accuracy(prepared.y[valid_idx], predicted), 4),
                         "Macro F1": round(mx.precision_recall_f1(
                             prepared.y[valid_idx], predicted)["f1"], 4),
                         "Leaves": model.tree.count_leaves()})
        progress.progress(index / len(splits))
    progress.empty()

    frame = pd.DataFrame(rows)
    show_dataframe(frame)
    metric = "RMSE" if prepared.task == "regression" else "Accuracy"
    show_figure(charts.plot_cv_folds(rows, metric))
    st.caption(f"Mean {metric.lower()} across folds: **{frame[metric].mean():.4f}** "
               f"(± {frame[metric].std():.4f}). A wide spread means the result depends heavily on "
               "which rows happened to land in the training set.")


def _errors(state) -> None:
    prepared = state.prepared
    tree = state.tree
    _train, test = state.split_indices()
    if not len(test):
        st.info("No test set to inspect.")
        return

    X_test, y_test = state.test_data()
    predictions = tree.predict(X_test)
    if prepared.task == "regression":
        residuals = np.asarray([float(p) for p in predictions]) - np.asarray(y_test, dtype=float)
        frame = prepared.frame.iloc[test].copy()
        frame["predicted"] = [float(p) for p in predictions]
        frame["residual"] = residuals
        frame = frame.reindex(frame["residual"].abs().sort_values(ascending=False).index)
        st.markdown("**Largest errors first**")
        show_dataframe(frame.head(25))
        return

    wrong = [i for i, (truth, guess) in enumerate(zip(y_test, predictions))
             if str(truth) != str(guess)]
    st.markdown(f"**{len(wrong)} misclassified test sample(s)**")
    if not wrong:
        st.success("The tree gets every test sample right.")
        return
    frame = prepared.frame.iloc[test[wrong]].copy()
    frame.insert(0, "predicted", [predictions[i] for i in wrong])
    frame.insert(0, "row", test[wrong])
    show_dataframe(frame)
    st.caption("Copy a row number here into the **Predict** page to see exactly which branch sent "
               "it to the wrong leaf.")


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

    st.title("5 · Evaluate")
    _pruned_banner(state)
    st.caption("Accuracy on the data it was trained on tells you almost nothing. Everything here "
               "is about how the tree behaves on rows it has never seen.")

    _headline(state)
    st.divider()

    if state.prepared.task == "classification":
        st.subheader("Where the mistakes are")
        _confusion(state)
        st.divider()

    st.subheader("Overfitting as depth increases")
    _depth_curve(state)
    st.divider()

    st.subheader("Cross-validation")
    st.caption("One train/test split is a single roll of the dice. k-fold rotates every row "
               "through the validation set once.")
    _cross_validation(state)
    st.divider()

    st.subheader("The samples it gets wrong")
    _errors(state)
