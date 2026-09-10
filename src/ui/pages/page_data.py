"""1 · Data — pick or upload a dataset, declare column types, split it."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st

from ...datasets.loaders import (
    CATEGORICAL,
    NUMERIC,
    DatasetError,
    dataset_from_frame,
    infer_feature_types,
    list_builtin,
    load_builtin,
    read_table,
)
from ...datasets.preprocess import (
    EQUAL_FREQUENCY,
    EQUAL_WIDTH,
    MISSING_DROP,
    MISSING_IMPUTE,
    MISSING_KEEP,
    Preprocessor,
)
from ...viz import charts
from ..components import render_sidebar_status, safe_frame, show_dataframe, show_figure
from ..state import get_state


def _source_section(state) -> pd.DataFrame | None:
    """Returns the raw frame chosen by the user, or None if nothing is chosen."""
    tab_builtin, tab_upload = st.tabs(["Bundled datasets", "Upload your own CSV"])

    with tab_builtin:
        catalogue = list_builtin()
        names = [row["name"] for row in catalogue]
        default = names.index(state.dataset.name) if state.dataset and state.dataset.name in names else 0
        chosen = st.selectbox("Dataset", names, index=default, key="data_builtin")
        meta = next(row for row in catalogue if row["name"] == chosen)
        st.caption(f"{meta['rows']} rows · target `{meta['target']}` · {meta['task']} · "
                   f"source: {meta['source']}")
        st.info(meta["description"])
        if st.button("Load this dataset", type="primary", key="data_load_builtin"):
            dataset = load_builtin(chosen)
            st.session_state["_raw_frame"] = dataset.df
            st.session_state["_raw_name"] = dataset.name
            st.session_state["_raw_target"] = dataset.target
            st.session_state["_raw_types"] = dataset.feature_types
            st.session_state["_raw_task"] = dataset.task
            st.session_state["_raw_description"] = dataset.description
            st.rerun()

    with tab_upload:
        st.caption("Any CSV with one row per sample. The separator and encoding are detected "
                   "automatically; you choose the target column below.")
        upload = st.file_uploader("CSV file", type=["csv", "tsv", "txt"], key="data_upload")
        if upload is not None and st.button("Load this file", type="primary", key="data_load_upload"):
            try:
                frame = read_table(upload)
            except DatasetError as error:
                st.error(str(error))
            else:
                st.session_state["_raw_frame"] = frame
                st.session_state["_raw_name"] = upload.name
                st.session_state["_raw_target"] = frame.columns[-1]
                st.session_state["_raw_types"] = infer_feature_types(frame, frame.columns[-1])
                st.session_state["_raw_task"] = None
                st.session_state["_raw_description"] = f"Uploaded file: {upload.name}"
                st.rerun()

    return st.session_state.get("_raw_frame")


def _configure(state, frame: pd.DataFrame) -> None:
    st.subheader("Columns")
    columns = list(frame.columns)
    stored_target = st.session_state.get("_raw_target", columns[-1])
    target = st.selectbox(
        "Target column (what the tree should predict)", columns,
        index=columns.index(stored_target) if stored_target in columns else len(columns) - 1,
        key="data_target",
    )

    inferred = infer_feature_types(frame, target)
    stored_types: Dict[str, str] = {**inferred, **st.session_state.get("_raw_types", {})}

    editor_rows = []
    for column in columns:
        if column == target:
            continue
        editor_rows.append(
            {
                "Feature": column,
                "Use": True,
                "Type": stored_types.get(column, CATEGORICAL),
                "Distinct values": int(frame[column].nunique(dropna=True)),
                "Missing": int(frame[column].isna().sum()),
            }
        )
    edited = st.data_editor(
        pd.DataFrame(editor_rows),
        width="stretch",
        hide_index=True,
        disabled=["Feature", "Distinct values", "Missing"],
        column_config={
            "Use": st.column_config.CheckboxColumn(help="Untick to leave a column out of the tree."),
            "Type": st.column_config.SelectboxColumn(
                options=[CATEGORICAL, NUMERIC],
                help="numeric: the tree may compare with <=. categorical: it can only match values.",
            ),
        },
        key="data_types_editor",
    )

    selected = [row["Feature"] for _i, row in edited.iterrows() if row["Use"]]
    overrides = {row["Feature"]: row["Type"] for _i, row in edited.iterrows()}

    st.subheader("Preparation")
    left, right = st.columns(2)
    with left:
        missing_strategy = st.radio(
            "Missing values in the features",
            [MISSING_KEEP, MISSING_IMPUTE, MISSING_DROP],
            horizontal=True, key="data_missing",
            help="keep: leave them for C4.5 to handle. impute: fill with the mean/mode. "
                 "drop: remove those rows.",
        )
        task_default = st.session_state.get("_raw_task")
        task = st.radio(
            "Task", ["classification", "regression"],
            index=0 if (task_default or "classification") == "classification" else 1,
            horizontal=True, key="data_task",
            help="Regression (a numeric target) is supported by CART only.",
        )
    with right:
        numeric_features = [f for f in selected if overrides.get(f) == NUMERIC]
        discretize = st.multiselect(
            "Discretise these numeric columns into bins", numeric_features,
            default=[], key="data_discretize",
            help="ID3 cannot compare numbers. Binning turns a numeric column into interval "
                 "labels so ID3 can use it — C4.5 and CART do not need this.",
        )
        bin_count = st.slider("Number of bins", 2, 8, 3, key="data_bins",
                              disabled=not discretize)
        bin_strategy = st.radio("Binning", [EQUAL_WIDTH, EQUAL_FREQUENCY], horizontal=True,
                                key="data_binstrategy", disabled=not discretize,
                                help="equal_width: same interval size. equal_frequency: same "
                                     "number of samples per bin.")

    st.subheader("Train / test split")
    split_left, split_mid, split_right = st.columns(3)
    test_size = split_left.slider("Test set share", 0.0, 0.5, 0.3, step=0.05, key="data_testsize")
    stratify = split_mid.checkbox("Stratify by class", value=True, key="data_stratify",
                                  disabled=task == "regression",
                                  help="Keeps each class in the same proportion in both halves.")
    seed = split_right.number_input("Random seed", 0, 9999, 42, key="data_seed")

    if st.button("Apply and prepare data", type="primary", key="data_apply"):
        if not selected:
            st.error("Select at least one feature column.")
            return
        try:
            dataset = dataset_from_frame(
                frame, target, name=st.session_state.get("_raw_name", "Dataset"),
                type_overrides=overrides, task=task,
            )
        except DatasetError as error:
            st.error(str(error))
            return
        dataset.description = st.session_state.get("_raw_description", dataset.description)
        preprocessor = Preprocessor(
            missing_strategy=missing_strategy, discretize=discretize,
            n_bins=int(bin_count), bin_strategy=bin_strategy,
        )
        try:
            prepared = preprocessor.prepare(dataset, features=selected)
        except ValueError as error:
            st.error(str(error))
            return
        state.set_dataset(dataset, preprocessor, prepared, selected)
        state.split_config = {"test_size": float(test_size), "stratify": bool(stratify),
                              "random_state": int(seed)}
        st.success(f"Prepared **{dataset.name}**: {len(prepared)} rows, "
                   f"{len(prepared.feature_names)} features, target `{target}`.")


def _overview(state) -> None:
    prepared = state.prepared
    dataset = state.dataset
    st.subheader("What the model will see")

    train, test = state.split_indices()
    metrics = st.columns(4)
    metrics[0].metric("Rows", len(prepared))
    metrics[1].metric("Features", len(prepared.feature_names))
    metrics[2].metric("Train / test", f"{len(train)} / {len(test)}")
    metrics[3].metric("Classes" if dataset.task == "classification" else "Target",
                      len(prepared.class_names) if dataset.task == "classification" else "numeric")

    if prepared.notes:
        with st.expander("Preparation notes", expanded=False):
            for note in prepared.notes:
                st.markdown(f"- {note}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Prepared data (first 25 rows)**")
        show_dataframe(prepared.frame.head(25))
    with right:
        if dataset.task == "classification":
            counts = prepared.frame[prepared.target].value_counts().to_dict()
            show_figure(charts.plot_class_distribution(
                counts, list(counts.keys()), title="Target distribution"))
            smallest = min(counts.values())
            largest = max(counts.values())
            if largest > 3 * smallest:
                st.warning(f"The classes are unbalanced ({largest} vs {smallest}). Accuracy alone "
                           "will look better than the model really is — read the per-class scores "
                           "on the Evaluate page.")
        else:
            st.markdown("**Target summary**")
            st.dataframe(prepared.frame[prepared.target].describe().to_frame("value"),
                         width="stretch")

    with st.expander("Column types and missing values", expanded=False):
        rows = []
        for name in prepared.feature_names:
            column = prepared.frame[name]
            rows.append({
                "Feature": name,
                "Type": prepared.feature_types.get(name, CATEGORICAL),
                "Distinct": int(pd.Series(column).nunique(dropna=True)),
                "Missing": int(pd.isna(column).sum()),
                "Example values": ", ".join(str(v) for v in pd.Series(column).dropna().unique()[:4]),
            })
        show_dataframe(pd.DataFrame(rows))


def render() -> None:
    state = get_state()
    render_sidebar_status(state)

    st.title("1 · Data")
    st.caption("Choose what the tree learns from. Everything downstream — the build steps, the "
               "diagram, the explanations — is rebuilt from this page.")

    frame = _source_section(state)
    if frame is None:
        st.info("Pick a bundled dataset above, or upload your own CSV, to get started.")
        return

    st.divider()
    _configure(state, frame)

    if state.is_data_ready():
        st.divider()
        _overview(state)
