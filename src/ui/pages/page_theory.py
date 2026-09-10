"""7 · Theory — the ideas behind the numbers, with an interactive calculator."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from ...core import criteria as crit
from ...viz import charts
from ..components import render_formula, render_sidebar_status, show_dataframe, show_figure
from ..state import get_state


def _calculator() -> None:
    st.markdown("Type any class distribution and watch the three impurities respond.")
    left, right = st.columns([1, 1])
    with left:
        n_classes = st.slider("Number of classes", 2, 5, 2, key="th_k")
        counts: Dict[str, float] = {}
        for index in range(n_classes):
            name = chr(ord("A") + index)
            counts[name] = st.number_input(f"Count of class {name}", 0, 1000,
                                           [9, 5, 0, 0, 0][index] if index < 5 else 0,
                                           key=f"th_count_{index}")
    labels = [name for name, count in counts.items() for _ in range(int(count))]
    with right:
        if not labels:
            st.info("Give at least one class a non-zero count.")
            return
        render_formula(crit.entropy_c(labels))
        st.divider()
        render_formula(crit.gini_c(labels))
        st.divider()
        render_formula(crit.misclassification_error_c(labels))

    if n_classes == 2:
        total = sum(counts.values())
        if total:
            show_figure(charts.plot_entropy_curve(list(counts.values())[0] / total))


def _split_lab() -> None:
    st.markdown("Now split those samples into branches and watch the gain appear.")
    st.caption("Enter one row per branch: how many samples of each class end up there.")

    default = pd.DataFrame(
        [{"Branch": "left", "Class A": 3, "Class B": 5},
         {"Branch": "right", "Class A": 6, "Class B": 0}]
    )
    edited = st.data_editor(default, num_rows="dynamic", key="th_split", width="stretch")

    children = []
    for _index, row in edited.iterrows():
        labels = ["A"] * int(row.get("Class A", 0) or 0) + ["B"] * int(row.get("Class B", 0) or 0)
        if labels:
            children.append((str(row["Branch"]), labels, None))
    if len(children) < 2:
        st.info("Two non-empty branches are needed for a split.")
        return

    parent_labels = [label for _name, labels, _w in children for label in labels]
    impurity = crit.get_impurity("entropy")
    parent = impurity.explain(parent_labels)
    weighted = crit.weighted_impurity_c(children, impurity)
    gain = crit.impurity_gain_c(parent, weighted, "this split", impurity)
    info = crit.split_info_c(children, "this split")
    ratio = crit.gain_ratio_c(gain, info, "this split")

    left, right = st.columns(2, gap="large")
    with left:
        render_formula(parent, title="Before the split")
        for branch, child in weighted.children.items():
            render_formula(child, title=f"Branch [{branch}]")
    with right:
        render_formula(weighted)
        render_formula(gain)
        render_formula(info)
        render_formula(ratio)


COMPARISON = pd.DataFrame(
    [
        {
            "": "Split shape",
            "ID3": "one branch per value (multiway)",
            "C4.5": "multiway on categories, binary on numbers",
            "CART": "always binary",
        },
        {"": "Selection criterion", "ID3": "Information Gain",
         "C4.5": "Gain Ratio", "CART": "Gini gain (or MSE for regression)"},
        {"": "Numeric features", "ID3": "must be binned first",
         "C4.5": "midpoint thresholds", "CART": "midpoint thresholds"},
        {"": "Missing values", "ID3": "not handled",
         "C4.5": "fractional instances down every branch", "CART": "surrogate splits (here: largest branch)"},
        {"": "Reusing a feature", "ID3": "never on the same path",
         "C4.5": "categories no, numbers yes", "CART": "yes"},
        {"": "Regression", "ID3": "no", "C4.5": "no", "CART": "yes"},
        {"": "Pruning", "ID3": "none in the original",
         "C4.5": "pessimistic, error-based", "CART": "cost-complexity with a validation set"},
    ]
)


def render() -> None:
    state = get_state()
    render_sidebar_status(state)

    st.title("7 · The theory, briefly")
    st.caption("Everything the other pages compute, stated once.")

    tabs = st.tabs(["Impurity", "Gain and Gain Ratio", "The three algorithms",
                    "Overfitting & pruning", "Try it yourself"])

    with tabs[0]:
        st.markdown(
            "**Impurity** measures how mixed a node is. It is 0 when every sample shares one "
            "label and largest when the classes are evenly split — the node where you learn the "
            "least by looking."
        )
        st.latex(r"H(S) = -\sum_{i} p_i \log_2 p_i \qquad Gini(S) = 1 - \sum_i p_i^2"
                 r"\qquad Err(S) = 1 - \max_i p_i")
        show_figure(charts.plot_impurity_comparison())
        st.markdown(
            "- **Entropy** is measured in bits: it answers *how many yes/no questions would I "
            "still need to identify a sample's class?*\n"
            "- **Gini** is the chance that two samples drawn at random from the node disagree.\n"
            "- **Misclassification error** is flatter near the middle, which is why it makes a "
            "poor splitting criterion even though it is what we ultimately care about."
        )

    with tabs[1]:
        st.markdown("**Information Gain** is how much impurity a split removes:")
        st.latex(r"IG(S, A) = H(S) - \sum_{v \in values(A)} \frac{|S_v|}{|S|} H(S_v)")
        st.markdown(
            "It has one fatal weakness. A column of unique values — a row id, a date — splits the "
            "data into one-sample branches, every one of them perfectly pure, so its gain is "
            "maximal. The tree it produces is useless: it has memorised the table."
        )
        st.markdown("**Gain Ratio** fixes this by dividing by how fragmented the split is:")
        st.latex(r"SplitInfo(S, A) = -\sum_v \frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|}"
                 r"\qquad GainRatio(S,A) = \frac{IG(S,A)}{SplitInfo(S,A)}")
        st.info("Load the **PlayTennis + Day (ID column)** dataset on the Data page and train ID3 "
                "and C4.5 in turn. ID3 splits on `Day`; C4.5 refuses to.")
        st.markdown("**For CART** the same idea uses Gini, and for a numeric target, variance:")
        st.latex(r"\Delta Gini(S,A) = Gini(S) - \sum_v \frac{|S_v|}{|S|}Gini(S_v)")
        st.latex(r"\Delta MSE(S,A) = MSE(S) - \sum_v \frac{|S_v|}{|S|}MSE(S_v)")

    with tabs[2]:
        show_dataframe(COMPARISON)
        st.markdown(
            "All three are **greedy**: they take the best split available right now and never go "
            "back. That is what makes them fast, and it is also why none of them is guaranteed to "
            "find the smallest tree that fits the data."
        )

    with tabs[3]:
        st.markdown(
            "A tree grown until every leaf is pure has learned the training set exactly — including "
            "its noise. **Pruning** removes subtrees that do not pay for themselves."
        )
        st.markdown("**Reduced error pruning** collapses a subtree whenever a held-out set says "
                    "accuracy does not drop.")
        st.markdown("**Cost-complexity pruning** (CART) prices each subtree per leaf:")
        st.latex(r"\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}")
        st.markdown("Cutting in ascending $\\alpha$ gives a nested sequence of trees; the "
                    "validation curve picks one.")
        st.markdown("**Pessimistic pruning** (C4.5) needs no held-out data. It assumes the "
                    "training error is optimistic and inflates it to the upper end of a binomial "
                    "confidence interval — small leaves get inflated most, so they collapse first.")
        st.info("The **3 · Tree** page runs all three and shows every cut it makes.")

    with tabs[4]:
        st.subheader("Impurity calculator")
        _calculator()
        st.divider()
        st.subheader("Split laboratory")
        _split_lab()
