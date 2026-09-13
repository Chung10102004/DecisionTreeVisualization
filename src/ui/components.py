"""Reusable Streamlit widgets.

``render_formula`` is the important one: every number the app reports goes
through it, so the user always sees *formula -> numbers -> result* rather than a
bare value.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from ..core.criteria import Computation, get_impurity
from ..core.node import Node, Tree
from ..core.trace import BuildStep, STOP_REASONS, TrainingTrace
from ..viz.mathfmt import DEFAULT_FORMATTER, FormulaBlock, MathFormatter
from ..viz.tree_dot import RenderOptions, TreeDotRenderer, dot_to_svg, has_graphviz_binary
from .state import AppState

ALGORITHMS = ["ID3", "C4.5", "CART"]


# --------------------------------------------------------------------------- #
# Formula display — the heart of the "show the working" requirement
# --------------------------------------------------------------------------- #
def render_formula(
    computation: Optional[Computation],
    caption: Optional[str] = None,
    title: Optional[str] = None,
    formatter: MathFormatter = DEFAULT_FORMATTER,
    show_title: bool = True,
) -> None:
    """Three stacked lines: the symbolic formula, the numbers, the result."""
    if computation is None:
        return
    block = formatter.render(computation, title=title)
    if show_title and block.title:
        st.markdown(f"**{block.title}**")
    st.latex(block.symbolic)
    st.latex(r"= " + block.substituted)
    st.latex(r"= \mathbf{%s}" % block.result)
    if block.note:
        st.caption(block.note)
    if caption:
        st.caption(caption)


def render_formula_inline(computation: Optional[Computation],
                          formatter: MathFormatter = DEFAULT_FORMATTER) -> None:
    """The same content on one line, for tight spaces such as a timeline card."""
    if computation is None:
        return
    block = formatter.render(computation)
    st.latex(block.one_line())
    if block.note:
        st.caption(block.note)


def render_formula_chain(computations: Sequence[Computation], title: str,
                         expanded: bool = False,
                         formatter: MathFormatter = DEFAULT_FORMATTER) -> None:
    with st.expander(title, expanded=expanded):
        for computation in computations:
            render_formula(computation, formatter=formatter)
            st.divider()


def render_blocks(blocks: Sequence[FormulaBlock]) -> None:
    for block in blocks:
        st.markdown(f"**{block.title}**")
        st.latex(block.symbolic)
        st.latex(r"= " + block.substituted)
        st.latex(r"= \mathbf{%s}" % block.result)
        if block.note:
            st.caption(block.note)
        st.write("")


# --------------------------------------------------------------------------- #
# Data display helpers
# --------------------------------------------------------------------------- #
def safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make a frame Arrow-safe: object columns become strings, NaN becomes '-'."""
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out


def show_dataframe(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(safe_frame(frame), width="stretch", **kwargs)


def show_figure(figure, width: int = 720) -> None:
    """Render a matplotlib figure at a fixed pixel width.

    ``st.pyplot`` scales the image to whatever column it lands in, which blows the
    fonts up in a wide column and shrinks them in a narrow one.  Rasterising once
    at a known size keeps the typography identical everywhere — and closing the
    figure afterwards stops matplotlib accumulating them across reruns.
    """
    import io

    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight",
                   facecolor=figure.get_facecolor())
    plt.close(figure)
    buffer.seek(0)
    st.image(buffer, width=width)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def require_data(state: AppState) -> None:
    if not state.is_data_ready():
        st.warning("Load a dataset on the **1 · Data** page first.")
        st.stop()


def require_model(state: AppState) -> None:
    require_data(state)
    if not state.is_model_ready():
        st.warning("Train a tree on the **2 · Build** page first.")
        st.stop()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar_status(state: AppState) -> None:
    with st.sidebar:
        st.markdown("### Session")
        if state.dataset is None:
            st.info("No dataset loaded yet.")
            return
        dataset = state.dataset
        st.markdown(f"**Dataset** · {dataset.name}")
        st.caption(f"{dataset.n_rows} rows · {len(state.prepared.feature_names) if state.prepared else 0} "
                   f"features · target `{dataset.target}` · {dataset.task}")
        if state.is_model_ready():
            tree = state.tree
            stats = tree.stats()
            st.markdown(f"**Model** · {tree.algorithm}")
            st.caption(f"{stats['nodes']} nodes · {stats['leaves']} leaves · depth {stats['depth']}"
                       + (" · pruned" if state.use_pruned else ""))
        else:
            st.caption("No model trained yet.")


# --------------------------------------------------------------------------- #
# Algorithm configuration
# --------------------------------------------------------------------------- #
def render_algorithm_config(state: AppState, key_prefix: str = "cfg") -> Dict[str, Any]:
    """The hyper-parameter form; every control carries a one-line explanation."""
    dataset = state.dataset
    is_regression = dataset is not None and dataset.task == "regression"

    left, right = st.columns([1, 1])
    with left:
        if is_regression:
            algorithm = "CART"
            st.selectbox("Algorithm", ["CART"], index=0, key=f"{key_prefix}_algo",
                         help="ID3 and C4.5 are classification-only; a numeric target needs CART.")
        else:
            algorithm = st.selectbox(
                "Algorithm", ALGORITHMS, index=ALGORITHMS.index(state.algorithm)
                if state.algorithm in ALGORITHMS else 0, key=f"{key_prefix}_algo",
                help="ID3: multiway splits on Information Gain. C4.5: Gain Ratio, numeric "
                     "thresholds, missing values. CART: binary splits on Gini.",
            )

        if algorithm == "CART":
            impurity_options = ["mse", "mae"] if is_regression else ["gini", "entropy", "error"]
        else:
            impurity_options = ["entropy"]
        # Keyed widgets keep their old value across reruns, so switching algorithm
        # would leave CART on entropy or C4.5 on min_samples_leaf=1. Reset the
        # algorithm-specific defaults whenever the algorithm actually changes.
        default_leaf = 2 if algorithm == "C4.5" else 1
        if st.session_state.get(f"{key_prefix}_algo_seen") != algorithm:
            st.session_state[f"{key_prefix}_algo_seen"] = algorithm
            st.session_state[f"{key_prefix}_impurity"] = impurity_options[0]
            st.session_state[f"{key_prefix}_msl"] = default_leaf
        impurity = st.selectbox(
            "Impurity measure", impurity_options, index=0, key=f"{key_prefix}_impurity",
            help="The quantity the split is trying to reduce.",
        )

        max_depth = st.slider("max_depth", 1, 12, 4, key=f"{key_prefix}_depth",
                              help="Hard ceiling on the depth. Lower means a simpler tree.")
        unlimited = st.checkbox("No depth limit", value=False, key=f"{key_prefix}_nodepth",
                                help="Grow until every leaf is pure or another rule stops it.")

    with right:
        min_samples_split = st.slider(
            "min_samples_split", 2, 40, 2, key=f"{key_prefix}_mss",
            help="A node with fewer samples than this becomes a leaf without being tested.",
        )
        min_samples_leaf = st.slider(
            "min_samples_leaf", 1, 30, default_leaf, key=f"{key_prefix}_msl",
            help="A split is rejected if it would leave any branch smaller than this. "
                 "This is what keeps ID-like columns out of a C4.5 tree.",
        )
        min_impurity_decrease = st.slider(
            "min_impurity_decrease", 0.0, 0.3, 0.0, step=0.005, key=f"{key_prefix}_mid",
            help="A split whose gain is below this is not worth making.",
        )
        expansion_order = st.radio(
            "Expansion order", ["bfs", "dfs"], horizontal=True, key=f"{key_prefix}_order",
            help="Only changes the order the steps are recorded in — bfs grows the picture "
                 "level by level, which is easier to follow.",
        )

    params: Dict[str, Any] = {
        "impurity": impurity,
        "max_depth": None if unlimited else int(max_depth),
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
        "min_impurity_decrease": float(min_impurity_decrease),
        "expansion_order": expansion_order,
    }

    if algorithm == "C4.5":
        with st.expander("C4.5 options", expanded=False):
            params["missing_strategy"] = st.radio(
                "Missing values", ["fractional", "majority", "drop"], horizontal=True,
                key=f"{key_prefix}_missing",
                help="fractional: send a weighted copy down every branch (real C4.5). "
                     "majority: send the whole sample to the biggest branch. drop: ignore it.",
            )
            params["use_gain_filter"] = st.checkbox(
                "Only consider splits with at least average gain", value=True,
                key=f"{key_prefix}_filter",
                help="Quinlan's guard: without it a tiny gain over a tinier SplitInfo can win.",
            )
    else:
        params["missing_strategy"] = "majority"

    state.algorithm = algorithm
    return {"algorithm": algorithm, **params}


def build_model(algorithm: str, params: Dict[str, Any], task: str = "classification"):
    """Instantiate the right learner for the chosen algorithm."""
    from ..core.c45 import C45Tree
    from ..core.cart import CARTRegressor, CARTTree
    from ..core.id3 import ID3Tree

    kwargs = {k: v for k, v in params.items() if k != "algorithm"}
    if algorithm == "ID3":
        kwargs.pop("use_gain_filter", None)
        return ID3Tree(**kwargs)
    if algorithm == "C4.5":
        return C45Tree(**kwargs)
    kwargs.pop("use_gain_filter", None)
    return CARTRegressor(**kwargs) if task == "regression" else CARTTree(**kwargs)


# --------------------------------------------------------------------------- #
# Tree rendering
# --------------------------------------------------------------------------- #
def render_options_from_state(state: AppState, **overrides) -> RenderOptions:
    options = RenderOptions()
    for key, value in {**state.render_options, **overrides}.items():
        if hasattr(options, key):
            setattr(options, key, value)
    return options


def render_tree_view(
    tree: Tree,
    state: Optional[AppState] = None,
    options: Optional[RenderOptions] = None,
    highlight_path: Optional[Sequence[int]] = None,
    highlight_edges: Optional[Sequence] = None,
    highlight_node: Optional[int] = None,
    dim_nodes: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    show_downloads: bool = True,
    key: str = "tree",
) -> str:
    """Draw the tree and offer the DOT/SVG downloads. Returns the DOT source."""
    if options is None:
        options = render_options_from_state(state) if state is not None else RenderOptions()
    renderer = TreeDotRenderer(options)
    dot = renderer.render(
        tree, highlight_path=highlight_path, highlight_edges=highlight_edges,
        highlight_node=highlight_node, dim_nodes=dim_nodes, title=title,
    )
    # Natural size + horizontal scroll beats stretching: a 3-node tree blown up to
    # full width is unreadable, and a 40-node tree squashed into it is worse.
    st.graphviz_chart(dot, use_container_width=False)

    if show_downloads:
        columns = st.columns(3)
        columns[0].download_button("Download .dot", dot, file_name=f"{key}.dot",
                                   mime="text/vnd.graphviz", width="stretch")
        svg = dot_to_svg(dot) if has_graphviz_binary() else None
        if svg:
            columns[1].download_button("Download .svg", svg, file_name=f"{key}.svg",
                                       mime="image/svg+xml", width="stretch")
        else:
            columns[1].caption("Install `graphviz` (`sudo apt install graphviz`) for SVG export.")
        import json

        columns[2].download_button("Download .json", json.dumps(tree.to_dict(), indent=2, default=str),
                                   file_name=f"{key}.json", mime="application/json", width="stretch")
    return dot


def render_class_legend(tree: Tree) -> None:
    """Colour key — identity must never be carried by colour alone."""
    from ..viz import theme

    classes = [str(c) for c in (tree.class_names or [])]
    if not classes:
        return
    colors = theme.class_colors(classes)
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{colors[name]};'
        f'display:inline-block;"></span><span style="font-size:0.85rem;">{name}</span></span>'
        for name in classes
    )
    st.markdown(f"<div style='margin:4px 0 10px;'>{chips}</div>", unsafe_allow_html=True)


def render_node_card(node: Node, tree: Tree) -> None:
    """Compact summary of one node."""
    columns = st.columns(4)
    columns[0].metric("Node", f"#{node.node_id}")
    columns[1].metric("Depth", node.depth)
    columns[2].metric("Samples", f"{node.n_samples:g}")
    columns[3].metric(node.impurity_name.capitalize(), f"{node.impurity:.4f}")
    conditions = tree.path_conditions(node.node_id)
    st.caption("Path from the root: " + (" AND ".join(conditions) if conditions else "(this is the root)"))
    if node.is_leaf:
        st.success(f"Leaf — predicts **{node.prediction}** "
                   f"({STOP_REASONS.get(node.stop_reason or '', node.stop_reason or 'stopped')})")
    else:
        st.info(f"Splits on **{node.test_description()}**"
                + (f" · gain {node.gain:.4f}" if node.gain is not None else ""))


# --------------------------------------------------------------------------- #
# Step navigator
# --------------------------------------------------------------------------- #
def render_step_navigator(trace: TrainingTrace, state: AppState, key: str = "nav") -> int:
    """Slider plus first/prev/next/last buttons; returns the selected step index."""
    total = len(trace)
    if total == 0:
        return 0

    # A keyed slider ignores its ``value`` argument on reruns, so the buttons have
    # to write into the slider's own session-state entry *before* it is created.
    slider_key = f"{key}_slider"
    stored = st.session_state.get(slider_key, min(state.current_step + 1, total))
    stored = int(min(max(stored, 1), total))

    buttons = st.columns([1, 1, 1, 1, 6])
    if buttons[0].button("⏮", key=f"{key}_first", help="First step", width="stretch"):
        stored = 1
    if buttons[1].button("◀", key=f"{key}_prev", help="Previous step", width="stretch"):
        stored = max(1, stored - 1)
    if buttons[2].button("▶", key=f"{key}_next", help="Next step", width="stretch"):
        stored = min(total, stored + 1)
    if buttons[3].button("⏭", key=f"{key}_last", help="Last step", width="stretch"):
        stored = total
    st.session_state[slider_key] = stored

    with buttons[4]:
        if total > 1:
            current = st.slider(
                "Build step", 1, total, key=slider_key,
                help="Scrub through the construction: each step is one node the learner "
                     "processed.",
            ) - 1
        else:
            st.caption("The tree has a single step.")
            current = 0

    state.current_step = current
    step = trace.steps[current]
    st.caption(f"Step {current + 1} of {total} — node #{step.node_id} at depth {step.depth} "
               f"({step.action})")
    return current
