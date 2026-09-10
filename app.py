"""Decision Tree Explainer — Streamlit entry point.

This file only wires pages together; every piece of behaviour lives in ``src/``:

    src/core      the algorithms (ID3, C4.5, CART), written from scratch
    src/datasets  loading and preparing data
    src/viz       DOT, matplotlib figures, and LaTeX formatting
    src/ui        the Streamlit layer

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.ui.pages import (
    page_build,
    page_compare,
    page_data,
    page_evaluate,
    page_predict,
    page_theory,
    page_tree,
)

st.set_page_config(
    page_title="Decision Tree Explainer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Every page module exposes a function called ``render``, so each one needs an
# explicit url_path — Streamlit would otherwise infer the same path for all of them.
PAGES = [
    st.Page(page_data.render, title="1 · Data", icon="📂", url_path="data", default=True),
    st.Page(page_build.render, title="2 · Build", icon="🪜", url_path="build"),
    st.Page(page_tree.render, title="3 · Tree", icon="🌳", url_path="tree"),
    st.Page(page_predict.render, title="4 · Predict", icon="🔎", url_path="predict"),
    st.Page(page_evaluate.render, title="5 · Evaluate", icon="📊", url_path="evaluate"),
    st.Page(page_compare.render, title="6 · Compare", icon="⚖️", url_path="compare"),
    st.Page(page_theory.render, title="7 · Theory", icon="📘", url_path="theory"),
]


def main() -> None:
    with st.sidebar:
        st.markdown("## 🌳 Decision Tree Explainer")
        st.caption("ID3 · C4.5 · CART, implemented from scratch — every number shown with the "
                   "formula it came from.")
    navigation = st.navigation(PAGES, position="sidebar")
    navigation.run()


if __name__ == "__main__":
    main()
