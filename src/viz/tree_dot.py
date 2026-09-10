"""Rendering a tree as Graphviz DOT.

Streamlit's ``st.graphviz_chart`` renders DOT in the browser, so this module only
has to produce a correct string — no Graphviz binary is required.  Node fill
colour encodes the majority class, and its intensity encodes purity, so a glance
at the diagram already tells you where the tree is confident.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.node import Node, Tree

from . import theme

PALETTE = theme.CATEGORICAL

HIGHLIGHT = theme.HIGHLIGHT
PENDING_FILL = "#F2F1EE"
PENDING_LINE = theme.TEXT_MUTED
LEAF_LINE = theme.TEXT_SECONDARY
NODE_LINE = theme.TEXT_SECONDARY
TEXT = theme.TEXT_PRIMARY
MUTED = theme.TEXT_MUTED


def _blend(hex_color: str, ratio: float) -> str:
    """Mix a colour toward the chart surface; ratio 1.0 keeps it fully saturated."""
    return theme.blend_to_surface(hex_color, ratio)


def _escape(text: Any) -> str:
    """Escape for a Graphviz HTML-like label."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


@dataclass
class RenderOptions:
    """Every display toggle the Tree page exposes, in one place."""

    show_impurity: bool = True
    show_samples: bool = True
    show_counts: bool = True
    show_distribution_bar: bool = True
    show_percentages: bool = False
    show_node_ids: bool = True
    orientation: str = "TB"          # TB (top-down) or LR (left-right)
    max_display_depth: Optional[int] = None
    focus_node_id: Optional[int] = None
    font: str = "Helvetica"
    font_size: int = 11


class TreeDotRenderer:
    """Turns a :class:`Tree` into DOT, with optional highlighting."""

    def __init__(self, options: Optional[RenderOptions] = None) -> None:
        self.options = options or RenderOptions()

    # ------------------------------------------------------------- colouring
    def class_colors(self, tree: Tree) -> Dict[str, str]:
        classes = [str(c) for c in (tree.class_names or [])]
        if not classes:
            classes = sorted({str(k) for n in tree.root.iter_nodes() for k in n.class_counts})
        return theme.class_colors(classes)

    def _node_fill(self, node: Node, colors: Dict[str, str], task: str) -> str:
        if node.is_pending:
            return PENDING_FILL
        if task == "regression" or not node.class_counts:
            return theme.SEQUENTIAL[1]
        majority = str(node.majority_class())
        base = colors.get(majority, PALETTE[0])
        n_classes = max(len(colors), 2)
        floor = 1.0 / n_classes
        purity = node.confidence()
        strength = 0.18 + 0.62 * max(0.0, (purity - floor) / max(1e-9, 1 - floor))
        return _blend(base, strength)

    # ----------------------------------------------------------------- label
    def _distribution_bar(self, node: Node, colors: Dict[str, str]) -> str:
        total = float(sum(node.class_counts.values()))
        if total <= 0:
            return ""
        cells = []
        for label, count in node.class_counts.items():
            if count <= 0:
                continue
            width = max(4, int(round(120 * count / total)))
            cells.append(
                f'<TD BGCOLOR="{colors.get(str(label), PALETTE[0])}" '
                f'WIDTH="{width}" HEIGHT="7" FIXEDSIZE="TRUE"></TD>'
            )
        if not cells:
            return ""
        return ('<TR><TD><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
                f'<TR>{"".join(cells)}</TR></TABLE></TD></TR>')

    def _label(self, node: Node, tree: Tree, colors: Dict[str, str], hidden_below: int = 0) -> str:
        options = self.options
        rows: List[str] = []

        if node.is_pending:
            rows.append(f'<TR><TD><FONT COLOR="{MUTED}"><B>node #{node.node_id}</B></FONT></TD></TR>')
            rows.append(f'<TR><TD><FONT COLOR="{MUTED}">not expanded yet</FONT></TD></TR>')
            rows.append(f'<TR><TD><FONT COLOR="{MUTED}">samples = {node.n_samples:g}</FONT></TD></TR>')
            return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">' + "".join(rows) + "</TABLE>>"

        header = node.test_description() if not node.is_leaf else f"predict: {node.prediction}"
        if hidden_below:
            header = f"{node.test_description()}"
        prefix = f"#{node.node_id} · " if options.show_node_ids else ""
        rows.append(f"<TR><TD><B>{_escape(prefix + header)}</B></TD></TR>")

        if options.show_impurity:
            rows.append(f"<TR><TD>{_escape(node.impurity_name)} = {node.impurity:.3f}</TD></TR>")
        if options.show_samples:
            rows.append(f"<TR><TD>samples = {node.n_samples:g}</TD></TR>")
        if options.show_counts and node.class_counts:
            if options.show_percentages:
                text = ", ".join(f"{k}: {v:.0%}" for k, v in node.probabilities().items())
            else:
                text = ", ".join(f"{k}={v:g}" for k, v in node.class_counts.items())
            rows.append(f"<TR><TD>{_escape(text)}</TD></TR>")
        if tree.task == "regression" and node.prediction is not None:
            rows.append(f"<TR><TD>mean = {float(node.prediction):.4g}</TD></TR>")
        if options.show_distribution_bar and node.class_counts:
            bar = self._distribution_bar(node, colors)
            if bar:
                rows.append(bar)
        if node.is_leaf and node.stop_reason:
            rows.append(f'<TR><TD><FONT COLOR="{MUTED}" POINT-SIZE="9">'
                        f"[{_escape(node.stop_reason)}]</FONT></TD></TR>")
        if hidden_below:
            rows.append(f'<TR><TD><FONT COLOR="{MUTED}" POINT-SIZE="9">'
                        f"… {hidden_below} more node(s) hidden</FONT></TD></TR>")

        return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">' + "".join(rows) + "</TABLE>>"

    # ---------------------------------------------------------------- render
    def render(
        self,
        tree: Tree,
        highlight_path: Optional[Sequence[int]] = None,
        highlight_edges: Optional[Sequence[Tuple[int, str]]] = None,
        highlight_node: Optional[int] = None,
        dim_nodes: Optional[Sequence[int]] = None,
        max_display_depth: Optional[int] = None,
        focus_node_id: Optional[int] = None,
        title: Optional[str] = None,
    ) -> str:
        options = self.options
        colors = self.class_colors(tree)
        path = set(highlight_path or [])
        edges = set(highlight_edges or [])
        dim = set(dim_nodes or [])
        depth_limit = max_display_depth if max_display_depth is not None else options.max_display_depth
        focus_id = focus_node_id if focus_node_id is not None else options.focus_node_id

        root = tree.root
        if focus_id is not None:
            root = tree.nodes_by_id.get(focus_id, tree.root)
        base_depth = root.depth

        lines: List[str] = [
            "digraph DecisionTree {",
            f"  rankdir={options.orientation};",
            "  bgcolor=\"transparent\";",
            "  splines=polyline;",
            "  nodesep=0.35;",
            "  ranksep=0.5;",
            f'  node [shape=box, style="rounded,filled", fontname="{options.font}", '
            f'fontsize={options.font_size}, fontcolor="{TEXT}", penwidth=1.2, margin="0.12,0.08"];',
            f'  edge [fontname="{options.font}", fontsize={max(options.font_size - 2, 8)}, '
            f'color="{NODE_LINE}", arrowsize=0.7];',
        ]
        if title:
            lines.append(f'  labelloc="t"; fontname="{options.font}"; '
                         f'fontsize={options.font_size + 2}; label="{_escape(title)}";')

        def emit(node: Node) -> None:
            relative_depth = node.depth - base_depth
            truncated = depth_limit is not None and relative_depth >= depth_limit and node.children
            hidden = (node.count_nodes() - 1) if truncated else 0

            fill = self._node_fill(node, colors, tree.task)
            border = PENDING_LINE if node.is_pending else (LEAF_LINE if node.is_leaf else NODE_LINE)
            width = 1.2
            style = "rounded,filled,dashed" if node.is_pending else "rounded,filled"

            if node.node_id in dim:
                fill, border = "#EFEFEF", "#C0C0C0"
            if node.node_id in path:
                border, width = HIGHLIGHT, 2.6
            if highlight_node is not None and node.node_id == highlight_node:
                border, width = HIGHLIGHT, 3.2
                style += ",bold"

            lines.append(
                f'  n{node.node_id} [label={self._label(node, tree, colors, hidden)}, '
                f'fillcolor="{fill}", color="{border}", penwidth={width}, style="{style}"];'
            )

            if truncated or node.is_pending:
                return

            for key, child in node.children.items():
                emit(child)
                is_hot = (node.node_id, key) in edges
                label = _escape(node.branch_condition(key))
                attributes = [f'label=" {label} "']
                if is_hot:
                    attributes += [f'color="{HIGHLIGHT}"', "penwidth=2.4",
                                   f'fontcolor="{HIGHLIGHT}"']
                lines.append(f"  n{node.node_id} -> n{child.node_id} [{', '.join(attributes)}];")

        emit(root)
        lines.append("}")
        return "\n".join(lines)

    def render_node_zoom(self, tree: Tree, node_id: int) -> str:
        """Just one node and its immediate children — the current step, enlarged."""
        node = tree.nodes_by_id.get(node_id)
        if node is None:
            return self.render(tree)
        return self.render(tree, focus_node_id=node_id, max_display_depth=1, highlight_node=node_id)

    def legend_dot(self, tree: Tree) -> str:
        """A tiny standalone graph mapping colours to class names."""
        colors = self.class_colors(tree)
        cells = "".join(
            f'<TR><TD BGCOLOR="{color}" WIDTH="18" HEIGHT="12" FIXEDSIZE="TRUE"></TD>'
            f"<TD ALIGN=\"LEFT\">{_escape(name)}</TD></TR>"
            for name, color in colors.items()
        )
        return (
            "digraph Legend {\n  bgcolor=\"transparent\";\n"
            f'  node [shape=plaintext, fontname="{self.options.font}", '
            f'fontsize={self.options.font_size}, fontcolor="{TEXT}"];\n'
            f'  legend [label=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">{cells}</TABLE>>];\n'
            "}"
        )


def render_text_tree(tree: Tree, node: Optional[Node] = None, prefix: str = "",
                     is_last: bool = True, is_root: bool = True) -> str:
    """An indented text version — always readable, even for a huge tree."""
    node = node or tree.root
    lines: List[str] = []

    if is_root:
        head = node.test_description() if not node.is_leaf else f"predict: {node.prediction}"
        lines.append(f"[#{node.node_id}] {head}   "
                     f"(n={node.n_samples:g}, {node.impurity_name}={node.impurity:.3f})")
    children = list(node.children.items())
    for index, (key, child) in enumerate(children):
        last = index == len(children) - 1
        connector = "└── " if last else "├── "
        condition = node.branch_condition(key)
        if child.is_leaf:
            body = f"=> {child.prediction}"
        else:
            body = child.test_description()
        lines.append(
            f"{prefix}{connector}{node.feature_name} {condition}: [#{child.node_id}] {body}   "
            f"(n={child.n_samples:g}, {child.impurity_name}={child.impurity:.3f})"
        )
        extension = "    " if last else "│   "
        nested = render_text_tree(tree, child, prefix + extension, last, is_root=False)
        if nested:
            lines.append(nested)
    return "\n".join(line for line in lines if line)


def has_graphviz_binary() -> bool:
    """True when a local ``dot`` exists, which is only needed for PNG/SVG export."""
    import shutil

    return shutil.which("dot") is not None


def dot_to_svg(dot: str) -> Optional[bytes]:
    """Render to SVG with the local binary, or return None if it is not installed."""
    import subprocess

    if not has_graphviz_binary():
        return None
    try:
        result = subprocess.run(["dot", "-Tsvg"], input=dot.encode("utf-8"),
                                capture_output=True, check=True, timeout=30)
        return result.stdout
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - environment dependent
        return None
