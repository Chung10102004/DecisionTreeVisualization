"""Presentation layer for the formulas.

``core.criteria`` computes each quantity and records the symbolic form plus the
form with real numbers substituted.  This module lays that out: three stacked
lines (*symbolic -> substituted -> result*), chains of related formulas, the
candidates table, and a full "worked example" a student could copy into a report.

Returns strings and dataframes only — no Streamlit here, so it is unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..core import latex as tex
from ..core.criteria import Computation
from ..core.splitter import SplitCandidate
from ..core.trace import ACTION_LEAF, ACTION_SPLIT, STOP_REASONS, BuildStep


@dataclass
class FormulaBlock:
    """One formula ready for display: the three lines plus an optional note."""

    title: str
    symbolic: str
    substituted: str
    result: str
    note: Optional[str] = None
    caption: Optional[str] = None

    def lines(self) -> List[str]:
        return [self.symbolic, self.substituted, self.result]

    def one_line(self) -> str:
        return f"{self.symbolic} = {self.substituted} = {self.result}"

    def as_markdown(self) -> str:
        parts = [f"**{self.title}**", "", f"$$\\begin{{aligned}}"
                 f"&{self.symbolic} \\\\ &= {self.substituted} \\\\ &= {self.result}"
                 f"\\end{{aligned}}$$"]
        if self.note:
            parts.append(f"> {self.note}")
        return "\n".join(parts)


class MathFormatter:
    """Formats :class:`Computation` objects; precision is a display choice only."""

    def __init__(self, precision: int = 3, show_symbolic: bool = True) -> None:
        self.precision = int(precision)
        self.show_symbolic = show_symbolic

    def number(self, value: Any) -> str:
        return tex.num(value, self.precision)

    def render(self, comp: Computation, title: Optional[str] = None) -> FormulaBlock:
        return FormulaBlock(
            title=title or _title_for(comp),
            symbolic=comp.symbolic,
            substituted=comp.substituted,
            result=self.number(comp.result),
            note=comp.note,
        )

    def render_chain(self, comps: Sequence[Computation], titles: Optional[Sequence[str]] = None
                     ) -> List[FormulaBlock]:
        titles = list(titles) if titles else [None] * len(comps)
        return [self.render(c, t) for c, t in zip(comps, titles)]

    # ------------------------------------------------------------- node maths
    def node_impurity(self, step: BuildStep) -> FormulaBlock:
        return self.render(step.impurity_computation, title=f"Impurity of node #{step.node_id}")

    def candidate_chain(self, candidate: SplitCandidate) -> List[FormulaBlock]:
        """Parent impurity -> each child -> weighted -> gain -> split info -> ratio."""
        blocks: List[FormulaBlock] = []
        comps = candidate.computations
        if "parent_impurity" in comps:
            blocks.append(self.render(comps["parent_impurity"], "Impurity before the split"))

        weighted = comps.get("weighted_impurity")
        if weighted is not None:
            for branch, child in weighted.children.items():
                label = candidate.branch_labels.get(branch, branch)
                blocks.append(self.render(child, f"Branch [{label}]"))
            blocks.append(self.render(weighted, "Weighted impurity of the children"))
        for key, title in (("gain", None), ("split_info", "Split information"),
                           ("gain_ratio", "Gain ratio")):
            if key in comps:
                blocks.append(self.render(comps[key], title))
        return blocks

    def selection(self, step: BuildStep) -> Optional[FormulaBlock]:
        if step.selection_computation is None:
            return None
        return self.render(step.selection_computation, "Choosing the winning split")

    def stop_rule(self, step: BuildStep) -> Optional[FormulaBlock]:
        if step.stop_computation is None:
            return None
        return self.render(step.stop_computation, "Stopping rule that fired")

    # ---------------------------------------------------------------- tables
    def candidates_dataframe(self, step: BuildStep) -> pd.DataFrame:
        """One row per candidate split, best first, losers included."""
        rows: List[Dict[str, Any]] = []
        for rank, candidate in enumerate(step.ranked_candidates, start=1):
            rows.append(
                {
                    "Rank": rank if candidate.is_valid else None,
                    "Feature": candidate.feature_name,
                    "Test": candidate.describe(),
                    "Branches": candidate.n_branches,
                    "Branch sizes": ", ".join(
                        f"{candidate.branch_labels.get(k, k)}={candidate.child_sizes.get(k, 0):g}"
                        for k in candidate.partitions
                    ),
                    "Child impurities": ", ".join(
                        f"{v:.3f}" for v in candidate.child_impurities.values()
                    ),
                    "Weighted impurity": round(candidate.weighted_impurity, 4),
                    "Gain": round(candidate.gain, 4),
                    "Split info": round(candidate.split_info, 4),
                    "Gain ratio": round(candidate.gain_ratio, 4),
                    "Usable": "yes" if candidate.is_valid else "no",
                    "Why not": candidate.reject_reason or "",
                }
            )
        return pd.DataFrame(rows)

    def child_counts_dataframe(self, candidate: SplitCandidate) -> pd.DataFrame:
        rows = []
        for key in candidate.partitions:
            row: Dict[str, Any] = {"Branch": candidate.branch_labels.get(key, key)}
            row.update({str(k): v for k, v in candidate.child_class_counts.get(key, {}).items()})
            row["n"] = candidate.child_sizes.get(key, 0)
            row["Impurity"] = round(candidate.child_impurities.get(key, 0.0), 4)
            rows.append(row)
        return pd.DataFrame(rows).fillna(0)

    # -------------------------------------------------------- worked example
    def worked_example(self, step: BuildStep, impurity_display: str, score_name: str) -> str:
        """The whole step written out the way it would be solved on paper."""
        lines: List[str] = [
            f"### Step {step.step_id + 1} — Node #{step.node_id} (depth {step.depth})",
            "",
            f"**Path to this node:** {step.path_text()}",
            "",
            f"**Samples here:** {step.n_samples:g}  |  **Class counts:** {step.counts_text() or '-'}",
            "",
            "#### 1. Impurity before splitting",
            "",
            self.render(step.impurity_computation).as_markdown(),
            "",
        ]

        if step.action == ACTION_LEAF:
            lines += [
                "#### 2. Why this node stops here",
                "",
                f"{STOP_REASONS.get(step.stop_reason or '', step.stop_reason or 'no split was possible').capitalize()}.",
                "",
            ]
            block = self.stop_rule(step)
            if block:
                lines += [block.as_markdown(), ""]
            lines += [f"**Result:** leaf predicting `{step.prediction}`.", ""]
            return "\n".join(lines)

        lines += ["#### 2. Score every candidate split", "",
                  markdown_table(self.candidates_dataframe(step)), ""]

        if step.chosen is not None:
            lines += [f"#### 3. Full calculation for the winner ({step.chosen.describe()})", ""]
            for block in self.candidate_chain(step.chosen):
                lines += [block.as_markdown(), ""]

        selection = self.selection(step)
        if selection:
            lines += ["#### 4. The selection itself", "", selection.as_markdown(), ""]

        if step.chosen is not None:
            lines += [
                f"**Result:** split on `{step.chosen.feature_name}` into "
                f"{step.chosen.n_branches} branches "
                f"({step.chosen.branch_summary()}).",
                "",
            ]
        return "\n".join(lines)

    def worked_example_document(self, trace, title: str = "Decision tree — worked example") -> str:
        """Every step of a fit, concatenated into one report-ready document."""
        from ..core.criteria import get_impurity

        impurity = get_impurity(trace.impurity_name)
        header = [
            f"# {title}",
            "",
            f"**Algorithm:** {trace.algorithm}  |  **Impurity:** {impurity.display}  "
            f"|  **Selection criterion:** {trace.score_name}",
            "",
            f"**Parameters:** {trace.params}",
            "",
        ]
        if trace.notes:
            header += ["**Notes**", ""] + [f"- {note}" for note in trace.notes] + [""]
        body = [self.worked_example(step, impurity.display, trace.score_name) for step in trace.steps]
        return "\n".join(header + body)


def markdown_table(frame: pd.DataFrame) -> str:
    """A GitHub-flavoured table without pulling in ``tabulate``."""
    if frame.empty:
        return "_(no rows)_"
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for _index, row in frame.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _title_for(comp: Computation) -> str:
    return {
        "entropy": "Entropy",
        "gini": "Gini index",
        "error": "Misclassification error",
        "mse": "Mean squared error",
        "mae": "Mean absolute error",
        "weighted_impurity": "Weighted impurity of the children",
        "gain": "Gain",
        "split_info": "Split information",
        "gain_ratio": "Gain ratio",
        "argmax": "Choosing the winning split",
        "comparison": "Test at this node",
        "probability": "Leaf probability",
        "threshold": "Threshold",
        "threshold_choice": "Threshold search",
        "subset_choice": "Subset search",
        "alpha": "Cost-complexity alpha",
        "pessimistic": "Pessimistic error comparison",
    }.get(comp.name, comp.name.replace("_", " ").capitalize())


DEFAULT_FORMATTER = MathFormatter()
