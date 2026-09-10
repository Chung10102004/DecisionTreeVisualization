"""Training and prediction traces — the record of *how* the tree happened.

``TrainingTrace`` holds one :class:`BuildStep` per node the learner touched, in
the exact order it touched them, which is what the Build page scrubs through.
``PredictionTrace`` holds one :class:`PredictStep` per hop a sample takes on its
way to a leaf, which is what the Predict page replays.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import criteria as crit
from .node import Node, Tree
from .splitter import SplitCandidate, rank_candidates

ACTION_SPLIT = "SPLIT"
ACTION_LEAF = "LEAF"

STOP_REASONS = {
    "pure": "every sample in this node has the same label",
    "max_depth": "the node sits at the configured maximum depth",
    "min_samples_split": "the node holds too few samples to be worth splitting",
    "min_samples_leaf": "any split would create a branch below min_samples_leaf",
    "no_features": "no unused feature is left to test",
    "no_valid_split": "no candidate split separates the samples",
    "no_gain": "the best split does not reduce impurity enough",
    "empty": "the node received no samples",
}


@dataclass
class BuildStep:
    """Everything the learner knew and decided at one node."""

    step_id: int
    node_id: int
    parent_id: Optional[int]
    depth: int
    path_conditions: List[str]

    sample_indices: np.ndarray
    n_samples: float
    class_counts: Dict[Any, float]
    impurity_name: str
    impurity_value: float
    impurity_computation: crit.Computation

    candidates: List[SplitCandidate] = field(default_factory=list)
    chosen: Optional[SplitCandidate] = None
    selection_computation: Optional[crit.Computation] = None
    action: str = ACTION_LEAF
    stop_reason: Optional[str] = None
    stop_computation: Optional[crit.Computation] = None
    prediction: Any = None
    narrative: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def ranked_candidates(self) -> List[SplitCandidate]:
        return rank_candidates(self.candidates)

    @property
    def node_label(self) -> str:
        return f"Node #{self.node_id}"

    def counts_text(self) -> str:
        return ", ".join(f"{k}={v:g}" for k, v in self.class_counts.items())

    def path_text(self) -> str:
        return " AND ".join(self.path_conditions) if self.path_conditions else "(root — no conditions yet)"


def narrate(step: BuildStep, impurity_display: str, score_name: str) -> str:
    """One plain-English sentence describing what happened at this node."""
    where = f"{step.node_label} (depth {step.depth})"
    counts = step.counts_text()
    head = (f"{where} holds {step.n_samples:g} samples ({counts}); "
            f"{impurity_display} = {step.impurity_value:.4f}.")

    if step.action == ACTION_SPLIT and step.chosen is not None:
        n_valid = sum(1 for c in step.candidates if c.is_valid)
        return (f"{head} Of the {n_valid} usable candidate split(s), "
                f"'{step.chosen.describe()}' has the highest {score_name} "
                f"({step.chosen.score:.4f}), so the node is split on {step.chosen.feature_name} "
                f"into {step.chosen.n_branches} branches.")

    reason = STOP_REASONS.get(step.stop_reason or "", step.stop_reason or "no split was possible")
    return f"{head} It becomes a LEAF predicting '{step.prediction}' because {reason}."


@dataclass
class TrainingTrace:
    """The ordered log of a single ``fit`` call."""

    algorithm: str
    impurity_name: str
    score_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    steps: List[BuildStep] = field(default_factory=list)
    tree: Optional[Tree] = None
    preamble: List[crit.Computation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def step(self, index: int) -> BuildStep:
        return self.steps[index]

    def step_for_node(self, node_id: int) -> Optional[BuildStep]:
        for step in self.steps:
            if step.node_id == node_id:
                return step
        return None

    # --------------------------------------------------------- partial trees
    def tree_at_step(self, k: int) -> Tree:
        """The tree as it looked right after step ``k`` finished.

        Nodes the learner has not reached yet are kept as *pending* placeholders
        so the picture grows rather than jumping around between steps.
        """
        if self.tree is None:  # pragma: no cover - fit always sets it
            raise ValueError("Trace has no tree attached.")
        k = max(-1, min(int(k), len(self.steps) - 1))
        processed = {step.node_id for step in self.steps[: k + 1]}

        def clone(node: Node) -> Node:
            new = copy.copy(node)
            new.children = {}
            new.child_labels = dict(node.child_labels)
            if node.node_id in processed:
                new.is_pending = False
                for key, child in node.children.items():
                    new.children[key] = clone(child)
            else:
                new.is_pending = True
                new.is_leaf = True
                new.child_labels = {}
            return new

        partial = copy.copy(self.tree)
        partial.root = clone(self.tree.root)
        return partial

    def frontier_at_step(self, k: int) -> List[Node]:
        """Nodes that exist but are still waiting in the queue at step ``k``."""
        return [n for n in self.tree_at_step(k).root.iter_nodes() if n.is_pending]

    def current_node_id(self, k: int) -> Optional[int]:
        if 0 <= k < len(self.steps):
            return self.steps[k].node_id
        return None

    # ---------------------------------------------------------------- tables
    def summary_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for step in self.steps:
            rows.append(
                {
                    "Step": step.step_id + 1,
                    "Node": step.node_id,
                    "Depth": step.depth,
                    "Parent": step.parent_id if step.parent_id is not None else "-",
                    "Samples": step.n_samples,
                    "Class counts": step.counts_text(),
                    self.impurity_name.capitalize(): round(step.impurity_value, 4),
                    "Action": step.action,
                    "Split on": step.chosen.describe() if step.chosen else "-",
                    self.score_name: round(step.chosen.score, 4) if step.chosen else None,
                    "Stop reason": step.stop_reason or "-",
                }
            )
        return rows

    def summary_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.summary_rows())


# --------------------------------------------------------------------------- #
# Prediction side
# --------------------------------------------------------------------------- #
@dataclass
class PredictStep:
    """One hop of a sample down the tree."""

    index: int
    node_id: int
    depth: int
    feature: Optional[str]
    sample_value: Any
    test: str
    branch_key: Optional[str]
    branch_label: Optional[str]
    outcome_text: str
    node_class_counts: Dict[Any, float]
    node_probabilities: Dict[Any, float]
    n_samples: float
    is_leaf: bool
    unseen_value: bool = False
    comparison: Optional[crit.Computation] = None
    narrative: str = ""


@dataclass
class PredictionTrace:
    """The full root-to-leaf story for one sample."""

    sample: Dict[str, Any]
    steps: List[PredictStep] = field(default_factory=list)
    path_node_ids: List[int] = field(default_factory=list)
    final_leaf_id: Optional[int] = None
    predicted_class: Any = None
    probabilities: Dict[Any, float] = field(default_factory=dict)
    probability_computation: Optional[crit.Computation] = None
    rule_text: str = ""
    confidence: float = 0.0
    support: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def path_edges(self) -> List[tuple]:
        """``(parent_id, branch_key)`` pairs, used to highlight edges in the diagram."""
        return [(s.node_id, s.branch_key) for s in self.steps if s.branch_key is not None]

    def as_text(self) -> str:
        lines = [f"Sample: {self.sample}"]
        for step in self.steps:
            lines.append(f"  Step {step.index + 1}: {step.narrative}")
        lines.append(f"  => Prediction: {self.predicted_class} (confidence {self.confidence:.1%})")
        return "\n".join(lines)
