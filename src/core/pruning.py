"""Post-pruning: three ways to trade fit for simplicity, each step recorded.

Every pruner returns ``(pruned_tree, steps)`` and never mutates the tree it was
given, so the UI can show before and after side by side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import criteria as crit
from . import latex as tex
from . import metrics as mx
from .node import Node, Tree


@dataclass
class PruneStep:
    """One decision made by a pruner, kept so the Tree page can replay it."""

    step_id: int
    node_id: int
    action: str  # "prune" | "keep" | "candidate"
    reason: str
    leaves_before: int
    leaves_after: int
    metric_name: str = "accuracy"
    metric_before: float = 0.0
    metric_after: float = 0.0
    alpha: Optional[float] = None
    computation: Optional[crit.Computation] = None


def _collapse(node: Node) -> None:
    """Turn an internal node into a leaf, keeping its statistics."""
    node.is_leaf = True
    node.children = {}
    node.child_labels = {}
    node.feature_index = None
    node.feature_name = None
    node.split_type = None
    node.threshold = None
    node.categories = None
    node.gain = None
    node.stop_reason = "pruned"
    if node.prediction is None:
        node.prediction = node.majority_class()


def _internal_nodes(tree: Tree) -> List[Node]:
    return [n for n in tree.root.iter_nodes() if not n.is_leaf and n.children]


# --------------------------------------------------------------------------- #
# Reduced error pruning
# --------------------------------------------------------------------------- #
class ReducedErrorPruner:
    """Greedily collapse any subtree that a held-out set says is not earning its keep."""

    name = "Reduced Error Pruning"

    def __init__(self, allow_ties: bool = True) -> None:
        self.allow_ties = allow_ties

    def prune(self, tree: Tree, X_val: Any, y_val: Any) -> Tuple[Tree, List[PruneStep]]:
        working = tree.copy()
        steps: List[PruneStep] = []
        baseline = mx.accuracy(y_val, working.predict(X_val))
        improved = True

        while improved:
            improved = False
            for node in sorted(_internal_nodes(working), key=lambda n: -n.depth):
                saved_children = node.children
                saved_labels = node.child_labels
                saved = (node.feature_index, node.feature_name, node.split_type,
                         node.threshold, node.categories, node.gain, node.is_leaf, node.stop_reason)
                leaves_before = working.count_leaves()

                _collapse(node)
                candidate_score = mx.accuracy(y_val, working.predict(X_val))
                keeps = candidate_score > baseline or (self.allow_ties and candidate_score >= baseline)

                if keeps:
                    steps.append(
                        PruneStep(
                            step_id=len(steps), node_id=node.node_id, action="prune",
                            reason=(f"Collapsing node #{node.node_id} keeps validation accuracy at "
                                    f"{candidate_score:.4f} (was {baseline:.4f}) while removing "
                                    f"{leaves_before - working.count_leaves()} leaves."),
                            leaves_before=leaves_before, leaves_after=working.count_leaves(),
                            metric_before=baseline, metric_after=candidate_score,
                            computation=crit.rule_check_c(
                                "rep", "accuracy after pruning", candidate_score, ">=",
                                "accuracy before pruning", baseline),
                        )
                    )
                    baseline = candidate_score
                    improved = True
                    break

                # Undo: put the subtree back exactly as it was.
                node.children = saved_children
                node.child_labels = saved_labels
                (node.feature_index, node.feature_name, node.split_type, node.threshold,
                 node.categories, node.gain, node.is_leaf, node.stop_reason) = saved
                steps.append(
                    PruneStep(
                        step_id=len(steps), node_id=node.node_id, action="keep",
                        reason=(f"Collapsing node #{node.node_id} would drop validation accuracy to "
                                f"{candidate_score:.4f} (below {baseline:.4f}), so the subtree stays."),
                        leaves_before=leaves_before, leaves_after=leaves_before,
                        metric_before=baseline, metric_after=candidate_score,
                    )
                )
        return working, steps


# --------------------------------------------------------------------------- #
# Cost-complexity pruning (CART)
# --------------------------------------------------------------------------- #
class CostComplexityPruner:
    """Breiman's weakest-link pruning: repeatedly cut the cheapest subtree.

    ``alpha_eff(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)`` is the price per leaf
    of keeping the subtree rooted at ``t``.  Cutting in ascending alpha produces a
    nested sequence of trees, one of which the validation curve picks out.
    """

    name = "Cost-Complexity Pruning"

    def __init__(self, task: str = "classification") -> None:
        self.task = task

    def _node_error(self, node: Node, total: float) -> float:
        if self.task == "regression":
            return (node.impurity * node.n_samples) / total if total else 0.0
        counts = node.class_counts
        if not counts:
            return 0.0
        wrong = sum(counts.values()) - max(counts.values())
        return wrong / total if total else 0.0

    def _subtree_error(self, node: Node, total: float) -> float:
        if node.is_leaf or not node.children:
            return self._node_error(node, total)
        return sum(self._subtree_error(child, total) for child in node.children.values())

    def alpha_sequence(self, tree: Tree) -> List[Tuple[float, Tree, int]]:
        """The nested sequence ``(alpha, tree, n_leaves)``, from full tree upward."""
        total = float(tree.root.n_samples) or 1.0
        current = tree.copy()
        sequence: List[Tuple[float, Tree, int]] = [(0.0, current.copy(), current.count_leaves())]

        while _internal_nodes(current):
            best_node, best_alpha = None, math.inf
            for node in _internal_nodes(current):
                leaves = node.count_leaves()
                if leaves <= 1:
                    continue
                alpha = (self._node_error(node, total) - self._subtree_error(node, total)) / (leaves - 1)
                if alpha < best_alpha:
                    best_node, best_alpha = node, alpha
            if best_node is None:
                break
            _collapse(best_node)
            sequence.append((float(max(best_alpha, 0.0)), current.copy(), current.count_leaves()))
        return sequence

    def prune(
        self, tree: Tree, X_val: Any, y_val: Any, X_train: Any = None, y_train: Any = None
    ) -> Tuple[Tree, List[PruneStep]]:
        """Walk the sequence, score each subtree, keep the best on validation."""
        sequence = self.alpha_sequence(tree)
        steps: List[PruneStep] = []
        best_tree, best_score = tree.copy(), -math.inf

        for index, (alpha, candidate, leaves) in enumerate(sequence):
            if self.task == "regression":
                score = -mx.root_mean_squared_error(y_val, [float(v) for v in candidate.predict(X_val)])
                metric_name = "negative RMSE"
            else:
                score = mx.accuracy(y_val, candidate.predict(X_val))
                metric_name = "accuracy"
            steps.append(
                PruneStep(
                    step_id=index, node_id=-1, action="candidate",
                    reason=(f"alpha = {alpha:.5f} leaves a tree with {leaves} leaves and "
                            f"{metric_name} {score:.4f} on the validation set."),
                    leaves_before=sequence[0][2], leaves_after=leaves,
                    metric_name=metric_name, metric_before=0.0, metric_after=score, alpha=alpha,
                    computation=crit.Computation(
                        name="alpha",
                        symbolic=r"\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}",
                        substituted=r"\alpha = %s,\ |leaves| = %s" % (tex.num(alpha, 5), tex.num(leaves)),
                        result=alpha,
                        operands={"alpha": alpha, "leaves": leaves, "score": score},
                    ),
                )
            )
            if score > best_score:
                best_score, best_tree = score, candidate.copy()
        return best_tree, steps


# --------------------------------------------------------------------------- #
# Pessimistic (error-based) pruning — C4.5
# --------------------------------------------------------------------------- #
Z_FOR_CF = {0.50: 0.00, 0.25: 0.69, 0.10: 1.28, 0.05: 1.65, 0.01: 2.33}


def upper_error_bound(errors: float, n: float, confidence: float = 0.25) -> float:
    """C4.5's pessimistic estimate: the upper end of the error's confidence interval.

    With no validation data, C4.5 assumes the training error is optimistic and
    inflates it; small leaves get inflated the most, which is what makes them
    worth collapsing.
    """
    if n <= 0:
        return 0.0
    z = Z_FOR_CF.get(round(confidence, 2), 0.69)
    f = errors / n
    numerator = f + (z * z) / (2 * n) + z * math.sqrt(max(f / n - (f * f) / n + (z * z) / (4 * n * n), 0.0))
    return numerator / (1 + (z * z) / n)


class PessimisticPruner:
    """C4.5 error-based pruning — needs no held-out data at all."""

    name = "Pessimistic (error-based) Pruning"

    def __init__(self, confidence: float = 0.25) -> None:
        self.confidence = confidence

    def _leaf_predicted_errors(self, node: Node) -> float:
        counts = node.class_counts
        n = float(sum(counts.values()))
        wrong = n - (max(counts.values()) if counts else 0.0)
        return n * upper_error_bound(wrong, n, self.confidence)

    def _subtree_predicted_errors(self, node: Node) -> float:
        if node.is_leaf or not node.children:
            return self._leaf_predicted_errors(node)
        return sum(self._subtree_predicted_errors(child) for child in node.children.values())

    def prune(self, tree: Tree, X_val: Any = None, y_val: Any = None) -> Tuple[Tree, List[PruneStep]]:
        working = tree.copy()
        steps: List[PruneStep] = []
        changed = True
        while changed:
            changed = False
            for node in sorted(_internal_nodes(working), key=lambda n: -n.depth):
                as_leaf = self._leaf_predicted_errors(node)
                as_subtree = self._subtree_predicted_errors(node)
                leaves_before = working.count_leaves()
                computation = crit.Computation(
                    name="pessimistic",
                    symbolic=r"E_{\text{leaf}} \leq E_{\text{subtree}}",
                    substituted=r"%s \leq %s" % (tex.num(as_leaf), tex.num(as_subtree)),
                    result=as_leaf,
                    operands={"leaf": as_leaf, "subtree": as_subtree, "cf": self.confidence},
                    note=("Predicted errors use the upper bound of a binomial confidence interval "
                          f"at CF={self.confidence}."),
                )
                if as_leaf <= as_subtree + 1e-9:
                    _collapse(node)
                    steps.append(
                        PruneStep(
                            step_id=len(steps), node_id=node.node_id, action="prune",
                            reason=(f"Node #{node.node_id} as a leaf has a predicted error of {as_leaf:.3f} "
                                    f"versus {as_subtree:.3f} for its subtree, so the subtree is removed."),
                            leaves_before=leaves_before, leaves_after=working.count_leaves(),
                            metric_name="predicted errors", metric_before=as_subtree, metric_after=as_leaf,
                            computation=computation,
                        )
                    )
                    changed = True
                    break
                steps.append(
                    PruneStep(
                        step_id=len(steps), node_id=node.node_id, action="keep",
                        reason=(f"Node #{node.node_id} as a leaf would predict {as_leaf:.3f} errors, worse "
                                f"than its subtree's {as_subtree:.3f}, so the subtree stays."),
                        leaves_before=leaves_before, leaves_after=leaves_before,
                        metric_name="predicted errors", metric_before=as_subtree, metric_after=as_leaf,
                        computation=computation,
                    )
                )
        return working, steps


PRUNERS = {
    "rep": ReducedErrorPruner,
    "ccp": CostComplexityPruner,
    "pessimistic": PessimisticPruner,
}
