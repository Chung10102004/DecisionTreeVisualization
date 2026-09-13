"""The shared tree-growing engine.

The loop is deliberately *iterative* over an explicit queue rather than
recursive: one iteration == one node == one :class:`BuildStep`, which is what
makes the Build page's step slider possible at all.

Subclasses (ID3 / C4.5 / CART) only say which impurity to use, which splitter to
build, and how a candidate is scored.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import criteria as crit
from . import latex as tex
from .node import Node, Tree, is_missing
from .splitter import SplitCandidate, Splitter, argmax_computation, rank_candidates
from .trace import (
    ACTION_LEAF,
    ACTION_SPLIT,
    BuildStep,
    PredictionTrace,
    PredictStep,
    TrainingTrace,
    narrate,
)


class BaseDecisionTree(ABC):
    """Common machinery for every algorithm in this project."""

    algorithm: str = "BASE"
    default_impurity: str = "entropy"
    score_key: str = "gain"
    consume_features: bool = False
    splitter_class: type = Splitter

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[int] = None,
        expansion_order: str = "bfs",
        impurity: Optional[str] = None,
        missing_strategy: str = "majority",
        max_thresholds: Optional[int] = None,
        task: str = "classification",
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features
        self.expansion_order = expansion_order
        self.impurity_key = impurity or self.default_impurity
        self.missing_strategy = missing_strategy
        self.max_thresholds = max_thresholds
        self.task = task
        self.random_state = random_state

        self.tree_: Optional[Tree] = None
        self.trace_: Optional[TrainingTrace] = None
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------- subclass API
    @abstractmethod
    def _make_splitter(self) -> Splitter:
        """Build the splitter that decides which candidates exist."""

    def _select(self, candidates: Sequence[SplitCandidate]) -> Optional[SplitCandidate]:
        """Pick the winning candidate. Overridden by C4.5 for its gain filter."""
        ranked = rank_candidates(candidates, only_valid=True)
        return ranked[0] if ranked else None

    def _leaf_value(self, indices: np.ndarray) -> Any:
        if self.task == "regression":
            values = np.asarray(self._y[indices], dtype=float)
            weights = self._w[indices]
            total = float(np.sum(weights))
            return float(np.sum(values * weights) / total) if total > 0 else 0.0
        return crit.majority_class(self._y[indices], self._w[indices])

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[Sequence[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "BaseDecisionTree":
        self._ingest(X, y, feature_names, feature_types, sample_weights)
        self.impurity = crit.get_impurity(self.impurity_key)
        # The trace has to exist before the splitter is built: subclasses record
        # setup warnings (ID3 meeting a numeric column, for instance) while
        # constructing theirs.
        self.trace_ = TrainingTrace(
            algorithm=self.algorithm,
            impurity_name=self.impurity.key,
            score_name=self._score_display(),
            params=self.get_params(),
        )
        self._splitter = self._make_splitter()
        self._node_counter = 0
        self._nodes: Dict[int, Node] = {}

        root = self._new_node(np.arange(len(self._y)), depth=0, parent=None, branch_key=None, branch_label=None)
        queue: deque = deque([(root, list(range(len(self._feature_names))))])

        while queue:
            node, available = queue.popleft() if self.expansion_order == "bfs" else queue.pop()
            children = self._process_node(node, available)
            for child, child_available in children:
                queue.append((child, child_available))

        self.tree_ = Tree(
            root=root,
            feature_names=list(self._feature_names),
            feature_types=dict(self._feature_types),
            class_names=self._class_names,
            algorithm=self.algorithm,
            task=self.task,
            impurity_name=self.impurity.key,
            params=self.get_params(),
            n_train=int(self._n_original),
        )
        self.trace_.tree = self.tree_
        return self

    def _ingest(self, X, y, feature_names, feature_types, sample_weights) -> None:
        """Normalise inputs into object arrays the rest of the class relies on."""
        if hasattr(X, "columns"):
            feature_names = feature_names or list(X.columns)
            X = X.values
        self._X = np.asarray(X, dtype=object)
        if self._X.ndim == 1:
            self._X = self._X.reshape(-1, 1)
        self._y = np.asarray(list(y), dtype=object)
        if len(self._X) != len(self._y):
            raise ValueError(f"X has {len(self._X)} rows but y has {len(self._y)}.")

        self._feature_names = list(feature_names) if feature_names is not None else [
            f"feature_{i}" for i in range(self._X.shape[1])
        ]
        self._feature_types = dict(feature_types or {})
        for name in self._feature_names:
            self._feature_types.setdefault(name, "categorical")

        self._w = (np.ones(len(self._y), dtype=float) if sample_weights is None
                   else np.asarray(sample_weights, dtype=float))
        self._n_original = len(self._y)
        if self.task == "regression":
            self._class_names = []
        else:
            self._class_names = sorted({v for v in self._y.tolist()}, key=str)

    # ------------------------------------------------------------ node logic
    def _new_node(
        self,
        indices: np.ndarray,
        depth: int,
        parent: Optional[Node],
        branch_key: Optional[str],
        branch_label: Optional[str],
    ) -> Node:
        node = Node(
            node_id=self._node_counter,
            depth=depth,
            parent_id=parent.node_id if parent else None,
            branch_key=branch_key,
            branch_label=branch_label,
            sample_indices=np.asarray(indices, dtype=int),
            n_samples=float(np.sum(self._w[indices])) if len(indices) else 0.0,
            impurity_name=self.impurity_key,
        )
        self._node_counter += 1
        self._nodes[node.node_id] = node
        if len(indices):
            if self.task == "regression":
                node.class_counts = {}
                node.impurity = self.impurity.value(self._y[indices], self._w[indices])
            else:
                node.class_counts = crit.class_counts(self._y[indices], self._w[indices])
                node.impurity = self.impurity.value(self._y[indices], self._w[indices])
        node.prediction = self._leaf_value(indices) if len(indices) else None
        return node

    def _available_features(self, available: Sequence[int]) -> List[int]:
        """Optionally restrict to a random subset (teaching aid for random forests)."""
        pool = list(available)
        if self.max_features and 0 < self.max_features < len(pool):
            chosen = self._rng.choice(pool, size=int(self.max_features), replace=False)
            return sorted(int(c) for c in chosen)
        return pool

    def _process_node(self, node: Node, available: Sequence[int]) -> List[Tuple[Node, List[int]]]:
        indices = node.sample_indices
        step_id = len(self.trace_.steps)
        impurity_comp = self.impurity.explain(
            self._y[indices], self._w[indices], symbol=f"S_{{{node.node_id}}}"
        ) if len(indices) else crit.Computation("empty", "I(S)", "0", 0.0)

        step = BuildStep(
            step_id=step_id,
            node_id=node.node_id,
            parent_id=node.parent_id,
            depth=node.depth,
            path_conditions=self._path_conditions(node),
            sample_indices=indices,
            n_samples=node.n_samples,
            class_counts=dict(node.class_counts),
            impurity_name=self.impurity.key,
            impurity_value=node.impurity,
            impurity_computation=impurity_comp,
            prediction=node.prediction,
        )

        stop_reason, stop_comp = self._pre_split_stop(node, indices, available)
        if stop_reason is not None:
            return self._finish_leaf(node, step, stop_reason, stop_comp)

        usable = self._available_features(available)
        step.candidates = self._splitter.generate(indices, usable, impurity_comp)
        chosen = self._select(step.candidates)
        step.selection_computation = argmax_computation(step.candidates, chosen, self._score_display())

        if chosen is None:
            return self._finish_leaf(node, step, "no_valid_split", None)

        if chosen.gain < self.min_impurity_decrease:
            comp = crit.rule_check_c(
                "min_impurity_decrease", "best gain", chosen.gain, "<",
                "min_impurity_decrease", self.min_impurity_decrease,
            )
            return self._finish_leaf(node, step, "no_gain", comp)

        return self._finish_split(node, step, chosen, available)

    def _pre_split_stop(
        self, node: Node, indices: np.ndarray, available: Sequence[int]
    ) -> Tuple[Optional[str], Optional[crit.Computation]]:
        """Checks that do not need any candidate to be generated first."""
        if len(indices) == 0:
            return "empty", None
        if self.task == "classification" and crit.is_pure(self._y[indices]):
            return "pure", None
        if self.task == "regression" and node.impurity <= 0:
            return "pure", None
        if self.max_depth is not None and node.depth >= self.max_depth:
            return "max_depth", crit.rule_check_c(
                "max_depth", "depth", node.depth, ">=", "max_depth", self.max_depth
            )
        if node.n_samples < self.min_samples_split:
            return "min_samples_split", crit.rule_check_c(
                "min_samples_split", "n", node.n_samples, "<", "min_samples_split", self.min_samples_split
            )
        if not list(available):
            return "no_features", None
        return None, None

    def _finish_leaf(
        self, node: Node, step: BuildStep, reason: str, comp: Optional[crit.Computation]
    ) -> List[Tuple[Node, List[int]]]:
        node.is_leaf = True
        node.stop_reason = reason
        node.children = {}
        step.action = ACTION_LEAF
        step.stop_reason = reason
        step.stop_computation = comp
        step.prediction = node.prediction
        step.narrative = narrate(step, self.impurity.display, self._score_display())
        self.trace_.steps.append(step)
        return []

    def _finish_split(
        self, node: Node, step: BuildStep, chosen: SplitCandidate, available: Sequence[int]
    ) -> List[Tuple[Node, List[int]]]:
        partitions = self._handle_missing(node, chosen)

        node.is_leaf = False
        node.feature_index = chosen.feature_index
        node.feature_name = chosen.feature_name
        node.split_type = chosen.split_type
        node.threshold = chosen.threshold
        node.categories = chosen.categories
        node.gain = chosen.gain
        node.score = chosen.score
        node.child_labels = dict(chosen.branch_labels)

        child_available = self._child_features(available, chosen)

        children: List[Tuple[Node, List[int]]] = []
        for key, child_indices in partitions.items():
            child = self._new_node(
                child_indices, node.depth + 1, node,
                branch_key=key, branch_label=chosen.branch_labels.get(key, key),
            )
            node.children[key] = child
            children.append((child, child_available))

        step.action = ACTION_SPLIT
        step.chosen = chosen
        step.prediction = node.prediction
        step.narrative = narrate(step, self.impurity.display, self._score_display())
        self.trace_.steps.append(step)
        return children

    def _child_features(self, available: Sequence[int], chosen: SplitCandidate) -> List[int]:
        """Which features the children may still test.

        ID3 (and C4.5 for categorical tests) removes a feature once used, because
        a multiway split leaves nothing more to learn from that column.
        """
        if self.consume_features:
            return [f for f in available if f != chosen.feature_index]
        return list(available)

    def _append_row(self, source_index: int, weight: float) -> int:
        """Append a re-weighted copy of a row; used by C4.5 fractional instances."""
        self._X = np.vstack([self._X, self._X[source_index][None, :]])
        self._y = np.concatenate([self._y, self._y[source_index : source_index + 1]])
        self._w = np.concatenate([self._w, np.array([weight], dtype=float)])
        self._splitter.X = self._X
        self._splitter.y = self._y
        self._splitter.weights = self._w
        return len(self._y) - 1

    def _handle_missing(self, node: Node, chosen: SplitCandidate) -> Dict[str, np.ndarray]:
        """Place samples whose split feature is missing. Base rule: heaviest branch."""
        partitions = {k: np.asarray(v, dtype=int) for k, v in chosen.partitions.items()}
        placed = {int(i) for idx in partitions.values() for i in idx}
        missing = [int(i) for i in node.sample_indices if int(i) not in placed]
        if not missing:
            return partitions
        if self.missing_strategy == "drop":
            self.trace_.notes.append(
                f"Node #{node.node_id}: dropped {len(missing)} sample(s) with a missing "
                f"{chosen.feature_name}."
            )
            return partitions
        heaviest = max(partitions.items(), key=lambda kv: float(np.sum(self._w[kv[1]])))[0]
        partitions[heaviest] = np.concatenate([partitions[heaviest], np.array(missing, dtype=int)])
        self.trace_.notes.append(
            f"Node #{node.node_id}: {len(missing)} sample(s) with a missing {chosen.feature_name} "
            f"were sent to the largest branch ('{chosen.branch_labels.get(heaviest, heaviest)}')."
        )
        return partitions

    def _path_conditions(self, node: Node) -> List[str]:
        """Tests that had to hold for a sample to reach this node."""
        stack: List[str] = []
        current: Optional[Node] = node
        while current is not None and current.parent_id is not None:
            parent = self._nodes.get(current.parent_id)
            if parent is None:
                break
            stack.append(f"{parent.feature_name} {current.branch_label}")
            current = parent
        return list(reversed(stack))

    def _score_display(self) -> str:
        if self.score_key == "gain_ratio":
            return "Gain Ratio"
        return crit.get_impurity(self.impurity_key).gain_name

    # -------------------------------------------------------------- params
    def get_params(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "impurity": self.impurity_key,
            "criterion": self._score_display(),
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_features": self.max_features,
            "expansion_order": self.expansion_order,
            "missing_strategy": self.missing_strategy,
            "task": self.task,
        }

    # ---------------------------------------------------------- inference API
    @property
    def tree(self) -> Tree:
        if self.tree_ is None:
            raise RuntimeError("The model has not been fitted yet — call fit() first.")
        return self.tree_

    def predict(self, X: Any) -> np.ndarray:
        return self.tree.predict(X)

    def predict_proba(self, X: Any) -> List[Dict[Any, float]]:
        return self.tree.predict_proba(X)

    def get_trace(self) -> TrainingTrace:
        if self.trace_ is None:
            raise RuntimeError("The model has not been fitted yet — call fit() first.")
        return self.trace_

    def explain(self, sample: Any) -> PredictionTrace:
        return explain_sample(self.tree, sample)


# --------------------------------------------------------------------------- #
# Prediction explanation
# --------------------------------------------------------------------------- #
def explain_sample(tree: Tree, sample: Any) -> PredictionTrace:
    """Replay one sample's journey through a fitted tree, hop by hop."""
    row = tree._as_row(sample)
    sample_dict = {name: row[i] for i, name in enumerate(tree.feature_names)}
    trace = PredictionTrace(sample=sample_dict)

    path = tree.decision_path(sample)
    conditions: List[str] = []

    for order, (node, branch_key, matched) in enumerate(path):
        value = row[node.feature_index] if node.feature_index is not None else None
        probs = node.probabilities()
        counts_text = ", ".join(f"{k}={v:g}" for k, v in node.class_counts.items()) or "-"

        if node.is_leaf or branch_key is None:
            prediction = node.prediction if node.prediction is not None else node.majority_class()
            narrative = (f"Reached leaf #{node.node_id} at depth {node.depth}. "
                         f"{node.n_samples:g} training sample(s) landed here ({counts_text}), "
                         f"so the prediction is '{prediction}'.")
            trace.steps.append(
                PredictStep(
                    index=order, node_id=node.node_id, depth=node.depth, feature=None,
                    sample_value=None, test="(leaf)", branch_key=None, branch_label=None,
                    outcome_text=f"predict {prediction}", node_class_counts=dict(node.class_counts),
                    node_probabilities=probs, n_samples=node.n_samples, is_leaf=True,
                    narrative=narrative,
                )
            )
            trace.path_node_ids.append(node.node_id)
            trace.final_leaf_id = node.node_id
            trace.predicted_class = prediction
            trace.probabilities = probs
            trace.confidence = node.confidence()
            trace.support = node.n_samples
            trace.probability_computation = crit.probability_c(
                node.class_counts, prediction, where=f"leaf #{node.node_id}"
            )
            break

        label = node.branch_condition(branch_key)
        comparison = _comparison_for(node, value, branch_key)
        if not matched:
            reason = ("the value is missing" if is_missing(value)
                      else f"the value '{value}' was never seen for {node.feature_name} during training")
            outcome = f"fallback to the largest branch [{label}] because {reason}"
            trace.warnings.append(
                f"Node #{node.node_id}: {reason}; the sample followed the largest branch instead."
            )
        else:
            outcome = f"go to branch [{label}]"

        narrative = (f"At node #{node.node_id} (depth {node.depth}) the test is "
                     f"\"{node.test_description()}\"; the sample has {node.feature_name} = "
                     f"{_pretty(value)}, so {outcome}. "
                     f"{node.n_samples:g} training sample(s) are here ({counts_text}).")

        trace.steps.append(
            PredictStep(
                index=order, node_id=node.node_id, depth=node.depth, feature=node.feature_name,
                sample_value=value, test=node.test_description(), branch_key=branch_key,
                branch_label=label, outcome_text=outcome, node_class_counts=dict(node.class_counts),
                node_probabilities=probs, n_samples=node.n_samples, is_leaf=False,
                unseen_value=not matched, comparison=comparison, narrative=narrative,
            )
        )
        trace.path_node_ids.append(node.node_id)
        conditions.append(f"{node.feature_name} {label}")

    trace.rule_text = (
        "IF " + " AND ".join(conditions) + f" THEN {trace.predicted_class}"
        if conditions else f"THEN {trace.predicted_class}"
    )
    return trace


def _pretty(value: Any) -> str:
    if is_missing(value):
        return "(missing)"
    if isinstance(value, float):
        return f"{value:.4g}"
    return f"'{value}'"


def _comparison_for(node: Node, value: Any, branch_key: str) -> crit.Computation:
    """The exact test at this node, with the sample's own value substituted in."""
    from .node import IN, LEFT, SPLIT_CATEGORICAL_BINARY, SPLIT_NUMERIC_BINARY

    if node.split_type == SPLIT_NUMERIC_BINARY:
        return crit.comparison_c(node.feature_name, value, "<=", node.threshold, branch_key == LEFT)
    if node.split_type == SPLIT_CATEGORICAL_BINARY:
        inside = r",\ ".join(tex.text(c) for c in (node.categories or ()))
        outcome = branch_key == IN
        return crit.Computation(
            name="comparison",
            symbolic=r"x\left[%s\right] \in \{%s\}" % (tex.text(node.feature_name), inside),
            substituted=r"%s \in \{%s\}" % (tex.text(value), inside),
            result=1.0 if outcome else 0.0,
            operands={"feature": node.feature_name, "value": value, "categories": node.categories},
            note=r"\text{TRUE}" if outcome else r"\text{FALSE}",
        )
    return crit.comparison_c(node.feature_name, value, "==", value, True)
