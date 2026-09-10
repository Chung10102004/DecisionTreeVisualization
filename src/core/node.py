"""Tree data structure: a node, and the tree that owns the nodes.

The node knows how to *route* a sample, which is the single source of truth for
both ``predict`` and the step-by-step explanation on the Predict page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import latex as tex

SPLIT_CATEGORICAL_MULTIWAY = "categorical_multiway"
SPLIT_NUMERIC_BINARY = "numeric_binary"
SPLIT_CATEGORICAL_BINARY = "categorical_binary"

LEFT, RIGHT = "left", "right"
IN, OUT = "in", "out"


def is_missing(value: Any) -> bool:
    """True for None / NaN / empty string — the three shapes missing data takes here."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() in ("", "?", "nan", "NaN"):
        return True
    try:
        return bool(np.asarray(value).dtype.kind == "f" and np.isnan(value))
    except (TypeError, ValueError):
        return False


@dataclass
class Node:
    """One node of the tree; a leaf is simply a node with ``is_leaf=True``."""

    node_id: int
    depth: int
    parent_id: Optional[int] = None
    branch_key: Optional[str] = None
    branch_label: Optional[str] = None

    sample_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    sample_weights: Optional[np.ndarray] = None
    n_samples: float = 0.0
    class_counts: Dict[Any, float] = field(default_factory=dict)
    impurity: float = 0.0
    impurity_name: str = "entropy"

    is_leaf: bool = False
    is_pending: bool = False
    prediction: Any = None
    stop_reason: Optional[str] = None

    feature_index: Optional[int] = None
    feature_name: Optional[str] = None
    split_type: Optional[str] = None
    threshold: Optional[float] = None
    categories: Optional[Tuple[Any, ...]] = None
    gain: Optional[float] = None
    score: Optional[float] = None

    children: Dict[str, "Node"] = field(default_factory=dict)
    child_labels: Dict[str, str] = field(default_factory=dict)

    # ---------------------------------------------------------------- queries
    def probabilities(self) -> Dict[Any, float]:
        total = float(sum(self.class_counts.values()))
        if total <= 0:
            return {k: 0.0 for k in self.class_counts}
        return {k: v / total for k, v in self.class_counts.items()}

    def majority_class(self) -> Any:
        if not self.class_counts:
            return self.prediction
        return max(self.class_counts.items(), key=lambda kv: (kv[1], str(kv[0])))[0]

    def confidence(self) -> float:
        probs = self.probabilities()
        return max(probs.values()) if probs else 0.0

    def is_pure(self) -> bool:
        non_zero = [c for c in self.class_counts.values() if c > 0]
        return len(non_zero) <= 1

    def test_description(self) -> str:
        """Human-readable test performed at this node, e.g. ``PetalLength <= 2.45``."""
        if self.is_leaf or self.feature_name is None:
            return ""
        if self.split_type == SPLIT_NUMERIC_BINARY:
            return f"{self.feature_name} <= {self.threshold:.4g}?"
        if self.split_type == SPLIT_CATEGORICAL_BINARY:
            inside = ", ".join(str(c) for c in (self.categories or ()))
            return f"{self.feature_name} in {{{inside}}}?"
        return f"{self.feature_name}?"

    def test_latex(self) -> str:
        if self.is_leaf or self.feature_name is None:
            return ""
        if self.split_type == SPLIT_NUMERIC_BINARY:
            return r"%s \leq %s" % (tex.text(self.feature_name), tex.num(self.threshold))
        if self.split_type == SPLIT_CATEGORICAL_BINARY:
            inside = r",\ ".join(tex.text(c) for c in (self.categories or ()))
            return r"%s \in \{%s\}" % (tex.text(self.feature_name), inside)
        return r"%s = ?" % tex.text(self.feature_name)

    # ---------------------------------------------------------------- routing
    def route(self, value: Any) -> Tuple[Optional["Node"], Optional[str], bool]:
        """Pick the child a sample value belongs to.

        Returns ``(child, branch_key, matched)``.  ``matched`` is False when the
        value is missing or is a category never seen during training; callers
        then fall back to the heaviest branch and say so in the explanation.
        """
        if self.is_leaf or not self.children:
            return None, None, False

        if is_missing(value):
            key = self._heaviest_branch()
            return self.children.get(key), key, False

        if self.split_type == SPLIT_NUMERIC_BINARY:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                key = self._heaviest_branch()
                return self.children.get(key), key, False
            key = LEFT if numeric <= float(self.threshold) else RIGHT
            return self.children.get(key), key, True

        if self.split_type == SPLIT_CATEGORICAL_BINARY:
            known = set(self.categories or ())
            key = IN if value in known else OUT
            child = self.children.get(key)
            return child, key, True

        key = str(value)
        if key in self.children:
            return self.children[key], key, True
        fallback = self._heaviest_branch()
        return self.children.get(fallback), fallback, False

    def _heaviest_branch(self) -> Optional[str]:
        if not self.children:
            return None
        return max(self.children.items(), key=lambda kv: kv[1].n_samples)[0]

    def branch_condition(self, key: str) -> str:
        """Human-readable label of one outgoing edge."""
        return self.child_labels.get(key, str(key))

    # -------------------------------------------------------------- traversal
    def iter_nodes(self) -> Iterator["Node"]:
        yield self
        for child in self.children.values():
            yield from child.iter_nodes()

    def leaves(self) -> List["Node"]:
        return [n for n in self.iter_nodes() if n.is_leaf]

    def count_nodes(self) -> int:
        return sum(1 for _ in self.iter_nodes())

    def count_leaves(self) -> int:
        return len(self.leaves())

    def subtree_depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(child.subtree_depth() for child in self.children.values())

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "n_samples": self.n_samples,
            "class_counts": {str(k): v for k, v in self.class_counts.items()},
            "impurity": self.impurity,
            "impurity_name": self.impurity_name,
            "is_leaf": self.is_leaf,
            "prediction": None if self.prediction is None else str(self.prediction),
            "stop_reason": self.stop_reason,
            "feature": self.feature_name,
            "split_type": self.split_type,
            "threshold": self.threshold,
            "categories": None if self.categories is None else [str(c) for c in self.categories],
            "gain": self.gain,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


@dataclass
class Rule:
    """One root-to-leaf path rendered as an IF ... THEN ... rule."""

    conditions: List[str]
    prediction: Any
    n_samples: float
    confidence: float
    leaf_id: int

    def as_text(self) -> str:
        if not self.conditions:
            return f"THEN {self.prediction}  [n={self.n_samples:g}, confidence={self.confidence:.1%}]"
        body = " AND ".join(self.conditions)
        return (f"IF {body} THEN {self.prediction}  "
                f"[n={self.n_samples:g}, confidence={self.confidence:.1%}]")


@dataclass
class Tree:
    """A fitted decision tree plus everything needed to explain it."""

    root: Node
    feature_names: List[str]
    feature_types: Dict[str, str]
    class_names: List[Any] = field(default_factory=list)
    algorithm: str = "ID3"
    task: str = "classification"
    impurity_name: str = "entropy"
    params: Dict[str, Any] = field(default_factory=dict)
    n_train: int = 0

    # ------------------------------------------------------------- structure
    @property
    def nodes_by_id(self) -> Dict[int, Node]:
        return {node.node_id: node for node in self.root.iter_nodes()}

    def node(self, node_id: int) -> Node:
        return self.nodes_by_id[node_id]

    def count_nodes(self) -> int:
        return self.root.count_nodes()

    def count_leaves(self) -> int:
        return self.root.count_leaves()

    def depth(self) -> int:
        return self.root.subtree_depth()

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)

    # ------------------------------------------------------------ prediction
    def _as_row(self, sample: Any) -> List[Any]:
        """Accept a dict, a Series, or a positional sequence of feature values."""
        if isinstance(sample, dict):
            return [sample.get(name, None) for name in self.feature_names]
        if hasattr(sample, "to_dict") and not isinstance(sample, np.ndarray):
            data = sample.to_dict()
            return [data.get(name, None) for name in self.feature_names]
        values = list(np.asarray(sample, dtype=object).ravel())
        if len(values) != len(self.feature_names):
            raise ValueError(
                f"Sample has {len(values)} values but the tree expects {len(self.feature_names)}."
            )
        return values

    def decision_path(self, sample: Any) -> List[Tuple[Node, Optional[str], bool]]:
        """Walk root -> leaf, recording the branch taken at every internal node."""
        row = self._as_row(sample)
        path: List[Tuple[Node, Optional[str], bool]] = []
        node = self.root
        guard = 0
        while not node.is_leaf and node.children:
            guard += 1
            if guard > 1000:  # pragma: no cover - structural corruption guard
                raise RuntimeError("decision_path exceeded 1000 hops; tree is malformed.")
            value = row[node.feature_index] if node.feature_index is not None else None
            child, key, matched = node.route(value)
            path.append((node, key, matched))
            if child is None:
                break
            node = child
        path.append((node, None, True))
        return path

    def predict_one(self, sample: Any) -> Any:
        leaf = self.decision_path(sample)[-1][0]
        return leaf.prediction if leaf.prediction is not None else leaf.majority_class()

    def predict(self, X: Any) -> np.ndarray:
        rows = X.values if hasattr(X, "values") else np.asarray(X, dtype=object)
        return np.array([self.predict_one(row) for row in rows], dtype=object)

    def predict_proba_one(self, sample: Any) -> Dict[Any, float]:
        leaf = self.decision_path(sample)[-1][0]
        probs = leaf.probabilities()
        return {label: probs.get(label, 0.0) for label in (self.class_names or probs.keys())}

    def predict_proba(self, X: Any) -> List[Dict[Any, float]]:
        rows = X.values if hasattr(X, "values") else np.asarray(X, dtype=object)
        return [self.predict_proba_one(row) for row in rows]

    # ----------------------------------------------------------------- rules
    def to_rules(self) -> List[Rule]:
        rules: List[Rule] = []

        def walk(node: Node, conditions: List[str]) -> None:
            if node.is_leaf or not node.children:
                rules.append(
                    Rule(
                        conditions=list(conditions),
                        prediction=node.prediction if node.prediction is not None else node.majority_class(),
                        n_samples=node.n_samples,
                        confidence=node.confidence(),
                        leaf_id=node.node_id,
                    )
                )
                return
            for key, child in node.children.items():
                walk(child, conditions + [f"{node.feature_name} {node.branch_condition(key)}"])

        walk(self.root, [])
        return rules

    def path_conditions(self, node_id: int) -> List[str]:
        """The chain of tests leading from the root down to ``node_id``."""
        by_id = self.nodes_by_id
        conditions: List[str] = []
        current = by_id.get(node_id)
        while current is not None and current.parent_id is not None:
            parent = by_id.get(current.parent_id)
            if parent is None:
                break
            conditions.append(f"{parent.feature_name} {parent.branch_condition(current.branch_key)}")
            current = parent
        return list(reversed(conditions))

    # ------------------------------------------------------------ importance
    def feature_importances(self) -> Dict[str, float]:
        """Weighted impurity decrease per feature, normalised to sum to 1."""
        total_n = float(self.root.n_samples) or 1.0
        scores = {name: 0.0 for name in self.feature_names}
        for node in self.root.iter_nodes():
            if node.is_leaf or node.feature_name is None or node.gain is None:
                continue
            scores[node.feature_name] += (node.n_samples / total_n) * float(node.gain)
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    def stats(self) -> Dict[str, Any]:
        leaves = self.root.leaves()
        return {
            "algorithm": self.algorithm,
            "task": self.task,
            "impurity": self.impurity_name,
            "nodes": self.count_nodes(),
            "leaves": len(leaves),
            "depth": self.depth(),
            "avg_samples_per_leaf": (sum(l.n_samples for l in leaves) / len(leaves)) if leaves else 0.0,
            "n_train": self.n_train,
        }

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "task": self.task,
            "impurity": self.impurity_name,
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "class_names": [str(c) for c in self.class_names],
            "params": self.params,
            "stats": self.stats(),
            "root": self.root.to_dict(),
        }

    def copy(self) -> "Tree":
        """Deep copy — used by the pruners so the original tree stays intact."""
        import copy as _copy

        return _copy.deepcopy(self)
