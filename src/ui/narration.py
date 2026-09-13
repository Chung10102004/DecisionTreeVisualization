"""Why each formula appears where it does in the build animation.

The formulas themselves come from ``src/core`` with the node's numbers already
substituted.  What they do not carry is the *motivation* — why the learner needs
this quantity at this moment — and that is what the Build page shows next to
each one.  Everything here is plain text keyed off the trace, so it stays in
step with whichever algorithm and impurity produced the numbers.
"""
from __future__ import annotations

from typing import Optional

from ..core.node import SPLIT_NUMERIC_BINARY
from ..core.splitter import SplitCandidate
from ..core.trace import BuildStep, TrainingTrace

_IMPURITY_WHY = {
    "entropy": (
        "Entropy $H(S) = -\\sum_k p_k \\log_2 p_k$ measures how mixed the labels are: "
        "0 when every sample agrees, $\\log_2 K$ when $K$ classes are spread evenly. "
        "{algorithm} needs this number before it looks at a single feature, because "
        "{score} is defined as the entropy a split *removes* — the parent's $H(S)$ is "
        "what every candidate will be subtracted from."
    ),
    "gini": (
        "The Gini index $1 - \\sum_k p_k^2$ is the chance that two samples drawn at "
        "random from this node disagree on the label: 0 for a pure node, up to "
        "$1 - 1/K$ for $K$ evenly mixed classes. {algorithm} scores every candidate "
        "split by how far it pulls this value down, so the node's own Gini is the "
        "baseline."
    ),
    "error": (
        "Misclassification error $1 - \\max_k p_k$ is the share of samples a majority "
        "vote gets wrong here. Every candidate split is scored by how much it "
        "reduces it."
    ),
    "mse": (
        "For a regression node the impurity is the mean squared error around the "
        "node's mean target — its variance. {algorithm} scores a split by how much of "
        "that variance the children remove, so it has to be measured first."
    ),
    "mae": (
        "For a regression node the impurity is the mean absolute error around the "
        "node's median target. {algorithm} scores a split by how much of it the "
        "children remove, so it has to be measured first."
    ),
}

_STOP_WHY = {
    "pure": "The impurity is exactly 0: every sample here carries the same label, so "
            "no split can improve on simply predicting it.",
    "max_depth": "The node sits at `max_depth = {max_depth}`. This is a pre-pruning "
                 "limit from the hyper-parameters, not something the data decided — "
                 "the leaf predicts the majority label of what landed here.",
    "min_samples_split": "The node holds {n:g} samples, fewer than "
                         "`min_samples_split = {min_samples_split}`. Splitting so few "
                         "points would be fitting noise, so the learner stops.",
    "min_samples_leaf": "Every candidate would leave a branch with fewer than "
                        "`min_samples_leaf = {min_samples_leaf}` samples, so none is "
                        "allowed.",
    "no_features": "Every feature has already been tested higher on this path. ID3 "
                   "(and C4.5 for categorical tests) retire a feature after a "
                   "multiway split, because it has nothing left to separate.",
    "no_valid_split": "Every candidate was rejected — see the table below for each "
                      "reason. No usable test separates these samples.",
    "no_gain": "The best candidate was scored, but its gain falls below "
               "`min_impurity_decrease = {min_impurity_decrease}`. Adding a node "
               "would not buy enough purity to be worth it.",
    "empty": "No samples reached this node, so it inherits its parent's prediction.",
}


def _fmt(template: str, **values) -> str:
    class _Safe(dict):
        def __missing__(self, key):  # leave unknown placeholders visible rather than crash
            return "{" + key + "}"

    return template.format_map(_Safe(values))


# --------------------------------------------------------------------------- #
def why_impurity(trace: TrainingTrace) -> str:
    template = _IMPURITY_WHY.get(
        trace.impurity_name,
        "{algorithm} measures the impurity of the node first: every candidate split "
        "is scored by how much of it the children remove.",
    )
    text = _fmt(template, algorithm=trace.algorithm, score=trace.score_name)
    if trace.params.get("task", "classification") == "classification":
        text += " If it is already 0 the node is pure and the search stops before it starts."
    return text


def why_scoring(trace: TrainingTrace) -> str:
    algorithm = trace.algorithm
    if algorithm == "ID3":
        return (
            "Information Gain $IG(S, A) = H(S) - \\sum_v \\frac{|S_v|}{|S|} H(S_v)$. Each "
            "candidate feature partitions the node into one branch per value; the "
            "children's entropies are averaged with weights $|S_v|/|S|$ so a branch "
            "holding two samples cannot outvote one holding forty. ID3 tests every "
            "feature not yet used on this path and keeps the whole table, so the "
            "losing candidates stay visible."
        )
    if algorithm == "C4.5":
        text = (
            "C4.5 also computes Information Gain but ranks by "
            "$\\text{Gain Ratio} = IG / \\text{SplitInfo}$, where "
            "$\\text{SplitInfo} = -\\sum_v \\frac{|S_v|}{|S|} \\log_2 \\frac{|S_v|}{|S|}$ "
            "is the entropy of the branch *sizes* themselves. A feature with many "
            "distinct values (an ID column at the extreme) earns a high gain almost "
            "for free by shattering the data; dividing by SplitInfo cancels that "
            "advantage. Numeric features are tried at every midpoint between sorted "
            "values, and only the best threshold enters this table."
        )
        if trace.params.get("use_gain_filter", True):
            text += (" Quinlan's extra rule: only candidates with at least the average "
                     "gain are eligible, so a tiny gain over a tiny SplitInfo cannot win "
                     "on a fluke.")
        return text
    if algorithm.startswith("CART"):
        impurity = {"gini": "Gini", "mse": "MSE", "mae": "MAE", "error": "Err",
                    "entropy": "H"}.get(trace.impurity_name, "I")
        return (
            "CART only ever makes binary splits: a numeric feature becomes $x \\le t$ "
            "for the best midpoint $t$, a categorical one becomes a subset test. Each "
            "candidate is scored by "
            "$\\Delta = %s(S) - \\sum_v \\frac{|S_v|}{|S|}\\,%s(S_v)$ — the impurity the two "
            "children remove once each is weighted by its share of the samples."
            % (impurity, impurity)
        )
    return (f"Every candidate split is scored by {trace.score_name}: the parent's "
            f"impurity minus the size-weighted impurity of the children it would create.")


def why_working(trace: TrainingTrace, candidate: Optional[SplitCandidate]) -> str:
    if candidate is None:
        return ""
    text = (f"This is the arithmetic behind the `{candidate.feature_name}` row of the table: "
            f"one impurity per child branch, the size-weighted combination, then the "
            f"difference from the parent. Every other row was produced the same way.")
    if trace.algorithm == "C4.5":
        text += " The SplitInfo line is what turns the gain into the Gain Ratio that C4.5 ranks by."
    if candidate.split_type == SPLIT_NUMERIC_BINARY and candidate.threshold_scan:
        text += (" The threshold chart shows the score at every candidate cut between two "
                 "sorted values; the chosen threshold is the peak.")
    return text


def why_choice(trace: TrainingTrace) -> str:
    text = (f"$A^{{*}} = \\arg\\max$ — the candidate with the highest {trace.score_name} wins.")
    decrease = trace.params.get("min_impurity_decrease")
    if decrease is not None:
        text += (f" It still has to clear one bar: had even the best gain fallen below "
                 f"`min_impurity_decrease = {decrease}`, the node would have stopped here "
                 f"as a leaf instead.")
    text += (" Once chosen, the node is split and each child joins the queue to be "
             "processed in turn — that is the dashed nodes appearing in the diagram.")
    if trace.algorithm in ("ID3", "C4.5"):
        text += (" For a categorical split the feature is retired on this path, since a "
                 "multiway split leaves nothing more to learn from that column.")
    return text


def why_stop(step: BuildStep, trace: TrainingTrace) -> str:
    template = _STOP_WHY.get(step.stop_reason or "",
                             "No split was possible at this node, so it becomes a leaf.")
    text = _fmt(template, n=step.n_samples, **trace.params)
    if trace.params.get("task", "classification") == "regression":
        text += (f" The leaf predicts `{step.prediction}` — the mean target of the "
                 f"{step.n_samples:g} samples that reached it.")
    else:
        text += (f" The leaf predicts `{step.prediction}` — the majority label among the "
                 f"{step.n_samples:g} samples that reached it.")
    return text

