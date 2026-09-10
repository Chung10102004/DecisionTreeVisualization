"""End-to-end smoke test of the core + viz layers, without touching Streamlit.

Trains every algorithm on every bundled dataset, walks every build step, renders
every formula and every figure, and explains a prediction.  Any exception, empty
LaTeX block, or malformed DOT string fails the run.

Usage:  .venv/bin/python scripts/smoke_e2e.py [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.c45 import C45Tree
from src.core.cart import CARTRegressor, CARTTree
from src.core.id3 import ID3Tree
from src.core.metrics import accuracy, root_mean_squared_error, train_test_split
from src.core.node import SPLIT_NUMERIC_BINARY
from src.core.pruning import CostComplexityPruner, PessimisticPruner, ReducedErrorPruner
from src.datasets.loaders import list_builtin, load_builtin, load_uploaded
from src.datasets.preprocess import Preprocessor, suggest_discretization
from src.viz import charts
from src.viz.boundary import plot_decision_boundary_2d
from src.viz.mathfmt import MathFormatter
from src.viz.tree_dot import RenderOptions, TreeDotRenderer, render_text_tree

FORMATTER = MathFormatter()
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def check_dot(dot: str, tree, label: str) -> None:
    check(dot.count("{") == dot.count("}"), f"{label}: unbalanced braces in DOT")
    check(dot.count("[label=<") == tree.count_nodes(),
          f"{label}: DOT has {dot.count('[label=<')} nodes, tree has {tree.count_nodes()}")
    declared = {line.split("[")[0].strip() for line in dot.splitlines() if line.strip().startswith("n")
                and "[label=<" in line}
    for line in dot.splitlines():
        if "->" in line:
            source, rest = line.split("->", 1)
            target = rest.split("[")[0].strip()
            check(source.strip() in declared, f"{label}: edge from undeclared {source.strip()}")
            check(target in declared, f"{label}: edge to undeclared {target}")


def check_computation(computation, label: str) -> None:
    if computation is None:
        return
    block = FORMATTER.render(computation)
    check(bool(block.symbolic.strip()), f"{label}: empty symbolic form")
    check(bool(block.substituted.strip()), f"{label}: empty substituted form")
    check(bool(block.result.strip()), f"{label}: empty result")
    for name, child in computation.children.items():
        check_computation(child, f"{label}/{name}")


def run_dataset(name: str, out_dir: Path) -> None:
    dataset = load_builtin(name)
    algorithms = ["CART"] if dataset.task == "regression" else ["ID3", "C4.5", "CART"]

    for algorithm in algorithms:
        label = f"{name} / {algorithm}"
        preprocessor = Preprocessor(
            missing_strategy="keep",
            discretize=suggest_discretization(dataset, algorithm),
            n_bins=3,
        )
        prepared = preprocessor.prepare(dataset)
        train, test = train_test_split(len(prepared), prepared.y, 0.3,
                                       stratify=dataset.task == "classification")

        if algorithm == "ID3":
            model = ID3Tree(max_depth=5)
        elif algorithm == "C4.5":
            model = C45Tree(max_depth=5)
        elif dataset.task == "regression":
            model = CARTRegressor(max_depth=4, min_samples_leaf=5)
        else:
            model = CARTTree(max_depth=5)

        model.fit(prepared.X[train], prepared.y[train],
                  prepared.feature_names, prepared.feature_types)
        tree, trace = model.tree, model.get_trace()

        check(len(trace) == tree.count_nodes(), f"{label}: step count != node count")

        # Every step, every candidate, every formula.
        for step in trace.steps:
            check(bool(step.narrative.strip()), f"{label}: step {step.step_id} has no narrative")
            check_computation(step.impurity_computation, f"{label}/step{step.step_id}/impurity")
            check_computation(step.selection_computation, f"{label}/step{step.step_id}/argmax")
            check_computation(step.stop_computation, f"{label}/step{step.step_id}/stop")
            for candidate in step.candidates:
                for key, computation in candidate.computations.items():
                    check_computation(computation, f"{label}/step{step.step_id}/{candidate.feature_name}/{key}")
            FORMATTER.candidates_dataframe(step)
            FORMATTER.worked_example(step, "Entropy", trace.score_name)

        # Every partial tree renders.
        renderer = TreeDotRenderer(RenderOptions())
        for k in range(len(trace)):
            partial = trace.tree_at_step(k)
            check_dot(renderer.render(partial, highlight_node=trace.steps[k].node_id),
                      partial, f"{label}/partial{k}")
        check_dot(renderer.render(tree), tree, f"{label}/full")
        renderer.render(tree, max_display_depth=1)
        renderer.render(tree, focus_node_id=tree.root.node_id)
        check(bool(render_text_tree(tree).strip()), f"{label}: empty text tree")
        check(len(tree.to_rules()) == tree.count_leaves(), f"{label}: rules != leaves")

        # Explanations.
        for position in list(test[:3]) + list(train[:2]):
            row = {n: prepared.X[position][i] for i, n in enumerate(prepared.feature_names)}
            explanation = model.explain(row)
            check(explanation.predicted_class == tree.predict_one(row),
                  f"{label}: explain disagrees with predict")
            check(bool(explanation.rule_text.strip()), f"{label}: empty rule text")
            for predict_step in explanation.steps:
                check(bool(predict_step.narrative.strip()), f"{label}: empty predict narrative")
                check_computation(predict_step.comparison, f"{label}/predict/comparison")
            check_computation(explanation.probability_computation, f"{label}/predict/probability")

        # Unseen category and missing value must not raise.
        weird = {n: "***never-seen***" for n in prepared.feature_names}
        model.explain(weird)
        model.explain({n: None for n in prepared.feature_names})

        # Figures.
        step = trace.steps[0]
        figures = [
            charts.plot_candidate_gains(step, trace.score_name),
            charts.plot_impurity_breakdown(step),
            charts.plot_class_distribution(step.class_counts),
            charts.plot_feature_importance(tree.feature_importances()),
        ]
        numeric = next((c for c in step.candidates if c.split_type == SPLIT_NUMERIC_BINARY), None)
        if numeric is not None:
            figures.append(charts.plot_threshold_scan(numeric, trace.score_name))
        safe = f"{name}_{algorithm}".replace(" ", "_").replace("/", "-").replace("+", "")
        for index, figure in enumerate(figures):
            figure.savefig(out_dir / f"{safe}_{index}.png", dpi=80)
        plt.close("all")

        # Pruning.
        if dataset.task == "classification" and len(test):
            for pruner in (ReducedErrorPruner(), CostComplexityPruner(), PessimisticPruner()):
                pruned, steps = pruner.prune(tree, prepared.X[test], prepared.y[test])
                check(pruned.count_leaves() <= tree.count_leaves(),
                      f"{label}/{pruner.name}: pruning grew the tree")
                for prune_step in steps:
                    check(bool(prune_step.reason.strip()),
                          f"{label}/{pruner.name}: prune step without a reason")
                    check_computation(prune_step.computation, f"{label}/{pruner.name}/computation")

        score = (accuracy(prepared.y[test], tree.predict(prepared.X[test]))
                 if dataset.task == "classification"
                 else -root_mean_squared_error(prepared.y[test],
                                               [float(v) for v in tree.predict(prepared.X[test])]))
        print(f"  {algorithm:6s} leaves={tree.count_leaves():3d} depth={tree.depth():2d} "
              f"steps={len(trace):3d} score={score:.4f}")


def run_boundary(out_dir: Path) -> None:
    dataset = load_builtin("Iris")
    prepared = Preprocessor().prepare(dataset)
    model = CARTTree(max_depth=4).fit(prepared.X, prepared.y,
                                      prepared.feature_names, prepared.feature_types)
    figure = plot_decision_boundary_2d(
        model.tree, prepared.frame[prepared.feature_names], prepared.y,
        "PetalLength", "PetalWidth", prepared.feature_types, resolution=90,
    )
    figure.savefig(out_dir / "boundary.png", dpi=80)
    plt.close("all")
    print("  decision boundary rendered")


def run_upload(out_dir: Path) -> None:
    """The external-CSV path: a file the app has never seen, with renamed columns."""
    import pandas as pd

    frame = pd.read_csv(ROOT / "data" / "iris.csv")
    frame.columns = ["sl", "sw", "pl", "pw", "kind"]
    path = out_dir / "external.csv"
    frame.to_csv(path, index=False)

    dataset = load_uploaded(str(path), target="kind")
    check(dataset.target == "kind", "upload: wrong target")
    check(len(dataset.features) == 4, "upload: wrong feature count")
    check(dataset.task == "classification", "upload: wrong task")
    prepared = Preprocessor().prepare(dataset)
    model = CARTTree(max_depth=3).fit(prepared.X, prepared.y,
                                      prepared.feature_names, prepared.feature_types)
    score = accuracy(prepared.y, model.predict(prepared.X))
    check(score > 0.9, f"upload: accuracy too low ({score:.3f})")
    print(f"  uploaded external CSV -> accuracy {score:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/dt_smoke")
    arguments = parser.parse_args()
    out_dir = Path(arguments.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in list_builtin():
        print(f"{entry['name']} ({entry['rows']} rows)")
        try:
            run_dataset(entry["name"], out_dir)
        except Exception:  # noqa: BLE001 - the point is to catch everything
            failures.append(f"{entry['name']}: {traceback.format_exc()}")

    print("Decision boundary")
    run_boundary(out_dir)
    print("External CSV upload")
    run_upload(out_dir)

    print()
    if failures:
        print(f"FAILED — {len(failures)} problem(s):")
        for failure in failures[:20]:
            print(f"  - {failure}")
        return 1
    print(f"ALL CHECKS PASSED — figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
