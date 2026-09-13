"""The Build page's player: reveal states, single-screen boards, and payload size."""
from __future__ import annotations

import json

import pytest

from src.core.c45 import C45Tree
from src.core.cart import CARTTree
from src.core.id3 import ID3Tree
from src.core.playback import total_ticks
from src.core.trace import ACTION_SPLIT
from src.viz import theater


@pytest.fixture
def trace(play_tennis):
    df, features, types, target = play_tennis
    return ID3Tree().fit(df[features], df[target], feature_types=types).get_trace()


def test_reveal_starts_with_a_pending_root_and_ends_with_nothing_queued(trace):
    states = theater.reveal_states(trace)
    assert len(states) == len(trace) + 1
    assert states[0] == {"processed": [], "pending": [trace.tree.root.node_id]}
    assert states[-1]["pending"] == []
    assert sorted(states[-1]["processed"]) == sorted(trace.tree.nodes_by_id)


def test_each_step_moves_its_node_from_pending_to_processed(trace):
    states = theater.reveal_states(trace)
    for k, step in enumerate(trace.steps):
        assert step.node_id in states[k]["pending"]
        assert step.node_id in states[k + 1]["processed"]
        assert not set(states[k + 1]["processed"]) & set(states[k + 1]["pending"])


def test_one_frame_per_beat_and_the_tree_grows_on_the_deciding_beat(trace):
    frames = theater.build_frames(trace)
    assert len(frames) == total_ticks(trace)
    assert frames[-1].final and "complete" in frames[-1].banner.lower()
    for frame in frames:
        step = trace.steps[frame.step - 1]
        deciding = frame.phase in ("choice", "stop")
        assert frame.state == (frame.step if deciding else frame.step - 1)
        assert frame.title and frame.here
        assert frame.lines, f"{frame.phase} frame shows no formula"
        if step.action == ACTION_SPLIT:
            assert frame.phase in ("impurity", "working", "choice")
        else:
            assert frame.phase in ("impurity", "stop")


def test_the_board_is_cumulative_within_a_step(trace):
    """Later beats add blocks; nothing already on screen disappears."""
    frames = theater.build_frames(trace)
    by_step: dict = {}
    for frame in frames:
        by_step.setdefault(frame.step, []).append(frame)
    for beats in by_step.values():
        shown = [len(f.lines) for f in beats]
        assert shown == sorted(shown) and shown[0] >= 1
        for earlier, later in zip(beats, beats[1:]):
            assert earlier.lines == later.lines[: len(earlier.lines)]


def test_split_board_has_impurity_winner_others_and_decision(trace):
    step = trace.steps[0]
    title, sections = theater.build_sections(step, trace)
    kinds = [s.kind for s in sections]
    assert kinds == ["impurity", "working", "others", "choice"]
    assert [s.beat for s in sections] == [0, 1, 1, 2]

    impurity, working, others, choice = sections
    assert r"\frac{9}{14}" in impurity.lines[0] and "0.940" in impurity.lines[0]

    # the winner: one line per branch with its own numbers, then the formula, then the numbers
    branch_lines = [l for l in working.lines if "Outlook = " in l]
    assert len(branch_lines) == 3
    assert any("0.971" in l for l in branch_lines)
    assert working.lines[-2].startswith(r"IG(S, \text{Outlook}) = H(S)")
    assert working.lines[-1].endswith("0.247")
    assert "0.940" in working.lines[-1]

    # the rest: results only, one per column, none of them the winner
    assert "Similarly" in others.title
    assert len(others.lines) == 3
    assert not any("Outlook" in l for l in others.lines)
    assert all(l.count("=") == 1 for l in others.lines)

    assert "IG(S, A)" in choice.lines[0]
    assert choice.lines[1].endswith(r"\Rightarrow \text{Outlook}")
    assert choice.lines[1].count("{:}") <= theater.MAX_ARGMAX


def test_leaf_board_is_impurity_then_the_stop_rule(trace):
    leaf = next(s for s in trace.steps if s.action != ACTION_SPLIT)
    title, sections = theater.build_sections(leaf, trace)
    assert [s.kind for s in sections] == ["impurity", "stop"]
    assert "leaf" in title.lower()
    assert str(leaf.prediction) in sections[1].title
    assert sections[1].lines and "0" in sections[1].lines[0]


def test_c45_working_adds_split_info_and_gain_ratio(weather_numeric):
    df, features, types, target = weather_numeric
    tr = C45Tree().fit(df[features], df[target], feature_types=types).get_trace()
    step = next(s for s in tr.steps if s.action == ACTION_SPLIT)
    _title, sections = theater.build_sections(step, tr)
    working = next(s for s in sections if s.kind == "working")
    assert any(l.startswith("SplitInfo") for l in working.lines)
    assert any(l.startswith("GainRatio") for l in working.lines)
    choice = next(s for s in sections if s.kind == "choice")
    assert "GainRatio(S, A)" in choice.lines[0]


def test_numeric_branches_read_as_inequalities(iris):
    df, features, types, target = iris
    tr = CARTTree(max_depth=2).fit(df[features], df[target], feature_types=types).get_trace()
    _title, sections = theater.build_sections(tr.steps[0], tr)
    working = next(s for s in sections if s.kind == "working")
    assert any("≤" in l for l in working.lines)
    assert any("> 2.45" in l for l in working.lines)


def test_frames_say_why_the_node_exists(trace):
    frames = theater.build_frames(trace)
    root = [f for f in frames if f.node_id == trace.tree.root.node_id][0]
    assert root.here.startswith("Root")
    child = [f for f in frames if f.depth == 1][0]
    assert child.here.startswith("Here because")
    assert any(cond in child.here for cond in trace.steps[child.step - 1].path_conditions)


def test_payload_ships_one_drawing_and_one_board_per_step(titanic):
    df, features, types, target = titanic
    tr = C45Tree().fit(df[features], df[target], feature_types=types).get_trace()
    assert len(tr) > 100  # the unpruned tree is big
    payload = theater.theater_payload(tr, key="t", prefer_svg=False)
    assert set(payload["tree"]) == {"dot"}
    assert len(payload["states"]) == len(tr) + 1
    assert len(payload["boards"]) == len(tr)
    assert all("sections" not in f for f in payload["frames"])
    assert len(json.dumps(payload)) < 1_500_000
    html = theater.theater_html(tr, key="t", prefer_svg=False)
    assert "</script>" in html and "<\\/" in html  # payload is script-safe


def test_decision_block_explains_c45_average_gain_filter_when_it_bites():
    """When the top Gain Ratio loses to the average-gain rule, the board says why."""
    from src.core.metrics import train_test_split
    from src.datasets.loaders import load_builtin
    from src.datasets.preprocess import Preprocessor

    prepared = Preprocessor().prepare(load_builtin("Weather (numeric)"))
    train, _test = train_test_split(len(prepared), prepared.y, 0.3, True, 42)
    model = C45Tree(max_depth=4, min_samples_leaf=1).fit(
        prepared.X[train], prepared.y[train], prepared.feature_names, prepared.feature_types)
    tr = model.get_trace()
    step = tr.steps[0]
    ranked = [c for c in step.ranked_candidates if c.is_valid]
    assert ranked[0] is not step.chosen, "fixture must exercise the filter"

    choice = theater.build_sections(step, tr)[1][-1]
    assert len(choice.lines) == 4
    assert choice.lines[0].startswith(r"\bar{g}")
    assert "shortlist" in choice.lines[1]
    assert choice.lines[3].endswith(r"\Rightarrow \text{%s}" % step.chosen.feature_name)
    assert ranked[0].feature_name not in choice.lines[3]
    assert ranked[0].feature_name in choice.note and "set aside" in choice.note
