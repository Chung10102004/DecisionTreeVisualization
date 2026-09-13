"""The Build page's player: reveal states, frames, and the size of what is shipped."""
from __future__ import annotations

import json

import pytest

from src.core.c45 import C45Tree
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
        # nothing is ever both decided and queued
        assert not set(states[k + 1]["processed"]) & set(states[k + 1]["pending"])


def test_one_frame_per_phase_and_the_tree_grows_on_the_deciding_phase(trace):
    frames = theater.build_frames(trace)
    assert len(frames) == total_ticks(trace)
    assert frames[-1].final and "complete" in frames[-1].banner.lower()
    for frame in frames:
        step = trace.steps[frame.step - 1]
        deciding = frame.phase in ("choice", "stop")
        assert frame.state == (frame.step if deciding else frame.step - 1)
        assert frame.title and frame.what and frame.here
        assert frame.lines, f"{frame.phase} frame has no formula"
        if step.action == ACTION_SPLIT:
            assert frame.phase in ("impurity", "score", "working", "choice")
        else:
            assert frame.phase in ("impurity", "stop")


def test_frames_say_why_the_node_exists(trace):
    frames = theater.build_frames(trace)
    root = [f for f in frames if f.node_id == trace.tree.root.node_id][0]
    assert root.here.startswith("Root")
    child = [f for f in frames if f.depth == 1][0]
    assert child.here.startswith("Here because")
    assert any(cond in child.here for cond in trace.steps[child.step - 1].path_conditions)


def test_choice_frame_names_the_winner_and_keeps_the_argmax_short(weather_numeric):
    df, features, types, target = weather_numeric
    tr = C45Tree().fit(df[features], df[target], feature_types=types).get_trace()
    choice = [f for f in theater.build_frames(tr) if f.phase == "choice"][0]
    chosen = tr.steps[choice.step - 1].chosen
    line = choice.lines[0]
    assert line.endswith(r"\Rightarrow " + r"\text{%s}" % chosen.feature_name)
    assert line.count("{:}") <= theater.MAX_ARGMAX
    working = [f for f in theater.build_frames(tr) if f.phase == "working"][0]
    assert len(working.lines) == 3  # gain, split info, gain ratio for C4.5


def test_payload_ships_one_drawing_however_many_steps(titanic):
    df, features, types, target = titanic
    tr = C45Tree().fit(df[features], df[target], feature_types=types).get_trace()
    assert len(tr) > 100  # the unpruned tree is big
    payload = theater.theater_payload(tr, key="t", prefer_svg=False)
    assert set(payload["tree"]) == {"dot"}
    assert len(payload["states"]) == len(tr) + 1
    assert len(json.dumps(payload)) < 1_500_000
    html = theater.theater_html(tr, key="t", prefer_svg=False)
    assert "</script>" in html and "<\\/" in html  # payload is script-safe
