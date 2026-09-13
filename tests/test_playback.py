"""The Build page's playback cursor and the per-formula explanations."""
from __future__ import annotations

import pytest

from src.core.c45 import C45Tree
from src.core.cart import CARTRegressor
from src.core.id3 import ID3Tree
from src.core.trace import ACTION_LEAF, ACTION_SPLIT
from src.core import playback
from src.core.playback import Cursor
from src.ui import narration


@pytest.fixture
def trace(play_tennis):
    df, features, types, target = play_tennis
    return ID3Tree().fit(df[features], df[target], feature_types=types).get_trace()


def test_split_steps_have_four_phases_and_leaves_two(trace):
    for step in trace.steps:
        expected = 3 if step.action == ACTION_SPLIT else 1
        assert playback.max_phase(step) == expected


def test_advance_walks_every_phase_of_every_step_exactly_once(trace):
    cursor = playback.first()
    visited = [cursor]
    while True:
        cursor, finished = playback.advance(cursor, trace)
        if finished:
            break
        visited.append(cursor)
    assert len(visited) == playback.total_ticks(trace)
    assert visited[-1] == playback.last(trace)
    assert len(set(visited)) == len(visited)
    # a leaf goes impurity -> stop, then straight on to the next node
    for a, b in zip(visited, visited[1:]):
        if a.step != b.step:
            assert a.phase == playback.max_phase(trace.steps[a.step]) and b.phase == 0


def test_retreat_is_the_inverse_of_advance(trace):
    cursor = playback.first()
    for _ in range(playback.total_ticks(trace) - 1):
        forward, _ = playback.advance(cursor, trace)
        assert playback.retreat(forward, trace) == cursor
        cursor = forward
    assert playback.retreat(playback.first(), trace) == playback.first()


def test_advance_at_the_end_reports_finished_and_stays(trace):
    end = playback.last(trace)
    assert playback.advance(end, trace) == (end, True)


def test_scrubbing_to_a_step_reveals_the_whole_step(trace):
    for index, step in enumerate(trace.steps):
        cursor = playback.at_step(trace, index)
        assert cursor.step == index
        assert cursor.is_complete(step)


def test_clamp_keeps_out_of_range_cursors_inside_the_trace(trace):
    assert Cursor(-3, -1).clamp(trace) == Cursor(0, 0)
    assert Cursor(99, 99).clamp(trace) == playback.last(trace)


def test_progress_runs_from_first_tick_to_one(trace):
    assert 0 < playback.progress(playback.first(), trace) < 1
    assert playback.progress(playback.last(trace), trace) == pytest.approx(1.0)


def test_phase_labels_are_distinct_within_a_step(trace):
    for step in trace.steps:
        labels = [playback.phase_label(step, p) for p in range(playback.max_phase(step) + 1)]
        assert len(set(labels)) == len(labels)


# ----------------------------------------------------------------- narration
def _traces(play_tennis, weather_numeric, auto_mpg):
    df, f, t, y = play_tennis
    id3 = ID3Tree().fit(df[f], df[y], feature_types=t).get_trace()
    df, f, t, y = weather_numeric
    c45 = C45Tree().fit(df[f], df[y], feature_types=t).get_trace()
    df, f, t, y = auto_mpg
    cart = CARTRegressor(max_depth=3).fit(df[f], df[y], feature_types=t).get_trace()
    return id3, c45, cart


def test_every_algorithm_gets_a_reason_for_every_formula(play_tennis, weather_numeric, auto_mpg):
    for tr in _traces(play_tennis, weather_numeric, auto_mpg):
        assert tr.algorithm in narration.why_impurity(tr)
        assert narration.why_scoring(tr).strip()
        assert narration.why_choice(tr).strip()
        for step in tr.steps:
            if step.action == ACTION_LEAF:
                text = narration.why_stop(step, tr)
                assert str(step.prediction) in text
                assert "{" not in text  # every placeholder was filled
            else:
                assert step.chosen.feature_name in narration.why_working(tr, step.chosen)


def test_scoring_explanations_name_the_right_criterion(play_tennis, weather_numeric, auto_mpg):
    id3, c45, cart = _traces(play_tennis, weather_numeric, auto_mpg)
    assert "Information Gain" in narration.why_scoring(id3)
    assert "Gain Ratio" in narration.why_scoring(c45) and "SplitInfo" in narration.why_scoring(c45)
    assert "binary" in narration.why_scoring(cart)
    assert "variance" in narration.why_impurity(cart).lower()
    assert "mean target" in narration.why_stop(cart.steps[-1], cart)

