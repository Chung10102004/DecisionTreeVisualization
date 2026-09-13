"""The playback cursor for the build animation.

A build step (one node) is shown in *phases* — the node's impurity, the working
for the best candidate (with the other columns' results listed beside it), and
the choice — so the formulas appear one block at a time rather than all at once.
This module is the pure arithmetic of moving that cursor around: which phases a
step has, and what comes next or before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .trace import ACTION_SPLIT, BuildStep, TrainingTrace

# Phase indices, shared with the page so the reveal logic reads as names.
PHASE_IMPURITY = 0
PHASE_WORKING = 1
PHASE_CHOICE = 2
PHASE_STOP = 1  # leaves skip straight from impurity to the stop rule

SPEED_OPTIONS: List[float] = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
DEFAULT_SPEED = 2.0


def max_phase(step: BuildStep) -> int:
    """The last phase of a step: 2 for a split, 1 for a leaf."""
    return PHASE_CHOICE if step.action == ACTION_SPLIT else PHASE_STOP


def phase_label(step: BuildStep, phase: int) -> str:
    """Short caption for the progress line."""
    if phase <= PHASE_IMPURITY:
        return "Measuring how impure the node is"
    if step.action != ACTION_SPLIT:
        return "Deciding to stop — the node becomes a leaf"
    if phase == PHASE_WORKING:
        return "Computing the gain of the best column — the others follow the same formula"
    return "Choosing the split — the tree grows"


@dataclass(frozen=True)
class Cursor:
    """Where the animation is: which step, and how much of it is revealed."""

    step: int = 0
    phase: int = 0

    def clamp(self, trace: TrainingTrace) -> "Cursor":
        total = len(trace)
        if total == 0:
            return Cursor(0, 0)
        step = min(max(self.step, 0), total - 1)
        phase = min(max(self.phase, 0), max_phase(trace.steps[step]))
        return Cursor(step, phase)

    def is_complete(self, step: BuildStep) -> bool:
        return self.phase >= max_phase(step)


def first() -> Cursor:
    return Cursor(0, 0)


def last(trace: TrainingTrace) -> Cursor:
    if len(trace) == 0:
        return Cursor(0, 0)
    index = len(trace) - 1
    return Cursor(index, max_phase(trace.steps[index]))


def at_step(trace: TrainingTrace, index: int) -> Cursor:
    """Jump to a step with everything in it revealed — what scrubbing wants."""
    return Cursor(index, 10 ** 6).clamp(trace)


def advance(cursor: Cursor, trace: TrainingTrace) -> Tuple[Cursor, bool]:
    """One tick forward. Returns ``(new_cursor, finished)``.

    ``finished`` is True when the cursor was already on the last phase of the last
    step and therefore did not move.
    """
    cursor = cursor.clamp(trace)
    step = trace.steps[cursor.step]
    if cursor.phase < max_phase(step):
        return Cursor(cursor.step, cursor.phase + 1), False
    if cursor.step < len(trace) - 1:
        return Cursor(cursor.step + 1, 0), False
    return cursor, True


def retreat(cursor: Cursor, trace: TrainingTrace) -> Cursor:
    """One tick back: the previous phase, or the last phase of the previous step."""
    cursor = cursor.clamp(trace)
    if cursor.phase > 0:
        return Cursor(cursor.step, cursor.phase - 1)
    if cursor.step > 0:
        return Cursor(cursor.step - 1, max_phase(trace.steps[cursor.step - 1]))
    return cursor


def total_ticks(trace: TrainingTrace) -> int:
    return sum(max_phase(step) + 1 for step in trace.steps)


def ticks_done(cursor: Cursor, trace: TrainingTrace) -> int:
    """How many phases have been revealed so far, for the progress bar."""
    cursor = cursor.clamp(trace)
    before = sum(max_phase(step) + 1 for step in trace.steps[: cursor.step])
    return before + cursor.phase + 1


def progress(cursor: Cursor, trace: TrainingTrace) -> float:
    total = total_ticks(trace)
    return ticks_done(cursor, trace) / total if total else 1.0
