"""Display-accurate timing helpers."""

from __future__ import annotations

import random
from typing import Optional


def sample_duration(min_s: float, max_s: float, rng: Optional[random.Random] = None) -> float:
    r = rng.random() if rng is not None else random.random()
    return min_s + r * (max_s - min_s)


def wait_seconds(win, duration_s: float, *, clock=None, draw_fn=None) -> float:
    """Wait approximately duration_s while optionally redrawing each frame.

    Returns the flip timestamp of the final frame (PsychoPy timebase).
    """
    from psychopy import core

    if clock is None:
        clock = core.Clock()
    clock.reset()
    last_flip = None
    while clock.getTime() < duration_s:
        if draw_fn is not None:
            draw_fn()
        last_flip = win.flip()
    return last_flip if last_flip is not None else win.flip()


def flip_onset(win, draw_fn) -> float:
    """Draw once and return the onset flip timestamp."""
    draw_fn()
    return win.flip()
