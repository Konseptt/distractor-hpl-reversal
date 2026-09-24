"""Visual-angle stimulus helpers for the additional-singleton search array."""

from __future__ import annotations

import math
from typing import Any, Optional

# PsychoPy imports are deferred so unit tests / schedule tools need no display.


def position_xy(index: int, n_positions: int, eccentricity_deg: float) -> tuple[float, float]:
    """Position index 0 at top (12 o'clock), increasing clockwise."""
    angle = -math.pi / 2 + (2 * math.pi * index / n_positions)
    return eccentricity_deg * math.cos(angle), eccentricity_deg * math.sin(angle)


def make_monitor(config: dict[str, Any]):
    from psychopy import monitors

    mon_cfg = config["monitor"]
    mon = monitors.Monitor(mon_cfg["name"])
    mon.setWidth(mon_cfg["width_cm"])
    mon.setDistance(mon_cfg["distance_cm"])
    mon.setSizePix(tuple(mon_cfg["resolution"]))
    return mon


def create_window(config: dict[str, Any], *, simulate: bool = False):
    from psychopy import visual

    mon = make_monitor(config)
    disp = config["display"]
    size = tuple(config["monitor"]["resolution"])
    return visual.Window(
        size=size,
        fullscr=bool(disp.get("fullscreen", True)) and not simulate,
        monitor=mon,
        units=disp.get("units", "deg"),
        color=disp.get("background", [0.5, 0.5, 0.5]),
        allowGUI=simulate,
        checkTiming=not simulate,
    )


class SearchDisplay:
    """Reusable search-array stimuli (circles + diamond target + internal line)."""

    def __init__(self, win, config: dict[str, Any]):
        from psychopy import visual

        self.win = win
        self.config = config
        disp = config["display"]
        self.n_positions = int(disp["positions"])
        self.eccentricity = float(disp["eccentricity_deg"])
        self.item_size = float(disp["item_size_deg"])
        self.nontarget_color = disp["nontarget_color"]
        self.distractor_color = disp["distractor_color"]
        self.line_length = float(disp["line_length_deg"])
        self.line_width = float(disp["line_width_deg"])

        self.fixation = visual.TextStim(
            win,
            text="+",
            color="white",
            height=float(disp["fixation_size_deg"]),
            pos=(0, 0),
        )
        self.items = [
            visual.Circle(
                win,
                radius=self.item_size / 2,
                fillColor=self.nontarget_color,
                lineColor=self.nontarget_color,
                edges=64,
            )
            for _ in range(self.n_positions)
        ]
        # Diamond = rotated square
        self.target = visual.Rect(
            win,
            width=self.item_size,
            height=self.item_size,
            fillColor=self.nontarget_color,
            lineColor=self.nontarget_color,
            ori=45,
        )
        self.target_line = visual.Line(
            win,
            start=(-self.line_length / 2, 0),
            end=(self.line_length / 2, 0),
            lineColor="black",
            lineWidth=2,
        )
        self.message = visual.TextStim(win, text="", color="white", height=0.6, wrapWidth=20)

    def draw_fixation(self) -> None:
        self.fixation.draw()

    def configure_trial(
        self,
        *,
        target_location: int,
        target_orientation: str,
        distractor_present: bool,
        distractor_location: Optional[int],
    ) -> None:
        for i, stim in enumerate(self.items):
            xy = position_xy(i, self.n_positions, self.eccentricity)
            stim.pos = xy
            stim.fillColor = self.nontarget_color
            stim.lineColor = self.nontarget_color
            stim.opacity = 1.0
            if i == target_location:
                stim.opacity = 0.0  # hide circle under diamond
            elif distractor_present and i == distractor_location:
                stim.fillColor = self.distractor_color
                stim.lineColor = self.distractor_color

        txy = position_xy(target_location, self.n_positions, self.eccentricity)
        self.target.pos = txy
        self.target.fillColor = self.nontarget_color
        self.target.lineColor = self.nontarget_color
        self.target_line.pos = txy
        if target_orientation == "horizontal":
            self.target_line.start = (-self.line_length / 2, 0)
            self.target_line.end = (self.line_length / 2, 0)
        else:
            self.target_line.start = (0, -self.line_length / 2)
            self.target_line.end = (0, self.line_length / 2)

    def draw_array(self) -> None:
        for stim in self.items:
            if stim.opacity > 0:
                stim.draw()
        self.target.draw()
        self.target_line.draw()
        self.fixation.draw()

    def show_message(self, text: str) -> None:
        self.message.text = text
        self.message.draw()
