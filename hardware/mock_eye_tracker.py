"""Mock eye tracker for development and --simulate mode."""

from __future__ import annotations

import time
from typing import Optional

from hardware.eye_tracker import EyeTracker, GazeSample


class MockTracker(EyeTracker):
    def __init__(self, sample_rate_hz: float = 500.0):
        self.sample_rate_hz = sample_rate_hz
        self._recording = False
        self._t0 = time.perf_counter()
        self._messages: list[tuple[float, str]] = []
        self._sample_index = 0

    def initialize(self) -> None:
        self._t0 = time.perf_counter()

    def calibrate(self) -> None:
        return  # no-op

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def get_latest_sample(self) -> Optional[GazeSample]:
        if not self._recording:
            return None
        now = time.perf_counter() - self._t0
        self._sample_index += 1
        # Gaze near fixation (0, 0) with tiny noise for realism
        noise = ((self._sample_index % 7) - 3) * 0.01
        return GazeSample(
            experiment_timestamp=now,
            tracker_timestamp=now * 1000.0,
            x=noise,
            y=-noise,
            pupil=3.5,
            validity=1.0,
        )

    def send_message(self, message: str) -> None:
        self._messages.append((time.perf_counter() - self._t0, message))

    def close(self) -> None:
        self._recording = False

    @property
    def messages(self) -> list[tuple[float, str]]:
        return list(self._messages)
