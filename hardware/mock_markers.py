"""Mock event markers (no LSL required)."""

from __future__ import annotations

import time
from typing import Any, Optional

from hardware.marker_interface import EventMarker


class MockMarkers(EventMarker):
    def __init__(self):
        self._t0 = time.perf_counter()
        self.sent: list[tuple[float, str, Optional[dict[str, Any]]]] = []

    def initialize(self) -> None:
        self._t0 = time.perf_counter()
        self.sent.clear()

    def send_marker(self, code: str, metadata: Optional[dict[str, Any]] = None) -> float:
        ts = time.perf_counter() - self._t0
        self.sent.append((ts, code, metadata))
        return ts

    def close(self) -> None:
        return
