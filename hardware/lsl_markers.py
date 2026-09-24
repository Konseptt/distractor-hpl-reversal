"""LSL event marker output via pylsl (optional dependency)."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from hardware.marker_interface import EventMarker


class LSLMarkers(EventMarker):
    """Push string markers to an LSL outlet named 'PsychoPyMarkers'."""

    def __init__(self, stream_name: str = "PsychoPyMarkers", source_id: str = "hpl_reversal"):
        self.stream_name = stream_name
        self.source_id = source_id
        self._outlet = None
        self._t0 = time.perf_counter()

    def initialize(self) -> None:
        try:
            from pylsl import StreamInfo, StreamOutlet
        except ImportError as exc:
            raise ImportError(
                "pylsl is required for LSL markers. Install pylsl or set hardware.eeg_markers: mock"
            ) from exc

        info = StreamInfo(
            name=self.stream_name,
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id=self.source_id,
        )
        self._outlet = StreamOutlet(info)
        self._t0 = time.perf_counter()

    def send_marker(self, code: str, metadata: Optional[dict[str, Any]] = None) -> float:
        if self._outlet is None:
            raise RuntimeError("LSLMarkers not initialized")
        ts = time.perf_counter() - self._t0
        payload = code if not metadata else f"{code}|{json.dumps(metadata, default=str)}"
        self._outlet.push_sample([payload])
        return ts

    def close(self) -> None:
        self._outlet = None
