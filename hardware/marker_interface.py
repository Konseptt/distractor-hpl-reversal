"""Abstract EEG / LSL event marker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class EventMarker(ABC):
    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def send_marker(self, code: str, metadata: Optional[dict[str, Any]] = None) -> float:
        """Send marker; return local experiment timestamp used for logging."""

    @abstractmethod
    def close(self) -> None:
        ...
