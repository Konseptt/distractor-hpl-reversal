"""Abstract eye-tracker interface.

Concrete adapters (EyeLink, Tobii, …) should subclass EyeTracker without
changing the behavioral task. Version 1 ships MockTracker only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class GazeSample:
    experiment_timestamp: float
    tracker_timestamp: float
    x: float
    y: float
    pupil: float
    validity: float


class EyeTracker(ABC):
    """Hardware-independent eye-tracking API."""

    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def calibrate(self) -> None:
        ...

    @abstractmethod
    def start_recording(self) -> None:
        ...

    @abstractmethod
    def stop_recording(self) -> None:
        ...

    @abstractmethod
    def get_latest_sample(self) -> Optional[GazeSample]:
        ...

    @abstractmethod
    def send_message(self, message: str) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


# Future adapters (not implemented in v1):
# class EyeLinkTracker(EyeTracker): ...
# class TobiiTracker(EyeTracker): ...
# class GazeContingentController:  # observational in v1; hook for later causal designs
