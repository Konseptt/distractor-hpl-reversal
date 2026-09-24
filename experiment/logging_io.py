"""Session logging: trials.csv, events.csv, gaze.csv, metadata.json."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


TRIAL_FIELDS = [
    "participant_id",
    "session_id",
    "experiment_mode",
    "block",
    "phase",
    "trial_number_global",
    "trial_number_block",
    "trial_relative_to_reversal",
    "is_practice",
    "target_location",
    "target_orientation",
    "distractor_present",
    "distractor_location",
    "distractor_is_HPL",
    "current_HPL",
    "response_key",
    "correct_response",
    "accuracy",
    "reaction_time_ms",
    "fixation_duration_ms",
    "array_actual_onset_time",
    "response_time",
    "trial_start_time",
    "fixation_break",
    "timed_out",
    "frame_intervals_mean",
    "frame_intervals_sd",
    "random_seed",
]

EVENT_FIELDS = [
    "timestamp",
    "trial_id",
    "block",
    "event_name",
    "event_code",
    "metadata",
]

GAZE_FIELDS = [
    "experiment_timestamp",
    "tracker_timestamp",
    "trial_id",
    "block",
    "phase",
    "x",
    "y",
    "pupil",
    "validity",
]


def git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def psychopy_version() -> Optional[str]:
    try:
        import psychopy

        return getattr(psychopy, "__version__", None)
    except Exception:
        return None


class SessionLogger:
    def __init__(
        self,
        data_root: Path,
        *,
        participant_id: str,
        session_id: str,
        config: dict[str, Any],
        extra_metadata: Optional[dict[str, Any]] = None,
        random_seed: int,
    ):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = data_root / f"{participant_id}_{session_id}_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "schedules").mkdir(exist_ok=True)

        self.participant_id = participant_id
        self.session_id = session_id
        self.config = config
        self.random_seed = random_seed

        self.trials_path = self.session_dir / "trials.csv"
        self.events_path = self.session_dir / "events.csv"
        self.gaze_path = self.session_dir / "gaze.csv"
        self.metadata_path = self.session_dir / "metadata.json"

        self._trial_file = open(self.trials_path, "w", newline="", encoding="utf-8")
        self._event_file = open(self.events_path, "w", newline="", encoding="utf-8")
        self._gaze_file = open(self.gaze_path, "w", newline="", encoding="utf-8")
        self._trial_writer = csv.DictWriter(self._trial_file, fieldnames=TRIAL_FIELDS)
        self._event_writer = csv.DictWriter(self._event_file, fieldnames=EVENT_FIELDS)
        self._gaze_writer = csv.DictWriter(self._gaze_file, fieldnames=GAZE_FIELDS)
        self._trial_writer.writeheader()
        self._event_writer.writeheader()
        self._gaze_writer.writeheader()

        meta = {
            "participant_id": participant_id,
            "session_id": session_id,
            "datetime_utc": datetime.now(timezone.utc).isoformat(),
            "computer_name": platform.node(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "psychopy_version": psychopy_version(),
            "experiment_version": config.get("experiment", {}).get("version"),
            "git_commit": git_commit_hash(Path(__file__).resolve().parents[1]),
            "screen_resolution": config.get("monitor", {}).get("resolution"),
            "monitor": config.get("monitor"),
            "config": config,
            "random_seed": random_seed,
            "p_cond_note": (
                "Distractor HPL uses conditional probability given target≠HPL; "
                "see trial_generator.conditional_hpl_probability"
            ),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        self.metadata_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def log_trial(self, row: dict[str, Any]) -> None:
        out = {k: row.get(k, "") for k in TRIAL_FIELDS}
        self._trial_writer.writerow(out)
        self._trial_file.flush()

    def log_event(
        self,
        *,
        timestamp: float,
        trial_id: Optional[int],
        block: Optional[int],
        event_name: str,
        event_code: str,
        metadata: Any = None,
    ) -> None:
        self._event_writer.writerow(
            {
                "timestamp": timestamp,
                "trial_id": "" if trial_id is None else trial_id,
                "block": "" if block is None else block,
                "event_name": event_name,
                "event_code": event_code,
                "metadata": "" if metadata is None else json.dumps(metadata, default=str),
            }
        )
        self._event_file.flush()

    def log_gaze(self, row: dict[str, Any]) -> None:
        out = {k: row.get(k, "") for k in GAZE_FIELDS}
        self._gaze_writer.writerow(out)

    def flush_gaze(self) -> None:
        self._gaze_file.flush()

    def close(self) -> None:
        for f in (self._trial_file, self._event_file, self._gaze_file):
            try:
                f.close()
            except Exception:
                pass
