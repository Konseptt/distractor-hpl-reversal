"""EEG / behavioral event marker codes.

Every marker sent to LSL (or mock) must also be written to events.csv
with the same local experiment timestamp.
"""

from __future__ import annotations

EVENT_CODES: dict[str, str] = {
    "EXP_START": "S01",
    "EXP_END": "S02",
    "BLOCK_START": "S10",
    "BLOCK_END": "S11",
    "FIXATION_ON": "S20",
    "ARRAY_ON_ABSENT": "S30",
    "ARRAY_ON_HPL": "S31",
    "ARRAY_ON_LP": "S32",
    "RESP_CORRECT": "S40",
    "RESP_INCORRECT": "S41",
    "RESP_TIMEOUT": "S42",
    "PHASE_ACQUISITION": "S50",
    "PHASE_REVERSAL": "S51",
}

# Reverse lookup: code -> name
CODE_TO_NAME: dict[str, str] = {v: k for k, v in EVENT_CODES.items()}


def get_code(event_name: str) -> str:
    """Return marker code for a named event; raises KeyError if unknown."""
    return EVENT_CODES[event_name]


def array_onset_code(*, distractor_present: bool, distractor_is_hpl: bool) -> str:
    """Select ARRAY_ON_* code from trial features."""
    if not distractor_present:
        return EVENT_CODES["ARRAY_ON_ABSENT"]
    if distractor_is_hpl:
        return EVENT_CODES["ARRAY_ON_HPL"]
    return EVENT_CODES["ARRAY_ON_LP"]


def response_code(*, timed_out: bool, correct: bool) -> str:
    if timed_out:
        return EVENT_CODES["RESP_TIMEOUT"]
    if correct:
        return EVENT_CODES["RESP_CORRECT"]
    return EVENT_CODES["RESP_INCORRECT"]
