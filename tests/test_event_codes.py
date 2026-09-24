"""Tests for event marker codes."""

from experiment.event_codes import (
    CODE_TO_NAME,
    EVENT_CODES,
    array_onset_code,
    get_code,
    response_code,
)


def test_codes_unique():
    codes = list(EVENT_CODES.values())
    assert len(codes) == len(set(codes))


def test_required_codes_present():
    required = {
        "EXP_START",
        "EXP_END",
        "BLOCK_START",
        "BLOCK_END",
        "FIXATION_ON",
        "ARRAY_ON_ABSENT",
        "ARRAY_ON_HPL",
        "ARRAY_ON_LP",
        "RESP_CORRECT",
        "RESP_INCORRECT",
        "RESP_TIMEOUT",
        "PHASE_ACQUISITION",
        "PHASE_REVERSAL",
    }
    assert required <= set(EVENT_CODES.keys())


def test_reverse_lookup():
    for name, code in EVENT_CODES.items():
        assert CODE_TO_NAME[code] == name


def test_array_onset_helpers():
    assert array_onset_code(distractor_present=False, distractor_is_hpl=False) == "S30"
    assert array_onset_code(distractor_present=True, distractor_is_hpl=True) == "S31"
    assert array_onset_code(distractor_present=True, distractor_is_hpl=False) == "S32"


def test_response_helpers():
    assert response_code(timed_out=True, correct=False) == "S42"
    assert response_code(timed_out=False, correct=True) == "S40"
    assert response_code(timed_out=False, correct=False) == "S41"


def test_get_code():
    assert get_code("FIXATION_ON") == "S20"
