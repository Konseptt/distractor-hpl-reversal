"""Counterbalancing, reversal, and stable-mode tests."""

from pathlib import Path

from experiment.trial_generator import (
    generate_session_schedules,
    load_config,
    resolve_hpl_sequence,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config.yaml")


def test_odd_participant_left_then_right():
    seq = resolve_hpl_sequence(
        mode="reversal",
        participant_id="1",
        n_blocks=6,
        acquisition_blocks=3,
        left_hpl=6,
        right_hpl=2,
    )
    assert [h for _, h in seq[:3]] == [6, 6, 6]
    assert [h for _, h in seq[3:]] == [2, 2, 2]
    assert [p for p, _ in seq[:3]] == ["acquisition"] * 3
    assert [p for p, _ in seq[3:]] == ["reversal"] * 3


def test_even_participant_right_then_left():
    seq = resolve_hpl_sequence(
        mode="reversal",
        participant_id="2",
        n_blocks=6,
        acquisition_blocks=3,
        left_hpl=6,
        right_hpl=2,
    )
    assert [h for _, h in seq[:3]] == [2, 2, 2]
    assert [h for _, h in seq[3:]] == [6, 6, 6]


def test_stable_does_not_reverse():
    seq = resolve_hpl_sequence(
        mode="stable",
        participant_id="1",
        n_blocks=6,
        acquisition_blocks=3,
        left_hpl=6,
        right_hpl=2,
    )
    assert all(h == 6 for _, h in seq)
    assert all(p == "stable" for p, _ in seq)


def test_stable_even_uses_right():
    seq = resolve_hpl_sequence(
        mode="stable",
        participant_id="4",
        n_blocks=6,
        acquisition_blocks=3,
        left_hpl=6,
        right_hpl=2,
    )
    assert all(h == 2 for _, h in seq)


def test_reversal_happens_at_correct_trial():
    cfg = {**CONFIG, "random_seed": 10, "experiment": {**CONFIG["experiment"], "mode": "reversal"}}
    session = generate_session_schedules(cfg, participant_id="1", session_id="a")
    assert session["reversal_global_index"] == 241  # 3*80 + 1
    all_exp = [t for block in session["blocks"] for t in block]
    last_acq = [t for t in all_exp if t.phase == "acquisition"][-1]
    first_rev = [t for t in all_exp if t.phase == "reversal"][0]
    assert last_acq.trial_relative_to_reversal == -1
    assert first_rev.trial_relative_to_reversal == 1
    assert last_acq.current_HPL == 6
    assert first_rev.current_HPL == 2


def test_stable_has_virtual_reversal_index():
    cfg = {**CONFIG, "random_seed": 10, "experiment": {**CONFIG["experiment"], "mode": "stable"}}
    session = generate_session_schedules(cfg, participant_id="1", session_id="a")
    exp = CONFIG["experiment"]
    assert session["reversal_global_index"] == exp["acquisition_blocks"] * exp["trials_per_block"] + 1
    all_exp = [t for block in session["blocks"] for t in block]
    assert all(t.trial_relative_to_reversal is not None for t in all_exp)
