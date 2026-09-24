"""Probability and session-level validation tests."""

from pathlib import Path

from experiment.trial_generator import (
    conditional_hpl_probability,
    generate_session_schedules,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config.yaml")


def test_conditional_hpl_formula():
    p = conditional_hpl_probability(0.70, 8)
    assert abs(p - (0.70 / 0.875)) < 1e-9


def test_session_hpl_rates_near_config():
    cfg = dict(CONFIG)
    cfg = {**cfg, "random_seed": 99}
    session = generate_session_schedules(cfg, participant_id="1", session_id="s1")
    all_exp = [t for block in session["blocks"] for t in block]
    dp = [t for t in all_exp if t.distractor_present]
    hpl = [t for t in dp if t.distractor_is_HPL]
    rate = len(hpl) / len(dp)
    assert abs(rate - 0.70) <= 0.05


def test_distractor_absent_rate():
    cfg = {**CONFIG, "random_seed": 5}
    session = generate_session_schedules(cfg, participant_id="3", session_id="s1")
    all_exp = [t for block in session["blocks"] for t in block]
    da = sum(1 for t in all_exp if not t.distractor_present)
    assert abs(da / len(all_exp) - 0.20) <= 0.05
