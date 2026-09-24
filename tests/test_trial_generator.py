"""Tests for trial schedule generation constraints."""

from pathlib import Path

import pytest

from experiment.trial_generator import (
    ScheduleGenerationError,
    compute_stats,
    generate_block_schedule,
    generate_session_schedules,
    hash_seed,
    load_config,
    resolve_hpl_sequence,
    validate_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config.yaml")


def _rng(seed=12345):
    import random

    return random.Random(seed)


def _gen_block(hpl=6, n=80, seed=1, flat=False, practice=False):
    cfg = CONFIG
    return generate_block_schedule(
        block=1,
        phase="acquisition",
        hpl=None if practice else hpl,
        n_trials=n,
        n_positions=cfg["display"]["positions"],
        distractor_present_prob=cfg["probabilities"]["distractor_present"],
        hpl_given_distractor=cfg["probabilities"]["hpl_given_distractor"],
        keys=cfg["experiment"]["keys"],
        max_consecutive_hpl=cfg["sequence"]["max_consecutive_hpl"],
        max_consecutive_orientation=cfg["sequence"]["max_consecutive_orientation"],
        max_attempts=cfg["sequence"]["max_generation_attempts"],
        rng=_rng(seed),
        is_practice=practice,
        flat_distractor=flat or practice,
        trial_number_global_start=1,
        reversal_global_index=241,
    )


def test_target_never_overlaps_distractor():
    trials = _gen_block()
    for t in trials:
        if t.distractor_present:
            assert t.distractor_location != t.target_location


def test_hpl_probability_approximately_correct():
    trials = _gen_block(n=80, seed=7)
    stats = compute_stats(
        trials,
        n_positions=8,
        distractor_present_prob=0.8,
        hpl_given_distractor=0.7,
        rate_tolerance=0.05,
        target_count_max_diff=2,
        orientation_tolerance=0.08,
    )
    assert abs(stats.hpl_among_dp_pct - 0.7) <= 0.05
    assert not stats.errors


def test_target_positions_balanced():
    trials = _gen_block(n=80, seed=9)
    stats = validate_schedule(
        trials,
        n_positions=8,
        distractor_present_prob=0.8,
        hpl_given_distractor=0.7,
        rate_tolerance=0.05,
        target_count_max_diff=2,
        orientation_tolerance=0.08,
    )
    counts = list(stats.target_counts.values())
    assert max(counts) - min(counts) <= 2


def test_target_orientations_balanced():
    trials = _gen_block(n=80, seed=11)
    n_h = sum(1 for t in trials if t.target_orientation == "horizontal")
    assert abs(n_h / len(trials) - 0.5) <= 0.08


def test_practice_no_hpl_bias():
    trials = _gen_block(practice=True, n=20, seed=3)
    assert all(t.is_practice for t in trials)
    assert all(not t.distractor_is_HPL for t in trials)
    assert all(t.current_HPL is None for t in trials)


def test_seed_reproduces_schedule():
    a = _gen_block(seed=42)
    b = _gen_block(seed=42)
    assert [(t.target_location, t.distractor_location, t.target_orientation) for t in a] == [
        (t.target_location, t.distractor_location, t.target_orientation) for t in b
    ]


def test_hash_seed_stable():
    assert hash_seed("001", "1") == hash_seed("001", "1")
    assert hash_seed("001", "1") != hash_seed("002", "1")
