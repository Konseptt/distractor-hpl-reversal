#!/usr/bin/env python3
"""Validate a saved block schedule JSON against config tolerances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.trial_generator import (
    TrialSpec,
    load_config,
    print_stats,
    validate_schedule,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("schedule_json", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--practice", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    payload = json.loads(args.schedule_json.read_text(encoding="utf-8"))
    trials = [TrialSpec(**t) for t in payload["trials"]]
    stats = validate_schedule(
        trials,
        n_positions=cfg["display"]["positions"],
        distractor_present_prob=cfg["probabilities"]["distractor_present"],
        hpl_given_distractor=cfg["probabilities"]["hpl_given_distractor"],
        rate_tolerance=cfg["validation"]["rate_tolerance"],
        target_count_max_diff=cfg["validation"]["target_count_max_diff"],
        orientation_tolerance=cfg["validation"]["orientation_tolerance"],
        expect_hpl_bias=not args.practice,
        raise_on_error=False,
    )
    print_stats(stats, title=str(args.schedule_json))
    return 1 if stats.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
