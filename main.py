#!/usr/bin/env python3
"""Entry point for the HPL distractor-suppression experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.trial_generator import load_config, generate_session_schedules, compute_stats
from hardware.mock_eye_tracker import MockTracker
from hardware.mock_markers import MockMarkers


def build_markers(kind: str):
    if kind == "lsl":
        from hardware.lsl_markers import LSLMarkers

        return LSLMarkers()
    return MockMarkers()


def build_eye_tracker(kind: str):
    if kind == "real":
        raise NotImplementedError(
            "Real eye-tracker adapters are not implemented in v1. "
            "Use hardware.eye_tracker: mock, or implement EyeLinkTracker/TobiiTracker."
        )
    return MockTracker()


def prompt_meta(args) -> dict:
    participant_id = args.participant or input("Participant ID: ").strip()
    session_id = args.session or input("Session ID: ").strip() or "1"
    age = args.age
    handedness = args.handedness
    if not args.simulate and age is None:
        age = input("Age (optional): ").strip() or None
    if not args.simulate and handedness is None:
        handedness = input("Handedness (optional, L/R): ").strip() or None
    return {
        "participant_id": participant_id,
        "session_id": session_id,
        "age": age,
        "handedness": handedness,
    }


def run_headless_simulate(config: dict, meta: dict, data_root: Path) -> int:
    """Fast schedule + logging simulation without PsychoPy display."""
    import random
    from experiment.logging_io import SessionLogger
    from experiment.trial_generator import save_block_schedule
    from experiment.event_codes import EVENT_CODES, array_onset_code, response_code

    session = generate_session_schedules(config, meta["participant_id"], meta["session_id"])
    logger = SessionLogger(
        data_root,
        participant_id=meta["participant_id"],
        session_id=meta["session_id"],
        config=config,
        random_seed=session["seed"],
        extra_metadata={
            "age": meta.get("age"),
            "handedness": meta.get("handedness"),
            "simulate": True,
            "headless": True,
            "p_cond": session["p_cond"],
            "hpl_sequence": session["hpl_sequence"],
            "reversal_global_index": session["reversal_global_index"],
        },
    )
    markers = MockMarkers()
    markers.initialize()
    et = MockTracker()
    et.initialize()
    et.start_recording()

    rng = random.Random(session["seed"] + 1)
    sim = config.get("simulation", {})
    n_pos = int(config["display"]["positions"])
    capture = float(sim.get("capture_cost_ms", 0)) / 1000.0
    lr = float(sim.get("learning_rate", 0))
    w = [1.0 / n_pos] * n_pos  # learned P(distractor at location)
    t = 0.0

    def emit(name, trial_id=None, block=None, metadata=None):
        nonlocal t
        code = EVENT_CODES[name]
        markers.send_marker(code, metadata)
        et.send_message(f"{name}:{code}")
        logger.log_event(
            timestamp=t,
            trial_id=trial_id,
            block=block,
            event_name=name,
            event_code=code,
            metadata=metadata,
        )
        t += 0.001

    emit("EXP_START")
    save_block_schedule(
        logger.session_dir / "schedules" / "block_00_practice.json",
        session["practice"],
        {"phase": "practice"},
    )
    emit("BLOCK_START", block=0, metadata={"practice": True})
    emit("BLOCK_END", block=0)

    if config["experiment"]["mode"] == "reversal":
        emit("PHASE_ACQUISITION", block=1)
    first_rev = True
    for block_trials in session["blocks"]:
        b = block_trials[0].block
        phase = block_trials[0].phase
        hpl = block_trials[0].current_HPL
        save_block_schedule(
            logger.session_dir / "schedules" / f"block_{b:02d}.json",
            block_trials,
            {"phase": phase, "hpl": hpl},
        )
        if phase == "reversal" and first_rev and config["experiment"]["mode"] == "reversal":
            emit("PHASE_REVERSAL", block=b)
            first_rev = False
        emit("BLOCK_START", block=b, metadata={"phase": phase, "hpl": hpl})
        for trial in block_trials:
            trial_start = t
            emit("FIXATION_ON", trial_id=trial.trial_number_global, block=b)
            t += float(config["timing"]["simulate_fixation"])
            onset = t
            code = array_onset_code(
                distractor_present=trial.distractor_present,
                distractor_is_hpl=trial.distractor_is_HPL,
            )
            name = {
                "S30": "ARRAY_ON_ABSENT",
                "S31": "ARRAY_ON_HPL",
                "S32": "ARRAY_ON_LP",
            }[code]
            emit(name, trial_id=trial.trial_number_global, block=b)

            mean_rt = float(sim.get("mean_rt_ms", 550)) / 1000.0
            sd = float(sim.get("rt_sd_ms", 80)) / 1000.0
            rt = rng.gauss(mean_rt, sd)
            if trial.distractor_present:
                loc = trial.distractor_location
                rt += capture * (1.0 - w[loc])
                w = [wi + lr * ((i == loc) - wi) for i, wi in enumerate(w)]
            rt = max(0.15, rt)
            timed_out = rt > float(config["timing"]["search_timeout"])
            correct = (not timed_out) and (rng.random() < float(sim.get("response_accuracy", 0.9)))
            key = trial.correct_response if correct else (
                "j" if trial.correct_response == "f" else "f"
            )
            if timed_out:
                key = ""
                correct = False
            rname = {
                "S40": "RESP_CORRECT",
                "S41": "RESP_INCORRECT",
                "S42": "RESP_TIMEOUT",
            }[response_code(timed_out=timed_out, correct=correct)]
            t = onset + min(rt, float(config["timing"]["search_timeout"]))
            emit(rname, trial_id=trial.trial_number_global, block=b)
            t += float(config["timing"]["simulate_iti"])

            sample = et.get_latest_sample()
            if sample:
                logger.log_gaze(
                    {
                        "experiment_timestamp": t,
                        "tracker_timestamp": sample.tracker_timestamp,
                        "trial_id": trial.trial_number_global,
                        "block": b,
                        "phase": phase,
                        "x": sample.x,
                        "y": sample.y,
                        "pupil": sample.pupil,
                        "validity": sample.validity,
                    }
                )

            logger.log_trial(
                {
                    "participant_id": meta["participant_id"],
                    "session_id": meta["session_id"],
                    "experiment_mode": config["experiment"]["mode"],
                    "block": trial.block,
                    "phase": trial.phase,
                    "trial_number_global": trial.trial_number_global,
                    "trial_number_block": trial.trial_number_block,
                    "trial_relative_to_reversal": trial.trial_relative_to_reversal
                    if trial.trial_relative_to_reversal is not None
                    else "",
                    "is_practice": False,
                    "target_location": trial.target_location,
                    "target_orientation": trial.target_orientation,
                    "distractor_present": trial.distractor_present,
                    "distractor_location": trial.distractor_location
                    if trial.distractor_location is not None
                    else "",
                    "distractor_is_HPL": trial.distractor_is_HPL,
                    "current_HPL": trial.current_HPL if trial.current_HPL is not None else "",
                    "response_key": key,
                    "correct_response": trial.correct_response,
                    "accuracy": int(correct),
                    "reaction_time_ms": "" if timed_out else round(rt * 1000, 3),
                    "fixation_duration_ms": float(config["timing"]["simulate_fixation"]) * 1000,
                    "array_actual_onset_time": onset,
                    "response_time": onset + min(rt, float(config["timing"]["search_timeout"])),
                    "trial_start_time": trial_start,
                    "fixation_break": False,
                    "timed_out": timed_out,
                    "frame_intervals_mean": "",
                    "frame_intervals_sd": "",
                    "random_seed": session["seed"],
                }
            )
        emit("BLOCK_END", block=b)

    emit("EXP_END")
    logger.flush_gaze()
    logger.close()
    markers.close()
    et.close()

    # Print summary matching plan format
    cfg = config
    n_pos = cfg["display"]["positions"]
    errors = 0
    print("\n========== SIMULATION SUMMARY ==========")
    by_phase: dict = {}
    for block in session["blocks"]:
        for tr in block:
            by_phase.setdefault(tr.phase, []).append(tr)
    for i, phase in enumerate(["acquisition", "reversal", "stable"], start=1):
        trials = by_phase.get(phase)
        if not trials:
            continue
        hpl = trials[0].current_HPL
        side = "LEFT" if hpl == cfg["spatial"]["left_hpl"] else "RIGHT"
        stats = compute_stats(
            trials,
            n_positions=n_pos,
            distractor_present_prob=cfg["probabilities"]["distractor_present"],
            hpl_given_distractor=cfg["probabilities"]["hpl_given_distractor"],
            rate_tolerance=cfg["validation"]["rate_tolerance"],
            target_count_max_diff=cfg["validation"]["target_count_max_diff"],
            orientation_tolerance=cfg["validation"]["orientation_tolerance"],
        )
        label = f"PHASE {i}" if phase != "stable" else "PHASE (STABLE)"
        print(f"\n{label}")
        print(f"HPL: {side} ({hpl})")
        print(f"HPL distractors: {stats.hpl_among_dp_pct:.1%}")
        print("Target-location balance:")
        for k, v in sorted(stats.target_counts.items()):
            print(f"  {k}: {v}")
        errors += len(stats.errors)
    print(f"\nErrors:\n{errors}")
    print(f"Data written to: {logger.session_dir}")
    print("========================================\n")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HPL distractor suppression experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--participant", type=str, default=None)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--age", type=str, default=None)
    parser.add_argument("--handedness", type=str, default=None)
    parser.add_argument("--mode", choices=["reversal", "stable"], default=None)
    parser.add_argument("--simulate", action="store_true", help="Fast mock run / headless OK")
    parser.add_argument(
        "--with-display",
        action="store_true",
        help="With --simulate, open a PsychoPy window instead of headless",
    )
    parser.add_argument("--validate-only", action="store_true", help="Generate schedules and exit")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.mode:
        config["experiment"]["mode"] = args.mode

    meta = prompt_meta(args)
    data_root = ROOT / "data"

    if args.validate_only:
        session = generate_session_schedules(
            config, meta["participant_id"], meta["session_id"]
        )
        print(f"Seed={session['seed']} p_cond={session['p_cond']:.3f}")
        print(f"HPL sequence: {session['hpl_sequence']}")
        return 0

    if args.simulate and not args.with_display:
        return run_headless_simulate(config, meta, data_root)

    # PsychoPy path — requires Python 3.10–3.12
    vi = sys.version_info
    if vi >= (3, 13):
        print(
            f"ERROR: Interactive PsychoPy needs Python 3.10–3.12 "
            f"(you have {vi.major}.{vi.minor}.{vi.micro}).\n\n"
            "Create a venv and re-run:\n"
            "  cd <repo root>\n"
            "  /opt/homebrew/bin/python3.12 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            "  python main.py --participant 99 --session trial\n\n"
            "Or use headless simulation (no display):\n"
            "  python3 main.py --simulate --participant 99 --session trial",
            file=sys.stderr,
        )
        return 1

    if args.simulate:
        config = {**config, "hardware": {**config["hardware"], "eye_tracker": "mock", "eeg_markers": "mock"}}
        config["display"] = {**config["display"], "fullscreen": False}

    eye = build_eye_tracker("mock" if args.simulate else config["hardware"]["eye_tracker"])
    markers = build_markers("mock" if args.simulate else config["hardware"]["eeg_markers"])

    try:
        from experiment.task import Experiment
    except ImportError as exc:
        print(
            f"ERROR: PsychoPy import failed ({exc}).\n"
            "Install into a Python 3.10–3.12 venv (see README).",
            file=sys.stderr,
        )
        return 1

    exp = Experiment(
        config,
        participant_id=meta["participant_id"],
        session_id=meta["session_id"],
        age=meta.get("age"),
        handedness=meta.get("handedness"),
        eye_tracker=eye,
        markers=markers,
        simulate=args.simulate,
        data_root=data_root,
    )
    result = exp.run()
    return int(result.get("errors", 0) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
