"""Main PsychoPy experiment loop."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Optional

from experiment.event_codes import EVENT_CODES, array_onset_code, response_code
from experiment.instructions import BREAK_TEXT, END_TEXT, INSTRUCTIONS, PRACTICE_END
from experiment.logging_io import SessionLogger
from experiment.stimuli import SearchDisplay, create_window
from experiment.timing import flip_onset, sample_duration
from experiment.trial_generator import (
    TrialSpec,
    generate_session_schedules,
    save_block_schedule,
    compute_stats,
)
from hardware.eye_tracker import EyeTracker
from hardware.marker_interface import EventMarker


class Experiment:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        participant_id: str,
        session_id: str,
        age: Optional[str] = None,
        handedness: Optional[str] = None,
        eye_tracker: EyeTracker,
        markers: EventMarker,
        simulate: bool = False,
        data_root: Optional[Path] = None,
    ):
        self.config = config
        self.participant_id = participant_id
        self.session_id = session_id
        self.age = age
        self.handedness = handedness
        self.eye_tracker = eye_tracker
        self.markers = markers
        self.simulate = simulate
        self.data_root = data_root or Path(__file__).resolve().parents[1] / "data"

        self.rng = random.Random()
        self.win = None
        self.display: Optional[SearchDisplay] = None
        self.logger: Optional[SessionLogger] = None
        self.session = None
        self.clock_t0 = None
        self._exp_clock = None

    def _now(self) -> float:
        if self._exp_clock is not None:
            return self._exp_clock.getTime()
        return time.perf_counter()

    def _emit(self, event_name: str, *, trial_id=None, block=None, metadata=None) -> float:
        code = EVENT_CODES[event_name]
        # Prefer experiment clock for behavioral log; marker backend may use its own.
        ts_marker = self.markers.send_marker(code, metadata=metadata)
        ts = self._now()
        self.eye_tracker.send_message(f"{event_name}:{code}")
        if self.logger is not None:
            self.logger.log_event(
                timestamp=ts,
                trial_id=trial_id,
                block=block,
                event_name=event_name,
                event_code=code,
                metadata=metadata if metadata is not None else {"marker_ts": ts_marker},
            )
        return ts

    def _wait_key(self, keys=("space",), timeout=None):
        if self.simulate:
            return "space", 0.0
        from psychopy import event, core

        event.clearEvents()
        clock = core.Clock()
        while True:
            these = event.getKeys(keyList=list(keys) + ["escape"], timeStamped=clock)
            if these:
                key, rt = these[0]
                if key == "escape":
                    raise KeyboardInterrupt("Escape pressed")
                return key, rt
            if timeout is not None and clock.getTime() >= timeout:
                return None, clock.getTime()
            if self.win is not None:
                self.win.flip()

    def _show_text_and_wait(self, text: str) -> None:
        assert self.display is not None and self.win is not None
        self.display.show_message(text)
        self.win.flip()
        self._wait_key(("space",))

    def setup(self) -> None:
        from psychopy import core

        self.session = generate_session_schedules(
            self.config, self.participant_id, self.session_id
        )
        seed = self.session["seed"]
        self.rng = random.Random(seed + 999)

        self.logger = SessionLogger(
            self.data_root,
            participant_id=self.participant_id,
            session_id=self.session_id,
            config=self.config,
            random_seed=seed,
            extra_metadata={
                "age": self.age,
                "handedness": self.handedness,
                "simulate": self.simulate,
                "p_cond": self.session["p_cond"],
                "hpl_sequence": self.session["hpl_sequence"],
                "reversal_global_index": self.session["reversal_global_index"],
            },
        )

        self.win = create_window(self.config, simulate=self.simulate)
        self.display = SearchDisplay(self.win, self.config)
        self._exp_clock = core.Clock()
        self.clock_t0 = self._exp_clock

        self.eye_tracker.initialize()
        if not self.simulate:
            self.eye_tracker.calibrate()
        self.markers.initialize()
        self.eye_tracker.start_recording()

    def run(self) -> dict[str, Any]:
        self.setup()
        assert self.session is not None
        summary_errors = 0
        try:
            self._emit("EXP_START", block=None, trial_id=None)
            self._show_text_and_wait(INSTRUCTIONS)

            # Practice
            practice = self.session["practice"]
            save_block_schedule(
                self.logger.session_dir / "schedules" / "block_00_practice.json",
                practice,
                {"phase": "practice", "hpl": None},
            )
            self._emit("BLOCK_START", block=0, trial_id=None, metadata={"practice": True})
            self.eye_tracker.send_message("BLOCK_START:practice")
            for trial in practice:
                self._run_trial(trial, feedback=True)
            self._emit("BLOCK_END", block=0, trial_id=None, metadata={"practice": True})
            self._show_text_and_wait(PRACTICE_END)

            # Experimental blocks
            mode = self.config["experiment"]["mode"]
            first_exp = True
            phase_reversal_emitted = False
            for block_trials in self.session["blocks"]:
                b = block_trials[0].block
                phase = block_trials[0].phase
                hpl = block_trials[0].current_HPL
                save_block_schedule(
                    self.logger.session_dir / "schedules" / f"block_{b:02d}.json",
                    block_trials,
                    {"phase": phase, "hpl": hpl},
                )
                if first_exp:
                    self._emit("PHASE_ACQUISITION", block=b, trial_id=None)
                    first_exp = False
                if (
                    mode == "reversal"
                    and phase == "reversal"
                    and not phase_reversal_emitted
                ):
                    self._emit("PHASE_REVERSAL", block=b, trial_id=None)
                    phase_reversal_emitted = True

                self._emit(
                    "BLOCK_START",
                    block=b,
                    trial_id=None,
                    metadata={"phase": phase, "hpl": hpl},
                )
                for trial in block_trials:
                    self._run_trial(trial, feedback=False)
                self._emit("BLOCK_END", block=b, trial_id=None)

                if b < self.config["experiment"]["blocks"]:
                    self._show_text_and_wait(BREAK_TEXT)

            self._emit("EXP_END")
            self._show_text_and_wait(END_TEXT)
        except KeyboardInterrupt:
            summary_errors += 1
            print("Experiment aborted by user.")
        finally:
            self.cleanup()

        return self._simulation_summary() if self.simulate else {"errors": summary_errors}

    def _simulate_response(self, trial: TrialSpec):
        sim = self.config.get("simulation", {})
        acc = float(sim.get("response_accuracy", 0.9))
        mean_rt = float(sim.get("mean_rt_ms", 550)) / 1000.0
        sd = float(sim.get("rt_sd_ms", 80)) / 1000.0
        rt = max(0.15, self.rng.gauss(mean_rt, sd))
        timeout = float(self.config["timing"]["search_timeout"])
        if rt > timeout:
            return None, True, rt
        correct = self.rng.random() < acc
        key = trial.correct_response if correct else (
            self.config["experiment"]["keys"]["vertical"]
            if trial.target_orientation == "horizontal"
            else self.config["experiment"]["keys"]["horizontal"]
        )
        return key, False, rt

    def _run_trial(self, trial: TrialSpec, *, feedback: bool) -> None:
        assert self.display is not None and self.win is not None and self.logger is not None
        from psychopy import event, core

        timing = self.config["timing"]
        keys_cfg = self.config["experiment"]["keys"]
        allowed = [keys_cfg["horizontal"], keys_cfg["vertical"]]

        self._current_trial_id = trial.trial_number_global
        self._current_block = trial.block
        self._current_phase = trial.phase

        trial_start = self._now()
        self.eye_tracker.send_message(f"TRIAL_START:{trial.trial_number_global}")

        # Fixation
        if self.simulate:
            fix_dur = float(timing["simulate_fixation"])
        else:
            fix_dur = sample_duration(timing["fixation_min"], timing["fixation_max"], self.rng)

        fix_onset = flip_onset(self.win, self.display.draw_fixation)
        self._emit(
            "FIXATION_ON",
            trial_id=trial.trial_number_global,
            block=trial.block,
            metadata={"scheduled_ms": fix_dur * 1000},
        )

        # Pre-stimulus monitoring (observational; no gaze-contingent display change)
        if self.simulate:
            fixation_break = False
            core.wait(fix_dur)
        else:
            # redraw fixation while waiting; monitor gaze
            fix_clock = core.Clock()
            fixation_break = False
            radius = float(self.config["fixation_monitoring"]["radius_deg"])
            while fix_clock.getTime() < fix_dur:
                sample = self.eye_tracker.get_latest_sample()
                if sample is not None:
                    self.logger.log_gaze(
                        {
                            "experiment_timestamp": self._now(),
                            "tracker_timestamp": sample.tracker_timestamp,
                            "trial_id": trial.trial_number_global,
                            "block": trial.block,
                            "phase": trial.phase,
                            "x": sample.x,
                            "y": sample.y,
                            "pupil": sample.pupil,
                            "validity": sample.validity,
                        }
                    )
                    if sample.validity >= 0.5 and math.hypot(sample.x, sample.y) > radius:
                        fixation_break = True
                self.display.draw_fixation()
                self.win.flip()
            self.logger.flush_gaze()

        if fixation_break and self.config["fixation_monitoring"].get("reject_on_break", False):
            # Still run trial; flag only unless configured — reject path reserved
            pass

        # Search array
        self.display.configure_trial(
            target_location=trial.target_location,
            target_orientation=trial.target_orientation,
            distractor_present=trial.distractor_present,
            distractor_location=trial.distractor_location,
        )
        array_onset = flip_onset(self.win, self.display.draw_array)
        arr_name = {
            "S30": "ARRAY_ON_ABSENT",
            "S31": "ARRAY_ON_HPL",
            "S32": "ARRAY_ON_LP",
        }[
            array_onset_code(
                distractor_present=trial.distractor_present,
                distractor_is_hpl=trial.distractor_is_HPL,
            )
        ]
        self._emit(arr_name, trial_id=trial.trial_number_global, block=trial.block)

        timeout = float(timing["search_timeout"])
        timed_out = False
        response_key = None
        rt_s = None
        resp_time = None

        if self.simulate:
            response_key, timed_out, rt_s = self._simulate_response(trial)
            core.wait(min(rt_s, 0.001))
            resp_time = self._now()
        else:
            event.clearEvents()
            resp_clock = core.Clock()
            while True:
                these = event.getKeys(keyList=allowed + ["escape"], timeStamped=resp_clock)
                if these:
                    key, rt_s = these[0]
                    if key == "escape":
                        raise KeyboardInterrupt("Escape pressed")
                    response_key = key
                    resp_time = self._now()
                    break
                if resp_clock.getTime() >= timeout:
                    timed_out = True
                    rt_s = timeout
                    resp_time = self._now()
                    break
                self.display.draw_array()
                self.win.flip()

        correct = (
            (not timed_out)
            and response_key is not None
            and response_key.lower() == trial.correct_response.lower()
        )
        resp_event = {
            "S40": "RESP_CORRECT",
            "S41": "RESP_INCORRECT",
            "S42": "RESP_TIMEOUT",
        }[response_code(timed_out=timed_out, correct=correct)]
        self._emit(
            resp_event,
            trial_id=trial.trial_number_global,
            block=trial.block,
            metadata={"key": response_key, "rt_ms": None if rt_s is None else rt_s * 1000},
        )

        if feedback:
            if timed_out:
                msg = "Too slow"
            elif correct:
                msg = "Correct"
            else:
                msg = "Incorrect"
            self.display.show_message(msg)
            self.win.flip()
            fb = timing["simulate_feedback"] if self.simulate else timing["feedback_duration"]
            core.wait(float(fb))

        # ITI blank
        self.win.flip()
        iti = (
            float(timing["simulate_iti"])
            if self.simulate
            else sample_duration(timing["iti_min"], timing["iti_max"], self.rng)
        )
        core.wait(iti)

        # Frame timing if available
        frame_mean = frame_sd = ""
        try:
            intervals = self.win.frameIntervals[-30:]
            if intervals:
                frame_mean = sum(intervals) / len(intervals)
                if len(intervals) > 1:
                    mu = frame_mean
                    frame_sd = (sum((x - mu) ** 2 for x in intervals) / (len(intervals) - 1)) ** 0.5
        except Exception:
            pass

        self.logger.log_trial(
            {
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "experiment_mode": self.config["experiment"]["mode"],
                "block": trial.block,
                "phase": trial.phase,
                "trial_number_global": trial.trial_number_global,
                "trial_number_block": trial.trial_number_block,
                "trial_relative_to_reversal": trial.trial_relative_to_reversal
                if trial.trial_relative_to_reversal is not None
                else "",
                "is_practice": trial.is_practice,
                "target_location": trial.target_location,
                "target_orientation": trial.target_orientation,
                "distractor_present": trial.distractor_present,
                "distractor_location": trial.distractor_location
                if trial.distractor_location is not None
                else "",
                "distractor_is_HPL": trial.distractor_is_HPL,
                "current_HPL": trial.current_HPL if trial.current_HPL is not None else "",
                "response_key": response_key or "",
                "correct_response": trial.correct_response,
                "accuracy": int(correct),
                "reaction_time_ms": "" if timed_out or rt_s is None else round(rt_s * 1000, 3),
                "fixation_duration_ms": round(fix_dur * 1000, 3),
                "array_actual_onset_time": array_onset,
                "response_time": resp_time if resp_time is not None else "",
                "trial_start_time": trial_start,
                "fixation_break": bool(fixation_break),
                "timed_out": bool(timed_out),
                "frame_intervals_mean": frame_mean,
                "frame_intervals_sd": frame_sd,
                "random_seed": self.session["seed"],
            }
        )
        self.eye_tracker.send_message(f"TRIAL_END:{trial.trial_number_global}")

    def _simulation_summary(self) -> dict[str, Any]:
        assert self.session is not None
        cfg = self.config
        n_pos = cfg["display"]["positions"]
        errors = 0
        print("\n========== SIMULATION SUMMARY ==========")
        # Phase summaries
        by_phase: dict[str, list] = {}
        for block in self.session["blocks"]:
            for t in block:
                by_phase.setdefault(t.phase, []).append(t)

        phase_order = ["acquisition", "reversal", "stable"]
        for i, phase in enumerate(phase_order, start=1):
            trials = by_phase.get(phase)
            if not trials:
                continue
            hpl = trials[0].current_HPL
            stats = compute_stats(
                trials,
                n_positions=n_pos,
                distractor_present_prob=cfg["probabilities"]["distractor_present"],
                hpl_given_distractor=cfg["probabilities"]["hpl_given_distractor"],
                rate_tolerance=cfg["validation"]["rate_tolerance"],
                target_count_max_diff=cfg["validation"]["target_count_max_diff"],
                orientation_tolerance=cfg["validation"]["orientation_tolerance"],
                expect_hpl_bias=True,
            )
            label = f"PHASE {i}" if phase != "stable" else "PHASE (STABLE)"
            print(f"\n{label}")
            print(f"HPL: {hpl}")
            print(f"HPL distractors: {stats.hpl_among_dp_pct:.1%}")
            print("Target-location balance:")
            for k, v in sorted(stats.target_counts.items()):
                print(f"  {k}: {v}")
            errors += len(stats.errors)
            if stats.errors:
                for e in stats.errors:
                    print(f"  ERROR: {e}")

        print(f"\nErrors:\n{errors}")
        print("========================================\n")
        return {"errors": errors}

    def cleanup(self) -> None:
        try:
            self.eye_tracker.stop_recording()
            self.eye_tracker.close()
        except Exception:
            pass
        try:
            self.markers.close()
        except Exception:
            pass
        if self.logger is not None:
            self.logger.close()
        if self.win is not None:
            try:
                self.win.close()
            except Exception:
                pass
