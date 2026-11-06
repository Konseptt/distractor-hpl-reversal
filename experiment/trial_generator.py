"""Pre-block trial schedule generation and validation.

Target location is sampled first (uniform / balanced). Distractors are then
placed among remaining positions with a *conditional* HPL probability so that
overall P(distractor at HPL | distractor present) matches the configured rate
despite target–distractor mutual exclusion.

p_cond = hpl_given_distractor / (1 - 1/n_positions)
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


EXPERIMENT_VERSION = "1.0.0"


@dataclass
class TrialSpec:
    block: int
    phase: str  # acquisition | reversal | practice | stable
    trial_number_block: int
    trial_number_global: int
    trial_relative_to_reversal: Optional[int]
    is_practice: bool
    target_location: int
    target_orientation: str  # horizontal | vertical
    distractor_present: bool
    distractor_location: Optional[int]
    distractor_is_HPL: bool
    current_HPL: Optional[int]
    correct_response: str  # key letter


@dataclass
class ScheduleStats:
    n_trials: int
    distractor_present_pct: float
    distractor_absent_pct: float
    hpl_among_dp_pct: float
    target_counts: dict[int, int]
    distractor_counts: dict[int, int]
    orientation_counts: dict[str, int]
    p_cond: float
    errors: list[str] = field(default_factory=list)


def hash_seed(participant_id: str, session_id: str, version: str = EXPERIMENT_VERSION) -> int:
    """Deterministic 32-bit seed from participant + session + version."""
    payload = f"{participant_id}|{session_id}|{version}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def conditional_hpl_probability(hpl_given_distractor: float, n_positions: int) -> float:
    """Boost HPL rate when target ≠ HPL so overall ≈ configured rate."""
    if n_positions < 2:
        raise ValueError("n_positions must be >= 2")
    denom = 1.0 - (1.0 / n_positions)
    if denom <= 0:
        raise ValueError("invalid n_positions for conditional HPL")
    p = hpl_given_distractor / denom
    return min(1.0, max(0.0, p))


def resolve_hpl_sequence(
    *,
    mode: str,
    participant_id: str,
    n_blocks: int,
    acquisition_blocks: int,
    left_hpl: int,
    right_hpl: int,
) -> list[tuple[str, int]]:
    """Return per-block (phase_name, hpl_index) for experimental blocks 1..n."""
    pid_digits = "".join(ch for ch in str(participant_id) if ch.isdigit())
    if not pid_digits:
        raise ValueError(
            f"participant_id must contain a digit for odd/even counterbalancing; got {participant_id!r}"
        )
    odd = int(pid_digits) % 2 == 1

    if mode == "reversal":
        first, second = (left_hpl, right_hpl) if odd else (right_hpl, left_hpl)
        out: list[tuple[str, int]] = []
        for b in range(1, n_blocks + 1):
            if b <= acquisition_blocks:
                out.append(("acquisition", first))
            else:
                out.append(("reversal", second))
        return out

    if mode == "stable":
        hpl = left_hpl if odd else right_hpl
        return [("stable", hpl) for _ in range(n_blocks)]

    raise ValueError(f"Unknown experiment mode: {mode!r}")


def _balanced_targets(n: int, n_positions: int, rng: Any) -> list[int]:
    base = [i % n_positions for i in range(n)]
    rng.shuffle(base)
    return base


def _balanced_orientations(n: int, rng: Any) -> list[str]:
    n_h = n // 2
    n_v = n - n_h
    orients = ["horizontal"] * n_h + ["vertical"] * n_v
    rng.shuffle(orients)
    return orients


def _max_run(seq: list[Any]) -> int:
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _max_run_true(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _repair_runs(seq: list[Any], max_run: int, rng: Any, attempts: int = 200) -> list[Any]:
    """Greedy local swaps to reduce consecutive runs."""
    seq = list(seq)
    for _ in range(attempts):
        if _max_run(seq) <= max_run:
            return seq
        # find a long run and swap an element out of it
        run_start = 0
        while run_start < len(seq):
            run_end = run_start + 1
            while run_end < len(seq) and seq[run_end] == seq[run_start]:
                run_end += 1
            if run_end - run_start > max_run:
                # swap middle of run with a different value later
                mid = run_start + max_run
                for j in range(run_end, len(seq)):
                    if seq[j] != seq[mid]:
                        seq[mid], seq[j] = seq[j], seq[mid]
                        break
                else:
                    for j in range(0, run_start):
                        if seq[j] != seq[mid]:
                            seq[mid], seq[j] = seq[j], seq[mid]
                            break
            run_start = run_end
        if _max_run(seq) <= max_run:
            return seq
        rng.shuffle(seq)  # last resort reshuffle then continue
    return seq


def _distribute_lp_locations(
    n_lp: int,
    hpl: int,
    n_positions: int,
    rng: Any,
) -> list[int]:
    """Evenly distribute LP distractor placements across non-HPL positions."""
    others = [i for i in range(n_positions) if i != hpl]
    if not others:
        raise ValueError("no LP positions available")
    base = n_lp // len(others)
    rem = n_lp % len(others)
    bag: list[int] = []
    for i, loc in enumerate(others):
        bag.extend([loc] * (base + (1 if i < rem else 0)))
    rng.shuffle(bag)
    return bag


def generate_block_schedule(
    *,
    block: int,
    phase: str,
    hpl: Optional[int],
    n_trials: int,
    n_positions: int,
    distractor_present_prob: float,
    hpl_given_distractor: float,
    keys: dict[str, str],
    max_consecutive_hpl: int,
    max_consecutive_orientation: int,
    max_attempts: int,
    rng: Any,
    is_practice: bool = False,
    trial_number_global_start: int = 1,
    reversal_global_index: Optional[int] = None,
    flat_distractor: bool = False,
) -> list[TrialSpec]:
    """Generate one block of trials with constraints and quotas."""
    p_cond = conditional_hpl_probability(hpl_given_distractor, n_positions)

    last_error = ""
    for attempt in range(max_attempts):
        try:
            trials = _try_generate_block(
                block=block,
                phase=phase,
                hpl=hpl,
                n_trials=n_trials,
                n_positions=n_positions,
                distractor_present_prob=distractor_present_prob,
                hpl_given_distractor=hpl_given_distractor,
                p_cond=p_cond,
                keys=keys,
                max_consecutive_hpl=max_consecutive_hpl,
                max_consecutive_orientation=max_consecutive_orientation,
                rng=rng,
                is_practice=is_practice,
                trial_number_global_start=trial_number_global_start,
                reversal_global_index=reversal_global_index,
                flat_distractor=flat_distractor or is_practice,
            )
            return trials
        except ScheduleGenerationError as exc:
            last_error = str(exc)
            continue
    raise ScheduleGenerationError(
        f"Failed to generate valid block {block} after {max_attempts} attempts. Last error: {last_error}"
    )


class ScheduleGenerationError(Exception):
    pass


def _try_generate_block(
    *,
    block: int,
    phase: str,
    hpl: Optional[int],
    n_trials: int,
    n_positions: int,
    distractor_present_prob: float,
    hpl_given_distractor: float,
    p_cond: float,
    keys: dict[str, str],
    max_consecutive_hpl: int,
    max_consecutive_orientation: int,
    rng: Any,
    is_practice: bool,
    trial_number_global_start: int,
    reversal_global_index: Optional[int],
    flat_distractor: bool,
) -> list[TrialSpec]:
    n_dp = int(round(n_trials * distractor_present_prob))
    n_da = n_trials - n_dp

    present_flags = [True] * n_dp + [False] * n_da
    rng.shuffle(present_flags)

    targets = _repair_runs(
        _balanced_targets(n_trials, n_positions, rng),
        max_run=max_consecutive_orientation,  # soft; locations less critical
        rng=rng,
        attempts=50,
    )
    orientations = _repair_runs(
        _balanced_orientations(n_trials, rng),
        max_run=max_consecutive_orientation,
        rng=rng,
    )

    distractor_locs: list[Optional[int]] = [None] * n_trials
    distractor_is_hpl_flags: list[bool] = [False] * n_trials
    dp_indices = [i for i, p in enumerate(present_flags) if p]

    if flat_distractor or hpl is None:
        for i in dp_indices:
            choices = [p for p in range(n_positions) if p != targets[i]]
            distractor_locs[i] = rng.choice(choices)
            distractor_is_hpl_flags[i] = False
    else:
        n_hpl_quota = int(round(n_dp * hpl_given_distractor))
        eligible = [i for i in dp_indices if targets[i] != hpl]
        ineligible = [i for i in dp_indices if targets[i] == hpl]
        if len(eligible) < n_hpl_quota:
            raise ScheduleGenerationError(
                f"only {len(eligible)} eligible HPL slots but quota={n_hpl_quota}"
            )

        rng.shuffle(eligible)
        hpl_trials = set(eligible[:n_hpl_quota])
        lp_trials = eligible[n_hpl_quota:] + ineligible
        n_lp = len(lp_trials)
        lp_bag = _distribute_lp_locations(n_lp, hpl, n_positions, rng)

        for i in hpl_trials:
            distractor_locs[i] = hpl
            distractor_is_hpl_flags[i] = True

        lp_idx = 0
        for i in lp_trials:
            tloc = targets[i]
            placed = False
            for k in range(lp_idx, len(lp_bag)):
                cand = lp_bag[k]
                if cand != tloc:
                    lp_bag[lp_idx], lp_bag[k] = lp_bag[k], lp_bag[lp_idx]
                    distractor_locs[i] = lp_bag[lp_idx]
                    lp_idx += 1
                    placed = True
                    break
            if not placed:
                choices = [p for p in range(n_positions) if p != tloc and p != hpl]
                if not choices:
                    choices = [p for p in range(n_positions) if p != tloc]
                distractor_locs[i] = rng.choice(choices)
            distractor_is_hpl_flags[i] = distractor_locs[i] == hpl

        # Repair HPL runs: move HPL status from inside an over-long run to an
        # LP trial where it creates no new over-long run; swap the two
        # distractor locations so LP location balance is preserved.
        flags = distractor_is_hpl_flags

        def run_len(j: int) -> int:
            lo = hi = j
            while lo > 0 and flags[lo - 1]:
                lo -= 1
            while hi < n_trials - 1 and flags[hi + 1]:
                hi += 1
            return hi - lo + 1

        for _ in range(n_trials):
            mid = next(
                (i for i in range(max_consecutive_hpl, n_trials)
                 if all(flags[i - max_consecutive_hpl:i + 1])),
                None,
            )
            if mid is None:
                break
            flags[mid] = False
            cands = []
            for j in range(n_trials):
                if present_flags[j] and not flags[j] and j != mid and targets[j] != hpl:
                    flags[j] = True
                    if run_len(j) <= max_consecutive_hpl:
                        cands.append(j)
                    flags[j] = False
            if not cands:
                raise ScheduleGenerationError("cannot repair HPL run")
            j = rng.choice(cands)
            lp_loc = distractor_locs[j]
            if lp_loc == targets[mid]:
                lp_loc = rng.choice(
                    [p for p in range(n_positions) if p != targets[mid] and p != hpl]
                )
            distractor_locs[mid] = lp_loc
            distractor_locs[j] = hpl
            flags[j] = True
        if _max_run_true(flags) > max_consecutive_hpl:
            raise ScheduleGenerationError("HPL run constraint unsatisfied")

        # Verify exact quota preserved
        n_hpl_final = sum(1 for i in dp_indices if distractor_is_hpl_flags[i])
        if n_hpl_final != n_hpl_quota:
            raise ScheduleGenerationError(
                f"HPL quota drift: got {n_hpl_final} expected {n_hpl_quota}"
            )
    # Build TrialSpec list
    trials: list[TrialSpec] = []
    for i in range(n_trials):
        tloc = targets[i]
        dloc = distractor_locs[i]
        if present_flags[i] and dloc is not None and dloc == tloc:
            raise ScheduleGenerationError("target/distractor overlap")
        orient = orientations[i]
        gnum = trial_number_global_start + i
        rel = None
        if (not is_practice) and reversal_global_index is not None:
            if gnum < reversal_global_index:
                rel = gnum - reversal_global_index
            elif gnum >= reversal_global_index:
                rel = gnum - reversal_global_index + 1  # skip 0
        trials.append(
            TrialSpec(
                block=block,
                phase=phase if not is_practice else "practice",
                trial_number_block=i + 1,
                trial_number_global=gnum if not is_practice else -(i + 1),
                trial_relative_to_reversal=rel if not is_practice else None,
                is_practice=is_practice,
                target_location=tloc,
                target_orientation=orient,
                distractor_present=present_flags[i],
                distractor_location=dloc if present_flags[i] else None,
                distractor_is_HPL=bool(distractor_is_hpl_flags[i]) if present_flags[i] else False,
                current_HPL=None if is_practice else hpl,
                correct_response=keys[orient],
            )
        )

    # Soft check orientation runs
    if _max_run([t.target_orientation for t in trials]) > max_consecutive_orientation + 1:
        raise ScheduleGenerationError("orientation run too long")

    return trials


def compute_stats(
    trials: list[TrialSpec],
    *,
    n_positions: int,
    distractor_present_prob: float,
    hpl_given_distractor: float,
    rate_tolerance: float,
    target_count_max_diff: int,
    orientation_tolerance: float,
    expect_hpl_bias: bool = True,
) -> ScheduleStats:
    n = len(trials)
    dp = [t for t in trials if t.distractor_present]
    da = [t for t in trials if not t.distractor_present]
    hpl_trials = [t for t in dp if t.distractor_is_HPL]

    p_cond = conditional_hpl_probability(hpl_given_distractor, n_positions)
    target_counts = Counter(t.target_location for t in trials)
    distractor_counts = Counter(
        t.distractor_location for t in dp if t.distractor_location is not None
    )
    orientation_counts = Counter(t.target_orientation for t in trials)

    errors: list[str] = []

    # Overlap
    for t in trials:
        if t.distractor_present and t.distractor_location == t.target_location:
            errors.append(
                f"overlap trial block={t.block} i={t.trial_number_block}"
            )

    dp_pct = len(dp) / n if n else 0.0
    da_pct = len(da) / n if n else 0.0
    hpl_pct = len(hpl_trials) / len(dp) if dp else 0.0

    if abs(dp_pct - distractor_present_prob) > rate_tolerance:
        errors.append(
            f"distractor_present_pct={dp_pct:.3f} expected≈{distractor_present_prob:.3f}"
        )

    if expect_hpl_bias and dp:
        if abs(hpl_pct - hpl_given_distractor) > rate_tolerance:
            errors.append(
                f"hpl_among_dp_pct={hpl_pct:.3f} expected≈{hpl_given_distractor:.3f}"
            )

    # Target balance
    if target_counts:
        counts = [target_counts.get(i, 0) for i in range(n_positions)]
        if max(counts) - min(counts) > target_count_max_diff:
            errors.append(f"target imbalance counts={counts}")

    # Orientation balance
    n_h = orientation_counts.get("horizontal", 0)
    n_v = orientation_counts.get("vertical", 0)
    if n:
        if abs(n_h / n - 0.5) > orientation_tolerance:
            errors.append(f"orientation imbalance H={n_h} V={n_v}")

    return ScheduleStats(
        n_trials=n,
        distractor_present_pct=dp_pct,
        distractor_absent_pct=da_pct,
        hpl_among_dp_pct=hpl_pct,
        target_counts={i: target_counts.get(i, 0) for i in range(n_positions)},
        distractor_counts={i: distractor_counts.get(i, 0) for i in range(n_positions)},
        orientation_counts=dict(orientation_counts),
        p_cond=p_cond,
        errors=errors,
    )


def validate_schedule(
    trials: list[TrialSpec],
    *,
    n_positions: int,
    distractor_present_prob: float,
    hpl_given_distractor: float,
    rate_tolerance: float,
    target_count_max_diff: int,
    orientation_tolerance: float,
    expect_hpl_bias: bool = True,
    raise_on_error: bool = True,
) -> ScheduleStats:
    stats = compute_stats(
        trials,
        n_positions=n_positions,
        distractor_present_prob=distractor_present_prob,
        hpl_given_distractor=hpl_given_distractor,
        rate_tolerance=rate_tolerance,
        target_count_max_diff=target_count_max_diff,
        orientation_tolerance=orientation_tolerance,
        expect_hpl_bias=expect_hpl_bias,
    )
    if stats.errors and raise_on_error:
        raise ScheduleGenerationError(
            "Schedule validation failed:\n  - " + "\n  - ".join(stats.errors)
        )
    return stats


def print_stats(stats: ScheduleStats, title: str = "Schedule stats") -> None:
    print(f"\n=== {title} ===")
    print(f"N={stats.n_trials}  DP={stats.distractor_present_pct:.1%}  "
          f"DA={stats.distractor_absent_pct:.1%}  "
          f"HPL|DP={stats.hpl_among_dp_pct:.1%}  p_cond={stats.p_cond:.3f}")
    print("Target counts by position:")
    for k, v in sorted(stats.target_counts.items()):
        print(f"  {k}: {v}")
    print("Distractor counts by position:")
    for k, v in sorted(stats.distractor_counts.items()):
        print(f"  {k}: {v}")
    print(f"Orientations: {stats.orientation_counts}")
    if stats.errors:
        print("Errors:")
        for e in stats.errors:
            print(f"  - {e}")
    else:
        print("Errors: 0")


def generate_session_schedules(config: dict[str, Any], participant_id: str, session_id: str) -> dict[str, Any]:
    """Generate practice + all experimental blocks; return dict with trials and meta."""
    import random

    exp = config["experiment"]
    probs = config["probabilities"]
    spatial = config["spatial"]
    seq = config["sequence"]
    val = config["validation"]
    n_positions = int(config["display"]["positions"])

    seed = config.get("random_seed")
    if seed is None:
        seed = hash_seed(participant_id, session_id, exp.get("version", EXPERIMENT_VERSION))
    rng = random.Random(seed)

    mode = exp["mode"]
    n_blocks = int(exp["blocks"])
    acq = int(exp["acquisition_blocks"])
    trials_per_block = int(exp["trials_per_block"])
    practice_n = int(exp["practice_trials"])
    keys = exp["keys"]

    hpl_seq = resolve_hpl_sequence(
        mode=mode,
        participant_id=participant_id,
        n_blocks=n_blocks,
        acquisition_blocks=acq,
        left_hpl=int(spatial["left_hpl"]),
        right_hpl=int(spatial["right_hpl"]),
    )

    # First reversal trial's global number. Set in stable mode too (virtual
    # reversal) so the control group gets a matched trial_relative_to_reversal.
    reversal_global_index: Optional[int] = acq * trials_per_block + 1

    p_cond = conditional_hpl_probability(probs["hpl_given_distractor"], n_positions)

    # Practice
    practice = generate_block_schedule(
        block=0,
        phase="practice",
        hpl=None,
        n_trials=practice_n,
        n_positions=n_positions,
        distractor_present_prob=probs["distractor_present"],
        hpl_given_distractor=probs["hpl_given_distractor"],
        keys=keys,
        max_consecutive_hpl=seq["max_consecutive_hpl"],
        max_consecutive_orientation=seq["max_consecutive_orientation"],
        max_attempts=seq["max_generation_attempts"],
        rng=rng,
        is_practice=True,
        flat_distractor=True,
    )
    validate_schedule(
        practice,
        n_positions=n_positions,
        distractor_present_prob=probs["distractor_present"],
        hpl_given_distractor=probs["hpl_given_distractor"],
        rate_tolerance=val["rate_tolerance"],
        target_count_max_diff=val["target_count_max_diff"],
        orientation_tolerance=val["orientation_tolerance"],
        expect_hpl_bias=False,
    )

    experimental: list[list[TrialSpec]] = []
    global_start = 1
    for b_idx, (phase, hpl) in enumerate(hpl_seq, start=1):
        block_trials = generate_block_schedule(
            block=b_idx,
            phase=phase,
            hpl=hpl,
            n_trials=trials_per_block,
            n_positions=n_positions,
            distractor_present_prob=probs["distractor_present"],
            hpl_given_distractor=probs["hpl_given_distractor"],
            keys=keys,
            max_consecutive_hpl=seq["max_consecutive_hpl"],
            max_consecutive_orientation=seq["max_consecutive_orientation"],
            max_attempts=seq["max_generation_attempts"],
            rng=rng,
            is_practice=False,
            trial_number_global_start=global_start,
            reversal_global_index=reversal_global_index,
            flat_distractor=False,
        )
        stats = validate_schedule(
            block_trials,
            n_positions=n_positions,
            distractor_present_prob=probs["distractor_present"],
            hpl_given_distractor=probs["hpl_given_distractor"],
            rate_tolerance=val["rate_tolerance"],
            target_count_max_diff=val["target_count_max_diff"],
            orientation_tolerance=val["orientation_tolerance"],
            expect_hpl_bias=True,
        )
        print_stats(stats, title=f"Block {b_idx} ({phase}, HPL={hpl})")
        experimental.append(block_trials)
        global_start += trials_per_block

    return {
        "seed": seed,
        "p_cond": p_cond,
        "mode": mode,
        "hpl_sequence": [{"block": i + 1, "phase": p, "hpl": h} for i, (p, h) in enumerate(hpl_seq)],
        "reversal_global_index": reversal_global_index,
        "practice": practice,
        "blocks": experimental,
    }


def trials_to_dicts(trials: list[TrialSpec]) -> list[dict[str, Any]]:
    return [asdict(t) for t in trials]


def save_block_schedule(path: Path, trials: list[TrialSpec], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "trials": trials_to_dicts(trials)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_config(path: Path | str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
