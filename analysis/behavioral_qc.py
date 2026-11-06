#!/usr/bin/env python3
"""Offline behavioral QC for a completed session trials.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def moving_bin_mean(x: np.ndarray, y: np.ndarray, bin_width: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    xs, ys = [], []
    for i in range(len(x)):
        lo = max(0, i - bin_width // 2)
        hi = min(len(x), i + bin_width // 2 + 1)
        xs.append(x[i])
        ys.append(np.nanmean(y[lo:hi]))
    return np.asarray(xs), np.asarray(ys)


CATEGORY_COLORS = {
    "acq_HPL": "#c0392b",   # location that was HPL during acquisition
    "rev_HPL": "#1f4e79",   # location that is HPL during reversal
    "other_LP": "#7f8c8d",
    "absent": "#bdc3c7",
}


def location_category(df: pd.DataFrame) -> pd.Series:
    """Label each trial's distractor location relative to BOTH phases' HPLs,
    so suppression at the old HPL can be tracked after reversal."""
    hpl = pd.to_numeric(df["current_HPL"], errors="coerce")
    acq = hpl[df["phase"].isin(["acquisition", "stable"])].mode()
    rev = hpl[df["phase"] == "reversal"].mode()
    acq = acq.iloc[0] if len(acq) else None
    rev = rev.iloc[0] if len(rev) else None
    loc = pd.to_numeric(df["distractor_location"], errors="coerce")
    present = df["distractor_present"].astype(str).str.lower().isin(["true", "1"])
    cat = pd.Series("other_LP", index=df.index)
    cat[present & (loc == acq)] = "acq_HPL"
    if rev is not None:
        cat[present & (loc == rev)] = "rev_HPL"
    cat[~present] = "absent"
    return cat


def run_qc(trials_csv: Path, *, bin_width: int = 20, out_dir: Path | None = None) -> None:
    df = pd.read_csv(trials_csv)
    df = df[df["is_practice"].astype(str).str.lower().isin(["false", "0"])].copy()
    out_dir = out_dir or trials_csv.parent

    print("=== Behavioral QC ===")
    print(f"N experimental trials: {len(df)}")
    print(f"Accuracy: {df['accuracy'].mean():.3f}")

    # RT analyses on correct trials only (errors / timeouts → NaN)
    rt = pd.to_numeric(df["reaction_time_ms"], errors="coerce").where(df["accuracy"] == 1)
    print(f"Median correct RT (ms): {rt.median():.1f}")

    def cond_rt(mask, label):
        sub = rt[mask]
        print(f"  {label}: n={mask.sum()} median_RT={sub.median():.1f}")

    print("RT by distractor condition:")
    cond_rt(df["distractor_present"].astype(str).str.lower().isin(["true", "1"]), "distractor present")
    cond_rt(~df["distractor_present"].astype(str).str.lower().isin(["true", "1"]), "distractor absent")
    present = df["distractor_present"].astype(str).str.lower().isin(["true", "1"])
    cond_rt(present & df["distractor_is_HPL"].astype(str).str.lower().isin(["true", "1"]), "HPL distractor")
    cond_rt(present & ~df["distractor_is_HPL"].astype(str).str.lower().isin(["true", "1"]), "LP distractor")

    print("RT across blocks:")
    for b, g in df.groupby("block"):
        print(f"  block {b}: median_RT={rt[g.index].median():.1f} "
              f"acc={g['accuracy'].mean():.3f}")

    loc_cat = location_category(df)
    print("Median correct RT by phase x distractor location:")
    print(rt.groupby([df["phase"], loc_cat]).median().unstack().round(1).to_string())

    # Post-reversal adaptation plot, one line per distractor location
    rel = pd.to_numeric(df.get("trial_relative_to_reversal"), errors="coerce")
    if rel.notna().any():
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            for cat, color in CATEGORY_COLORS.items():
                valid = rel.notna() & rt.notna() & (loc_cat == cat)
                if valid.sum() < 2:
                    continue
                xs, ys = moving_bin_mean(rel[valid].to_numpy(), rt[valid].to_numpy(), bin_width=bin_width)
                ax.plot(xs, ys, color=color, lw=2, label=cat)
            ax.axvline(0, color="gray", ls="--", lw=1)
            ax.set_xlabel("Trial relative to reversal")
            ax.set_ylabel("Correct RT (ms, moving bin)")
            ax.set_title("Post-reversal RT adaptation by distractor location")
            ax.legend(frameon=False)
            fig.tight_layout()
            out = out_dir / "qc_rt_relative_to_reversal.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"Wrote plot: {out}")
        except Exception as exc:
            print(f"Plot skipped: {exc}")
    else:
        print("No trial_relative_to_reversal values (stable mode or missing); plot skipped.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trials_csv", type=Path)
    parser.add_argument("--bin-width", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    run_qc(args.trials_csv, bin_width=args.bin_width, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
