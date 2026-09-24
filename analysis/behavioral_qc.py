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


def run_qc(trials_csv: Path, *, bin_width: int = 20, out_dir: Path | None = None) -> None:
    df = pd.read_csv(trials_csv)
    df = df[df["is_practice"].astype(str).str.lower().isin(["false", "0"])].copy()
    out_dir = out_dir or trials_csv.parent

    print("=== Behavioral QC ===")
    print(f"N experimental trials: {len(df)}")
    print(f"Accuracy: {df['accuracy'].mean():.3f}")

    rt = pd.to_numeric(df["reaction_time_ms"], errors="coerce")
    print(f"Median RT (ms): {rt.median():.1f}")

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
        print(f"  block {b}: median_RT={pd.to_numeric(g['reaction_time_ms'], errors='coerce').median():.1f} "
              f"acc={g['accuracy'].mean():.3f}")

    # Post-reversal adaptation plot
    rel = pd.to_numeric(df.get("trial_relative_to_reversal"), errors="coerce")
    if rel.notna().any():
        valid = rel.notna() & rt.notna()
        xs, ys = moving_bin_mean(rel[valid].to_numpy(), rt[valid].to_numpy(), bin_width=bin_width)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(xs, ys, color="#1f4e79", lw=2)
            ax.axvline(0, color="gray", ls="--", lw=1)
            ax.set_xlabel("Trial relative to reversal")
            ax.set_ylabel("RT (ms, moving bin)")
            ax.set_title("Post-reversal RT adaptation")
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
