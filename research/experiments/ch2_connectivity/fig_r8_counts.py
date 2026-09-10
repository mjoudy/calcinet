"""
R.8 (shared input vs directionality) — COUNT-based PLOT stage, local.

Same underlying question as fig_r8_violin.py (are false positives more
symmetric than true edges?) but as a plain count instead of a continuous
score: for each predicted edge, is the REVERSE direction ALSO predicted
connected (bidirectional — looks like shared input) or not (one-directional —
looks like a real edge)? Needs bidir_n_*/bidir_frac_* fields from
fig_r8_compute.py (added alongside this script — older cached npz won't have
them).

Usage:
  python research/experiments/ch2_connectivity/fig_r8_counts.py --data <r8_*.npz> --out figures/fig_R8_counts_<tag>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs

GROUPS = [("E_true", "true E", fs.C_E, 1.0),
          ("E_false", "E false pos.", fs.C_E, 0.45),
          ("I_true", "true I", fs.C_I, 1.0),
          ("I_false", "I false pos.", fs.C_I, 0.45)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/fig_R8_counts")
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=True)
    N = int(z["N"])
    missing = [k for k, *_ in GROUPS if f"bidir_frac_{k}" not in z.files]
    if missing:
        raise SystemExit(f"missing bidir_frac_* for {missing} — re-run the "
                          "updated fig_r8_compute.py to produce this npz")

    labels = [lab for _, lab, _, _ in GROUPS]
    fracs = [float(z[f"bidir_frac_{k}"]) for k, *_ in GROUPS]
    ns = [int(z[f"n_{k}"]) for k, *_ in GROUPS]
    cols = [col for _, _, col, alpha in GROUPS]
    alphas = [alpha for *_, alpha in GROUPS]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.6))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.84, bottom=0.12)
    x = np.arange(len(labels))
    bars = ax.bar(x, fracs, color=cols, width=0.6)
    for bar, a_ in zip(bars, alphas):
        bar.set_alpha(0.35 if a_ < 1.0 else 0.55)
    for xi, f, n in zip(x, fracs, ns):
        ax.text(xi, f + 0.02, f"{f*100:.0f}%\n(n={n:,})", ha="center",
                va="bottom", fontsize=8.5, color=fs.INK)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set(ylabel="% of predicted edges where the REVERSE direction\n"
                   "is ALSO predicted connected (bidirectional)",
           ylim=(0, 1.12))
    ax.grid(True, axis="y", color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
    fs.despine(ax)
    ax.set_title(f"Bidirectional vs one-directional predictions at N={N} "
                 "(single-lag OLS)\ncount-based version of the symmetry check",
                 fontsize=12, color=fs.INK)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
