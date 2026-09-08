"""
R.7 (hidden neurons / shared input) — PLOT stage, ALTERNATE layout.

Alternative to fig_r7_plot.py's fig_R7 (metric-per-panel, colored by N): here
every metric is a line on ONE shared axis, styled by class (colour) and
recall-vs-precision (linestyle), and each network size gets its OWN figure
rather than sharing panels. Same underlying r7_<net>.npz/.csv, same house
palette -- purely a different way to lay the same numbers out, kept as a
separate script/output so both can be compared side by side before picking
one (2026-09-04, user request).

Usage:
  python experiments/ch2_connectivity/fig_r7_plot_bynet.py --root ~/calcium_results/r7 \
      --nets n1250r4 n12500r4 --out-prefix figures/fig_R7
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs

# (csv column, legend label, colour, linestyle, marker). The SAME grammar
# repeats in every colour family, so it only has to be learned once: circle
# + solid = recall-type/primary quantity, square + dashed = precision-type/
# secondary quantity -- this is the fix for recall/precision (and ROC/PR)
# pairs being hard to tell apart when they only differed by linestyle.
SERIES = [
    ("corr", "correlation", fs.INK, "-", "o"),
    ("roc_auc", "ROC-AUC", fs.ACCENT2, "-", "o"),
    ("pr_ap", "PR-AUC", fs.ACCENT2, "--", "s"),
    ("E_rec", "excitatory recall", fs.C_E, "-", "o"),
    ("E_prec", "excitatory precision", fs.C_E, "--", "s"),
    ("I_rec", "inhibitory recall", fs.C_I, "-", "o"),
    ("I_prec", "inhibitory precision", fs.C_I, "--", "s"),
]


def load(root: Path, net: str):
    rows = sorted(csv.DictReader(open(root / f"r7_{net}.csv")),
                  key=lambda r: float(r["frac"]))
    return rows


def fig_one_net(rows, N, out):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.13)

    frac = np.array([float(r["frac"]) for r in rows])
    for key, label, col, ls, marker in SERIES:
        y = np.array([float(r[key]) for r in rows])
        mfc = col if marker == "o" else "white"   # hollow squares read as "secondary"
        ax.plot(frac * 100, y, marker=marker, ls=ls, color=col, lw=2, ms=7,
                mfc=mfc, mew=1.6, label=label)
    ax.set_xscale("log")
    ax.set(xlabel="observed fraction of the network (%)", ylabel="metric value",
           ylim=(0, 1.02))
    ax.invert_xaxis()      # more hidden -> to the right
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)
    ax.legend(fontsize=8.5, loc="lower left", ncol=2)

    fig.suptitle(f"Hiding neurons degrades connectivity recovery (N={N}, OLS)",
                 fontsize=13, color=fs.INK, x=0.1, ha="left", y=0.97)
    fs.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/mjoudy/calcium_results/r7")
    ap.add_argument("--nets", nargs="+", default=["n1250r4", "n12500r4"])
    ap.add_argument("--out-prefix", default="figures/fig_R7")
    args = ap.parse_args()

    fs.apply_style()
    root = Path(args.root)
    for net in args.nets:
        f = root / f"r7_{net}.csv"
        if not f.exists():
            print(f"[warn] {f} not found, skipping"); continue
        rows = load(root, net)
        N = int(float(rows[0]["n_obs"]) / float(rows[0]["frac"]))
        fig_one_net(rows, N, f"{args.out_prefix}_n{N}")


if __name__ == "__main__":
    main()
