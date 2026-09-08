"""
R.7 (hidden neurons / shared input) — PLOT stage, local.

Two figures from the r7_<net>.npz files:

  fig_R7             the DEGRADATION curves. Six measures vs. observed
                     fraction, at matched samples per OBSERVED neuron (so any
                     drop is hidden-input confounding, not less data), one
                     line per network size. Top row: correlation, ROC-AUC,
                     PR-AUC (threshold-free ranking metrics, Section 2.2.3).
                     Bottom row: excitatory recall/precision, inhibitory
                     recall (the operating-point metrics, same density-
                     matched threshold used throughout this thesis).

  fig_R7_mechanism   the MECHANISM panel, one per network size: among
                     observed pairs that are NOT truly connected, mean
                     inferred |weight| binned by their number of shared
                     presynaptic neurons (over the FULL network, hidden
                     included). If shared input is what manufactures false
                     edges, this should rise with the overlap. Bins with
                     fewer than --min-count sampled pairs are dropped (too
                     noisy to plot) rather than shown as a misleadingly
                     smooth line through near-empty bins.

Usage:
  python experiments/ch2_connectivity/fig_r7_plot.py --root ~/calcium_results/r7 \
      --nets n1250r4 n12500r4 --out figures/fig_R7
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

# top row = threshold-free ranking metrics; bottom row = operating-point
# (density-matched threshold) metrics -- mirrors the Section 2.2.3 split.
MEASURES = [("corr", "correlation"), ("roc_auc", "ROC-AUC (connected vs. none)"),
            ("pr_ap", "PR-AUC (connected vs. none)"), ("E_rec", "excitatory recall"),
            ("E_prec", "excitatory precision"), ("I_rec", "inhibitory recall")]


def load(root: Path, net: str):
    """Per-fraction metrics come from the .csv (it has every scored field,
    including roc_auc/pr_ap, which the .npz never saved); N and the mechanism
    panel arrays (shared_bins/shared_meanw/shared_count/shared_frac) exist
    only in the .npz, so both files are read and merged."""
    z = np.load(root / f"r7_{net}.npz", allow_pickle=False)
    rows = sorted(csv.DictReader(open(root / f"r7_{net}.csv")),
                  key=lambda r: float(r["frac"]))
    numeric = [c for c in rows[0] if c != "net"]
    out = {c: np.array([float(r[c]) for r in rows]) for c in numeric}
    out["N"] = int(z["N"])
    for k in ("shared_bins", "shared_meanw", "shared_count", "shared_frac"):
        if k in z.files:
            out[k] = z[k]
    return out


def fig_degradation(data, out):
    nets = sorted(data, key=lambda n: data[n]["N"])
    colors = dict(zip(nets, plt.cm.viridis(np.linspace(0.15, 0.85, len(nets)))))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.09,
                        hspace=0.32, wspace=0.24)
    flat = axes.flat
    for i, (key, label) in enumerate(MEASURES):
        ax = flat[i]
        for net in nets:
            d = data[net]
            ax.plot(d["frac"] * 100, d[key], "o-", color=colors[net], lw=2,
                    ms=6, label=f"N = {int(d['N'])}" if i == 0 else None)
        ax.set_xscale("log")
        ax.set(xlabel="observed fraction of the network (%)", ylabel=label,
               ylim=(0, 1.02))
        ax.invert_xaxis()      # more hidden -> to the right
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
        fs.despine(ax)
    flat[0].legend(fontsize=9.5, loc="lower left")
    fig.suptitle("Hiding neurons degrades connectivity recovery, at matched "
                 "samples per OBSERVED neuron (OLS)",
                 fontsize=13, color=fs.INK, x=0.06, ha="left", y=0.975)
    fs.save(fig, out)


def fig_mechanism(data, out, min_count=5):
    nets = sorted(data, key=lambda n: data[n]["N"])
    fig, axes = plt.subplots(1, len(nets), figsize=(6.0 * len(nets), 4.4), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.14, wspace=0.26)
    for c, net in enumerate(nets):
        d = data[net]
        ax = axes[0][c]
        if "shared_bins" not in d:
            ax.text(0.5, 0.5, "no mechanism panel cached", ha="center", va="center",
                    transform=ax.transAxes, color=fs.MUTED)
            continue
        bins, mw, cnt = d["shared_bins"], d["shared_meanw"], d["shared_count"]
        keep = np.isfinite(mw) & (cnt >= min_count)
        ax.plot(bins[keep], mw[keep], "o-", color=fs.ACCENT2, lw=1.8, ms=4)
        ax.set(xlabel="shared presynaptic neurons (full network)",
               ylabel="mean $|\\hat{A}_{ij}|$ among non-connected pairs",
               title=f"N = {int(d['N'])}  (f = {float(d['shared_frac']):.2g} observed)")
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
        fs.despine(ax)
    fig.suptitle("Spurious weight rises with how much presynaptic input a pair "
                "shares -- the mechanism behind the degradation curves above",
                fontsize=12.5, color=fs.INK, x=0.08, ha="left", y=0.97)
    fs.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/mjoudy/calcium_results/r7")
    ap.add_argument("--nets", nargs="+", default=["n1250r4", "n12500r4"])
    ap.add_argument("--out", default="figures/fig_R7")
    ap.add_argument("--out-mechanism", default="figures/fig_R7_mechanism")
    ap.add_argument("--min-count", type=int, default=5,
                    help="drop mechanism-panel bins with fewer sampled pairs than this")
    args = ap.parse_args()

    fs.apply_style()
    root = Path(args.root)
    data = {}
    for net in args.nets:
        f = root / f"r7_{net}.npz"
        if f.exists():
            data[net] = load(root, net)
        else:
            print(f"[warn] {f} not found, skipping")
    if not data:
        raise SystemExit(f"no r7_*.npz under {root}")

    fig_degradation(data, args.out)
    fig_mechanism(data, args.out_mechanism, min_count=args.min_count)


if __name__ == "__main__":
    main()
