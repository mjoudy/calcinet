"""
R.2 follow-up — PLOT. Does tuning the deconvolution's own smoothing window
let it reach the bin-size sweet spot?

Overlays the smoothing-window sweep (results/fig_data/r2_smooth_data.npz)
on the SAME ms x-axis as the existing "binned spikes (reference)" curve
(results/fig_data/r2_data.npz) so the two can be compared directly: does the
deconvolved feed's best window land on/near the reference curve's peak?

Usage:
  python experiments/ch1_proxy/fig_r2_smooth_plot.py \
      --smooth-data results/fig_data/r2_smooth_data.npz \
      --ref-data results/fig_data/r2_data.npz \
      --out figures/fig_R2_smooth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs


def panel(ax, ref, sm, measure, floor_ms):
    rx, ry = ref["spikes_x"], ref[f"spikes_{measure}"]
    ax.plot(rx, ry, "o-", color=fs.INK, lw=1.8, ms=6, label="binned spikes (reference)")
    w, y = sm["window_ms"], sm[f"{measure}"]
    ax.plot(w, y, "o-", color=fs.C_E, lw=1.8, ms=6, label="deconvolved (window tuned)")
    ax.axhline(float(sm[f"raw_{measure}"]), color=fs.C_I, lw=1.3, ls="--",
              label="raw calcium (fixed window)")
    ax.axvspan(w.min(), floor_ms, color=fs.MUTED, alpha=0.12, lw=0)
    ax.text(floor_ms, 0.03, " floor: window\n clamped to 5\n samples here",
           color=fs.MUTED, fontsize=7, va="bottom", ha="left")

    # mark both curves' own maxima, and a shared dotted line at the
    # reference ceiling, so "does deconvolution reach the sweet spot?" is a
    # direct visual read: how close does the blue star sit to the black one.
    iref = int(np.argmax(ry))
    ax.axhline(ry[iref], color=fs.MUTED, lw=1.0, ls=":")
    ax.plot(rx[iref], ry[iref], marker="*", ms=16, color=fs.INK,
            markeredgecolor=fs.INK, markeredgewidth=0.6, zorder=5,
            label="reference maximum")
    ibest = int(np.argmax(y))
    ax.plot(w[ibest], y[ibest], marker="*", ms=16, color=fs.C_E,
            markeredgecolor=fs.INK, markeredgewidth=0.6, zorder=5,
            label="deconvolved maximum")
    ax.set_xscale("log")
    ax.set(xlabel="temporal scale (ms)  —  bin size  /  smoothing window",
           ylabel="ROC-AUC" if measure == "auc" else "correlation", ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth-data", default="results/fig_data/r2_smooth_data.npz")
    ap.add_argument("--ref-data", default="results/fig_data/r2_data.npz")
    ap.add_argument("--out", default="figures/fig_R2_smooth")
    args = ap.parse_args()

    fs.apply_style()
    sm = np.load(args.smooth_data, allow_pickle=False)
    ref = np.load(args.ref_data, allow_pickle=False)

    cam_ms = float(sm["cam_ms"]); tau_ms = float(sm["tau_ms"])
    floor_ms = 5 * cam_ms   # smooth_win()'s 5-sample minimum, in ms, at this camera rate

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.15, wspace=0.28)
    panel(axes[0], ref, sm, "auc", floor_ms)
    panel(axes[1], ref, sm, "corr", floor_ms)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        f"Tuning the deconvolution window vs. the bin-size sweet spot  —  "
        f"n1250ai, N={int(sm['N'])}, camera={cam_ms:.1f} ms, dye τ={tau_ms:.0f} ms",
        fontsize=12.5, color=fs.INK, x=0.06, ha="left", y=0.97)
    fs.save(fig, args.out)
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
