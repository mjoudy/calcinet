"""
Timing signature: spikes vs calcium — COUNT-based PLOT (local).

Same question as fig_r8b_signal_plot.py (does a real edge's coupling peak at
the synaptic delay while a shared-input fake peaks near 0?) but as a plain
count instead of a continuous mean-magnitude-per-lag curve: for each pair, at
which lag is its coupling STRONGEST (peak_lag_ms, already saved per pair by
r8b_multilag.py)? Plot the fraction of true-excitatory vs false-positive
pairs whose peak falls at each lag, as a bar chart — a per-lag "hit rate"
instead of an average magnitude, less sensitive to a few strong pairs skewing
the mean.

Reads the same cached npz as fig_r8b_signal_plot.py — no re-run needed, the
per-pair fields (pred, a_est, g_sign, peak_lag_ms) were already saved.

Usage:
  python experiments/ch2_connectivity/fig_r8b_signal_counts.py \
      --spikes ~/calcium_results/r8b/r8b_n1250_r4_spikes.npz \
      --calcium ~/calcium_results/r8b/r8b_n1250_r4_feed.npz \
      --out figures/fig_R8b_signal_counts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs


def panel(ax, z, title):
    lags = z["lags_ms"]; delay = float(z["delay_ms"])
    pred, a_est, g_sign = z["pred"], z["a_est"], z["g_sign"]
    peak = z["peak_lag_ms"]
    pE = pred & (a_est > 0)
    classes = [("true E", pE & (g_sign > 0), fs.C_E, 1.0),
               ("E false pos.", pE & (g_sign == 0), fs.C_E, 0.45)]

    width = 0.35
    x = np.arange(len(lags))
    for k, (lab, m, col, alpha) in enumerate(classes):
        n = int(m.sum())
        counts = np.array([(peak[m] == l).sum() for l in lags], dtype=float)
        frac = counts / n if n else counts
        ax.bar(x + (k - 0.5) * width, frac, width=width, color=col,
               alpha=(0.35 if alpha < 1.0 else 0.55), label=f"{lab} (n={n:,})")
    delay_idx = np.argmin(np.abs(lags - delay))
    ax.axvline(delay_idx, color=fs.INK, ls=":", lw=1.3)
    ax.set_xticks(x); ax.set_xticklabels([f"{l:g}" for l in lags], fontsize=8.5)
    ax.set(xlabel="lag (ms)", ylabel="fraction of class peaking at this lag")
    ax.grid(True, axis="y", color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)
    ax.set_title(title)
    ax.text(delay_idx + 0.15, ax.get_ylim()[1] * 0.95, f"delay {delay:g}ms",
             fontsize=8.5, color=fs.INK, ha="left", va="top")
    ax.legend(fontsize=8.5, loc="center right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spikes", required=True)
    ap.add_argument("--calcium", required=True)
    ap.add_argument("--out", default="figures/fig_R8b_signal_counts")
    args = ap.parse_args()

    fs.apply_style()
    zs = np.load(args.spikes, allow_pickle=False)
    zc = np.load(args.calcium, allow_pickle=False)
    N = int(zs["N"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.14, wspace=0.22)
    panel(ax[0], zs, "on raw SPIKES: where does each pair's coupling peak?")
    panel(ax[1], zc, "through CALCIUM: where does each pair's coupling peak?")

    fig.suptitle(f"Count-based version: fraction of pairs peaking at each lag, "
                 f"true vs false-positive excitatory (N={N})",
                 fontsize=13, color=fs.INK, x=0.07, ha="left", y=0.97)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
