"""
Timing signature: spikes vs calcium — VIOLIN, count-based (local).

Same count-based data as fig_r8b_signal_counts.py (each pair's peak_lag_ms —
where its coupling is strongest, not how strong), shown as a violin instead of
a bar chart: full spread + median of the peak-lag distribution per class,
easier to compare true E vs false-positive E at a glance than the bar chart.

Violins are drawn on the LAG INDEX (0..9), not raw ms, because the lag grid
is unevenly spaced (0.1 ... 6 ms) and a kernel-density violin on the raw ms
values would blur across that uneven spacing; y-ticks are relabelled to the
actual ms values.

Reads the same cached npz as fig_r8b_signal_plot.py / fig_r8b_signal_counts.py
— no re-run needed.

Usage:
  python experiments/ch2_connectivity/fig_r8b_signal_violin.py \
      --spikes ~/calcium_results/r8b/r8b_n1250_r4_spikes.npz \
      --calcium ~/calcium_results/r8b/r8b_n1250_r4_feed.npz \
      --out figures/fig_R8b_signal_violin
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
    pred, a_est, g_sign, peak = z["pred"], z["a_est"], z["g_sign"], z["peak_lag_ms"]
    pE = pred & (a_est > 0)
    classes = [("true E", pE & (g_sign > 0), fs.C_E, 1.0),
               ("E false pos.", pE & (g_sign == 0), fs.C_E, 0.45)]

    # match on rounded ms values — peak_lag_ms is float32, lags may be float64,
    # so exact dict lookup can miss by floating-point noise.
    lag_to_idx = {round(float(l), 3): k for k, l in enumerate(lags)}
    positions = [1, 2]
    vals = []
    for lab, m, col, alpha in classes:
        idx = np.array([lag_to_idx[round(float(l), 3)] for l in peak[m]], dtype=float)
        vals.append(idx)

    parts = ax.violinplot(vals, positions=positions, showmedians=True,
                           showextrema=False, widths=0.8)
    for k, (body, (lab, m, col, alpha)) in enumerate(zip(parts["bodies"], classes)):
        body.set_facecolor(col); body.set_alpha(0.35 if alpha < 1.0 else 0.55)
        body.set_edgecolor(col)
        n = int(m.sum())
        ax.text(positions[k], len(lags) - 0.55, f"n={n:,}", ha="center",
                va="bottom", fontsize=8.5, color=fs.MUTED)
    parts["cmedians"].set_color(fs.INK); parts["cmedians"].set_linewidth(1.4)
    for k, v in enumerate(vals):
        med_idx = int(round(np.median(v)))
        ax.text(positions[k] + 0.12, np.median(v), f"{lags[med_idx]:g}ms",
                ha="left", va="center", fontsize=8.5, color=fs.INK)

    delay_idx = int(np.argmin(np.abs(lags - delay)))
    ax.axhline(delay_idx, color=fs.INK, ls=":", lw=1.3)
    ax.text(2.5, delay_idx, f" delay {delay:g}ms", fontsize=8.5, color=fs.INK,
             va="center")

    ax.set_xticks(positions); ax.set_xticklabels([lab for lab, *_ in classes], fontsize=9)
    ax.set_yticks(range(len(lags))); ax.set_yticklabels([f"{l:g}" for l in lags])
    ax.set(ylabel="lag (ms) at which coupling peaks", ylim=(-0.5, len(lags) - 0.1))
    ax.grid(True, axis="y", color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spikes", required=True)
    ap.add_argument("--calcium", required=True)
    ap.add_argument("--out", default="figures/fig_R8b_signal_violin")
    args = ap.parse_args()

    fs.apply_style()
    zs = np.load(args.spikes, allow_pickle=False)
    zc = np.load(args.calcium, allow_pickle=False)
    N = int(zs["N"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.1, wspace=0.3)
    panel(ax[0], zs, "on raw SPIKES")
    panel(ax[1], zc, "through CALCIUM")

    fig.suptitle(f"Count-based, as a violin: peak-lag distribution per class, "
                 f"true vs false-positive excitatory (N={N})",
                 fontsize=13, color=fs.INK, x=0.08, ha="left", y=0.97)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
