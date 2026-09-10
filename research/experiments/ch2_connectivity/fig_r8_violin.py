"""
R.8 (shared input vs directionality) — violin-only PLOT stage, local.

Same underlying data as fig_r8_plot.py (asym_{E,I}_{true,false} arrays), shown
as a single violin plot with per-class counts annotated (n=...), so it's clear
how many false positives each violin is actually summarising.

Two metrics, same formula |x-y|/(|x|+|y|), different inputs:
  --metric asym   (default) built from the continuous |A[i,j]|, |A[j,i]| weights.
  --metric asymc  built from BINARY connected/not-connected flags (is this
                  direction above the density threshold, yes/no) instead of
                  weights. Collapses to 0 (bidirectional) / 1 (one-directional)
                  per pair — needs fig_r8_compute.py's asymc_* fields.

Usage:
  python research/experiments/ch2_connectivity/fig_r8_violin.py --data ~/calcium_results/r8/r8_wrapup_n12500r4_T5000k.npz \
      --out figures/fig_R8_violin_n12500
  python research/experiments/ch2_connectivity/fig_r8_violin.py --metric asymc --data <r8_*.npz> \
      --out figures/fig_R8_violin_counts_n1250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs

LABELS = [("E_true", "n_E_true", "true E", fs.C_E, 1.0),
          ("E_false", "n_E_false", "E false pos.", fs.C_E, 0.45),
          ("I_true", "n_I_true", "true I", fs.C_I, 1.0),
          ("I_false", "n_I_false", "I false pos.", fs.C_I, 0.45)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--metric", choices=["asym", "asymc"], default="asym")
    ap.add_argument("--out", default="figures/fig_R8_violin")
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=True)
    N = int(z["N"])
    prefix = args.metric
    GROUPS = [(f"{prefix}_{stem}", nkey, lab, col, alpha)
              for stem, nkey, lab, col, alpha in LABELS]
    data = [(lab, col, alpha, z[key], (int(z[nkey]) if nkey in z.files else None))
            for key, nkey, lab, col, alpha in GROUPS
            if key in z.files and len(z[key]) > 0]
    if not data:
        raise SystemExit(f"no {prefix}_* fields in {args.data} — re-run the "
                          "updated fig_r8_compute.py to produce them")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.6))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.12)

    positions = list(range(1, len(data) + 1))
    vals = [v for _, _, _, v, _ in data]
    parts = ax.violinplot(vals, positions=positions, showmedians=True,
                           showextrema=False, widths=0.8)
    for i, (lab, col, alpha, v, n_true) in enumerate(data):
        body = parts["bodies"][i]
        body.set_facecolor(col); body.set_alpha(0.35 if alpha < 1.0 else 0.55)
        body.set_edgecolor(col)
        # n_true is the real per-class count; older cached npz files only kept
        # a 200k-capped subsample and never saved the true count, so fall back
        # to an honest "at least this many" label instead of a fake exact one.
        n_label = f"n={n_true:,}" if n_true is not None else f"n≥{len(v):,}"
        ax.text(positions[i], 1.05, n_label, ha="center", va="bottom",
                fontsize=8.5, color=fs.MUTED)
    parts["cmedians"].set_color(fs.INK); parts["cmedians"].set_linewidth(1.4)
    for i, (lab, col, alpha, v, n_true) in enumerate(data):
        med = np.median(v)
        # for asymc the median is nearly always exactly 0 or 1 (it's binary) —
        # the informative number there is the MEAN (= fraction one-directional).
        stat = np.mean(v) if args.metric == "asymc" else med
        stat_lab = f" mean {stat:.2f}" if args.metric == "asymc" else f" {stat:.2f}"
        ax.text(positions[i], med, stat_lab, ha="left", va="center",
                fontsize=8.5, color=fs.INK)

    ax.set_xticks(positions); ax.set_xticklabels([lab for lab, *_ in data], fontsize=9)
    if args.metric == "asymc":
        ylabel = "count-based asymmetry: is the reverse direction\nALSO predicted connected? (0=bidirectional, 1=one-directional)"
        subtitle = "false positives are bidirectional more often than true edges"
    else:
        ylabel = "directional asymmetry  |A[i,j]-A[j,i]| / (|A[i,j]|+|A[j,i]|)"
        subtitle = "false positives are more symmetric (lower) than true edges"
    ax.set(ylabel=ylabel, ylim=(0, 1.12))
    ax.grid(True, axis="y", color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
    fs.despine(ax)
    ax.set_title(f"Directional asymmetry by class at N={N} (single-lag OLS)\n{subtitle}",
                 fontsize=12, color=fs.INK)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
