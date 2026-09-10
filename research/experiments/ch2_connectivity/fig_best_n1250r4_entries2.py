"""
Trimmed version of fig_best_plot.py's "sample neurons" panel: ONE excitatory
and ONE inhibitory example (not 4), meant to run alongside
fig_best_n1250r4_signrank.py as an intuition-building companion rather than
standing on its own.

Usage:
  python research/experiments/ch2_connectivity/fig_best_n1250r4_entries2.py \
      --data ~/calcium_results/best/best_n1250r4.npz \
      --out figures/fig_best_n1250r4_entries2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs
from fig_best_plot import show_entries_bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "calcium_results/best/best_n1250r4.npz"))
    ap.add_argument("--out", default="figures/fig_best_n1250r4_entries2")
    args = ap.parse_args()

    fs.apply_style()
    import matplotlib.pyplot as plt

    z = np.load(args.data, allow_pickle=True)
    idx, is_exc, est = z["sample_idx"], z["sample_is_exc"], z["sample_est"]
    i_exc = int(np.flatnonzero(is_exc)[0])
    i_inh = int(np.flatnonzero(~is_exc)[0])

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    show_entries_bar(ax[0], est[i_exc], f"neuron #{int(idx[i_exc])} (excitatory)")
    show_entries_bar(ax[1], est[i_inh], f"neuron #{int(idx[i_inh])} (inhibitory)")
    fig.text(0.5, 0.98, "blue = positive   red = negative", ha="center",
             fontsize=9, color=fs.INK)
    fig.suptitle("Full inferred outgoing profile, one example neuron per class",
                 fontsize=12.5, color=fs.INK, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
