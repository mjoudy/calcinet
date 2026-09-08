"""
R.1 confusion matrix — NORMALIZATION OPTIONS (exploratory, local).

Renders every standard normalization of the 3x3 confusion matrix from
r1_data.npz side by side, so a normalization choice can be compared before
picking what goes into fig_R1 itself:

  raw        : counts, no normalization (log color, since "none" dwarfs
               everything else linearly)
  row  (true): P(predicted | true)   — recall view, currently used in fig_R1
  col  (pred): P(true | predicted)   — precision view
  all        : P(true, predicted)    — joint / global-normalized

Usage:
  python experiments/ch2_connectivity/fig_r1_confusion_options.py --data results/fig_data/r1_data.npz \
      --out figures/fig_R1_confusion_options
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs

CLASS_NAMES = ("E", "none", "I")


def fmt_count(v):
    v = int(round(v))
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}k"
    return str(v)


def draw(ax, color_mat, text_mat, title, vmax):
    ax.imshow(color_mat, cmap=fs.BLUE_SEQ, vmin=0, vmax=vmax, aspect="equal")
    for i in range(3):
        for j in range(3):
            light = color_mat[i, j] > 0.55 * vmax
            ax.text(j, i, text_mat[i, j], ha="center", va="center",
                    color="white" if light else fs.INK, fontsize=9.5)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=10.5)
    fs.despine(ax, keep=())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/fig_data/r1_data.npz")
    ap.add_argument("--out", default="figures/fig_R1_confusion_options")
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    cm = z["cm"]  # raw counts, rows=true, cols=predicted

    row_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)   # recall view
    col_norm = cm / np.clip(cm.sum(0, keepdims=True), 1, None)   # precision view
    all_norm = cm / cm.sum()                                     # joint view
    log_counts = np.log1p(cm)

    fig, ax = plt.subplots(2, 2, figsize=(11, 10.5))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.06,
                        hspace=0.4, wspace=0.4)
    ax = ax.ravel()

    draw(ax[0], log_counts, np.vectorize(fmt_count)(cm),
        "raw counts\n(log color scale)", log_counts.max())
    draw(ax[1], row_norm, np.vectorize(lambda v: f"{v:.2f}")(row_norm),
        "row-normalized: P(pred | true)\n— recall view (current fig_R1)", 1.0)
    draw(ax[2], col_norm, np.vectorize(lambda v: f"{v:.2f}")(col_norm),
        "column-normalized: P(true | pred)\n— precision view", 1.0)
    draw(ax[3], all_norm, np.vectorize(lambda v: f"{v:.3f}")(all_norm),
        "global-normalized: P(true, pred)\n— joint view", all_norm.max())

    fig.suptitle(f"Confusion matrix — normalization options  "
                 f"({str(z['label'])}, {str(z['method']).upper()}, N={int(z['N'])})",
                 fontsize=13.5, color=fs.INK, y=0.98)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
