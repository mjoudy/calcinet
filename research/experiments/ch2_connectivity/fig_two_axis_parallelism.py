"""
Schematic — BOTH chunking axes together: K nodes (outer, across-node -- see
fig_parallel_pooling_schematic.py) x GPU chunk-streaming (inner, within one
node -- see fig_chunking_cpu_vs_gpu.py), nested in a single figure.

Each node column shows its own mini chunk-stream (a few chunk_ms pieces
feeding a running GPU accumulator) producing that node's local Cxx_i; those
local Cxx_i are then pooled (summed) exactly as in the parallel-pooling
schematic. This is the "put it all together" figure.

Usage:
  python research/experiments/ch2_connectivity/fig_two_axis_parallelism.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs
from fig_parallel_pooling_schematic import (ADJ_CMAP, PURPLE, adj_icon, arrow, box, chip,
                                             cov_icon, label, mat, rng)

ORANGE = "#e08e2a"
GPU_FC = "#fdf1e2"


def node_column(ax, x, n_chunks=3):
    """One node's full inner axis: chunk row -> GPU accumulate (looped) -> local Cxx_i.
    Returns the local-Cxx_i matrix handle (for the outer pooling step)."""
    gpu = box(ax, (x - 0.15, 2.35), 2.9, 2.05, "", fc=GPU_FC, ec="#d9a45a", lw=1.1, zorder=1)
    label(ax, (x - 0.05, gpu["t"][1] - 0.18), "GPU", ha="left", fontsize=7.4,
          weight="bold", color="#8a5d1a")

    cw, gap = 0.42, 0.1
    xs = []
    for i in range(n_chunks):
        cx = x + i * (cw + gap)
        mat(ax, (cx, 3.75), cw, cw, rng.normal(size=(5, 5)), fs.BLUE_SEQ)
        xs.append(cx + cw / 2)

    acc = box(ax, (x + 0.15, 2.5), 2.3, 0.85, "Cxx_i, Cyx_i\n(running, in VRAM)",
               fc="white", ec=ORANGE, fontsize=6.8)
    for cx in xs:
        arrow(ax, (cx, 3.75), acc["t"], color=fs.INK, lw=1.3)
    arrow(ax, acc["l"], (acc["l"][0] - 0.3, acc["cy"] + 0.55), connectionstyle="arc3,rad=1.3",
          lw=1.1)

    out = mat(ax, (x + 0.55, 0.95), 1.35, 1.15, cov_icon(hash(str(x)) % 1000), fs.BLUE_SEQ)
    arrow(ax, acc["b"], out["t"])
    return out


def main():
    fs.apply_style()
    fig, ax = plt.subplots(figsize=(14.6, 11.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)

    ax.set_title("Both axes together — K nodes (outer) x GPU chunk-streaming (inner) -> pooled",
                  fontsize=13.5, color=fs.INK, loc="left", weight="bold", pad=10)

    Xw, Xh = 9.4, 1.1
    X0 = (0.4, 7.9)
    Xfull = mat(ax, X0, Xw, Xh, rng.normal(size=(10, 46)), fs.BLUE_SEQ)
    label(ax, (Xfull["cx"], Xfull["t"][1] + 0.2),
          "SAME data matrix X -- N never split, only T is (outer axis)",
          weight="bold", fontsize=9.4)

    edges = np.linspace(X0[0], X0[0] + Xw, 4)
    node_xs = [edges[0] + 0.15, edges[1] + 0.15, edges[2] + 0.55]
    chunk_labels = ["X_1", "X_2", "X_K"]
    for i in range(3):
        x0e, x1e = edges[i], edges[i + 1]
        if i > 0:
            ax.plot([x0e, x0e], [X0[1], X0[1] + Xh], color=fs.INK, lw=1.0, ls="--", zorder=4)
        label(ax, ((x0e + x1e) / 2, X0[1] + Xh / 2), chunk_labels[i], color="white",
              weight="bold", fontsize=10)
        arrow(ax, ((x0e + x1e) / 2, X0[1]), ((x0e + x1e) / 2 - (0.0 if i < 2 else 0.9), 6.15),
              color=fs.MUTED)

    label(ax, (node_xs[1] + 1.6, 7.15), "each node's own T/K ms\n(outer, across-node axis)",
          fontsize=8.4, color=fs.MUTED)
    label(ax, (node_xs[1] + 1.6, 6.6),
          "streamed through its own GPU in chunk_ms pieces\n(inner, within-node axis -- see "
          "fig_chunking_cpu_vs_gpu.png)", fontsize=8.0, color="#8a5d1a")

    outs = []
    for i, x in enumerate(node_xs):
        label(ax, (x + 1.35, 5.85), f"node {i+1 if i < 2 else 'K'}", fontsize=9.0,
              weight="bold")
        outs.append(node_column(ax, x))
        label(ax, (x + 1.2, 0.75), f"local Cxx_{i+1 if i < 2 else 'K'}", fontsize=8.0)

    eq_y = -0.75
    icon_w = 1.0
    xseq = [o["cx"] - icon_w / 2 for o in outs]
    signs = ["", "+", "+ ... +"]
    icons = []
    for o, xi, s in zip(outs, xseq, signs):
        if s:
            gap_mid = (icons[-1]["r"][0] + xi) / 2 if icons else xi
            label(ax, (gap_mid, eq_y + icon_w / 2), s, fontsize=13, weight="bold")
        icon = mat(ax, (xi, eq_y), icon_w, icon_w, cov_icon(500 + len(icons)), fs.BLUE_SEQ)
        arrow(ax, o["b"], icon["t"])
        icons.append(icon)
    eq_x = icons[-1]["r"][0] + 0.35
    label(ax, (eq_x, eq_y + icon_w / 2), "=", fontsize=15, weight="bold")
    Ptotal = mat(ax, (eq_x + 0.3, eq_y - 0.15), 1.3, 1.3, cov_icon(999), fs.BLUE_SEQ,
                  ec=ORANGE, lw=1.6)
    label(ax, (Ptotal["cx"], Ptotal["t"][1] + 0.22), "pooled Cxx", fontsize=8.6,
          color=fs.INK, weight="bold")
    label(ax, (Ptotal["cx"], Ptotal["b"][1] - 0.22), "(= exact sum, N x N)", fontsize=7.4)

    solve = box(ax, (Ptotal["r"][0] + 0.35, eq_y + 0.05), 1.55, 0.95, "solve\nonce",
                 fc="#f1ecfb", ec=PURPLE, fontsize=8.8)
    arrow(ax, Ptotal["r"], solve["l"])
    Ahat = mat(ax, (Ptotal["r"][0] + 0.5, eq_y - 1.55), 1.25, 1.15, adj_icon(3), ADJ_CMAP,
                vmin=-1, vmax=1)
    arrow(ax, solve["b"], Ahat["t"])
    label(ax, (Ahat["cx"], Ahat["b"][1] - 0.22), "estimated\nconnectivity", fontsize=7.8)

    ax.text(0.4, -2.6,
            "outer axis (nodes) is PARALLEL and INDEPENDENT -- summed at the end, needs\n"
            "the connectivity/noise-seed split described earlier. inner axis (GPU chunks) is\n"
            "SEQUENTIAL within one node -- a running accumulator, already how the code works.",
            fontsize=8.5, color=fs.MUTED, va="top")

    ax.set_xlim(-0.3, 13.2); ax.set_ylim(-3.4, 9.3); ax.axis("off"); ax.set_aspect("equal")
    fs.save(fig, "figures/fig_two_axis_parallelism")


if __name__ == "__main__":
    main()
