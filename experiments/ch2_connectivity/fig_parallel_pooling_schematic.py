"""
Schematic — how the pipeline runs today (one node, one long recording) vs. the
proposed K-way parallel pooling scheme for pushing N=12500 to a much longer
effective recording length.

Drawn at the MATRIX level (not just a flowchart): the data matrix X (N neurons
x T ms) is sliced along the TIME axis only -- N is never split, because every
chunk must still cover every neuron to estimate the one fixed adjacency matrix
-- and each chunk's local outer-product moment matrix (N x N) is literally
summed into the pooled Cxx/Cyx before a single solve. All matrix contents are
synthetic placeholders (fixed RNG seed) purely for visual texture, not data.

Usage:
  python experiments/ch2_connectivity/fig_parallel_pooling_schematic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs

PURPLE, RED, ORANGE = fs.ACCENT2, "#e34948", "#e08e2a"
ADJ_CMAP = LinearSegmentedColormap.from_list("adj_div", [fs.C_I, "#f4f3ef", fs.C_E])
rng = np.random.default_rng(0)


# ---- small building blocks -------------------------------------------------- #
def box(ax, xy, w, h, text, fc="white", ec=fs.INK, fontsize=9.0, lw=1.3, zorder=3):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             color=fs.INK, zorder=zorder + 1, linespacing=1.25)
    return dict(b=(x + w / 2, y), t=(x + w / 2, y + h), l=(x, y + h / 2),
                r=(x + w, y + h / 2), cx=x + w / 2, cy=y + h / 2, w=w, h=h)


def chip(ax, xy, text, fc, tc="white", w=1.05, h=0.4, fontsize=8.0):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                        linewidth=0, facecolor=fc, zorder=5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             color=tc, weight="bold", zorder=6)


def arrow(ax, p0, p1, color=fs.MUTED, lw=1.5, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
                         color=color, connectionstyle=connectionstyle,
                         shrinkA=0, shrinkB=0, zorder=2)
    ax.add_patch(a)


def mat(ax, xy, w, h, data, cmap, vmin=None, vmax=None, ec=fs.INK, lw=1.1, zorder=3):
    """Draw a small matrix icon (imshow'd array + border) -- synthetic texture only."""
    x, y = xy
    ax.imshow(data, extent=(x, x + w, y, y + h), origin="upper", cmap=cmap,
              vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest",
              zorder=zorder, resample=False)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=ec, linewidth=lw,
                            zorder=zorder + 1))
    return dict(b=(x + w / 2, y), t=(x + w / 2, y + h), l=(x, y + h / 2),
                r=(x + w, y + h / 2), cx=x + w / 2, cy=y + h / 2, w=w, h=h)


def label(ax, xy, text, fontsize=8.6, color=fs.MUTED, ha="center", va="center",
          style="normal", weight="normal"):
    ax.text(xy[0], xy[1], text, fontsize=fontsize, color=color, ha=ha, va=va,
             style=style, weight=weight)


def brace_h(ax, x0, x1, y, text, color=fs.MUTED, tick=0.08, down=True):
    d = -tick if down else tick
    ax.plot([x0, x0, x1, x1], [y, y + d, y + d, y], color=color, lw=1.0, zorder=1)
    label(ax, ((x0 + x1) / 2, y + d - 0.16 * (1 if down else -1)), text,
          va="top" if down else "bottom")


def cov_icon(seed, n=10):
    M = np.random.default_rng(seed).normal(size=(n, n))
    return (M + M.T) / 2


def adj_icon(seed, n=10, density=0.12):
    r = np.random.default_rng(seed)
    A = np.zeros((n, n))
    mask = r.random((n, n)) < density
    signs = r.choice([1.0, -1.0], size=mask.sum(), p=[0.55, 0.45])
    A[mask] = signs
    A += r.normal(scale=0.05, size=(n, n))
    np.fill_diagonal(A, 0)
    return A


# ---- panels ------------------------------------------------------------------ #
def panel_ladder(ax):
    ax.set_title("Three stages of scaling this pipeline up",
                  fontsize=12.5, color=fs.INK, loc="left", weight="bold", pad=8)

    stages = [
        ("CPU only", "N <= 1250\n~2-3 min / run", "CPU", "#8a8a86"),
        ("1 GPU · 1 node", "N = 12500 (today)\nup to ~10M ms measured,\n~14M ms hard ceiling",
         "GPU x1", fs.ACCENT),
        ("K GPUs · K nodes", "N = 12500 (proposed)\nK x T/K ms pooled\n~= 100M ms equivalent",
         "GPU x K", ORANGE),
    ]
    xs = [0.3, 4.5, 8.7]
    boxes = []
    for (title, body, tag, col), x in zip(stages, xs):
        b = box(ax, (x, 0.35), 3.5, 1.55, "", fc="#f7f7f5", fontsize=9.0)
        ax.text(x + 0.18, 0.35 + 1.55 - 0.28, title, fontsize=10.0, weight="bold",
                 color=fs.INK, ha="left", va="top")
        ax.text(x + 0.18, 0.35 + 1.55 - 0.62, body, fontsize=8.2, color=fs.MUTED,
                 ha="left", va="top", linespacing=1.3)
        chip(ax, (x + 3.5 - 1.15, 0.35 + 1.55 - 0.42), tag, col)
        boxes.append(b)
    arrow(ax, boxes[0]["r"], boxes[1]["l"])
    label(ax, ((boxes[0]["r"][0] + boxes[1]["l"][0]) / 2, boxes[0]["cy"] + 1.0),
          "N grows: the dense\nN x N solve needs a GPU", fontsize=7.6)
    arrow(ax, boxes[1]["r"], boxes[2]["l"])
    label(ax, ((boxes[1]["r"][0] + boxes[2]["l"][0]) / 2, boxes[1]["cy"] + 1.0),
          "T needed for N=12500 exceeds\none node's memory ceiling", fontsize=7.6)

    ax.set_xlim(-0.3, 12.6); ax.set_ylim(0.1, 2.3); ax.axis("off")
    ax.set_aspect("equal")


def panel_today(ax):
    ax.set_title("What we run today — one node, one continuous recording",
                  fontsize=12.5, color=fs.INK, loc="left", weight="bold", pad=10)
    chip(ax, (8.8, 2.35), "GPU x1", fs.ACCENT, w=1.2)

    sim = box(ax, (0.1, 1.55), 1.85, 0.85, "simulate + preprocess\nnode 1 · T ms · seed s",
               fc="#eaf1fc", fontsize=8.5)
    X = mat(ax, (2.35, 1.4), 3.6, 1.15, rng.normal(size=(10, 46)), fs.BLUE_SEQ)
    arrow(ax, sim["r"], X["l"])
    brace_h(ax, X["l"][0], X["r"][0], X["b"][1], "T ms  (whole recording, one node)")
    label(ax, (X["l"][0] - 0.18, X["t"][1] + 0.05), "N", ha="right", va="bottom")
    label(ax, (X["cx"], X["t"][1] + 0.22), "data matrix  X", weight="bold", fontsize=9.2)

    Cxx = mat(ax, (7.1, 1.4), 1.5, 1.5, cov_icon(1), fs.BLUE_SEQ)
    arrow(ax, X["r"], Cxx["l"])
    label(ax, (Cxx["cx"], Cxx["t"][1] + 0.22), "Cxx, Cyx  (N x N)", fontsize=8.8)
    label(ax, (Cxx["cx"], Cxx["b"][1] - 0.22), "= sum over all T ms\nof outer products",
          fontsize=7.6)

    Ahat = mat(ax, (7.1, -0.7), 1.5, 1.15, adj_icon(1), ADJ_CMAP, vmin=-1, vmax=1)
    arrow(ax, Cxx["b"], Ahat["t"])
    solve = box(ax, (4.6, -0.6), 1.85, 0.85, "solve\n(OLS / EN / Dale)", fc="#f1ecfb",
                 ec=PURPLE, fontsize=8.7)
    arrow(ax, Ahat["l"], solve["r"])
    label(ax, (Ahat["cx"], Ahat["b"][1] - 0.22), "estimated connectivity  (N x N)",
          fontsize=8.4)

    ax.text(0.1, -1.7,
            "measured: T=10M ms -> ~5-6 h, ~400 G/seed   ·   single-node ceiling on a 512G\n"
            "node: ~14.5M ms -- X (i.e. NEST's spike buffer) grows linearly with T and must\n"
            "fit in one node's memory for the whole run before Cxx/Cyx can be formed",
            fontsize=8.7, color=fs.MUTED, va="top")
    ax.set_xlim(-0.3, 10.0); ax.set_ylim(-2.5, 3.3); ax.axis("off")
    ax.set_aspect("equal")


def panel_parallel(ax):
    ax.set_title("Proposed — slice X in TIME only, solve K local moment matrices, sum them",
                  fontsize=12.2, color=fs.INK, loc="left", weight="bold", pad=10)
    chip(ax, (12.35, 6.0), "GPU x K", ORANGE, w=1.2)

    Xw, Xh = 8.6, 1.15
    X0 = (0.5, 5.05)
    Xfull = mat(ax, X0, Xw, Xh, rng.normal(size=(10, 46)), fs.BLUE_SEQ)
    label(ax, (Xfull["cx"], Xfull["t"][1] + 0.22), "SAME data matrix  X  --  N is never split "
          "(every chunk must cover every neuron)", weight="bold", fontsize=9.0)
    label(ax, (Xfull["l"][0] - 0.22, Xfull["cy"]), "N", ha="right")

    K, xs = 3, None
    edges = np.linspace(X0[0], X0[0] + Xw, 4)
    chunk_labels = ["X_1", "X_2", "X_K"]
    node_xs = [0.5, 4.05, 7.6]
    for i in range(3):
        x0, x1 = edges[i], edges[i + 1]
        if 0 < i:
            ax.plot([x0, x0], [X0[1], X0[1] + Xh], color=fs.INK, lw=1.1, ls="--", zorder=4)
        label(ax, ((x0 + x1) / 2, X0[1] + Xh / 2), chunk_labels[i], color="white",
              weight="bold", fontsize=9.5)
    brace_h(ax, edges[0], edges[1], X0[1], "T/K ms\nnoise seed 1")
    brace_h(ax, edges[1], edges[2], X0[1], "T/K ms\nnoise seed 2")
    brace_h(ax, edges[2], edges[3], X0[1], "T/K ms\nnoise seed K")

    mom_y = 2.75
    moms = []
    for i, x in enumerate(node_xs):
        src = ((edges[i] + edges[i + 1]) / 2, X0[1])
        m = mat(ax, (x + 0.55, mom_y), 1.3, 1.3, cov_icon(10 + i), fs.BLUE_SEQ)
        arrow(ax, src, (m["cx"], mom_y + 1.3), connectionstyle="arc3,rad=0.0")
        label(ax, (m["cx"], m["t"][1] + 0.22), f"local Cxx_{i+1 if i<2 else 'K'}", fontsize=8.2)
        label(ax, (m["cx"], m["b"][1] - 0.2), f"node {i+1 if i<2 else 'K'}", fontsize=7.8,
              color=fs.MUTED)
        moms.append(m)
    label(ax, (5.85, mom_y + 0.65), "...", fontsize=16, color=fs.MUTED)

    eq_y = 0.65
    icon_w = 1.0
    xseq = [m["cx"] - icon_w / 2 for m in moms]        # directly under each node's Cxx_i
    signs = ["", "+", "+ ... +"]
    icons = []
    for i, (m, xi, s) in enumerate(zip(moms, xseq, signs)):
        if s:
            gap_mid = (icons[-1]["r"][0] + xi) / 2 if icons else xi
            label(ax, (gap_mid, eq_y + icon_w / 2), s, fontsize=13, weight="bold")
        icon = mat(ax, (xi, eq_y), icon_w, icon_w, cov_icon(10 + i), fs.BLUE_SEQ)
        arrow(ax, m["b"], icon["t"])
        icons.append(icon)
    eq_x = icons[-1]["r"][0] + 0.35
    label(ax, (eq_x, eq_y + icon_w / 2), "=", fontsize=15, weight="bold")
    Ptotal = mat(ax, (eq_x + 0.3, eq_y - 0.15), 1.3, 1.3, cov_icon(99), fs.BLUE_SEQ,
                  ec=ORANGE, lw=1.6)
    label(ax, (Ptotal["cx"], Ptotal["t"][1] + 0.22), "pooled Cxx", fontsize=8.6,
          color=fs.INK, weight="bold")
    label(ax, (Ptotal["cx"], Ptotal["b"][1] - 0.22), "(= exact sum, N x N)\n(Cyx pooled the same way)",
          fontsize=7.4)

    solve = box(ax, (Ptotal["r"][0] + 0.35, eq_y + 0.05), 1.55, 0.95, "solve\nonce",
                 fc="#f1ecfb", ec=PURPLE, fontsize=8.8)
    arrow(ax, Ptotal["r"], solve["l"])
    Ahat = mat(ax, (Ptotal["r"][0] + 0.5, eq_y - 1.55), 1.25, 1.15, adj_icon(2), ADJ_CMAP,
                vmin=-1, vmax=1)
    arrow(ax, solve["b"], Ahat["t"])
    label(ax, (Ahat["cx"], Ahat["b"][1] - 0.22), "estimated\nconnectivity", fontsize=7.8)

    ax.text(0.5, -1.9,
            "example, K=7 nodes: 7 x 14.3M ms ~= 100M ms equivalent recording\n"
            "memory/node ~= 510 G (fits the 512G ceiling)   ·   wall-clock ~= 8 h (vs ~55 h serial)\n"
            "requires: connectivity fixed & exported once; only the noise seed differs per node",
            fontsize=8.7, color=fs.MUTED, va="top")
    ax.set_xlim(-0.3, 13.6); ax.set_ylim(-3.1, 6.7); ax.axis("off")
    ax.set_aspect("equal")


def main():
    fs.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 13.2),
                              gridspec_kw={"height_ratios": [1, 1.42]})
    fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.02, hspace=0.14)
    panel_today(axes[0])
    panel_parallel(axes[1])
    fig.suptitle("Scaling N=12500 to a much longer effective recording — at the matrix level",
                 fontsize=14, color=fs.INK, x=0.03, ha="left", y=0.99)
    fs.save(fig, "figures/fig_parallel_pooling_schematic")


if __name__ == "__main__":
    main()
