"""
Schematic — the INNER chunking axis: how one node streams its own T/K-ms slice
through the moment accumulator, CPU path vs GPU path. This is the axis that
sits INSIDE each node column of fig_parallel_pooling_schematic.py's "proposed"
panel (that figure only shows the OUTER, across-node axis).

Grounded directly in src/calcinet/io/streaming.py:
  - MomentAccumulator (CPU/numpy): everything -- the chunk, the outer product,
    the running Cxx_raw/Cyx_raw -- lives in plain host RAM. No transfer step.
  - TorchMomentAccumulator (GPU): add() explicitly does
    `torch.as_tensor(feed_chunk, ..., device=self.device)` for EVERY chunk (an
    explicit host->device copy), the matmul runs on the GPU, and Cxx_raw/Cyx_raw
    are allocated ON the device and stay resident there for the whole run;
    snapshot() is the one .cpu() call that brings the final result back.
  - chunk_ms is a generic streaming parameter used on BOTH paths in this
    codebase (usually 5000 ms either way) -- NOT something the code already
    sizes differently per device. The real, code-grounded difference is the
    extra host<->device transfer + the fact that only the fixed-size (N x N)
    accumulator needs to stay resident on the GPU, not the whole chunk history
    -- that's what makes streaming through a GPU tractable for large N at all.

Usage:
  python experiments/ch2_connectivity/fig_chunking_cpu_vs_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs
from fig_parallel_pooling_schematic import (arrow, box, chip, cov_icon, label, mat, rng)

ORANGE = "#e08e2a"
HOST_FC, GPU_FC = "#eef1f6", "#fdf1e2"


def chunk_row(ax, x0, y, n, w_each=0.55, gap=0.12, h=0.55, color=fs.ACCENT):
    xs = []
    for i in range(n):
        x = x0 + i * (w_each + gap)
        mat(ax, (x, y), w_each, h, rng.normal(size=(6, 6)), fs.BLUE_SEQ)
        xs.append(x + w_each / 2)
    return xs


def panel_cpu(ax):
    ax.set_title("CPU path — small N (e.g. N <= 1250)", fontsize=12, color=fs.INK,
                  loc="left", weight="bold", pad=8)
    chip(ax, (6.55, 6.35), "CPU", "#8a8a86", w=1.0)

    host = box(ax, (0.2, 0.3), 7.6, 6.2, "", fc=HOST_FC, ec="#b9b8ae", lw=1.2, zorder=1)
    label(ax, (host["l"][0] + 0.25, host["t"][1] - 0.3), "host RAM", ha="left",
          weight="bold", fontsize=9.5, color=fs.MUTED)

    xs = chunk_row(ax, 1.0, 5.1, 4)
    label(ax, (4.0, 5.85), "node i's own chunks, one at a time", fontsize=8.4)

    op = box(ax, (2.9, 3.55), 2.2, 0.9, "outer product\n(computed in RAM)", fc="white",
              fontsize=8.4)
    for x in xs:
        arrow(ax, (x, 5.08), op["t"], color=fs.INK, lw=2.0)

    acc = box(ax, (2.9, 1.75), 2.2, 0.95, "Cxx_raw, Cyx_raw\n(running sum, in RAM)",
               fc="white", ec=ORANGE, fontsize=8.2)
    arrow(ax, op["b"], acc["t"])
    arrow(ax, acc["l"], (acc["l"][0] - 0.5, acc["cy"] + 0.9), connectionstyle="arc3,rad=1.3")
    label(ax, (1.85, acc["cy"] + 0.75), "next\nchunk", fontsize=7.2, color=fs.MUTED)

    out = mat(ax, (5.6, 1.85), 1.3, 1.2, cov_icon(31), fs.BLUE_SEQ)
    arrow(ax, acc["r"], out["l"])
    label(ax, (out["cx"], out["b"][1] - 0.22), "local Cxx_i", fontsize=8.2)

    ax.text(0.4, 0.65, "no transfer step -- everything (chunk, matmul,\n"
            "running total) stays in one address space",
            fontsize=8.0, color=fs.MUTED)
    ax.set_xlim(-0.2, 8.2); ax.set_ylim(0, 6.9); ax.axis("off"); ax.set_aspect("equal")


def panel_gpu(ax):
    ax.set_title("GPU path — large N (e.g. N = 12500)", fontsize=12, color=fs.INK,
                  loc="left", weight="bold", pad=8)
    chip(ax, (6.3, 6.35), "GPU", fs.ACCENT, w=1.0)

    host = box(ax, (0.2, 4.7), 3.0, 1.8, "", fc=HOST_FC, ec="#b9b8ae", lw=1.2, zorder=1)
    label(ax, (host["l"][0] + 0.2, host["t"][1] - 0.25), "host RAM", ha="left",
          weight="bold", fontsize=9, color=fs.MUTED)
    gpu = box(ax, (0.2, 0.3), 7.6, 4.15, "", fc=GPU_FC, ec="#d9a45a", lw=1.2, zorder=1)
    label(ax, (gpu["l"][0] + 0.2, gpu["t"][1] - 0.28), "GPU / VRAM", ha="left",
          weight="bold", fontsize=9.5, color="#8a5d1a")

    mat(ax, (0.6, 5.15), 2.2, 0.95, rng.normal(size=(6, 18)), fs.BLUE_SEQ)
    label(ax, (1.7, 5.15 - 0.22), "next chunk waiting\n(node i's own data)", fontsize=7.6,
          color=fs.MUTED, va="top")

    xs = chunk_row(ax, 1.0, 3.35, 4, w_each=0.5, gap=0.1, h=0.5)
    for x in xs:
        arrow(ax, (x, 4.7), (x, 3.85), color="#8a5d1a")
    label(ax, (4.0, 4.35), "copy chunk to VRAM (per chunk, every time)",
          fontsize=7.6, color="#8a5d1a")

    op = box(ax, (2.9, 2.55), 2.2, 0.75, "outer product\n(matmul on GPU)", fc="white",
              fontsize=8.2)
    for x in xs:
        arrow(ax, (x, 3.33), op["t"], color=fs.INK, lw=2.0)

    acc = box(ax, (2.9, 1.05), 2.2, 0.9,
               "Cxx_raw, Cyx_raw\nresident IN VRAM\n(fixed N x N -- small)",
               fc="white", ec=ORANGE, fontsize=7.8)
    arrow(ax, op["b"], acc["t"])
    arrow(ax, acc["l"], (acc["l"][0] - 0.5, acc["cy"] + 1.7), connectionstyle="arc3,rad=1.4")
    label(ax, (1.65, acc["cy"] + 1.45), "next\nchunk", fontsize=7.0, color=fs.MUTED)

    out = mat(ax, (5.6, 1.1), 1.3, 1.1, cov_icon(32), fs.BLUE_SEQ)
    arrow(ax, acc["r"], out["l"], color="#8a5d1a")
    label(ax, (out["cx"] + 0.05, out["t"][1] + 0.55), "copy final\nCxx_i to host",
          fontsize=7.2, color="#8a5d1a")
    label(ax, (out["cx"], out["b"][1] - 0.22), "local Cxx_i (back on host)", fontsize=7.8)

    ax.text(0.4, -0.35,
            "the data itself (N x T/K) never has to fit in VRAM at once -- only the\n"
            "fixed-size (N x N) accumulator stays resident. That's what makes streaming\n"
            "through a GPU tractable for N too large to solve densely on CPU at all.",
            fontsize=8.0, color=fs.MUTED, va="top")
    ax.set_xlim(-0.2, 8.2); ax.set_ylim(-1.0, 6.9); ax.axis("off"); ax.set_aspect("equal")


def main():
    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.08)
    panel_cpu(axes[0])
    panel_gpu(axes[1])
    fig.suptitle("Inside one node — how a chunk of data reaches the moment accumulator",
                 fontsize=14.5, color=fs.INK, x=0.02, ha="left", y=0.985)
    fs.save(fig, "figures/fig_chunking_cpu_vs_gpu")


if __name__ == "__main__":
    main()
