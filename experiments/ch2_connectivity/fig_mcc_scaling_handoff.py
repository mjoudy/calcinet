"""
STOPGAP script -- hand-transcribed from a companion analysis chat's data
handoff (2026-09-04), NOT computed from a local metrics.csv, because the
MCC-carrying metrics.csv files (analyze_run.py's mcc3/mcc_pres/mcc_E/mcc_I/
mcc_type/mcc_neuron columns, added in commit 3daca8b1) have not yet been
rsynced off the cluster to this machine.

Once they are, delete this script and use experiments/ch2_connectivity/fig_mcc_table.py instead
(it already reads exactly this data straight from metrics.csv, aggregates
seeds, and renders a heatmap + N-scaling line plot in one go -- this script
duplicates only the size-scaling half of that, by hand, as a bridge).

Every number below is copied verbatim from the handoff's Table 3 (tuned-AI /
"r4" regime, OLS, density=0.10, mean over seeds). N=12500 is entirely
pending and is not plotted. N=5000 has two disjoint clusters of checkpoints
(100-1000k, then 10000-20000k); the big gap between them is NOT bridged with
a connecting line, since nothing was actually run in between.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs

DT = 0.1  # ms/sample

# T_k (thousand ms), n_seeds, MCC-3, MCC-E, MCC-I
DATA = {
    1250: dict(
        Tk=[12, 25, 50, 100, 200, 500, 1000, 2000, 5000, 7500, 10000, 15000, 20000],
        mcc3=[0.189, 0.280, 0.383, 0.483, 0.583, 0.690, 0.739, 0.766, 0.778, 0.786, 0.786, 0.782, 0.779],
        mcc_E=[0.145, 0.198, 0.289, 0.408, 0.525, 0.642, 0.696, 0.727, 0.740, 0.749, 0.749, 0.745, 0.742],
        mcc_I=[0.360, 0.549, 0.685, 0.742, 0.803, 0.876, 0.911, 0.927, 0.934, 0.937, 0.935, 0.931, 0.928],
    ),
    2500: dict(
        Tk=[100, 200, 500, 1000, 2000, 5000, 7500, 10000, 15000, 20000],
        mcc3=[0.352, 0.434, 0.554, 0.629, 0.680, 0.715, 0.722, 0.723, 0.721, 0.720],
        mcc_E=[0.251, 0.356, 0.497, 0.581, 0.637, 0.675, 0.683, 0.684, 0.683, 0.681],
        mcc_I=[0.661, 0.700, 0.766, 0.814, 0.849, 0.872, 0.877, 0.877, 0.875, 0.873],
    ),
    5000: dict(
        # two disjoint runs of checkpoints -- gap deliberately not bridged
        Tk=[[100, 1000], [10000, 15000, 20000]],
        mcc3=[[0.273, 0.510], [0.660, 0.669, 0.673]],
        mcc_E=[[0.150, 0.453], [0.622, 0.631, 0.636]],
        mcc_I=[[0.611, 0.724], [0.816, 0.822, 0.824]],
    ),
}

MEASURES = [("mcc3", "MCC-3 (E/none/I)"), ("mcc_E", "MCC-E (excitatory vs. rest)"),
            ("mcc_I", "MCC-I (inhibitory vs. rest)")]


def main():
    fs.apply_style()
    Ns = sorted(DATA)
    colors = dict(zip(Ns, plt.cm.viridis(np.linspace(0.15, 0.85, len(Ns)))))

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.2), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.09,
                        hspace=0.32, wspace=0.24)

    for col, (key, label) in enumerate(MEASURES):
        for N in Ns:
            c = colors[N]
            d = DATA[N]
            if N == 5000:
                # two disjoint clusters -- plot each separately, label once
                for k, (Tk_seg, y_seg) in enumerate(zip(d["Tk"], d[key])):
                    Tk_seg = np.asarray(Tk_seg); y_seg = np.asarray(y_seg)
                    axes[0][col].plot(Tk_seg * 1000, y_seg, "o-", color=c, lw=1.8, ms=5,
                                      label=f"N = {N}" if col == 0 and k == 0 else None)
                    axes[1][col].plot((Tk_seg * 1e6) / DT / N, y_seg, "o-", color=c,
                                      lw=1.8, ms=5)
            else:
                Tk = np.asarray(d["Tk"]); y = np.asarray(d[key])
                axes[0][col].plot(Tk * 1000, y, "o-", color=c, lw=1.8, ms=5,
                                  label=f"N = {N}" if col == 0 else None)
                axes[1][col].plot((Tk * 1e6) / DT / N, y, "o-", color=c, lw=1.8, ms=5)
        for row in (0, 1):
            ax = axes[row][col]
            ax.set_xscale("log"); ax.set(ylabel=label, ylim=(0, 1.02))
            ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
            fs.despine(ax)
        axes[0][col].set_xlabel("recording length (ms)")
        axes[1][col].set_xlabel("samples per neuron   T / N")

    axes[0][0].legend(fontsize=9, loc="lower right")
    fig.suptitle("MCC vs.\\ recording length (top) and samples per neuron (bottom), "
                "tuned-AI regime, OLS -- N=5000's gap between 1M and 10M ms is "
                "not yet run",
                fontsize=12.5, color=fs.INK, x=0.06, ha="left", y=0.975)
    fs.save(fig, "figures/fig_mcc_scaling_handoff")


if __name__ == "__main__":
    main()
