"""
R.4 ladder stability figure: rate / CV(ISI) / synchrony held (near-)constant
across a 10x network-size range, N = 1250 -> 12500.

This is the plot-only counterpart of the R.4 regime-tuning campaign
(experiments/ch2_connectivity/r4_tune_regime.py): one regime (g=8, V_reset=10, J fluctuation-scaled
so J*sqrt(C_E)=4.743, eta retuned per size) realised at four sizes so that
later scaling experiments vary N alone. The four points plotted here are the
already-verified cluster measurements recorded in
docs/brunel_network_config_reference.md (section "R.4 scaling ladder",
verified 2026-07-23) -- this script does not re-simulate anything, it only
turns that table into a figure.

Usage:
  python experiments/ch2_connectivity/fig_r4_ladder_stability.py --out figures/fig_r4_ladder_stability
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs

# N, C_E, J_ex, eta, rate (Hz), CV(ISI), synchrony -- verified 2026-07-23,
# 4 s sims / 1 s warm-up / seed 1 (docs/brunel_network_config_reference.md).
LADDER = [
    dict(N=1250,  C_E=100,  J_ex=0.474, eta=1.30, rate=14.8, cv=0.98, sync=0.010),
    dict(N=2500,  C_E=200,  J_ex=0.335, eta=1.50, rate=13.9, cv=0.98, sync=0.008),
    dict(N=5000,  C_E=400,  J_ex=0.237, eta=1.90, rate=14.4, cv=1.03, sync=0.007),
    dict(N=12500, C_E=1000, J_ex=0.150, eta=2.60, rate=13.9, cv=1.06, sync=0.007),
]
TARGET_RATE = 14.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/fig_r4_ladder_stability")
    args = ap.parse_args()

    fs.apply_style()
    import matplotlib.pyplot as plt

    N = np.array([d["N"] for d in LADDER], dtype=float)
    rate = np.array([d["rate"] for d in LADDER])
    cv = np.array([d["cv"] for d in LADDER])
    sync = np.array([d["sync"] for d in LADDER])
    eta = np.array([d["eta"] for d in LADDER])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.16, wspace=0.38)

    ax = axes[0]
    ax.axhspan(TARGET_RATE * 0.9, TARGET_RATE * 1.1, color=fs.GRID, zorder=0,
               label="±10% of 14 Hz")
    ax.plot(N, rate, "o-", color=fs.ACCENT2, lw=1.6, ms=6)
    for x, y, e in zip(N, rate, eta):
        ax.annotate(f"η={e:g}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=fs.MUTED)
    ax.set(xscale="log", xlabel="N", ylabel="firing rate (Hz)", ylim=(0, 20))
    ax.set_title("rate")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[1]
    ax.axhline(1.0, color=fs.GRID, lw=6, zorder=0, solid_capstyle="butt")
    ax.plot(N, cv, "o-", color=fs.C_I, lw=1.6, ms=6)
    ax.set(xscale="log", xlabel="N", ylabel="CV(ISI)", ylim=(0, 1.3))
    ax.set_title("irregularity")

    ax = axes[2]
    ax.plot(N, sync, "o-", color=fs.C_E, lw=1.6, ms=6)
    ax.set(xscale="log", xlabel="N", ylabel="synchrony index", ylim=(0, 0.02))
    ax.set_title("synchrony")

    for ax in axes:
        ax.set_xticks(N)
        ax.set_xticklabels([f"{int(n)}" for n in N])
        ax.grid(True, color=fs.GRID, lw=0.6, which="major")
        ax.set_axisbelow(True)
        fs.despine(ax)

    fig.suptitle("R.4 ladder: one AI regime held across a 10x size range "
                 "(g=8, J·√C_E=4.743, η retuned per N)",
                 fontsize=12, color=fs.INK, x=0.07, ha="left", y=0.97)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
