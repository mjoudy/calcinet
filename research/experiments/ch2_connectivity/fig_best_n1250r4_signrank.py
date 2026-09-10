"""
Sign reliability vs. magnitude, for the OLS estimate on the N=1250 ladder
network (replaces the 4-example "entries" bar chart with a population-wide,
quantitative version of the same claim: the LARGEST estimated entries are
essentially always correctly signed for their source neuron's true type;
sign becomes unreliable only once you're down in the near-zero noise floor).

Reads the same cached best_n1250r4.npz used by fig_best_plot.py.

Usage:
  python research/experiments/ch2_connectivity/fig_best_n1250r4_signrank.py \
      --data ~/calcium_results/best/best_n1250r4.npz \
      --out figures/fig_best_n1250r4_signrank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "calcium_results/best/best_n1250r4.npz"))
    ap.add_argument("--out", default="figures/fig_best_n1250r4_signrank")
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    gt, est = z["full_gt"], z["full_est"]
    n_exc, N = int(z["n_exc_full"]), gt.shape[0]

    expected_sign = np.tile(np.where(np.arange(N) < n_exc, 1.0, -1.0), (N, 1))
    mask = ~np.eye(N, dtype=bool)
    mag = np.abs(est)[mask]
    correct = (np.sign(est) == expected_sign)[mask]
    is_exc_col = np.tile(np.arange(N) < n_exc, (N, 1))[mask]

    order = np.argsort(-mag)
    mag_s, correct_s, isexc_s = mag[order], correct[order], is_exc_col[order]
    n = len(mag_s)

    pcts = np.geomspace(0.0005, 1.0, 60)
    frac_e, frac_i, frac_all = [], [], []
    for p in pcts:
        k = max(1, int(n * p))
        c, ie = correct_s[:k], isexc_s[:k]
        frac_e.append(c[ie].mean() if ie.any() else np.nan)
        frac_i.append(c[~ie].mean() if (~ie).any() else np.nan)
        frac_all.append(c.mean())

    fs.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(0.5, color=fs.GRID, lw=6, zorder=0, solid_capstyle="butt")
    ax.plot(pcts * 100, frac_e, color=fs.C_E, lw=1.8, label="excitatory sources")
    ax.plot(pcts * 100, frac_i, color=fs.C_I, lw=1.8, label="inhibitory sources")
    ax.plot(pcts * 100, frac_all, color=fs.INK, lw=1.2, ls="--", label="all sources")
    ax.set_xscale("log")
    ax.set_xlabel("largest-magnitude estimated entries (top X%, log scale)")
    ax.set_ylabel("fraction with the source's expected sign")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Sign reliability of the OLS estimate, by entry magnitude\n"
                 "(N=1250 ladder network)", fontsize=12, color=fs.INK)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, color=fs.GRID, lw=0.6, which="both")
    ax.set_axisbelow(True)
    fs.despine(ax)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
