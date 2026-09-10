"""
R.2 follow-up: overlay the original T=500k-ms camera-rate panel against the
new T=1M-ms rerun (4 focused points), to see directly whether the realistic
(~33ms) camera point's performance moved.

Usage:
  python research/experiments/ch1_proxy/fig_r2_camera_longT_compare.py \
      --baseline ~/calcium_results/fig_data/r2_data_r4.npz \
      --longer ~/calcium_results/fig_data/r2_camera_longT.npz \
      --out figures/fig_R2_camera_longT_compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs

STYLE = {
    ("deconv_rate", "base"): dict(color=fs.ACCENT, marker="o", ls="--", alpha=0.5,
                                   label="deconvolved, T=500k ms (original)"),
    ("deconv_rate", "long"): dict(color=fs.ACCENT, marker="o", ls="-", lw=2.4,
                                   label="deconvolved, T=1M ms (new)"),
    ("raw_rate", "base"): dict(color="#e34948", marker="o", ls="--", alpha=0.5,
                                label="raw calcium, T=500k ms (original)"),
    ("raw_rate", "long"): dict(color="#e34948", marker="o", ls="-", lw=2.4,
                                label="raw calcium, T=1M ms (new)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--longer", required=True)
    ap.add_argument("--out", default="figures/fig_R2_camera_longT_compare")
    args = ap.parse_args()

    fs.apply_style()
    zb = np.load(args.baseline, allow_pickle=False)
    zl = np.load(args.longer, allow_pickle=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.14, wspace=0.25)
    for ax, measure, ylab in zip(axes, ["auc", "corr"], ["ROC-AUC", "correlation"]):
        for key in ("deconv_rate", "raw_rate"):
            ax.plot(zb[f"{key}_x"], zb[f"{key}_{measure}"], **STYLE[(key, "base")])
            ax.plot(zl[f"{key}_x"], zl[f"{key}_{measure}"], **STYLE[(key, "long")])
        ax.axvline(33.0, color=fs.MUTED, lw=1.0, ls=":")
        ax.text(33.0, 0.02, " realistic ~30Hz camera", color=fs.MUTED, fontsize=7.5,
                rotation=90, va="bottom", ha="left")
        ax.set_xscale("log")
        ax.set(xlabel="camera frame interval (ms)", ylabel=ylab, ylim=(0, 1.02))
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)
    axes[0].legend(fontsize=7.8, loc="lower left")
    fig.suptitle("Does a longer recording move the realistic-camera point? "
                 "(dashed=original 500k ms, solid=new 1M ms)",
                 fontsize=12.5, color=fs.INK, x=0.08, ha="left", y=0.97)
    fs.save(fig, args.out)

    # numeric readout at the realistic point
    for key in ("deconv_rate", "raw_rate"):
        for measure in ("auc", "corr"):
            xb, yb = zb[f"{key}_x"], zb[f"{key}_{measure}"]
            xl, yl = zl[f"{key}_x"], zl[f"{key}_{measure}"]
            ib = np.argmin(np.abs(xb - 33.0)); il = np.argmin(np.abs(xl - 33.0))
            print(f"{key:12s} {measure:5s} @33ms:  500k={yb[ib]:.3f}  "
                  f"1M={yl[il]:.3f}  delta={yl[il]-yb[ib]:+.3f}")


if __name__ == "__main__":
    main()
