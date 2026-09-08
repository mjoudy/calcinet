"""
R.4 (data length x network size) — the scaling law, plotted LOCALLY.

Reads the small metrics.csv files produced on the cluster (one per size x
recording length) and draws the PRIMARY figure as one column: measure vs
recording length -> bigger N needs more data. (The old "vs T/N, curves
collapse" column was dropped from this figure per professor's feedback — it
was somewhat trivially built-in by the sweep design, matched samples-per-neuron
ratios at every N, rather than a discovered collapse.)

rows = correlation, excitatory recall, excitatory precision.

A SECONDARY appendix figure (--out-tn) revisits the "vs T/N" axis, but now to
directly test a specific question rather than just demo the collapse: does
matching samples-per-neuron erase the size gap for ALL three measures, or only
for correlation? (Prediction: correlation is a variance problem and should
collapse; excitatory recall/precision are a bias problem driven by raw N
(shared-input confounding scales with in-degree C_E=eps*N), so they should
stay separated by N even at matched T/N.) Zero new compute — same cached CSVs.

All four sizes share one AI regime (13.9-14.8 Hz, CV 0.98-1.06, synchrony
0.007-0.010), so N is the only variable. OLS only.

Usage:
  python experiments/ch2_connectivity/fig_r4_plot.py --root /home/mjoudy/calcium_results/hpc_metrics \
      --out figures/fig_R4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs

DT = 0.1  # ms per sample

# N -> directory prefix (the R.4 ladder; N=12500 reuses the low-rate A2 run)
LADDER = [(1250, "wrapup_n1250r4_T"), (2500, "wrapup_n2500r4_T"),
          (5000, "wrapup_n5000r4_T"), (12500, "wrapup_n12500r4_T")]
MEASURES = [("corr", "correlation"), ("E_rec", "excitatory recall"),
            ("E_prec", "excitatory precision"), ("roc_auc", "ROC-AUC (connected vs. none)"),
            ("pr_ap", "PR-AUC (connected vs. none)")]

# Canonical checkpoint grid, ms -- every N is meant to land on exactly these
# ten points (2026-08-14 decision: unify to N=1250's original grid rather than
# a coarser common one, since checkpointing within one run is free once the
# run's max length is fixed). Any directory whose T isn't on this list is a
# leftover exploratory point (e.g. the old single-seed 400k/800k probes at
# N=2500/5000) and is deliberately EXCLUDED here rather than plotted alongside
# properly-seeded points.
GRID_MS = [1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 7.5e6, 1e7, 1.5e7, 2e7]


def load_pooled_extras(root: Path, method="ols"):
    """-> {N: (T_ms, {measure: value})}

    A `*_pooled` directory (several short, independently-seeded chunks whose
    moments were summed together, then solved and scored once) is a single
    point estimate at a non-standard T -- not five independent network
    realizations at a GRID_MS checkpoint. Kept entirely separate from load()
    so it can never silently blend into the 5-seed ladder's mean/std; the
    caller is responsible for marking it visually distinct.
    """
    import json
    out = {}
    for N, prefix in LADDER:
        for d in sorted(root.glob(f"{prefix}*_pooled")):
            f = d / "metrics.json"
            if not f.exists():
                continue
            name = d.name[len(prefix):-len("_pooled")]  # e.g. "12000k"
            per_chunk_T_ms = float(name.rstrip("k")) * 1000.0
            m = json.loads(f.read_text())
            # pool_chunks_and_solve.py averages n_chunks already-per-sample-
            # normalized covariances, each from an INDEPENDENT chunk of the
            # SAME length per_chunk_T_ms -- by its own docstring, that is
            # statistically equivalent to one recording of n_chunks times
            # that length, not one recording of per_chunk_T_ms alone.
            T_ms = per_chunk_T_ms * m.get("n_chunks", 1)
            out[N] = (T_ms, {k: m[k] for k, _ in MEASURES if k in m})
    return out


def load(root: Path, method="ols"):
    """-> {N: (T_ms[], {measure: (mean[], std[], n_seeds[])})}

    Aggregates across every seed<k>/ row in metrics.csv at each recording
    length (analyze_run.py already writes one row per seed x method, so this
    is just grouping + mean/std -- no new compute needed). n_seeds=1 -> std=0,
    so old single-seed data still plots fine, just without a visible band.
    Only T values on GRID_MS are kept -- see the comment there.
    """
    out = {}
    for N, prefix in LADDER:
        by_T = {}
        for d in sorted(root.glob(f"{prefix}*k")):
            f = d / "metrics.csv"
            if not f.exists():
                continue
            T_ms = float(d.name.rsplit("_T", 1)[1].rstrip("k")) * 1000.0
            if not any(abs(T_ms - g) < 1.0 for g in GRID_MS):
                continue
            bucket = by_T.setdefault(T_ms, {k: [] for k, _ in MEASURES})
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    if row["method"] == method:
                        for k, _ in MEASURES:
                            bucket[k].append(float(row[k]))
        if by_T:
            Ts = sorted(by_T)
            T = np.array(Ts)
            vals = {}
            for k, _ in MEASURES:
                arrs = [np.array(by_T[t][k]) for t in Ts]
                vals[k] = (np.array([a.mean() for a in arrs]),
                          np.array([a.std(ddof=1) if len(a) > 1 else 0.0 for a in arrs]),
                          np.array([len(a) for a in arrs]))
            out[N] = (T, vals)
        else:
            print(f"[warn] no metrics for N={N} ({prefix}*)")
    return out


def plot_grid(data, colors, xfunc, xlabel, title, extras=None):
    """xfunc(N, T_ms array) -> x-values array. One figure, len(MEASURES) rows.
    Spread across seeds is shown as a shaded +-1 SD band, not error-bar caps
    (invisible when n_seeds=1, since std=0 there).

    extras (optional): {N: (T_ms, {measure: value})} -- a single pooled-chunk
    point per N, drawn with a dashed connector and a distinct open-diamond
    marker so it reads as a different KIND of estimate (one pooled solve, no
    seed-to-seed spread available) rather than a sixth point on the same
    5-seed ladder."""
    # 3x2 grid: 5 measures fill 5 cells row-major, the 6th (spare) cell holds
    # the legend instead of overlapping it onto a data panel.
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.06,
                        hspace=0.32, wspace=0.24)
    flat = axes.flat
    for r, (key, label) in enumerate(MEASURES):
        ax = flat[r]
        for (N, (T, vals)), c in zip(sorted(data.items()), colors):
            mean, std, n = vals[key]
            x = xfunc(N, T)
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.18, lw=0)
            ax.plot(x, mean, "o-", color=c, lw=1.9, ms=6, label=f"N = {N}")
            if extras and N in extras:
                Te, me = extras[N]
                if key in me:
                    xe = xfunc(N, np.array([Te]))[0]
                    ax.plot([x[-1], xe], [mean[-1], me[key]], "--", color=c, lw=1.3)
                    ax.plot([xe], [me[key]], "D", color=c, ms=7,
                            mfc="white", mew=1.6)
        ax.set_xscale("log")
        ax.set(xlabel=xlabel, ylabel=label, ylim=(0, 1.02))
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
        fs.despine(ax)
    handles, labels = flat[0].get_legend_handles_labels()
    if extras:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], ls="--", color=fs.MUTED, marker="D",
                              mfc="white", mec=fs.MUTED, ms=7,
                              label="pooled 7-chunk estimate (no seed spread)"))
    legend_ax = flat[len(MEASURES)]   # the spare 6th cell
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, fontsize=10.5, loc="center", frameon=False)
    fig.suptitle(title, fontsize=13, color=fs.INK, x=0.05, ha="left", y=0.975)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/mjoudy/calcium_results/hpc_metrics")
    ap.add_argument("--out", default="figures/fig_R4")
    ap.add_argument("--out-tn", default="figures/fig_R4_TN",
                    help="appendix figure: same measures vs samples-per-neuron T/N")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fs.apply_style()
    data = load(Path(args.root))
    if not data:
        raise SystemExit(f"no metrics.csv found under {args.root}")
    extras = load_pooled_extras(Path(args.root))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(data)))

    # ---- primary: vs recording length (ms) --------------------------------- #
    title = args.title or ("Scaling: recovery is set by data per neuron "
                           "(matched AI regime, ~14 Hz, OLS)")
    fig = plot_grid(data, colors, lambda N, T: T, "recording length (ms)", title,
                    extras=extras)
    fs.save(fig, args.out)

    # ---- appendix: vs samples per neuron T/N -------------------------------- #
    title_tn = ("Same measures at matched samples-per-neuron (T/N) — "
               "does the size gap survive?")
    fig2 = plot_grid(data, colors, lambda N, T: (T / DT) / N,
                     "samples per neuron   T / N", title_tn, extras=extras)
    fs.save(fig2, args.out_tn)


if __name__ == "__main__":
    main()
