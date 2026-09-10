"""
Stage 2b — pure-Lasso vs Elastic-Net L1/L2 sweep at the N=1250 operating point.

Runs a grid of regularizations on ONE fixed reference dataset (s1, tau100, T50k,
lag 2 ms), preprocessing the feed ONCE and re-solving on it, then scores every
setting three ways so the sparsity/precision story is unambiguous:

  1. STANDARD-threshold 2x2 detection confusion (|w| > 1% of max|w|) — the
     project's own rule; TP/FP/FN/TN + precision/recall/F1.
  2. DENSITY-MATCHED 2x2 confusion (keep the top-K |weights|, K = true #edges) ->
     precision == recall, the FAIR cross-method comparison that removes the
     "where is the threshold?" confound.
  3. 3-class confusion (E / none / I) at the standard threshold, for reference.

Grid: OLS + Ridge (dense references, closed form) and a Lasso strength ladder
(lam_l2 = 0) plus two Elastic-Net ratio points — so both questions are answered:
"does pure Lasso help?" and "does the L1/L2 ratio matter?".

Saves every matrix, a metrics CSV/JSON, and a comparison figure.

Run as a SLURM job (needs the reference dataset from SWEEP=prep):
    python research/experiments/ch2_connectivity/run_lasso_sweep.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # repo root

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calcinet.framework.config import ExperimentConfig
from calcinet.framework.runner import run_single
from calcinet.framework import metrics as M
from calcinet.io.dataset import SimulatedDataset
from calcinet.connectivity.solvers.fista import solve as fista


WORKDIR = os.environ.get("CALCIUM_AR_WORKDIR", ".")
SEED, LAG_MS, TAU, T, DT = 1, 2.0, 100.0, 50_000.0, 0.1
OUTDIR = os.path.join(WORKDIR, "results", "n1250_lasso_sweep")

# (label, lam_l1, lam_l2, kind)  kind in {"ols","ridge","fista"}
GRID = [
    ("OLS",             0.0,  0.0,  "ols"),
    ("Ridge",           0.0,  1e-3, "ridge"),
    ("EN L2-dom (now)", 1e-4, 1e-3, "fista"),
    ("EN L1-dom",       1e-3, 1e-4, "fista"),
    ("Lasso 1e-4",      1e-4, 0.0,  "fista"),
    ("Lasso 3e-4",      3e-4, 0.0,  "fista"),
    ("Lasso 1e-3",      1e-3, 0.0,  "fista"),
    ("Lasso 3e-3",      3e-3, 0.0,  "fista"),
    ("Lasso 1e-2",      1e-2, 0.0,  "fista"),
]


def _cfg():
    return ExperimentConfig(
        n_excitatory=1000, n_inhibitory=250, epsilon=0.1, g=5.0, eta=2.0,
        J_ex=0.8, sim_time=T, dt=DT, n_threads=8,
        tau=TAU, smooth_window_ms=3.1, tau_method="ransac",
        solver="fista", lag_ms=LAG_MS, lam=1e-4, lam_l2=1e-3, chunk_size=10_000,
        data_path=os.path.join(WORKDIR, "data", "N1250", f"s{SEED}_tau{int(TAU)}_T{int(T)}"),
        output_dir=os.path.join(OUTDIR, "_feed"), name="lasso_sweep_feed", seed=SEED,
    )


def _gram(feed_zarr, lag, chunk=10_000):
    """Mean-centred per-sample Gram matrices (XtX, XtY) — same accumulation as fista."""
    sig = zarr.open(feed_zarr, "r")
    N, Tt = sig.shape
    Cxx = np.zeros((N, N)); Cyx = np.zeros((N, N))
    sp = np.zeros(N); sn = np.zeros(N); n = 0
    for t0 in range(0, Tt - lag, chunk):
        t1 = min(t0 + chunk, Tt - lag)
        xp = np.asarray(sig[:, t0:t1]); xn = np.asarray(sig[:, t0 + lag:t1 + lag])
        Cxx += xp @ xp.T; Cyx += xn @ xp.T
        sp += xp.sum(1); sn += xn.sum(1); n += xp.shape[1]
    mp = sp / n; mn = sn / n
    XtX = (Cxx - n * np.outer(mp, mp)) / n
    XtY = (Cyx - n * np.outer(mn, mp)) / n
    return XtX, XtY


def solve_one(kind, l1, l2, feed_zarr, lag):
    if kind == "fista":
        return fista(feed_zarr, lag=lag, lam_l1=l1, lam_l2=l2, n_iter=500, chunk_size=10_000)
    XtX, XtY = _gram(feed_zarr, lag)           # OLS / Ridge in closed form (exact, dense)
    N = XtX.shape[0]
    return XtY @ np.linalg.inv(XtX + l2 * np.eye(N))


def score(A, adj, true_edges, off, y_true):
    A = A.copy(); np.fill_diagonal(A, 0.0)
    # standard threshold
    yt, yp = M._binary_labels(A, adj)
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p and r and (p + r) > 0 else float("nan")
    # density-matched threshold (precision == recall)
    s = np.abs(A[off]); order = np.argsort(s)[::-1]
    dpred = np.zeros_like(y_true); dpred[order[:true_edges]] = 1
    dtp = int(((dpred == 1) & (y_true == 1)).sum()); dfp = int(((dpred == 1) & (y_true == 0)).sum())
    dp = dtp / (dtp + dfp) if dtp + dfp else float("nan")
    return dict(
        support=int((np.abs(A[off]) > 0).sum()), auc=float(M.auc_roc(A, adj)),
        tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=r, f1=f,
        dens_prec=dp, dens_recall=dp,
    )


def render(rows, out):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.1, 1, 1], hspace=0.55, wspace=0.4)

    def draw2x2(ax, r, title):
        tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]
        ax.imshow([[tp, fp], [fn, tn]], cmap="Blues")
        labs = [["TP", "FP"], ["FN", "TN"]]; vals = [[tp, fp], [fn, tn]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labs[i][j]}\n{vals[i][j]}", ha="center", va="center", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["true edge", "true none"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["pred edge", "pred none"], fontsize=8)
        ax.set_title(title, fontsize=9)

    show = [r for r in rows if r["label"] in
            ("EN L2-dom (now)", "Lasso 3e-4", "Lasso 1e-3", "Lasso 3e-3", "Lasso 1e-2")]
    for j, r in enumerate(show):
        ax = fig.add_subplot(gs[0, j])
        draw2x2(ax, r, f"{r['label']}\nP={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}")
    fig.text(0.5, 0.965, "2x2 detection confusion at standard threshold (|w|>1% max)",
             ha="center", fontsize=12, weight="bold")

    lassos = sorted([r for r in rows if r["label"].startswith("Lasso")], key=lambda r: r["l1"])
    l1s = [r["l1"] for r in lassos]
    axc = fig.add_subplot(gs[1, 0:2])
    for key, mk in [("precision", "o-"), ("recall", "s-"), ("f1", "^-")]:
        axc.plot(l1s, [r[key] for r in lassos], mk, label=key)
    axc.set_xscale("log"); axc.set_ylim(0, 1.05); axc.grid(alpha=0.3)
    axc.set_xlabel("pure-Lasso L1 strength"); axc.legend(fontsize=8)
    axc.set_title("Pure Lasso: precision/recall/F1 vs L1", fontsize=10)

    axs = fig.add_subplot(gs[1, 2:4])
    axs.plot(l1s, [r["support"] for r in lassos], "o-", color="purple")
    axs.axhline(rows[0]["true_edges"], ls="--", color="green",
                label=f"true # edges ({rows[0]['true_edges']})")
    axs.set_xscale("log"); axs.set_yscale("log"); axs.grid(alpha=0.3)
    axs.set_xlabel("pure-Lasso L1 strength"); axs.set_ylabel("support (nnz)")
    axs.legend(fontsize=8); axs.set_title("Lasso self-selected sparsity vs true", fontsize=10)

    axa = fig.add_subplot(gs[1, 4])
    sr = sorted(rows, key=lambda r: -r["auc"])
    axa.barh([r["label"] for r in sr], [r["auc"] for r in sr], color="steelblue")
    axa.set_xlim(0.5, 0.95); axa.invert_yaxis(); axa.set_xlabel("AUC")
    axa.set_title("ranking quality (AUC)", fontsize=10); axa.tick_params(labelsize=7)

    axd = fig.add_subplot(gs[2, :])
    labels = [r["label"] for r in rows]; dp = [r["dens_prec"] for r in rows]
    colors = ["#c44" if l.startswith("Lasso") else "#888" for l in labels]
    b = axd.bar(labels, dp, color=colors)
    for rect, v in zip(b, dp):
        axd.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    axd.set_ylim(0, max(dp) * 1.2 if dp else 1); axd.set_ylabel("precision = recall")
    axd.set_title("FAIR comparison: precision at density-matched threshold. "
                  "Grey = dense/has-L2 · Red = pure Lasso. Higher = better.", fontsize=11)
    axd.tick_params(axis="x", labelsize=8)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[lasso_sweep] python: {sys.executable}")
    cfg = _cfg()
    if not os.path.exists(cfg.data_path):
        sys.exit(f"dataset missing: {cfg.data_path}\n-> run SWEEP=prep first.")

    # preprocess ONCE (EN run just to produce the feed zarr)
    result = run_single(cfg)
    feed_zarr = result.feed_zarr_path
    lag = round(LAG_MS / DT)

    ds = SimulatedDataset.load_or_generate(cfg, cfg.data_path)
    adj = np.asarray(ds.adj_true).copy(); np.fill_diagonal(adj, 0.0)
    N = adj.shape[0]; off = ~np.eye(N, dtype=bool)
    y_true = (adj.T[off] != 0).astype(int)
    true_edges = int(y_true.sum())
    print(f"[lasso_sweep] N={N}, true edges={true_edges}/{off.sum()} "
          f"(density {true_edges / off.sum():.3f}), lag={lag} samples")

    rows = []
    for label, l1, l2, kind in GRID:
        A = solve_one(kind, l1, l2, feed_zarr, lag)
        np.save(os.path.join(OUTDIR, f"A_{label.replace(' ', '_').replace('(', '').replace(')', '')}.npy"), A)
        row = dict(label=label, l1=l1, l2=l2, kind=kind, true_edges=true_edges,
                   **score(A, adj, true_edges, off, y_true))
        rows.append(row)
        print(f"  {label:16s} supp={row['support']:6d} auc={row['auc']:.2f} | "
              f"std P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.2f} | "
              f"dens P=R={row['dens_prec']:.2f}")

    fields = ["label", "l1", "l2", "kind", "support", "auc", "tp", "fp", "fn", "tn",
              "precision", "recall", "f1", "dens_prec", "dens_recall"]
    with open(os.path.join(OUTDIR, "lasso_sweep_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    json.dump(rows, open(os.path.join(OUTDIR, "lasso_sweep_metrics.json"), "w"), indent=1)

    fig_path = os.path.join(OUTDIR, "lasso_sweep.png")
    render(rows, fig_path)
    print(f"[lasso_sweep] DONE -> {fig_path}")


if __name__ == "__main__":
    main()
