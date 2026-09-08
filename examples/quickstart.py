#!/usr/bin/env python3
"""
calcinet quickstart — the whole two-stage pipeline on a tiny synthetic network.

Runs in seconds on a laptop CPU. No cluster, no downloads, no NEST.

    python examples/quickstart.py

What it does, end to end:

    known connectivity  ->  spikes (Hawkes)  ->  calcium fluorescence
        ->  spike proxy (stage 1: tau estimate + deconvolution)
        ->  connectivity estimate (stage 2: chunked ridge on the proxy)
        ->  metrics, a provenance record, a ledger row, and one figure

Outputs (all under examples/output/, git-ignored):
    quickstart.png   ground truth vs estimate, and the signal chain
    ledger.csv       one row: config + metrics + git commit + dirty flag
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from calcinet.connectivity.solvers import solve_ridge
from calcinet.framework.config import ExperimentConfig
from calcinet.framework.ledger import append_row
from calcinet.framework.metrics import compute_all
from calcinet.framework.provenance import git_revision
from calcinet.framework.result import ExperimentResult
from calcinet.proxy.signal_utils import get_signal_derivative_pair
from calcinet.proxy.tau_estimation import estimate_tau_robust
from calcinet.proxy.feed_reconstruction import reconstruct_feed
from calcinet.simulation.calcium_signal import simulate_calcium
from calcinet.simulation.hawkes_ground_truth import build_G, simulate_hawkes

OUT = Path(__file__).resolve().parent / "output"

# Small enough to run in seconds, large enough that the estimate means something.
N_EXC, N_INH = 32, 8
N            = N_EXC + N_INH
EPSILON      = 0.2          # connection probability
J_EXC, G     = 0.6, 4.0     # excitatory weight and inhibition ratio
SIM_TIME_MS  = 120_000.0
DT_MS        = 0.5          # fine enough that the decay is visible above the noise
SMOOTH_WIN   = 31           # Savitzky-Golay window (samples) for the derivative
TAU_MS       = 100.0        # true calcium decay constant
LAG_MS       = 10.0
SEED         = 7


def make_connectivity(rng: np.random.Generator) -> np.ndarray:
    """Sparse Dale-obeying weights. Convention: adj[source, target]."""
    adj = np.zeros((N, N))
    mask = rng.random((N, N)) < EPSILON
    np.fill_diagonal(mask, False)
    adj[mask] = J_EXC
    adj[N_EXC:, :] *= -G          # inhibitory rows flip sign
    adj[N_EXC:, :][~mask[N_EXC:, :]] = 0.0
    return adj


def bin_spikes(idx: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
    """Event list -> (N, T) binary matrix on the dt grid."""
    n_bins = int(SIM_TIME_MS / DT_MS)
    spikes = np.zeros((N, n_bins))
    keep = (times_ms >= 0) & (times_ms < SIM_TIME_MS)
    bins = (times_ms[keep] / DT_MS).astype(int)
    np.add.at(spikes, (idx[keep].astype(int), bins), 1.0)
    return np.clip(spikes, 0.0, 1.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # ---- ground truth + spikes -------------------------------------------
    adj_true = make_connectivity(rng)
    G_mat, info = build_G(adj_true, margin=0.4)
    idx, times_ms, stats = simulate_hawkes(
        G_mat, y0_hz=4.0, tau_syn_ms=10.0, sim_time_ms=SIM_TIME_MS, seed=SEED
    )
    spikes = bin_spikes(idx, times_ms)
    rate_hz = spikes.sum() / N / (SIM_TIME_MS / 1000.0)
    print(f"[1/5] {int(spikes.sum())} spikes, mean rate {rate_hz:.1f} Hz")

    # ---- calcium ----------------------------------------------------------
    calcium = simulate_calcium(spikes, tau=TAU_MS, dt=DT_MS, amplitude=1.0,
                               sigma_intra=0.01, sigma_extra=0.05, seed=SEED)
    print(f"[2/5] calcium traces {calcium.shape}")

    # ---- stage 1: spike proxy --------------------------------------------
    tau_est = estimate_tau_robust(calcium, window_length=SMOOTH_WIN, method="ransac", dt=DT_MS)
    smooth, deriv = get_signal_derivative_pair(calcium, window_length=SMOOTH_WIN, delta=DT_MS)
    feed = reconstruct_feed(smooth, deriv, tau_est)
    tau_mean = float(np.mean(np.atleast_1d(tau_est)))
    print(f"[3/5] tau: true {TAU_MS:.0f} ms, estimated {tau_mean:.0f} ms")

    # ---- stage 2: connectivity -------------------------------------------
    zarr_path = OUT / "feed.zarr"
    z = zarr.open(str(zarr_path), mode="w", shape=feed.shape,
                  chunks=(N, 4096), dtype="f8")
    z[:] = feed
    lag = max(1, int(LAG_MS / DT_MS))
    adj_hat = solve_ridge(zarr_path=str(zarr_path), lag=lag, lam=1.0, chunk_size=8192)
    print(f"[4/5] estimated connectivity {adj_hat.shape}")

    # ---- score, record, plot ---------------------------------------------
    np.fill_diagonal(adj_true, 0.0)
    metrics = compute_all(adj_inferred=adj_hat, adj_true=adj_true,
                          tau_est=tau_est, tau_true=TAU_MS)
    rev = git_revision()

    config = ExperimentConfig(
        n_excitatory=N_EXC, n_inhibitory=N_INH, sim_time=SIM_TIME_MS, dt=DT_MS,
        epsilon=EPSILON, g=G, J_ex=J_EXC, tau=TAU_MS, solver="ridge",
        lag_ms=LAG_MS, lam=1.0, name="quickstart", seed=SEED,
        output_dir=str(OUT),
    )
    result = ExperimentResult(
        config_path=str(OUT / "config.json"), metrics=metrics, loss_curve=[],
        duration_seconds=time.time() - t0,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        run_dir=str(OUT), spikes_path=None, calcium_path=None,
        feed_zarr_path=str(zarr_path), adj_true_path="", adj_inferred_path="",
        git_commit=rev["git_commit"], git_dirty=rev["git_dirty"],
    )
    ledger_path = append_row(result, config, ledger_path=OUT / "ledger.csv")

    corr = metrics.get("connectivity/pearson", float("nan"))
    auc  = metrics.get("connectivity/auc_roc", metrics.get("connectivity/auc", float("nan")))
    print(f"[5/5] pearson {corr:.3f}   auc {auc:.3f}   "
          f"git {rev['git_commit'][:8]} (dirty={rev['git_dirty']})")

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    t = slice(0, 6000)
    ax[0, 0].plot(spikes[0, t], lw=0.6);          ax[0, 0].set_title("spikes (neuron 0)")
    ax[0, 1].plot(calcium[0, t], lw=0.6);         ax[0, 1].set_title("calcium fluorescence")
    ax[0, 2].plot(feed[0, t], lw=0.6);            ax[0, 2].set_title("recovered spike proxy")
    for a in ax[0]:
        a.set_xlabel("time (ms)"); a.margins(x=0)
    v = np.abs(adj_true).max()
    ax[1, 0].imshow(adj_true.T, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[1, 0].set_title("ground truth (transposed)")
    shown = adj_hat.copy()
    np.fill_diagonal(shown, 0.0)   # self-terms would dominate the colour scale
    w = np.percentile(np.abs(shown), 99)
    ax[1, 1].imshow(shown, cmap="RdBu_r", vmin=-w, vmax=w)
    ax[1, 1].set_title("estimate")
    off = ~np.eye(N, dtype=bool)
    ax[1, 2].scatter(adj_true.T[off], adj_hat[off], s=3, alpha=0.3)
    ax[1, 2].set_xlabel("true weight"); ax[1, 2].set_ylabel("estimated")
    ax[1, 2].set_title(f"pearson r = {corr:.3f}")
    fig.suptitle("calcinet quickstart — calcium to directed connectivity", y=0.99)
    fig.tight_layout()
    fig_path = OUT / "quickstart.png"
    fig.savefig(fig_path, dpi=130)
    print(f"\nwrote {fig_path}\nwrote {ledger_path}")


if __name__ == "__main__":
    main()
