"""
R.2 follow-up — does TUNING the deconvolution's own smoothing window let it
reach the bin-size sweet spot, instead of just sitting near it by chance?

fig_R2's "binned spikes (reference)" panel showed a sweet spot in spike bin
size (~5 ms). We then marked where the deconvolved feed's fixed smoothing
window (SMOOTH_MS=3.1) falls on that same axis. This script actually SWEEPS
the smoothing window (the deconvolution's own "bin size" knob) and asks: does
its best point land near/on the reference curve's peak?

Camera + dye tau are held FIXED at a fast, floor-safe setting (cam=1ms,
tau=100ms) so the window itself is the only thing varying (fig_r2_compute's
smooth_win() floors the window at 5 samples -- a slow camera would make the
achievable window jump in big, confounding steps; 1ms keeps 5 samples <=5ms,
well below the smallest window tested).

Reuses fig_r2_compute.py's simulation + deconvolution + scoring code
directly (monkeypatches its module-level SMOOTH_MS before each call) so the
preprocessing math is identical to the rest of R.2, not a reimplementation.

Usage:
  python experiments/ch1_proxy/fig_r2_smooth_compute.py --out results/fig_data/r2_smooth_data.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "shared"))
import fig_r2_compute as r2c
from calcinet.simulation.brunel_network import BrunelNetwork
from wrapup_run import build_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="n1250ai")
    ap.add_argument("--T-ms", type=float, default=100_000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--cam-ms", type=float, default=1.0,
                    help="fixed camera frame interval (fast, floor-safe)")
    ap.add_argument("--tau-ms", type=float, default=100.0, help="fixed dye tau")
    ap.add_argument("--windows", type=float, nargs="+",
                    default=[0.5, 1, 2, 3.1, 5, 10, 20, 33, 50, 100])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = build_cfg(args.net); dt = cfg["dt"]
    print(f"simulating {args.net} for {args.T_ms:.0f} ms ...", flush=True)
    net = BrunelNetwork(n_excitatory=cfg["n_excitatory"], n_inhibitory=cfg["n_inhibitory"],
                        epsilon=cfg["epsilon"], g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
                        delay=cfg["delay"], V_reset=cfg["V_reset"], sim_time=args.T_ms,
                        dt=dt, n_threads=cfg["n_threads"], seed=args.seed)
    net.build(); net.run(densify=False)
    idx, tms = net.get_spike_events(); idx = idx.astype(np.int64)
    adj = net.get_adjacency(); np.fill_diagonal(adj, 0.0)
    N = adj.shape[0]

    # raw calcium reference (doesn't depend on the smoothing window at all)
    rng = np.random.default_rng(args.seed)
    Cxx, Cyx = r2c.accumulate(
        r2c.iter_calcium(idx, tms, N, dt, args.tau_ms, args.T_ms, False,
                         args.cam_ms, args.chunk, rng),
        N, max(1, round(r2c.LAG_MS / args.cam_ms)))
    raw_auc, raw_corr = r2c.score(Cxx, Cyx, adj)
    print(f"  raw calcium (fixed)  AUC={raw_auc:.3f}  corr={raw_corr:.3f}", flush=True)

    ws, aucs, corrs = [], [], []
    for w in sorted(args.windows):
        r2c.SMOOTH_MS = w                       # the knob under test
        rng = np.random.default_rng(args.seed)
        Cxx, Cyx = r2c.accumulate(
            r2c.iter_calcium(idx, tms, N, dt, args.tau_ms, args.T_ms, True,
                             args.cam_ms, args.chunk, rng),
            N, max(1, round(r2c.LAG_MS / args.cam_ms)))
        auc, corr = r2c.score(Cxx, Cyx, adj)
        ws.append(w); aucs.append(auc); corrs.append(corr)
        print(f"  window={w:<6g} ms   AUC={auc:.3f}  corr={corr:.3f}", flush=True)

    np.savez(args.out, net=args.net, N=N, T_ms=args.T_ms,
            cam_ms=args.cam_ms, tau_ms=args.tau_ms,
            window_ms=np.array(ws), auc=np.array(aucs), corr=np.array(corrs),
            raw_auc=raw_auc, raw_corr=raw_corr)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
