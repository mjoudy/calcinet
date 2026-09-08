"""
Memory-safe version of regime_probe.py's multi-config comparison figure.

regime_probe.py's probe() always densifies the full (N, T) spike matrix before
computing anything (via BrunelNetwork.run()'s default densify=True) -- at
N=12500 that is several GB, and running three configs back to back exhausted
this machine's RAM. This script computes every statistic and every plotted
quantity (raster subset, population-rate trace, ISI sample) directly from the
SPARSE spike-event list (net.run(densify=False); net.get_spike_events()), the
same approach already used in r4_tune_regime.py / check_cached_regime.py, so
memory scales with the number of spikes, not with N*T.

The rendering step (make_figure) is reused UNCHANGED from regime_probe.py --
only the data-collection step (probe -> sparse_probe) is replaced.

Usage:
  python experiments/ch2_connectivity/fig_brunel_regime_comparison.py --sim-time 10000 \
      --only brunel_SI_fig8C brunel_AI_fig8B brunel_AI_lowrate \
      --fig figures/fig_brunel_regime_comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calcinet.simulation.brunel_network import BrunelNetwork
sys.path.insert(0, str(Path(__file__).resolve().parent))  # siblings
from regime_probe import CONFIGS_BY_SCALE, make_figure


def sparse_probe(cfg, n_exc, n_inh, sim_time, dt, warmup_ms, n_threads, seed=1,
                  window_ms=1000.0, n_show=200, isi_sample=300, sync_sub=200):
    N = n_exc + n_inh
    net = BrunelNetwork(n_excitatory=n_exc, n_inhibitory=n_inh, epsilon=0.1,
                        g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
                        V_reset=cfg["V_reset"], delay=1.5,
                        sim_time=sim_time, dt=dt, n_threads=n_threads, seed=seed)
    net.build()
    net.run(densify=False)                       # <-- the actual fix
    idx, times = net.get_spike_events()           # sparse: (n_spikes,), (n_spikes,)
    del net

    keep = times >= warmup_ms
    idx, times = idx[keep], times[keep] - warmup_ms
    dur_s = (sim_time - warmup_ms) / 1000.0

    is_exc = idx < n_exc
    rate_E = is_exc.sum() / (n_exc * dur_s)
    rate_I = (~is_exc).sum() / (n_inh * dur_s)
    rate = len(times) / (N * dur_s)

    # ISI / CV from a subsample of neurons, full post-warmup duration
    order = np.lexsort((times, idx))
    i_s, t_s = idx[order], times[order]
    starts = np.flatnonzero(np.r_[True, np.diff(i_s) != 0])
    ends = np.r_[starts[1:], len(i_s)]
    neuron_of_run = i_s[starts]
    step = max(1, len(starts) // isi_sample)
    cvs, isi_list = [], []
    for k in range(0, len(starts), step):
        s, e = starts[k], ends[k]
        if e - s > 2:
            isi = np.diff(t_s[s:e])
            mu = isi.mean()
            if mu > 0:
                cvs.append(isi.std() / mu)
                isi_list.append(isi)
    cv_mean = float(np.mean(cvs)) if cvs else float("nan")
    silent = 1.0 - len(starts) / N
    isis = np.concatenate(isi_list) if isi_list else np.array([1.0])

    # synchrony: pairwise spike-count correlation on a random subset, 5 ms bins
    rng = np.random.default_rng(seed)
    sub = rng.choice(N, size=min(sync_sub, N), replace=False)
    pos = -np.ones(N, dtype=np.int64); pos[sub] = np.arange(len(sub))
    bin_ms = 5.0
    nb = max(2, int(dur_s * 1000.0 / bin_ms))
    counts = np.zeros((len(sub), nb))
    sel = pos[idx] >= 0
    b = np.clip((times[sel] / bin_ms).astype(np.int64), 0, nb - 1)
    np.add.at(counts, (pos[idx[sel]], b), 1.0)
    active = counts.sum(1) > 0
    if active.sum() > 2:
        C = np.corrcoef(counts[active])
        off = ~np.eye(C.shape[0], dtype=bool)
        sync = float(np.nanmean(C[off]))
    else:
        sync = float("nan")

    stats = dict(rate_E=float(rate_E), rate_I=float(rate_I), rate=float(rate),
                 cv=cv_mean, sync=sync, silent=float(silent))

    # --- payload for make_figure(): raster subset, pop-rate trace, ISI sample -
    win_ms = min(window_ms, sim_time - warmup_ms)
    n_bins = int(round(win_ms / dt))
    nE_show = int(round(n_show * n_exc / N))
    nI_show = n_show - nE_show
    show_idx = np.r_[np.arange(nE_show), np.arange(n_exc, n_exc + nI_show)]
    pos_show = -np.ones(N, dtype=np.int64); pos_show[show_idx] = np.arange(len(show_idx))

    in_win = times < win_ms
    t_win, idx_win = times[in_win], idx[in_win]
    raster = np.zeros((len(show_idx), n_bins), dtype=bool)
    sel2 = pos_show[idx_win] >= 0
    rows = pos_show[idx_win[sel2]]
    cols = np.clip((t_win[sel2] / dt).astype(np.int64), 0, n_bins - 1)
    raster[rows, cols] = True

    b5 = max(1, int(round(5.0 / dt)))
    nb5 = n_bins // b5
    pop_bins = np.clip((t_win / dt).astype(np.int64), 0, n_bins - 1)
    pop_hist = np.bincount(pop_bins, minlength=n_bins)[: nb5 * b5].reshape(nb5, b5).sum(1)
    pr = pop_hist / (N * b5 * dt / 1000.0)

    payload = dict(raster=raster, nE_show=nE_show, pr=pr, dt=dt,
                   window_ms=n_bins * dt, isis=isis)
    return stats, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="n12500", choices=list(CONFIGS_BY_SCALE))
    ap.add_argument("--n-exc", type=int, default=None)
    ap.add_argument("--n-inh", type=int, default=None)
    ap.add_argument("--sim-time", type=float, default=10000.0)
    ap.add_argument("--warmup-ms", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--window-ms", type=float, default=800.0)
    args = ap.parse_args()

    defaults = {"n12500": (10000, 2500), "n1250": (1000, 250)}[args.scale]
    args.n_exc = args.n_exc or defaults[0]
    args.n_inh = args.n_inh or defaults[1]
    CONFIGS = CONFIGS_BY_SCALE[args.scale]
    if args.only:
        CONFIGS = [c for c in CONFIGS if c[0] in set(args.only)]
        if not CONFIGS:
            raise SystemExit(f"no config matched {args.only}")

    print(f"scale={args.scale}  N={args.n_exc + args.n_inh}  sim_time={args.sim_time:.0f} ms "
          f"(sparse, densify=False)\n", flush=True)
    results = []
    for name, cfg in CONFIGS:
        st, pl = sparse_probe(cfg, args.n_exc, args.n_inh, args.sim_time, args.dt,
                              args.warmup_ms, args.n_threads, window_ms=args.window_ms)
        print(f"{name:22s} rate={st['rate']:6.1f} Hz  E={st['rate_E']:6.1f}  "
              f"I={st['rate_I']:6.1f}  CV={st['cv']:5.2f}  sync={st['sync']:6.3f}  "
              f"silent={st['silent']:5.2f}", flush=True)
        results.append((name, cfg, st, pl))

    make_figure(results, args.fig, args.n_exc + args.n_inh)


if __name__ == "__main__":
    main()
