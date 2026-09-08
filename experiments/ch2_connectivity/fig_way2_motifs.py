"""
Way 2 extended — motif-based exposure variables, following the Pernice (2011)
correlation-as-sum-of-motifs decomposition.

Original Way 2 (fig_way2.py) bins false-positive strength by the number of
DIRECT (1-hop) common presynaptic drivers, split observed vs hidden. That's
one term (g^(1,1)) of an infinite motif series. This script adds:

  1. 1-HOP count   — same as fig_way2.py, kept for comparison, axis fixed to
                      % change from each curve's own baseline (the raw
                      magnitude axis made the two curves hard to compare).
  2. 2-HOP strength — indirect common drivers, WEIGHTED (not just present/
                      absent): raw 2-hop reachability turned out saturated at
                      this density (~9.5%, N=1250) — almost every pair reaches
                      almost every other pair within 2 hops, so a binary count
                      carries no signal. Use the actual weighted 2-hop
                      influence G2 = adj @ adj (Pernice's G^2 term) instead,
                      same sum_k G2[k,i]*G2[k,j] construction as the weighted
                      1-hop metric below.
  3. WEIGHTED strength — sum_k w_ki * w_kj over common drivers instead of a
                      raw count (a driver's actual synaptic weight, not just
                      its presence), split observed vs hidden.
  4. CHAIN-MOTIF check — some "false positives" may not be shared-input
                      artifacts at all: if a real directed 2-hop path exists
                      between the pair in the TRUE adjacency (i->k->j or
                      j->k->i), the "confound" could be a genuine polysynaptic
                      effect. Path PRESENCE is saturated at this density (see
                      above), so this compares path STRENGTH (|G2[i,j]| +
                      |G2[j,i]|) for false positives vs. random non-edges,
                      not presence/absence.

All from cached moments (Cxx, Cyx, adj_true) — local, seconds, no GPU.

Usage:
  python experiments/ch2_connectivity/fig_way2_motifs.py --data ~/calcium_results/best_moments/n1250r4 \
      --out figures/fig_way2_motifs_n1250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import figstyle as fs


def ols_block(Cxx, Cyx, S):
    ix = np.ix_(S, S)
    A = Cyx[ix] @ np.linalg.inv(Cxx[ix] + 1e-9 * np.eye(len(S)))
    np.fill_diagonal(A, 0.0)
    return A


def binned_pct(count, w, min_n):
    """integer exposure -> mean |w| per value -> % change from the lowest bin."""
    vals = np.arange(int(count.min()), int(count.max()) + 1)
    xs, ys, ns = [], [], []
    for v in vals:
        m = count == v
        if m.sum() >= min_n:
            xs.append(v); ys.append(w[m].mean()); ns.append(int(m.sum()))
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
    pct = (ys / ys[0] - 1.0) * 100 if len(ys) else ys
    return xs, ys, pct, ns


def binned_quantile_pct(x, w, nbins, min_n):
    """continuous exposure -> quantile bins -> mean |w| -> % change from lowest bin."""
    qs = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    xs, ys, ns = [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() >= min_n:
            xs.append((lo + hi) / 2); ys.append(w[m].mean()); ns.append(int(m.sum()))
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
    pct = (ys / ys[0] - 1.0) * 100 if len(ys) else ys
    return xs, ys, pct, ns


def plot_panel(ax, xh, ph, xo, po, xlabel, title):
    ax.plot(xh, ph, "o-", color="#c0392b", lw=2, ms=5, label="HIDDEN drivers")
    ax.plot(xo, po, "o-", color="#2a78d6", lw=2, ms=5, label="OBSERVED drivers")
    ax.axhline(0, color=fs.GRID, lw=1)
    ax.set(xlabel=xlabel, ylabel="false-positive |weight|\n% change from lowest bin")
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title(title, fontsize=11)
    # total rise (lowest bin -> highest bin), not per-x-unit slope: the three
    # panels' x-axes are on very different scales (counts vs. weight sums), so
    # a raw regression slope isn't comparable across them — total % rise is.
    rh = ph[-1] - ph[0] if len(ph) else float("nan")
    ro = po[-1] - po[0] if len(po) else float("nan")
    ax.text(0.97, 0.05, f"total rise: hidden={rh:.1f}pp  observed={ro:.1f}pp",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=fs.MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--obs-frac", type=float, default=0.5)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--min-n", type=int, default=300, help="min pairs per bin")
    ap.add_argument("--qbins", type=int, default=8, help="quantile bins for weighted metric")
    ap.add_argument("--max-fp", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/fig_way2_motifs")
    args = ap.parse_args()

    d = Path(args.data)
    Cxx = np.load(d / "Cxx.npy"); Cyx = np.load(d / "Cyx.npy")
    adj = np.load(d / "adj_true.npy").astype(np.float64)
    N = Cxx.shape[0]; np.fill_diagonal(adj, 0.0)
    rng = np.random.default_rng(args.seed)

    types = np.sign(adj.sum(1)); types[types == 0] = 1
    E = np.flatnonzero(types > 0); I = np.flatnonzero(types < 0)
    rng.shuffle(E); rng.shuffle(I)
    nE, nI = int(args.obs_frac * len(E)), int(args.obs_frac * len(I))
    S = np.sort(np.concatenate([E[:nE], I[:nI]]))
    hidden = np.setdiff1d(np.arange(N), S)
    Sset = np.zeros(N, bool); Sset[S] = True
    print(f"N={N}  observed |S|={len(S)}  hidden={len(hidden)}")

    # ---- false positives (same procedure as fig_way2.py) ----
    A = ols_block(Cxx, Cyx, S)
    aa = np.abs(A)
    tau = np.quantile(aa[~np.eye(len(S), dtype=bool)], 1.0 - args.density)
    gb = adj[np.ix_(S, S)].T
    fp = (aa > tau) & (gb == 0); np.fill_diagonal(fp, False)
    fa, fb = np.nonzero(fp)
    if len(fa) > args.max_fp:
        k = rng.choice(len(fa), args.max_fp, replace=False); fa, fb = fa[k], fb[k]
    ti, tj = S[fa], S[fb]; w = aa[fa, fb]
    n_fp = len(fa)
    print(f"false positives: {n_fp}")

    Bfull = (adj != 0)                                         # source -> target
    G2 = adj @ adj                                             # weighted 2-hop influence

    # ---- 1-hop common drivers (binary presence — fine, network is sparse enough here) ----
    both1 = Bfull[:, ti] & Bfull[:, tj]
    obs1 = (both1 & Sset[:, None]).sum(0)
    hid1 = (both1 & (~Sset)[:, None]).sum(0)

    # ---- weighted shared-input strength: sum_k w_ki*w_kj, obs vs hidden 1-hop sources ----
    Ws = adj[np.ix_(S, S)]                          # sources in S, targets in S
    weighted_obs = (Ws.T @ Ws)[fa, fb]
    Wh = adj[np.ix_(hidden, S)]                     # sources hidden, targets in S
    weighted_hid = (Wh.T @ Wh)[fa, fb]

    # ---- 2-hop common drivers, WEIGHTED (raw reachability count saturates — see docstring) ----
    G2s = G2[np.ix_(S, S)]
    weighted2_obs = (G2s.T @ G2s)[fa, fb]
    G2h = G2[np.ix_(hidden, S)]
    weighted2_hid = (G2h.T @ G2h)[fa, fb]

    # ---- chain-motif check: real 2-hop path STRENGTH between the FP pair, vs baseline ----
    chain_strength = np.abs(G2[ti, tj]) + np.abs(G2[tj, ti])
    rng2 = np.random.default_rng(args.seed + 1)
    n_base = min(200_000, n_fp * 4)
    ri = rng2.integers(0, N, n_base); rj = rng2.integers(0, N, n_base)
    ok = (ri != rj) & (adj[ri, rj] == 0) & (adj[rj, ri] == 0)
    ri, rj = ri[ok], rj[ok]
    base_strength = np.abs(G2[ri, rj]) + np.abs(G2[rj, ri])
    print(f"\nchain-motif 2-hop path STRENGTH — false positives: mean={chain_strength.mean():.4f} "
          f"median={np.median(chain_strength):.4f}  |  random non-edges: mean="
          f"{base_strength.mean():.4f} median={np.median(base_strength):.4f}  "
          f"(mean ratio x{chain_strength.mean()/base_strength.mean():.2f})")

    # ---- plot ----
    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.14, wspace=0.32)

    xh1, yh1, ph1, _ = binned_pct(hid1, w, args.min_n)
    xo1, yo1, po1, _ = binned_pct(obs1, w, args.min_n)
    plot_panel(axes[0], xh1, ph1, xo1, po1, "# common presynaptic drivers (1-hop)",
               "1-HOP — same as fig_way2.py,\n% change instead of raw weight")

    xh2, yh2, ph2, _ = binned_quantile_pct(weighted2_hid, w, args.qbins, args.min_n)
    xo2, yo2, po2, _ = binned_quantile_pct(weighted2_obs, w, args.qbins, args.min_n)
    plot_panel(axes[1], xh2, ph2, xo2, po2, "sum of 2-hop driver weights  Σ G2[k,i]·G2[k,j]  (quantile bin)",
               "2-HOP WEIGHTED (NEW) — indirect\nshared input strength")

    xhw, yhw, phw, _ = binned_quantile_pct(weighted_hid, w, args.qbins, args.min_n)
    xow, yow, pow_, _ = binned_quantile_pct(weighted_obs, w, args.qbins, args.min_n)
    plot_panel(axes[2], xhw, phw, xow, pow_, "sum of driver weights  Σ w_ki·w_kj  (quantile bin)",
               "WEIGHTED (NEW) — driver\nstrength, not just count")

    fig.suptitle(f"N={N}, {int(args.obs_frac*100)}% observed — false-positive strength vs "
                 "shared-input exposure, three ways (motif-based)",
                 fontsize=13.5, color=fs.INK, x=0.06, ha="left", y=0.97)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
