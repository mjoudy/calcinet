# calcinet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22666410.svg)](https://doi.org/10.5281/zenodo.22666410) [![tests](https://github.com/mjoudy/calcinet/actions/workflows/tests.yml/badge.svg)](https://github.com/mjoudy/calcinet/actions/workflows/tests.yml)

**Linear VAR-based connectivity estimation from calcium imaging.**

`calcinet` recovers directed, signed connectivity between neurons from calcium
fluorescence, in two stages. Stage one inverts the calcium indicator: it
estimates each neuron's decay time constant by robust regression in phase space,
then deconvolves the trace into a continuous *spike proxy* — not discrete spike
times, but a drive signal that preserves relative rate. Stage two fits a linear
vector-autoregressive model to that proxy, so the estimated coefficient from
neuron *j* to neuron *i* carries both a direction and a sign. Most solvers
stream the proxy in time chunks and keep only N × N statistics in memory, never
the full N × T recording (see [How the solvers scale](#how-the-solvers-scale)),
which is what lets the same code run on a laptop at N = 40 and on a GPU node at
N = 12500.

```
   NEST simulation                  ┌──────────── stage 1 ────────────┐
   or CASCADE / real data  ──────>  │ tau estimation → deconvolution  │
                                    └────────────────┬────────────────┘
                                                     │  spike proxy
                                                     v
                                             zarr  (N × T, chunked)
                                                     │
                                    ┌────────────────┴── stage 2 ─────┐
                                    │  chunked solver (OLS / ridge /  │
                                    │  FISTA / torch, CPU or GPU)     │
                                    └────────────────┬────────────────┘
                                                     v
                                        connectivity matrix  (N × N)
                                                     │
                                     metrics ──> ledger.csv ──> figures
```

## Installation

Requires Python ≥ 3.11. NEST is only needed to *simulate* data; the estimation
path works without it.

**CPU (laptop, CPU cluster jobs)**

```bash
conda env create -f environment.yml       # creates 'phd_conda'
conda activate phd_conda
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

**GPU (NEMO2, L40S nodes)**

```bash
conda env create -f environment-gpu.yml -p $(ws_find calcium_ar)/env-gpu
$(ws_find calcium_ar)/env-gpu/bin/pip install -e .
```

The GPU environment is deliberately separate so the validated CPU environment
stays untouched. Verify CUDA *on a GPU node*, not the login node.

- **`environment.yml`** — the short, readable, curated dependency list: what the
  project depends on, pinned to validated versions. Start here.
- **`environment.lock.yml`** — a full byte-exact snapshot of a working
  environment (369 pins) for an identical rebuild. Prefer it on HPC.

PyTorch is intentionally absent from both: the correct build differs per machine,
so it is installed as a separate step.

## Quickstart

```bash
python examples/quickstart.py
```

Runs the entire pipeline on a tiny synthetic network in about 8 seconds — no
cluster, no downloads, no NEST. It generates spikes from a known connectivity
matrix via a Hawkes process, converts them to calcium, recovers the spike proxy,
estimates connectivity, and writes a figure plus a ledger row with the git
commit that produced it:

```
[1/5] 38891 spikes, mean rate 8.1 Hz
[2/5] calcium traces (40, 240000)
[3/5] tau: true 100 ms, estimated 101 ms
[4/5] estimated connectivity (40, 40)
[5/5] pearson 0.765   auc 0.968   git 4dd9148c (dirty=False)
```

## Repository layout

```
src/calcinet/          the installable, reusable tool
  proxy/               stage 1 — smoothing, tau estimation, deconvolution
  connectivity/        stage 2 — solvers, post-processing, diagnostics
  io/                  zarr streaming, chunked moment accumulation
  simulation/          Brunel (NEST), Hawkes and OU ground-truth generators
  framework/           experiment runner, config, metrics registry,
                       provenance, ledger
examples/quickstart.py runnable end-to-end demo
tests/                 pytest suite
docs/                  solver and ledger contracts, lab notebook, theory notes
results/               default output directory for runs (git-ignored)
research/              this thesis's own work — NOT part of the package
  experiments/         analysis and figure scripts: ch1_proxy, ch2_connectivity, shared
  research/slurm/               SLURM job scripts (NEMO2)
  research/slurm_logs/          logs of every cluster run, kept as evidence
  results/             ledgers and summary CSVs of the thesis runs
```

**The split is the important part.** `src/calcinet/` is code that works on
someone else's data — it is what you install and import. `research/` is the
record of how this thesis's results were produced: the analysis scripts, the
cluster job scripts and their logs, and the run ledgers. It is not packaged, it
is full of hard-coded network names and paths, and you do not need it to use
`calcinet` — it is kept as evidence and for reproducibility. **Research code
imports from the core; the core never imports from research code** —
`tests/test_layout.py` enforces this.

## How the solvers scale

A VAR fit needs only two lag-pair second moments of the spike proxy *x*:

```
Cxx = Σₜ x(t−lag) x(t−lag)ᵀ          Cyx = Σₜ x(t) x(t−lag)ᵀ
```

Both are N × N — all N neurons are regressed at once, so the cross-moment is a
matrix, not a vector — plus two length-N sums used for mean-centring. None of
them grows with the recording length T. The moment-based solvers therefore
stream the proxy from zarr in time chunks, add each chunk's contribution, and
solve once at the end: A = Cyx Cxx⁻¹ for OLS, with λI added to Cxx for ridge.
Because moments from different chunks, recordings or cluster nodes simply add,
a long recording can be split across nodes and pooled
(`calcinet.io.pool_chunks_and_solve`), and a saved set of moments can be
re-solved in seconds (`calcinet.connectivity.solve_from_cached`).

| Solver | How it uses the data | Memory |
|---|---|---|
| `ols`, `ridge` | one streaming pass accumulating the moments; exact closed-form solve | O(N²) |
| `torch_normal_eq` | the same, accumulated on the GPU | O(N²) |
| `fista` (elastic net) | one streaming pass for the moments, then iterates on them | O(N²) |
| `torch_gd`, `torch_minibatch`, `torch_linear_layer` | stream mini-batches over several epochs | O(N²) parameters, one chunk of data at a time |
| `sklearn_ols`, `sklearn_lasso` | load the full N × T matrix — reference implementations for small N | O(N·T) |

## Reproducibility

- **Provenance** — every run records the git commit and a `git_dirty` flag.
  `git_dirty=True` means the working tree had uncommitted changes, so the result
  is not reproducible from the commit alone.
- **Ledger** — one CSV row per run holding the full config, every metric, timing
  and provenance. Big arrays stay out of git; the ledger is the index into them.
  Schema and its known quirks: [`docs/ledger_schema.md`](docs/ledger_schema.md).
- **Locked environment** — `environment.lock.yml` pins the exact build of every
  package for an identical rebuild.
- **Metrics registry** — new metrics register with a decorator and appear
  automatically in every result and as a new ledger column.

Extending the solvers: [`docs/adding_a_solver.md`](docs/adding_a_solver.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite runs in a few seconds on a laptop. It checks that every module
imports, that every solver in the dispatch table returns a finite N × N matrix,
that the closed-form solvers recover a known VAR matrix (a check that fails on a
transposed result), that provenance is recorded, that chunked and single-pass
moment accumulation agree, and that the core package never imports research code.

Continuous integration ([`.github/workflows/tests.yml`](.github/workflows/tests.yml))
runs the same suite on every push and pull request: it builds the environment
from `environment.yml`, installs the CPU build of PyTorch and the package, and
runs `pytest`. It runs the tests only — there is no linting or type-checking step.

## Related repositories

Kept separate on purpose; neither is merged in.

- [**spikes-proxy**](https://github.com/mjoudy/spikes-proxy) — the original
  prototype of stage 1 (Savitzky-Golay reconstruction, RANSAC tau estimation).
  Superseded by `src/calcinet/proxy/`, which is the maintained version.
- [**Cascade fork**](https://github.com/mjoudy/Cascade) — a fork of
  [HelmchenLabSoftware/Cascade](https://github.com/HelmchenLabSoftware/Cascade)
  (deep-learning spike inference) carrying the phase-space tau method and
  `cut_spikes`. **It is GPL-3.0**, so code must not be copied from it into this
  MIT-licensed package.

## Declaration on the use of AI tools

During the preparation of this work the author used Anthropic's Claude, via
Claude Code, as a coding assistant. The research questions, the scientific
decisions and the interpretation of the results are the author's own. All
generated code and text were reviewed and verified by the author, who takes
full responsibility for the content of this repository.

Commits produced with this assistance carry a `Co-Authored-By` trailer, so its
extent can be inspected directly in the repository history.

## Citation

`calcinet` was developed as part of doctoral research at the Bernstein Center
Freiburg, Albert-Ludwigs-Universität Freiburg.

Archived on Zenodo. Cite the concept DOI, which always resolves to the latest
release:

> Joudy, M. *calcinet*. https://doi.org/10.5281/zenodo.22666410

See [`CITATION.cff`](CITATION.cff) for the machine-readable form — GitHub's
"Cite this repository" button reads it directly.

## Licence

MIT — see [`LICENSE`](LICENSE). All Python dependencies are permissive
(BSD-3-Clause, MIT, PSF). NEST is GPL-2.0-or-later and is imported lazily, only
when simulating: `calcinet` installs, imports and estimates without it.
