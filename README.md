# calcinet

**Linear VAR-based connectivity estimation from calcium imaging.**

`calcinet` recovers directed, signed connectivity between neurons from calcium
fluorescence, in two stages. Stage one inverts the calcium indicator: it
estimates each neuron's decay time constant by robust regression in phase space,
then deconvolves the trace into a continuous *spike proxy* — not discrete spike
times, but a drive signal that preserves relative rate. Stage two fits a linear
vector-autoregressive model to that proxy, so the estimated coefficient from
neuron *j* to neuron *i* carries both a direction and a sign. The solvers are
chunked and out-of-core, holding O(N²) accumulators rather than the full
(N × T) design matrix, which is what lets the same code run on a laptop at
N = 40 and on a GPU node at N = 12500.

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
experiments/           thesis-specific code — NOT installed
  ch1_proxy/           figures and analyses for the spike-proxy chapter
  ch2_connectivity/    figures and analyses for the connectivity chapter
  shared/              plotting style and helpers used by both
examples/quickstart.py runnable end-to-end demo
tests/                 pytest suite
slurm/                 SLURM job scripts (NEMO2)
docs/lab_notebook/     dated research notebook
results/               run outputs; git-ignored except ledger.csv
```

**The split is the important part.** `src/calcinet/` is code that works on
someone else's data — it is what you install and import. `experiments/` is the
code that produced specific results in the thesis; it is not packaged, and it is
full of hard-coded network names and paths. **Experiments import from the core;
the core never imports from experiments.** A test enforces the direction.

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

See [`CITATION.cff`](CITATION.cff). GitHub's "Cite this repository" button reads
it directly.

## Licence

MIT — see [`LICENSE`](LICENSE). All Python dependencies are permissive
(BSD-3-Clause, MIT, PSF). NEST is GPL-2.0-or-later and is imported lazily, only
when simulating: `calcinet` installs, imports and estimates without it.
