# Ledger schema

The ledger is the flat, human-readable record of what was run: one CSV row per
run, holding the full configuration, every computed metric, timing and the code
provenance. Big arrays stay out of git; these files are the scientific index into
them.

Ledgers live at `results/<experiment>/ledger.csv` and are the only thing under
`results/` that git tracks (see the `!results/**/ledger.csv` rule in
`.gitignore`).

## How a row is produced

`calcinet.framework.ledger.append_row(result, config, ledger_path)` writes one
row by merging two objects:

```
row = dataclasses.asdict(config)      # every ExperimentConfig field
row.update(result.summary())          # metrics + duration_s + timestamp
                                      #   + git_commit + git_dirty + run_dir
```

The writer is **column-aligned, not fixed-schema**: if a run introduces a new
config field or a newly registered metric, that becomes a new column and earlier
rows get `NaN`. This is deliberate — the schema grows as the project grows — but
it has consequences described under *Drift* below.

## Column groups

**1. Configuration** (28 columns, present in every ledger) — a verbatim dump of
`ExperimentConfig`:

| group | columns |
|---|---|
| network | `n_excitatory`, `n_inhibitory`, `sim_time`, `dt`, `epsilon`, `g`, `eta`, `J_ex`, `n_threads` |
| calcium | `tau`, `amplitude`, `sigma_intra`, `sigma_extra` |
| proxy (stage 1) | `smooth_window_ms`, `tau_method`, `spike_cut_window` |
| solver (stage 2) | `solver`, `lag_ms`, `lam`, `lam_l2`, `chunk_size`, `n_epochs`, `lr`, `device` |
| bookkeeping | `data_path`, `name`, `seed`, `output_dir` |

**2. Metrics** — one column per registered metric, named `<registry>/<function>`:
`connectivity/…`, `tau/…`, `solver/…`, `diag/…`. These come from the decorator
registries in `framework/metrics.py`, so the set depends on what was registered
when the run happened.

**3. Provenance and timing** — `duration_s`, `timestamp`, `git_commit`,
`git_dirty`, `run_dir`.

`git_dirty=True` means the working tree had uncommitted changes to tracked files,
so the row is **not** reproducible from `git_commit` alone.

## Drift: what is and is not consistent

Audited across the 9 ledgers in `results/` (34–41 columns each):

- **33 columns are universal** — all 28 config fields, plus
  `connectivity/pearson`, `diag/ei_ratio`, `duration_s`, `timestamp`, `run_dir`.
  The configuration block is completely consistent; any two ledgers can be
  concatenated on it safely.
- **10 columns appear in only some ledgers.** Most are simple growth: a metric
  registered later exists only in later runs (`connectivity/spearman` 6/9,
  `connectivity/f1` 7/9, `diag/daleianity` 3/9).

Two findings that are **not** just growth, and that a reader has to handle:

**`connectivity/auc` vs `connectivity/auc_roc` — the same metric, renamed.**
`auc_roc` is the currently registered name in `metrics.py`. Five ledgers carry
the old `auc` column and four carry `auc_roc`; **no ledger has both**, so no
single column name reads across all runs:

| old name (`auc`) | new name (`auc_roc`) |
|---|---|
| `combine_test`, `dale_candidates_test`, `dale_reg_test`, `methods_overview`, `unsup_rescale_test` | `data_size_test`, `oracle_ladder`, `postprocess_test`, `regularization_test` |

To read AUC across everything, coalesce the two:

```python
df["auc"] = df.get("connectivity/auc_roc").fillna(df.get("connectivity/auc"))
```

**No existing ledger records provenance.** `ExperimentResult.summary()` does emit
`git_commit` and `git_dirty`, and new runs get them (the quickstart example
writes both). But all 9 ledgers under `results/` were written before that was
added, so **none of them carry a commit hash**. Those results cannot be tied to a
code version from the ledger alone — only by the surrounding notebook entries.
Re-running is the only way to attach provenance to them.

## Reading a ledger

```python
import pandas as pd
df = pd.read_csv("results/methods_overview/ledger.csv")

# concatenating several ledgers is safe on the config block;
# metric columns align by name and fill NaN where absent
import glob
all_runs = pd.concat([pd.read_csv(p) for p in glob.glob("results/*/ledger.csv")],
                     ignore_index=True)
```

`ledger.rebuild()` regenerates a ledger from the run directories on disk and
deduplicates re-logged runs; `ledger.effect_of()` groups rows by one config field
to summarise a sweep.
