# Adding a solver

This page documents how connectivity solvers are wired into `calcinet`, and how
to add one. It describes the mechanism **as it is today** — including the parts
that are not as clean as they could be — so that anyone extending the package
knows exactly what to touch.

## The solver contract

A solver is a module under `src/calcinet/connectivity/solvers/` exposing a single
function named `solve`:

```python
def solve(zarr_path: str, lag: int = 10, chunk_size: int = 10_000, ...) -> np.ndarray:
    ...
```

| | |
|---|---|
| **Input** | `zarr_path` — path to a zarr array of the spike proxy ("feed"), shape `(N, T)`, `float64`. N neurons, T time samples. |
| **`lag`** | AR lag in **samples**, not milliseconds. The caller converts `lag_ms` using `dt`. The solver pairs `x(t)` with `x(t - lag)`. |
| **`chunk_size`** | Time samples to load per iteration. Solvers that do not stream still accept it, so every solver has a uniform signature. |
| **Output** | `A` — dense `np.ndarray` of shape `(N, N)`. |
| **Orientation** | `A[i, j]` is the influence **from j onto i** (target row, source column). This is the transpose of the ground-truth convention, where `adj_true[source, target]`. Metrics compare `A` against `adj_true.T`; see the note in `framework/metrics.py`. |
| **Diagonal** | Not required to be zero. Callers zero or ignore it. |
| **Memory** | Solvers are expected to hold `O(N²)` accumulators, not `O(N·T)`. Read the zarr in chunks rather than loading the full array. |

`chunked_ols.py` is the reference implementation and its docstring explains the
mean-centring correction that every closed-form solver here applies.

## Registration: what you actually have to edit

There is **no decorator-based registry for solvers**. Registration is manual and
touches three places. (Metrics *do* have a real registry — see the next section.)

**1. Write the module** — `src/calcinet/connectivity/solvers/my_solver.py`:

```python
"""One-line description of the estimator."""

import numpy as np
import zarr


def solve(zarr_path: str, lag: int = 10, chunk_size: int = 10_000,
          my_param: float = 1.0) -> np.ndarray:
    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    Cxx = np.zeros((N, N))
    Cyx = np.zeros((N, N))
    n_pairs = T - lag
    for t0 in range(0, n_pairs, chunk_size):
        t1 = min(t0 + chunk_size, n_pairs)
        x_prev = np.asarray(signals[:, t0:t1])
        x_now = np.asarray(signals[:, t0 + lag:t1 + lag])
        Cxx += x_prev @ x_prev.T
        Cyx += x_now @ x_prev.T

    return np.linalg.solve(Cxx.T + my_param * np.eye(N), Cyx.T).T
```

**2. Expose it** in `src/calcinet/connectivity/solvers/__init__.py`. Import
directly if the module is cheap, or use a lazy wrapper if it pulls a heavy
dependency (this is why the torch solvers are wrapped):

```python
def solve_my_solver(*args, **kwargs):
    from .my_solver import solve
    return solve(*args, **kwargs)
```

**3. Register it in the dispatch table** in
`src/calcinet/framework/runner.py` — add an entry to `_SOLVER_MAP` mapping the
config name to the wrapper's attribute name:

```python
_SOLVER_MAP = {
    ...
    "my_solver": "solve_my_solver",
}
```

If your solver needs arguments beyond `zarr_path`, `lag` and `chunk_size`, also
add a branch to `_call_solver` in the same file, following the existing pattern:

```python
if config.solver == "my_solver":
    return fn(**common, my_param=config.lam, chunk_size=config.chunk_size)
```

Finally, add the name to the comment listing valid solvers on
`ExperimentConfig.solver` in `framework/config.py`, and add the solver's
parameters as config fields if they are not already there.

**Be aware:** because the per-solver keyword arguments are hard-coded as
`if config.solver == ...` branches, adding a solver with new parameters means
editing the runner, not just dropping in a file. This is the main limitation of
the current design.

## Testing your solver

`tests/test_solvers.py` runs **every** entry in `_SOLVER_MAP` against the same
tiny synthetic input and asserts shape, dtype and finiteness. It also asserts
that the tested set equals `_SOLVER_MAP` exactly:

```python
def test_dispatch_table_is_fully_covered():
    assert set(SOLVER_KWARGS) == set(_SOLVER_MAP)
```

So adding a solver without adding its keyword arguments to `SOLVER_KWARGS` makes
the suite fail rather than silently skip it. That is deliberate.

If your solver is closed-form and consistent, add it to `CLOSED_FORM` in the same
file; it will then also be checked against a known VAR ground-truth matrix within
`atol=0.05`. That check is sensitive to a transposed result, which is the easiest
mistake to make here.

## The metrics registry — a real extension point

Unlike solvers, **metrics use a genuine decorator registry** and need no
central edit. In `src/calcinet/framework/metrics.py`:

```python
from calcinet.framework.metrics import connectivity_metrics

@connectivity_metrics.register
def my_metric(adj_inferred, adj_true, **kwargs) -> float:
    """Appears automatically in results and as a ledger column."""
    mask = ~np.eye(adj_true.shape[0], dtype=bool)
    return float(...)
```

Registered functions are keyed by `__name__` and surface as
`connectivity/my_metric` in `compute_all`, in every `ExperimentResult`, and as a
new ledger column. There are three registries — `connectivity_metrics`,
`tau_metrics` and `solver_metrics`.

One caveat: `MetricRegistry.compute` catches exceptions per metric and returns
`float("nan")` for a metric that raises. A metric that silently produces NaN is
usually a bug in the metric, not in the data.
