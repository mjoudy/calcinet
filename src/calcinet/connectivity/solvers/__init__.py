"""Connectivity solvers.

Every solver in this package is a module exposing a single function::

    solve(zarr_path: str, lag: int = 10, chunk_size: int = 10_000, ...) -> np.ndarray

Contract
--------
zarr_path   Path to a zarr array of the spike proxy ("feed"), shape (N, T),
            float64. Solvers stream it in chunks and hold O(N^2) accumulators,
            never the full (N, T) design matrix.
lag         AR lag in SAMPLES (not ms). Pairs x(t) with x(t - lag). The caller
            converts config.lag_ms using dt.
chunk_size  Time samples per iteration. Non-streaming solvers accept and ignore
            it so that all solvers share one signature.
returns     Dense (N, N) float array A, where A[i, j] is the influence of j on i
            (target row, source column). This is the TRANSPOSE of the
            ground-truth convention adj_true[source, target]; metrics compare A
            against adj_true.T. The diagonal is not required to be zero.

Adding a solver
---------------
There is no decorator registry here: the functions below are hand-written
wrappers, and dispatch is a name -> attribute table (`_SOLVER_MAP`) plus
per-solver keyword branches in ``calcinet.framework.runner``. Adding a solver
therefore means editing three places. The full procedure, and the reasoning
behind the lazy wrappers, is in ``docs/adding_a_solver.md``.

Metrics, by contrast, DO have a real decorator registry -- see
``calcinet.framework.metrics``.
"""

from .chunked_ols import solve as solve_ols
from .chunked_ridge import solve as solve_ridge


def solve_sklearn_ols(*args, **kwargs):
    from .sklearn_ols import solve
    return solve(*args, **kwargs)


def solve_torch_normal_eq(*args, **kwargs):
    from .torch_normal_eq import solve
    return solve(*args, **kwargs)


def solve_torch_minibatch(*args, **kwargs):
    from .torch_minibatch import solve
    return solve(*args, **kwargs)


def solve_torch_linear_layer(*args, **kwargs):
    from .torch_linear_layer import solve
    return solve(*args, **kwargs)


def solve_torch_gd(*args, **kwargs):
    from .torch_gd import solve
    return solve(*args, **kwargs)


def solve_sklearn_lasso(*args, **kwargs):
    from .sklearn_lasso import solve
    return solve(*args, **kwargs)


def solve_fista(*args, **kwargs):
    from .fista import solve
    return solve(*args, **kwargs)


# Dale-regularized solver (two-stage: needs types from a first pass, so it takes
# the data + types directly rather than the standard solve(zarr_path, ...) form).
from .dale_fista import dale_fista
