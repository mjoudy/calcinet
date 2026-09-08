"""Solver contract tests.

1. Every solver in the runner's dispatch table runs on the same tiny input and
   returns an (N, N) floating-point matrix with no NaNs.
2. The closed-form solvers recover a known VAR matrix on well-conditioned data.
"""

from __future__ import annotations

import numpy as np
import pytest

from calcinet.connectivity import solvers as slv
from calcinet.framework.runner import _SOLVER_MAP

# Per-solver keyword arguments, mirroring the dispatch in runner.py but with
# small iteration counts so the suite stays fast. Kept here (not imported) so a
# change to the runner's defaults shows up as a test failure rather than
# silently altering what is exercised.
SOLVER_KWARGS: dict[str, dict] = {
    "ols":                dict(chunk_size=512),
    "sklearn_ols":        dict(chunk_size=512),
    "ridge":              dict(lam=1.0, chunk_size=512),
    "torch_normal_eq":    dict(lam=1.0, chunk_size=512, device="cpu"),
    "torch_minibatch":    dict(lam=1e-3, chunk_size=512, n_epochs=2, lr=1e-3,
                               device="cpu"),
    "torch_linear_layer": dict(lam=1e-3, chunk_size=512, n_epochs=2, lr=1e-3,
                               device="cpu"),
    "torch_gd":           dict(chunk_size=512, n_epochs=2, lr=1e-3, device="cpu"),
    "sklearn_lasso":      dict(lam=1e-3, chunk_size=512),
    "fista":              dict(lam_l1=1e-3, lam_l2=1e-3, n_iter=20,
                               chunk_size=512),
}

# Solvers that estimate A in closed form and should recover ground truth.
CLOSED_FORM = ["ols", "sklearn_ols"]


def test_dispatch_table_is_fully_covered():
    """Every solver the runner can dispatch to is exercised below."""
    assert set(SOLVER_KWARGS) == set(_SOLVER_MAP)


@pytest.mark.parametrize("name", sorted(SOLVER_KWARGS))
def test_solver_runs_and_returns_square_matrix(name, small_zarr):
    if name.startswith("torch"):
        pytest.importorskip("torch")
    fn = getattr(slv, _SOLVER_MAP[name])
    A = fn(zarr_path=small_zarr, lag=1, **SOLVER_KWARGS[name])

    assert isinstance(A, np.ndarray), f"{name} did not return an ndarray"
    assert A.shape == (5, 5), f"{name} returned {A.shape}, expected (5, 5)"
    assert np.issubdtype(A.dtype, np.floating), f"{name} dtype {A.dtype}"
    assert np.isfinite(A).all(), f"{name} produced non-finite entries"


@pytest.mark.parametrize("name", CLOSED_FORM)
def test_closed_form_recovers_ground_truth(name, var_truth, var_zarr):
    """On noiseless-in-expectation VAR data the closed-form solve is consistent."""
    A_true, _ = var_truth
    fn = getattr(slv, _SOLVER_MAP[name])
    A_hat = fn(zarr_path=var_zarr, lag=1, **SOLVER_KWARGS[name])

    assert A_hat.shape == A_true.shape
    np.testing.assert_allclose(A_hat, A_true, atol=0.05)


def test_ridge_shrinks_towards_zero(var_zarr):
    """Sanity check on the regularisation path: heavier lam => smaller norm."""
    weak = slv.solve_ridge(zarr_path=var_zarr, lag=1, lam=1.0, chunk_size=512)
    strong = slv.solve_ridge(zarr_path=var_zarr, lag=1, lam=1e6, chunk_size=512)
    assert np.linalg.norm(strong) < np.linalg.norm(weak)
