"""Shared fixtures: tiny synthetic VAR data written to zarr.

Everything here is small enough to run in seconds on a laptop CPU. The point is
to prove the plumbing works and that the closed-form solvers recover a known
matrix -- not to reproduce any scientific result.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr


def make_var_data(N: int = 6, T: int = 20_000, lag: int = 1, seed: int = 0,
                  radius: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Generate a stable VAR process  x_t = A x_{t-lag} + e_t.

    Returns (A_true, X) with X of shape (N, T) -- the same (N, T) orientation the
    solvers expect from zarr.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0 / np.sqrt(N), size=(N, N))
    # Scale to a spectral radius < 1 so the process is stationary.
    A *= radius / np.max(np.abs(np.linalg.eigvals(A)))

    X = np.zeros((N, T))
    noise = rng.normal(0.0, 1.0, size=(N, T))
    for t in range(lag, T):
        X[:, t] = A @ X[:, t - lag] + noise[:, t]
    return A, X


def write_zarr(tmp_path, X: np.ndarray, name: str = "signals.zarr") -> str:
    """Write (N, T) array to a zarr store and return its path."""
    path = str(tmp_path / name)
    z = zarr.open(path, mode="w", shape=X.shape, chunks=(X.shape[0], 1024),
                  dtype="f8")
    z[:] = X
    return path


@pytest.fixture(scope="session")
def var_truth():
    """(A_true, X) for the ground-truth recovery test -- long T, tight recovery."""
    return make_var_data(N=6, T=20_000, lag=1, seed=0)


@pytest.fixture(scope="session")
def var_zarr(tmp_path_factory, var_truth):
    """Path to a zarr store holding the ground-truth VAR data."""
    _, X = var_truth
    return write_zarr(tmp_path_factory.mktemp("truth"), X)


@pytest.fixture(scope="session")
def small_zarr(tmp_path_factory):
    """A smaller store used for the 'every solver runs' smoke test."""
    _, X = make_var_data(N=5, T=2_000, lag=1, seed=1)
    return write_zarr(tmp_path_factory.mktemp("small"), X)
