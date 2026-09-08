"""Chunked moment accumulation matches accumulating the whole array at once."""

from __future__ import annotations

import numpy as np

from calcinet.io.streaming import MomentAccumulator

from conftest import make_var_data


def _accumulate(X: np.ndarray, lag: int, chunk: int | None):
    acc = MomentAccumulator(N=X.shape[0], lag=lag)
    if chunk is None:
        acc.add(X)
    else:
        for t0 in range(0, X.shape[1], chunk):
            acc.add(X[:, t0:t0 + chunk])
    return acc.snapshot()


def test_chunked_accumulation_matches_single_pass():
    _, X = make_var_data(N=4, T=3_000, lag=1, seed=2)
    lag = 1

    Cxx_one, Cyx_one = _accumulate(X, lag, chunk=None)

    for chunk in (97, 256, 1024):
        Cxx_chunked, Cyx_chunked = _accumulate(X, lag, chunk=chunk)
        np.testing.assert_allclose(Cxx_chunked, Cxx_one, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(Cyx_chunked, Cyx_one, rtol=1e-9, atol=1e-9)


def test_uneven_final_chunk_is_not_dropped():
    """A trailing chunk shorter than the chunk size still contributes."""
    _, X = make_var_data(N=3, T=1_001, lag=1, seed=3)
    Cxx_one, _ = _accumulate(X, lag=1, chunk=None)
    Cxx_chunked, _ = _accumulate(X, lag=1, chunk=100)   # 1001 = 10*100 + 1
    np.testing.assert_allclose(Cxx_chunked, Cxx_one, rtol=1e-9, atol=1e-9)
