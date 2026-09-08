"""The package and every subpackage import cleanly."""

import importlib

import pytest

MODULES = [
    "calcinet",
    "calcinet.io",
    "calcinet.proxy",
    "calcinet.simulation",
    "calcinet.connectivity.solvers",
    "calcinet.framework",
    "calcinet.framework.config",
    "calcinet.framework.ledger",
    "calcinet.framework.metrics",
    "calcinet.framework.persistence",
    "calcinet.framework.provenance",
    "calcinet.framework.result",
    "calcinet.framework.runner",
    "calcinet.io.streaming",
    "calcinet.framework.thresholding",
    "calcinet.connectivity.postprocessing",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name) is not None
