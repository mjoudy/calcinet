"""The core package never depends on the research code.

``src/calcinet/`` is the installable tool; ``research/`` is the thesis's own
analysis. Research code may import calcinet, never the reverse. This test fails
if any core module imports a module that exists only under ``research/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "calcinet"
RESEARCH = ROOT / "research"


def _research_module_names() -> set[str]:
    return {p.stem for p in RESEARCH.rglob("*.py")}


def _absolute_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_core_never_imports_research_code():
    if not RESEARCH.is_dir():
        pytest.skip("research/ not present (e.g. installed from a wheel)")
    research = _research_module_names()
    offenders = {
        str(p.relative_to(ROOT)): sorted(_absolute_import_roots(p) & research)
        for p in CORE.rglob("*.py")
        if _absolute_import_roots(p) & research
    }
    assert not offenders, f"core modules import research code: {offenders}"
