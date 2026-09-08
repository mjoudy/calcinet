"""Connectivity estimation: solvers, post-processing, and the solver registry.

``solvers`` holds the estimators themselves; see ``docs/adding_a_solver.md`` for
the interface contract and how a new solver is wired in.
"""

from . import solvers
from . import postprocessing
