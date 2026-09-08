"""Connectivity estimation: solvers and post-processing.

``solvers`` holds the estimators themselves. See ``docs/adding_a_solver.md`` for
the interface contract and the three places a new solver has to be wired into --
there is no automatic registry for solvers, only for metrics.
"""

from . import solvers
from . import postprocessing
