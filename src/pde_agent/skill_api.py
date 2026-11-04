"""The stable helper surface that learned solver skills import from.

Learned skills may import numpy, scipy, sympy, math and this module only.
ODE skills count their work with ``new_counts`` and ``ode_work`` like the built-in ODE methods.
"""

from .ode import NewtonFailed, ODEProblem, implicit_step, new_counts
from .problem import PDEProblem
from .solvers.base import (
    SolveResult,
    SolverDiverged,
    check_growth,
    evaluate,
    grid,
    growth_limit,
    laplacian_1d,
    mesh,
    time_grid,
)
from .solvers.ode import work as ode_work
from .solvers.poisson import interior_laplacian, laplacian_matrix

__all__ = [
    "NewtonFailed",
    "ODEProblem",
    "PDEProblem",
    "SolveResult",
    "SolverDiverged",
    "check_growth",
    "evaluate",
    "grid",
    "growth_limit",
    "interior_laplacian",
    "laplacian_1d",
    "laplacian_matrix",
    "mesh",
    "new_counts",
    "implicit_step",
    "ode_work",
    "time_grid",
]
