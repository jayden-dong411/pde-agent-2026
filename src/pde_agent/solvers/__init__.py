"""Built-in numerical methods for the basic equations."""

from . import advection, heat, poisson, wave
from .base import SolveResult, SolverDiverged, SolverRegistry, SolverSpec

BUILTIN_SOLVERS = poisson.SOLVERS + heat.SOLVERS + advection.SOLVERS + wave.SOLVERS


def builtin_registry() -> SolverRegistry:
    return SolverRegistry(BUILTIN_SOLVERS)


__all__ = [
    "BUILTIN_SOLVERS",
    "SolveResult",
    "SolverDiverged",
    "SolverRegistry",
    "SolverSpec",
    "builtin_registry",
]
