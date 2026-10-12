"""Solver interface, registry and shared numerical helpers.

Every solver, built in or learned, returns the solution at the final time on
the same tensor grid: ``n`` equal intervals per spatial dimension, so values
sit on ``linspace(a, b, n + 1)`` and ``u[i, j]`` is ``u(x_i, y_j)``. A common
grid lets the agent compare methods and estimate errors by refinement.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import scipy.sparse as sparse

from ..problem import PDEProblem


class SolverDiverged(RuntimeError):
    """Raised when a scheme blows up (usually a stability limit was exceeded)."""


@dataclass
class SolveResult:
    u: np.ndarray
    grid: tuple[np.ndarray, ...]
    n: int
    method: str = ""
    t: float | None = None
    steps: int = 0
    dt: float | None = None
    work: float = 0.0
    runtime: float = 0.0
    info: dict = field(default_factory=dict)


SolveFn = Callable[[PDEProblem, int, dict], SolveResult]


@dataclass(frozen=True)
class SolverSpec:
    """A numerical method the agent can choose.

    ``order`` is the expected convergence rate in ``n`` (error ~ n^-order).
    ``origin`` is ``builtin`` or ``learned``.
    """

    name: str
    kind: str
    dims: tuple[int, ...]
    boundaries: tuple[str, ...]
    order: float
    solve: SolveFn
    description: str = ""
    origin: str = "builtin"

    def supports(self, problem: PDEProblem) -> bool:
        return (
            problem.kind == self.kind
            and problem.dim in self.dims
            and problem.boundary in self.boundaries
        )

    def run(self, problem: PDEProblem, n: int, options: dict | None = None) -> SolveResult:
        start = time.perf_counter()
        result = self.solve(problem, int(n), dict(options or {}))
        result.runtime = time.perf_counter() - start
        result.method = self.name
        expected = (int(n) + 1,) * problem.dim
        if np.shape(result.u) != expected:
            raise ValueError(f"{self.name} returned shape {np.shape(result.u)}, expected {expected}")
        if not np.all(np.isfinite(result.u)):
            raise SolverDiverged(f"{self.name} produced non-finite values at n={n}")
        return result


class SolverRegistry:
    def __init__(self, specs: Iterable[SolverSpec] = ()):
        self._specs: dict[str, SolverSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: SolverSpec, replace: bool = False) -> None:
        if spec.name in self._specs and not replace:
            raise ValueError(f"solver {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> SolverSpec | None:
        return self._specs.get(name)

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def for_problem(self, problem: PDEProblem) -> list[SolverSpec]:
        return [s for s in self._specs.values() if s.supports(problem)]


# --- helpers shared by built-in and learned solvers -------------------------


def grid(problem: PDEProblem, n: int) -> tuple[np.ndarray, ...]:
    """1-D node coordinates per dimension: ``n`` intervals, ``n + 1`` nodes."""
    if n < 2:
        raise ValueError("need at least 2 intervals")
    return tuple(np.linspace(a, b, n + 1) for a, b in problem.domain)


def mesh(axes: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(np.meshgrid(*axes, indexing="ij"))


def evaluate(fn, points: tuple[np.ndarray, ...], t: float = 0.0) -> np.ndarray:
    """Evaluate a problem field ``fn(x, y, t)`` at 1-D or 2-D points."""
    if len(points) == 1:
        return fn(points[0], 0.0, t)
    return fn(points[0], points[1], t)


MAX_STEPS = 2_000_000


def time_grid(t_final: float, dt_target: float, max_steps: int = MAX_STEPS) -> tuple[int, float]:
    """Number of steps and the step that lands exactly on ``t_final``."""
    if not dt_target > 0:
        raise ValueError("time step must be positive")
    steps = max(1, math.ceil(t_final / dt_target - 1e-9))
    if steps > max_steps:
        raise ValueError(f"needs {steps} time steps (limit {max_steps}); this grid is too fine for the scheme")
    return steps, t_final / steps


def laplacian_1d(m: int, h: float) -> sparse.csc_matrix:
    """Second-difference matrix ``(u[i-1] - 2u[i] + u[i+1]) / h^2`` on m nodes."""
    main = np.full(m, -2.0)
    off = np.ones(m - 1)
    return sparse.diags([off, main, off], [-1, 0, 1], format="csc") / h**2


def check_growth(u: np.ndarray, limit: float, method: str, step: int) -> None:
    """Abort early when the solution has clearly blown up."""
    if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > limit:
        raise SolverDiverged(f"{method} diverged at step {step}")


def growth_limit(*arrays: np.ndarray) -> float:
    scale = max((float(np.max(np.abs(a))) for a in arrays if np.size(a)), default=0.0)
    return 1e6 * (1.0 + scale)
