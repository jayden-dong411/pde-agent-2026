"""Self-verification: how wrong is a numerical solution?

With an exact solution the error is measured directly. Without one the agent
estimates it from two nested grids (Richardson): if the error behaves like
``C n^-p``, then ``|u_n - u_rn| / (r^p - 1)`` approximates the error of the
finer solution ``u_rn``. Errors are relative to the size of the solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .problem import PDEProblem
from .solvers.base import SolveResult, SolverSpec, evaluate, mesh


def relative_error(u: np.ndarray, reference: np.ndarray) -> float:
    """Max-norm error relative to max|reference| (absolute if that is ~0)."""
    diff = float(np.max(np.abs(np.asarray(u) - np.asarray(reference))))
    scale = float(np.max(np.abs(reference)))
    return diff / scale if scale > 1e-12 else diff


def exact_values(problem: PDEProblem, result: SolveResult) -> np.ndarray:
    return evaluate(problem.fn("exact"), mesh(result.grid), problem.t_final or 0.0)


def exact_error(problem: PDEProblem, result: SolveResult) -> float | None:
    if not problem.has_exact():
        return None
    return relative_error(result.u, exact_values(problem, result))


def restrict(fine: SolveResult, n: int) -> np.ndarray:
    """Values of a fine-grid solution at the nodes of a nested coarse grid."""
    ratio, rem = divmod(fine.n, n)
    if rem or ratio < 1:
        raise ValueError(f"grid with n={n} is not nested in n={fine.n}")
    return fine.u[(slice(None, None, ratio),) * fine.u.ndim]


def richardson_error(coarse: SolveResult, fine: SolveResult, order: float) -> float:
    """Estimated relative error of ``fine`` from its difference to ``coarse``."""
    ratio = fine.n / coarse.n
    diff = relative_error(coarse.u, restrict(fine, coarse.n))
    return diff / (ratio ** max(order, 0.5) - 1.0)


def observed_order(e_coarse: float, e_fine: float, ratio: float) -> float | None:
    if not (e_coarse > 0 and e_fine > 0 and ratio > 1):
        return None
    return math.log(e_coarse / e_fine) / math.log(ratio)


@dataclass
class ConvergenceStudy:
    method: str
    ns: list[int]
    errors: list[float] = field(default_factory=list)
    orders: list[float | None] = field(default_factory=list)
    kind: str = "exact"

    @property
    def final_order(self) -> float | None:
        return self.orders[-1] if self.orders else None


def convergence_study(spec: SolverSpec, problem: PDEProblem, ns: list[int], options: dict | None = None) -> ConvergenceStudy:
    """Errors and observed orders over a sequence of refined grids.

    Uses the exact solution when there is one, otherwise successive
    differences (which shrink at the same rate as the error).
    """
    results = [spec.run(problem, n, options) for n in ns]
    if problem.has_exact():
        errors = [exact_error(problem, r) for r in results]
        kind = "exact"
    else:
        errors = [relative_error(c.u, restrict(f, c.n)) for c, f in zip(results, results[1:])]
        kind = "difference"
    study = ConvergenceStudy(method=spec.name, ns=list(ns), errors=errors, kind=kind)
    sizes = ns if kind == "exact" else ns[1:]
    study.orders = [observed_order(e0, e1, n1 / n0) for e0, e1, n0, n1 in zip(errors, errors[1:], sizes, sizes[1:])]
    return study
