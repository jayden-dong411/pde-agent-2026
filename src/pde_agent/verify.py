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
import sympy as sp
from scipy.interpolate import RectBivariateSpline, make_interp_spline

from .problem import Field, PDEProblem
from .solvers.base import SolveResult, SolverSpec, evaluate, mesh


# Hyperbolic problems whose solution jumps (or can jump: a nonlinear flux steepens smooth data into shocks).
# There the max-norm error is O(1) whenever a shock sits one cell off, however fine the grid,
# so the error is measured in L1, and a method that makes new extrema fails whatever its error.
SHOCK_KINDS = {"burgers_inviscid"}
TRANSPORT_KINDS = {"advection", "burgers_inviscid"}
_JUMP_FUNCS = {"sign", "Heaviside", "floor", "ceiling"}
OSCILLATION_LIMIT = 0.05  # new extrema beyond this fraction of the data's range count as oscillation
# (Lax-Wendroff and central schemes overshoot a square wave by 11-30%; WENO5 on a coarse grid by up to 3%)


def error_norm(problem) -> str:
    """``l1`` for transport problems with jumps or shocks, else ``max``."""
    if getattr(problem, "dim", 0) == 0 or problem.kind not in TRANSPORT_KINDS:
        return "max"
    if problem.kind in SHOCK_KINDS:
        return "l1"
    for name in ("initial", "exact", "boundary_value"):
        expr = problem.expr(name)
        if expr is not None and any(node.func.__name__ in _JUMP_FUNCS for node in sp.preorder_traversal(expr)):
            return "l1"
    return "max"


def relative_error(u: np.ndarray, reference: np.ndarray, norm: str = "max") -> float:
    """Error relative to the size of ``reference`` (absolute if that is ~0).

    ``max``: max-norm. ``l1``: mean absolute difference over mean |reference|
    (on an equal-spaced grid the same as the L1 norm ratio).
    """
    diff_abs = np.abs(np.asarray(u, dtype=float) - np.asarray(reference, dtype=float))
    ref_abs = np.abs(np.asarray(reference, dtype=float))
    if norm == "l1":
        diff, scale = float(np.mean(diff_abs)), float(np.mean(ref_abs))
    else:
        diff, scale = float(np.max(diff_abs)), float(np.max(ref_abs))
    return diff / scale if scale > 1e-12 else diff


def new_extrema(problem, result: SolveResult) -> float | None:
    """How far ``result`` overshoots the range of the initial and boundary data, as a fraction of that range.

    Only meaningful without a source (then the exact solution keeps to that range):
    None when the problem has a source or its data are not finite.
    """
    source = problem.expr("source")
    if source is not None and source != 0:
        return None
    view = problem.for_solver()
    with np.errstate(all="ignore"):
        x = result.grid[0]
        data = [np.asarray(view.fn("initial")(x, 0.0, 0.0), dtype=float) * np.ones_like(x)]
        if problem.boundary == "dirichlet" and view.expr("boundary_value") is not None:
            ends = x[[0, -1]]
            data += [np.asarray(view.fn("boundary_value")(ends, 0.0, t), dtype=float) * np.ones(2)
                     for t in (0.0, problem.t_final)]
    values = np.concatenate(data)
    if not np.all(np.isfinite(values)):
        return None
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 1e-12)
    u = np.asarray(result.u, dtype=float)
    return max(0.0, float(u.max()) - high, low - float(u.min())) / span


def exact_values(problem: PDEProblem, result: SolveResult) -> np.ndarray:
    if problem.dim == 0:  # an ODE: the state vector at the final time
        return problem.exact_at(problem.t_final)
    return evaluate(problem.fn("exact"), mesh(result.grid), problem.t_final or 0.0)


def exact_error(problem: PDEProblem, result: SolveResult) -> float | None:
    if not problem.has_exact():
        return None
    return relative_error(result.u, exact_values(problem, result), error_norm(problem))


def restrict(fine: SolveResult, n: int) -> np.ndarray:
    """Values of a fine-grid solution at the nodes of a nested coarse grid."""
    if not fine.grid:  # an ODE state at the final time: nothing to restrict
        return fine.u
    ratio, rem = divmod(fine.n, n)
    if rem or ratio < 1:
        raise ValueError(f"grid with n={n} is not nested in n={fine.n}")
    return fine.u[(slice(None, None, ratio),) * fine.u.ndim]


def richardson_error(coarse: SolveResult, fine: SolveResult, order: float, norm: str = "max") -> float:
    """Estimated relative error of ``fine`` from its difference to ``coarse``."""
    ratio = fine.n / coarse.n
    diff = relative_error(coarse.u, restrict(fine, coarse.n), norm)
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
        errors = [relative_error(c.u, restrict(f, c.n), error_norm(problem)) for c, f in zip(results, results[1:])]
        kind = "difference"
    study = ConvergenceStudy(method=spec.name, ns=list(ns), errors=errors, kind=kind)
    sizes = ns if kind == "exact" else ns[1:]
    study.orders = [observed_order(e0, e1, n1 / n0) for e0, e1, n0, n1 in zip(errors, errors[1:], sizes, sizes[1:])]
    return study


# --- resolution floor: can the grid see the problem's data? ----------------------
# A grid only sees its nodes. If the data alias there (every node of n = 16 is a
# zero of sin(16*pi*x)) a solver solves the wrong problem, and an error measured
# at the same nodes still comes out zero. The floor is the first grid whose node
# values interpolate the data to within RESOLUTION_LIMIT between the nodes.

RESOLUTION_LIMIT = 0.01  # largest mean interpolation error, as a fraction of max|data|, that counts as resolved
GOLD = (math.sqrt(5) - 1) / 2  # irrational probe offsets: the probes never fall on a node


def interpolation_mismatch(fn: Field, domain: list[tuple[float, float]], n: int, t: float) -> float:
    """Mean error of the cubic interpolant through ``fn`` at the nodes of an n-interval grid, relative to max|fn|.

    Measured at probe points that avoid the nodes. NaN when ``fn`` is not finite
    at a node or a probe (the solver reports that), 0 when ``fn`` vanishes.
    Expressions have no z, so a 3-D domain is checked on its x-y plane.
    """
    spans = domain[:2]
    m = 2048 if len(spans) == 1 else 96
    nodes = [np.linspace(a, b, n + 1) for a, b in spans]
    probes = [a + (b - a) * (np.arange(m) + ((k + 1) * GOLD) % 1.0) / m for k, (a, b) in enumerate(spans)]
    with np.errstate(all="ignore"):
        values = evaluate(fn, mesh(tuple(nodes)), t)
        truth = evaluate(fn, mesh(tuple(probes)), t)
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(truth))):
        return math.nan
    scale = float(np.max(np.abs(truth)))
    if scale < 1e-12:
        return 0.0
    if len(spans) == 1:
        fit = make_interp_spline(nodes[0], values, k=3)(probes[0])
    else:
        fit = RectBivariateSpline(nodes[0], nodes[1], values, kx=3, ky=3, s=0)(probes[0], probes[1])
    return float(np.mean(np.abs(fit - truth))) / scale


def resolution_fields(problem: PDEProblem, time_order: int) -> list[tuple[str, float]]:
    """The (field, time) pairs the grid has to resolve: the exact solution if known, else the data.

    ``time_order`` (0 steady, 1 or 2 time derivatives) says which initial data the equation takes.
    """
    if problem.has_exact():
        return [("exact", problem.t_final or 0.0)]
    checks: list[tuple[str, float]] = []
    if time_order >= 1 and problem.expr("initial") is not None:
        checks.append(("initial", 0.0))
    if time_order == 2:
        checks.append(("initial_velocity", 0.0))
    if problem.time_dependent:
        checks += [("source", 0.0), ("source", problem.t_final / 2), ("source", problem.t_final)]
    else:
        checks.append(("source", 0.0))
    return checks


def resolution_floor(problem: PDEProblem, tol: float, time_order: int, base_n: int, max_n: int) -> int | None:
    """First n in ``base_n, 2*base_n, ...`` (up to ``max_n``) whose grid resolves the problem's data, or None.

    Resolved means the interpolation mismatch of every checked field is at most
    ``max(tol, RESOLUTION_LIMIT)``. Fields that are not finite on the grid are
    skipped: the solver reports those, as it always did.
    """
    limit = max(tol, RESOLUTION_LIMIT)
    checks = [(problem.fn(name), t) for name, t in resolution_fields(problem, time_order)]
    n = base_n
    while n <= max_n:
        # "not >" rather than "<=": a NaN mismatch (non-finite data) counts as resolved
        if all(not interpolation_mismatch(fn, problem.domain, n, t) > limit for fn, t in checks):
            return n
        n *= 2
    return None
