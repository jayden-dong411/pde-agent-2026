"""1-D heat equation ``u_t = alpha * u_xx + f`` with Dirichlet boundaries.

The implicit schemes solve ``(I - c D2) u = rhs`` every step, with ``D2`` the
second difference and ``c = theta * alpha * dt / h^2`` fixed for a run. The
tridiagonal LU factors (LAPACK ``gttrf``) are computed once per ``(m, c)`` and
kept in a small cache, so every step is one ``gttrs`` back substitution, and
the extra runs of the checks and of a repeated solve on the same grid and step
reuse them. The explicit part is a stencil, not a sparse matrix product; a
source that does not depend on time is evaluated once.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from scipy.linalg.lapack import dgttrf, dgttrs

from ..problem import PDEProblem, T
from .base import SolveResult, SolverSpec, check_growth, grid, growth_limit, step_work, time_grid

CACHE_SIZE = 32  # factorizations kept: each is 4 vectors of length m
_FACTORS: OrderedDict = OrderedDict()
CACHE_STATS = {"hits": 0, "misses": 0}


class Tridiagonal:
    """LU factors of ``I - c * D2`` on m interior nodes (``D2`` = second difference without 1/h^2)."""

    def __init__(self, m: int, c: float):
        lower = np.full(m - 1, -c)
        diag = np.full(m, 1.0 + 2.0 * c)
        upper = np.full(m - 1, -c)
        *self.factors, info = dgttrf(lower, diag, upper)
        if info != 0:
            raise ValueError(f"singular implicit step matrix (gttrf info {info})")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        out, info = dgttrs(*self.factors, rhs)
        if info != 0:
            raise ValueError(f"gttrs failed (info {info})")
        return out


def factor(m: int, c: float) -> Tridiagonal:
    """The (cached) factorization of ``I - c D2``: one per grid and step, shared by every run that needs it."""
    key = (int(m), float(c))
    found = _FACTORS.get(key)
    if found is not None:
        _FACTORS.move_to_end(key)
        CACHE_STATS["hits"] += 1
        return found
    CACHE_STATS["misses"] += 1
    found = _FACTORS[key] = Tridiagonal(m, c)
    while len(_FACTORS) > CACHE_SIZE:
        _FACTORS.popitem(last=False)
    return found


def second_difference(u: np.ndarray) -> np.ndarray:
    """``u[i-1] - 2 u[i] + u[i+1]`` with zero values beyond both ends (boundary data go in the forcing)."""
    out = -2.0 * u
    out[1:] += u[:-1]
    out[:-1] += u[1:]
    return out


class _Setup:
    def __init__(self, problem: PDEProblem, n: int):
        self.alpha = problem.coefficients.get("alpha", 1.0)
        if self.alpha <= 0:
            raise ValueError("heat equation needs alpha > 0")
        (self.x,) = grid(problem, n)
        self.h = float(self.x[1] - self.x[0])
        self.inner = self.x[1:-1]
        self.ends = self.x[[0, -1]]
        self.g = problem.fn("boundary_value")
        source = problem.expr("source")
        if source is not None and not source.free_symbols & {T}:  # evaluate once, not every step
            fixed = np.asarray(problem.fn("source")(self.inner, 0.0, 0.0), dtype=float) * np.ones_like(self.inner)
            self.f = lambda x, y, t, fixed=fixed: fixed if np.shape(x) == fixed.shape else problem.fn("source")(x, y, t)
        else:
            self.f = problem.fn("source")
        self.u0 = problem.fn("initial")(self.x)
        self.t_final = problem.t_final

    def boundary_vector(self, t: float) -> np.ndarray:
        """Boundary contribution to the interior second difference."""
        left, right = self.g(self.ends, 0.0, t)
        b = np.zeros(self.inner.size)
        b[0] += left / self.h**2
        b[-1] += right / self.h**2
        return b

    def result(self, u_inner: np.ndarray, steps: int, dt: float, method_info: dict) -> SolveResult:
        u = np.empty_like(self.x)
        u[1:-1] = u_inner
        u[[0, -1]] = self.g(self.ends, 0.0, self.t_final)
        return SolveResult(
            u=u, grid=(self.x,), n=self.x.size - 1, t=self.t_final,
            steps=steps, dt=dt, work=float(self.x.size * steps), info=method_info,
        )


def solve_ftcs(problem: PDEProblem, n: int, options: dict) -> SolveResult:
    """Explicit Euler in time; stable for alpha*dt/h^2 <= 1/2."""
    s = _Setup(problem, n)
    r_target = options.get("r", 0.4)
    steps, dt = time_grid(s.t_final, r_target * s.h**2 / s.alpha)
    r = s.alpha * dt / s.h**2
    u = s.u0.copy()
    limit = growth_limit(s.u0, s.f(s.x, 0.0, 0.0))
    for k in range(steps):
        t = k * dt
        lap = u[:-2] - 2.0 * u[1:-1] + u[2:]
        u[1:-1] = u[1:-1] + r * lap + dt * s.f(s.inner, 0.0, t)
        u[[0, -1]] = s.g(s.ends, 0.0, t + dt)
        if k % 64 == 0:
            check_growth(u, limit, "heat.ftcs", k)
    check_growth(u, limit, "heat.ftcs", steps)
    return s.result(u[1:-1], steps, dt, {"r": r})


def _implicit(problem: PDEProblem, n: int, options: dict, theta: float, name: str, cfl: float) -> SolveResult:
    """theta-scheme: theta=1 backward Euler, theta=1/2 Crank-Nicolson."""
    s = _Setup(problem, n)
    steps, dt = time_grid(s.t_final, options.get("cfl", cfl) * s.h)
    m = s.inner.size
    r = s.alpha * dt / s.h**2
    lhs = factor(m, theta * r)
    explicit = (1.0 - theta) * r
    u = s.u0[1:-1].copy()
    limit = growth_limit(s.u0, s.f(s.x, 0.0, 0.0))
    forcing_old = s.alpha * s.boundary_vector(0.0) + s.f(s.inner, 0.0, 0.0)
    for k in range(steps):
        t_new = (k + 1) * dt
        forcing_new = s.alpha * s.boundary_vector(t_new) + s.f(s.inner, 0.0, t_new)
        rhs = u + dt * (theta * forcing_new + (1.0 - theta) * forcing_old)
        if explicit:
            rhs += explicit * second_difference(u)
        u = lhs.solve(rhs)
        forcing_old = forcing_new
        if k % 64 == 0:
            check_growth(u, limit, name, k)
    check_growth(u, limit, name, steps)
    return s.result(u, steps, dt, {"theta": theta, "cfl": dt / s.h})


def solve_backward_euler(problem: PDEProblem, n: int, options: dict) -> SolveResult:
    return _implicit(problem, n, options, theta=1.0, name="heat.backward_euler", cfl=0.5)


def solve_crank_nicolson(problem: PDEProblem, n: int, options: dict) -> SolveResult:
    return _implicit(problem, n, options, theta=0.5, name="heat.crank_nicolson", cfl=1.0)


def solve_bdf2(problem: PDEProblem, n: int, options: dict) -> SolveResult:
    """Second-order BDF: L-stable, so jumps in the initial data are damped instead of ringing as in Crank-Nicolson.

    The first step is two backward Euler half steps (enough for second order overall).
    """
    s = _Setup(problem, n)
    steps, dt = time_grid(s.t_final, options.get("cfl", 1.0) * s.h)
    m = s.inner.size
    r = s.alpha * dt / s.h**2
    half = factor(m, 0.5 * r)
    lhs = factor(m, (2.0 / 3.0) * r)
    limit = growth_limit(s.u0, s.f(s.x, 0.0, 0.0))

    def forcing(t: float) -> np.ndarray:
        return s.alpha * s.boundary_vector(t) + s.f(s.inner, 0.0, t)

    older = s.u0[1:-1].copy()
    u = older
    for t_new in (0.5 * dt, dt):
        u = half.solve(u + 0.5 * dt * forcing(t_new))
    for k in range(1, steps):
        t_new = (k + 1) * dt
        rhs = (4.0 * u - older) / 3.0 + (2.0 / 3.0) * dt * forcing(t_new)
        older, u = u, lhs.solve(rhs)
        if k % 64 == 0:
            check_growth(u, limit, "heat.bdf2", k)
    check_growth(u, limit, "heat.bdf2", steps)
    return s.result(u, steps, dt, {"cfl": dt / s.h})


def _h(problem: PDEProblem, n: int) -> float:
    return (problem.domain[0][1] - problem.domain[0][0]) / n


def _alpha(problem: PDEProblem) -> float:
    return max(problem.coefficients.get("alpha", 1.0), 1e-12)


def cost_ftcs(problem: PDEProblem, n: int) -> float:
    return step_work(problem, n, 0.4 * _h(problem, n) ** 2 / _alpha(problem))


def cost_backward_euler(problem: PDEProblem, n: int) -> float:
    return step_work(problem, n, 0.5 * _h(problem, n))


def cost_crank_nicolson(problem: PDEProblem, n: int) -> float:
    return step_work(problem, n, 1.0 * _h(problem, n))


def damping_crank_nicolson(problem: PDEProblem, n: int, k: float) -> float:
    """What is left of the mode sin(kx) at t_final: CN damps a fast mode by |1 - z/2| / |1 + z/2| per step."""
    dt = 1.0 * _h(problem, n)
    steps = max(1, math.ceil(problem.t_final / dt - 1e-9))
    z = _alpha(problem) * (problem.t_final / steps) * k**2
    return abs((1 - z / 2) / (1 + z / 2)) ** steps


def cost_bdf2(problem: PDEProblem, n: int) -> float:
    return step_work(problem, n, 1.0 * _h(problem, n)) + (n + 1)  # one extra solve to start


def damping_bdf2(problem: PDEProblem, n: int, k: float) -> float:
    """What is left of the mode sin(kx) at t_final: the larger root of (1 + 2z/3) r^2 - 4r/3 + 1/3 = 0, per step.

    BDF2 is L-stable: for a fast mode (large z) both roots go to zero, unlike Crank-Nicolson's -1.
    """
    dt = 1.0 * _h(problem, n)
    steps = max(1, math.ceil(problem.t_final / dt - 1e-9))
    z = _alpha(problem) * (problem.t_final / steps) * k**2
    roots = np.roots([1 + 2 * z / 3, -4 / 3, 1 / 3])
    start = 1.0 / (1.0 + z / 2) ** 2  # two backward Euler half steps
    return start * float(np.max(np.abs(roots))) ** max(0, steps - 1)


SOLVERS = (
    SolverSpec(
        name="heat.ftcs",
        kind="heat",
        dims=(1,),
        boundaries=("dirichlet",),
        order=2.0,
        solve=solve_ftcs,
        cost=cost_ftcs,
        description="Explicit forward-time centred-space; dt ~ h^2 for stability",
    ),
    SolverSpec(
        name="heat.backward_euler",
        kind="heat",
        dims=(1,),
        boundaries=("dirichlet",),
        order=1.0,
        solve=solve_backward_euler,
        cost=cost_backward_euler,
        description="Implicit backward Euler, unconditionally stable, first order in time",
    ),
    SolverSpec(
        name="heat.crank_nicolson",
        kind="heat",
        dims=(1,),
        boundaries=("dirichlet",),
        order=2.0,
        solve=solve_crank_nicolson,
        cost=cost_crank_nicolson,
        damping=damping_crank_nicolson,
        description="Implicit Crank-Nicolson, second order in space and time",
    ),
    SolverSpec(
        name="heat.bdf2",
        kind="heat",
        dims=(1,),
        boundaries=("dirichlet",),
        order=2.0,
        solve=solve_bdf2,
        cost=cost_bdf2,
        damping=damping_bdf2,
        description="Implicit BDF2, second order and L-stable: damps jumps in the initial data without ringing",
    ),
)
