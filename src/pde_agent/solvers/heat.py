"""1-D heat equation ``u_t = alpha * u_xx + f`` with Dirichlet boundaries."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu

from ..problem import PDEProblem
from .base import SolveResult, SolverSpec, check_growth, grid, growth_limit, laplacian_1d, step_work, time_grid


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
    lap = laplacian_1d(m, s.h)
    eye = sparse.identity(m, format="csc")
    lhs = splu((eye - theta * dt * s.alpha * lap).tocsc())
    rhs_op = (eye + (1.0 - theta) * dt * s.alpha * lap).tocsr()
    u = s.u0[1:-1].copy()
    limit = growth_limit(s.u0, s.f(s.x, 0.0, 0.0))
    forcing_old = s.alpha * s.boundary_vector(0.0) + s.f(s.inner, 0.0, 0.0)
    for k in range(steps):
        t_new = (k + 1) * dt
        forcing_new = s.alpha * s.boundary_vector(t_new) + s.f(s.inner, 0.0, t_new)
        rhs = rhs_op @ u + dt * (theta * forcing_new + (1.0 - theta) * forcing_old)
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
)
