"""1-D wave equation ``u_tt = c^2 * u_xx + f`` with Dirichlet boundaries."""

from __future__ import annotations

import numpy as np

from ..problem import PDEProblem
from .base import SolveResult, SolverSpec, check_growth, grid, growth_limit, time_grid


def solve_leapfrog(problem: PDEProblem, n: int, options: dict) -> SolveResult:
    """Central differences in space and time; stable for |c| dt / h <= 1."""
    c = problem.coefficients.get("c", 1.0)
    (x,) = grid(problem, n)
    h = float(x[1] - x[0])
    speed = abs(c) if abs(c) > 1e-12 else 1.0
    steps, dt = time_grid(problem.t_final, options.get("cfl", 0.8) * h / speed)
    g, f = problem.fn("boundary_value"), problem.fn("source")
    inner, ends = x[1:-1], x[[0, -1]]
    s2 = (c * dt / h) ** 2

    def lap(u):
        return u[:-2] - 2.0 * u[1:-1] + u[2:]

    u_prev = problem.fn("initial")(x)
    v0 = problem.fn("initial_velocity")(x)
    limit = growth_limit(u_prev, v0, f(x, 0.0, 0.0))
    # Taylor start: u(dt) = u0 + dt*v0 + dt^2/2 * u_tt(0)
    u = np.empty_like(u_prev)
    u[1:-1] = u_prev[1:-1] + dt * v0[1:-1] + 0.5 * (s2 * lap(u_prev) + dt**2 * f(inner, 0.0, 0.0))
    u[[0, -1]] = g(ends, 0.0, dt)
    for k in range(1, steps):
        u_next = np.empty_like(u)
        u_next[1:-1] = 2.0 * u[1:-1] - u_prev[1:-1] + s2 * lap(u) + dt**2 * f(inner, 0.0, k * dt)
        u_next[[0, -1]] = g(ends, 0.0, (k + 1) * dt)
        u_prev, u = u, u_next
        if k % 64 == 0:
            check_growth(u, limit, "wave.leapfrog", k)
    check_growth(u, limit, "wave.leapfrog", steps)
    return SolveResult(
        u=u, grid=(x,), n=n, t=problem.t_final, steps=steps, dt=dt,
        work=float(x.size * steps), info={"cfl": abs(c) * dt / h},
    )


SOLVERS = (
    SolverSpec(
        name="wave.leapfrog",
        kind="wave",
        dims=(1,),
        boundaries=("dirichlet",),
        order=2.0,
        solve=solve_leapfrog,
        description="Second-order leapfrog (central differences in space and time)",
    ),
)
