"""1-D advection ``u_t + c * u_x = f`` on a periodic interval."""

from __future__ import annotations

import numpy as np

from ..problem import PDEProblem
from .base import SolveResult, SolverSpec, check_growth, grid, growth_limit, time_grid


def _setup(problem: PDEProblem, n: int, cfl: float):
    c = problem.coefficients.get("c", 1.0)
    (x,) = grid(problem, n)
    h = float(x[1] - x[0])
    xp = x[:-1]  # the last node repeats the first one
    speed = abs(c) if abs(c) > 1e-12 else 1.0
    steps, dt = time_grid(problem.t_final, cfl * h / speed)
    u0 = problem.fn("initial")(xp)
    return c, x, xp, h, steps, dt, u0


def _result(x, u, steps, dt, problem, info) -> SolveResult:
    return SolveResult(
        u=np.append(u, u[0]), grid=(x,), n=x.size - 1, t=problem.t_final,
        steps=steps, dt=dt, work=float(u.size * steps), info=info,
    )


def solve_upwind(problem: PDEProblem, n: int, options: dict) -> SolveResult:
    """First-order upwind differences; stable for |c| dt / h <= 1."""
    c, x, xp, h, steps, dt, u = _setup(problem, n, options.get("cfl", 0.8))
    f = problem.fn("source")
    limit = growth_limit(u, f(xp, 0.0, 0.0))
    nu = c * dt / h
    for k in range(steps):
        t = k * dt
        if c >= 0:
            du = u - np.roll(u, 1)
        else:
            du = np.roll(u, -1) - u
        u = u - nu * du + dt * f(xp, 0.0, t)
        if k % 64 == 0:
            check_growth(u, limit, "advection.upwind", k)
    check_growth(u, limit, "advection.upwind", steps)
    return _result(x, u, steps, dt, problem, {"cfl": abs(nu)})


def solve_lax_wendroff(problem: PDEProblem, n: int, options: dict) -> SolveResult:
    """Second-order Lax-Wendroff with a second-order source treatment."""
    c, x, xp, h, steps, dt, u = _setup(problem, n, options.get("cfl", 0.8))
    f = problem.fn("source")
    limit = growth_limit(u, f(xp, 0.0, 0.0))
    nu = c * dt / h
    for k in range(steps):
        t_half = (k + 0.5) * dt
        up, um = np.roll(u, -1), np.roll(u, 1)
        fh = f(xp, 0.0, t_half)
        fx = (np.roll(fh, -1) - np.roll(fh, 1)) / (2.0 * h)
        u = u - 0.5 * nu * (up - um) + 0.5 * nu**2 * (up - 2.0 * u + um) + dt * fh - 0.5 * dt**2 * c * fx
        if k % 64 == 0:
            check_growth(u, limit, "advection.lax_wendroff", k)
    check_growth(u, limit, "advection.lax_wendroff", steps)
    return _result(x, u, steps, dt, problem, {"cfl": abs(nu)})


SOLVERS = (
    SolverSpec(
        name="advection.upwind",
        kind="advection",
        dims=(1,),
        boundaries=("periodic",),
        order=1.0,
        solve=solve_upwind,
        description="First-order upwind, monotone and robust",
    ),
    SolverSpec(
        name="advection.lax_wendroff",
        kind="advection",
        dims=(1,),
        boundaries=("periodic",),
        order=2.0,
        solve=solve_lax_wendroff,
        description="Second-order Lax-Wendroff",
    ),
)
