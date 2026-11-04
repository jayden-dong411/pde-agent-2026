"""Strong-stability-preserving Runge-Kutta, third order (example ODE skill).

This file shows the format of a learned ODE skill. It is given to the language
model as an example and is used by the tests.
"""

import numpy as np

from pde_agent.skill_api import SolveResult, check_growth, growth_limit, new_counts, ode_work

SOLVER = {
    "name": "ode.ssprk3",
    "kind": "ode",
    "dims": [0],
    "boundaries": ["ivp"],
    "order": 3,
    "description": "Shu-Osher SSP-RK3, explicit, third order",
}


def solve(problem, n, options):
    """n is the number of equal time steps; returns the state at t_final."""
    if n < 1:
        raise ValueError("need at least one time step")
    h = problem.t_final / n
    y = problem.y0.copy()
    counts = new_counts()
    limit = growth_limit(y, problem.f(0.0, y))
    with np.errstate(all="ignore"):
        for k in range(n):
            t = k * h
            y1 = y + h * problem.f(t, y)
            y2 = 0.75 * y + 0.25 * (y1 + h * problem.f(t + h, y1))
            y = y / 3.0 + 2.0 / 3.0 * (y2 + h * problem.f(t + 0.5 * h, y2))
            if k % 64 == 0:
                check_growth(y, limit, SOLVER["name"], k)
    check_growth(y, limit, SOLVER["name"], n)
    counts["f"] = 3 * n
    return SolveResult(u=y, grid=(), n=n, t=problem.t_final, steps=n, dt=h, work=ode_work(problem, counts))
