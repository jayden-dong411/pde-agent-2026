"""Reaction-diffusion u_t = D*u_xx - r*u + f with Crank-Nicolson (example skill).

This file shows the format of a learned skill. It is given to the language
model as an example and is used by the tests.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu

from pde_agent.skill_api import SolveResult, check_growth, grid, growth_limit, laplacian_1d, time_grid

# Only needed when the skill introduces a new equation kind.
EQUATION = {
    "kind": "reaction_diffusion",
    "form": "u_t = D * u_xx - r * u + f",
    "time_order": 1,
    "defaults": {"D": 1.0, "r": 1.0},
    "description": "Linear reaction-diffusion equation",
}


def operator(u, x, y, t, c):
    """L(u) as a sympy expression, so that the PDE reads L(u) = f."""
    import sympy as sp

    return sp.diff(u, t) - c["D"] * sp.diff(u, x, 2) + c["r"] * u


SOLVER = {
    "name": "reaction_diffusion.crank_nicolson",
    "kind": "reaction_diffusion",
    "dims": [1],
    "boundaries": ["dirichlet"],
    "order": 2,
    "description": "Crank-Nicolson in time, central differences in space",
}


def solve(problem, n, options):
    D = problem.coefficients.get("D", 1.0)
    r = problem.coefficients.get("r", 1.0)
    (x,) = grid(problem, n)
    h = float(x[1] - x[0])
    inner, ends = x[1:-1], x[[0, -1]]
    g, f = problem.fn("boundary_value"), problem.fn("source")
    steps, dt = time_grid(problem.t_final, options.get("cfl", 1.0) * h)

    m = inner.size
    eye = sparse.identity(m, format="csc")
    a = D * laplacian_1d(m, h) - r * eye
    lhs = splu((eye - 0.5 * dt * a).tocsc())
    rhs_op = (eye + 0.5 * dt * a).tocsr()

    def forcing(t):
        left, right = g(ends, 0.0, t)
        b = np.zeros(m)
        b[0], b[-1] = left, right
        return D * b / h**2 + f(inner, 0.0, t)

    u0 = problem.fn("initial")(x)
    u = u0[1:-1].copy()
    limit = growth_limit(u0)
    old = forcing(0.0)
    for k in range(steps):
        new = forcing((k + 1) * dt)
        u = lhs.solve(rhs_op @ u + 0.5 * dt * (old + new))
        old = new
        if k % 64 == 0:
            check_growth(u, limit, SOLVER["name"], k)

    full = np.empty_like(x)
    full[1:-1] = u
    full[[0, -1]] = g(ends, 0.0, problem.t_final)
    return SolveResult(u=full, grid=(x,), n=n, t=problem.t_final, steps=steps, dt=dt, work=float(x.size * steps))
