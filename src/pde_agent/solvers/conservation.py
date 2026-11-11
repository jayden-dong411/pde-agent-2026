"""Shock-capturing finite volumes for 1-D conservation laws ``u_t + F(u)_x = f``.

Two fluxes: linear advection ``F = c u`` and inviscid Burgers ``F = u^2 / 2``.
The grid nodes serve as cell centres, so results sit on the common node grid.
Periodic problems drop the repeated last node; Dirichlet problems hold the
boundary values in ghost cells (inflow or outflow, as the characteristics decide).

Methods: Rusanov (local Lax-Friedrichs, first order, monotone), MUSCL with a
minmod limiter and SSP-RK2 (second order away from extrema, no new extrema),
WENO5 with Lax-Friedrichs flux splitting and SSP-RK3 (fifth order on smooth
data, essentially non-oscillatory at jumps). ``work`` counts node updates times
flux evaluations per step: 1 for Rusanov, 4 for MUSCL, 12 for WENO5.
"""

from __future__ import annotations

import numpy as np

from ..problem import PDEProblem
from .base import SolveResult, SolverSpec, check_growth, grid, growth_limit, step_work, time_grid

GHOSTS = 3  # WENO5 reads three cells to each side
WEIGHT = {"rusanov": 1.0, "muscl": 4.0, "weno5": 12.0}
CFL = {"rusanov": 0.8, "muscl": 0.45, "weno5": 0.45}


def _flux(kind: str, coefficients: dict):
    if kind == "advection":
        c = coefficients.get("c", 1.0)
        return (lambda u: c * u), (lambda u: np.full_like(u, abs(c)))
    return (lambda u: 0.5 * u * u), np.abs


def _speed(problem: PDEProblem, xs: np.ndarray) -> float:
    if problem.kind == "advection":
        return abs(problem.coefficients.get("c", 1.0))
    values = [np.abs(np.asarray(problem.fn("initial")(xs, 0.0, 0.0), dtype=float) * np.ones_like(xs))]
    if problem.boundary == "dirichlet" and problem.expr("boundary_value") is not None:
        ends = xs[[0, -1]]
        values += [np.abs(np.asarray(problem.fn("boundary_value")(ends, 0.0, t), dtype=float) * np.ones(2))
                   for t in (0.0, problem.t_final)]
    top = float(max(np.max(v) for v in values))
    return top if np.isfinite(top) and top > 1e-12 else 1.0


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _weno_left(v0, v1, v2, v3, v4):
    """Fifth-order WENO value at the right face of the cell v2, from cells v0..v4 (left-biased)."""
    eps = 1e-6
    b0 = 13 / 12 * (v0 - 2 * v1 + v2) ** 2 + 0.25 * (v0 - 4 * v1 + 3 * v2) ** 2
    b1 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
    b2 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (3 * v2 - 4 * v3 + v4) ** 2
    a0, a1, a2 = 0.1 / (eps + b0) ** 2, 0.6 / (eps + b1) ** 2, 0.3 / (eps + b2) ** 2
    q0 = (2 * v0 - 7 * v1 + 11 * v2) / 6
    q1 = (-v1 + 5 * v2 + 2 * v3) / 6
    q2 = (2 * v2 + 5 * v3 - v4) / 6
    return (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)


def _face_fluxes(method: str, ue: np.ndarray, flux, speed) -> np.ndarray:
    """Numerical flux at the faces between the extended cells ``ue`` (m + 2 GHOSTS cells -> m + 1 faces)."""
    g = GHOSTS
    m = ue.size - 2 * g
    if method == "rusanov":
        left, right = ue[g - 1:g + m], ue[g:g + m + 1]
    elif method == "muscl":
        slope = _minmod(ue[1:-1] - ue[:-2], ue[2:] - ue[1:-1])  # for cells 1 .. size-2
        left = ue[g - 1:g + m] + 0.5 * slope[g - 2:g - 1 + m]
        right = ue[g:g + m + 1] - 0.5 * slope[g - 1:g + m]
    else:  # weno5 with global Lax-Friedrichs splitting
        alpha = float(np.max(speed(ue)))
        fp = 0.5 * (flux(ue) + alpha * ue)
        fm = 0.5 * (flux(ue) - alpha * ue)
        # face i+1/2 for i = g-1 .. g+m-1
        idx = np.arange(g - 1, g + m)
        plus = _weno_left(fp[idx - 2], fp[idx - 1], fp[idx], fp[idx + 1], fp[idx + 2])
        minus = _weno_left(fm[idx + 3], fm[idx + 2], fm[idx + 1], fm[idx], fm[idx - 1])
        return plus + minus
    a = np.maximum(speed(left), speed(right))
    return 0.5 * (flux(left) + flux(right)) - 0.5 * a * (right - left)


def _solve(problem: PDEProblem, n: int, options: dict, method: str) -> SolveResult:
    name = f"{problem.kind}.{method}"
    flux, speed = _flux(problem.kind, problem.coefficients)
    (x,) = grid(problem, n)
    h = float(x[1] - x[0])
    periodic = problem.boundary == "periodic"
    cells = x[:-1] if periodic else x
    steps, dt = time_grid(problem.t_final, options.get("cfl", CFL[method]) * h / _speed(problem, x))
    u = np.asarray(problem.fn("initial")(cells, 0.0, 0.0), dtype=float) * np.ones_like(cells)
    f = problem.fn("source")
    g_fn = None if periodic else problem.fn("boundary_value")
    limit = growth_limit(u, f(cells, 0.0, 0.0))
    g = GHOSTS

    def rhs(v: np.ndarray, t: float) -> np.ndarray:
        if periodic:
            ue = np.concatenate([v[-g:], v, v[:g]])
        else:
            left, right = np.asarray(g_fn(x[[0, -1]], 0.0, t), dtype=float) * np.ones(2)
            ue = np.concatenate([np.full(g, left), v, np.full(g, right)])
        faces = _face_fluxes(method, ue, flux, speed)
        return -(faces[1:] - faces[:-1]) / h + f(cells, 0.0, t)

    for k in range(steps):
        t = k * dt
        if method == "rusanov":
            u = u + dt * rhs(u, t)
        elif method == "muscl":  # SSP-RK2 (Heun)
            u1 = u + dt * rhs(u, t)
            u = 0.5 * u + 0.5 * (u1 + dt * rhs(u1, t + dt))
        else:  # SSP-RK3
            u1 = u + dt * rhs(u, t)
            u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1, t + dt))
            u = u / 3 + 2 / 3 * (u2 + dt * rhs(u2, t + 0.5 * dt))
        if k % 64 == 0:
            check_growth(u, limit, name, k)
    check_growth(u, limit, name, steps)
    if periodic:
        out = np.append(u, u[0])
    else:
        out = u.copy()
        # Dirichlet values belong to the boundary nodes only where the flow enters
        out[[0, -1]] = np.asarray(g_fn(x[[0, -1]], 0.0, problem.t_final), dtype=float) * np.ones(2)
        if problem.kind == "burgers_inviscid":
            if u[0] < 0:
                out[0] = u[0]
            if u[-1] > 0:
                out[-1] = u[-1]
        elif problem.coefficients.get("c", 1.0) >= 0:
            out[-1] = u[-1]
        else:
            out[0] = u[0]
    return SolveResult(u=out, grid=(x,), n=n, t=problem.t_final, steps=steps, dt=dt,
                       work=float(cells.size * steps * WEIGHT[method]), info={"cfl": dt * _speed(problem, x) / h})


def _cost(method: str):
    def cost(problem: PDEProblem, n: int) -> float:
        (x,) = grid(problem, n)
        h = float(x[1] - x[0])
        return WEIGHT[method] * step_work(problem, n, CFL[method] * h / _speed(problem, x))
    return cost


def _has_jumps(problem: PDEProblem) -> bool:
    from ..verify import error_norm

    return error_norm(problem) == "l1"


def _spec(kind: str, method: str, order: float, description: str, boundaries, traits=(), accepts=None) -> SolverSpec:
    return SolverSpec(
        name=f"{kind}.{method}", kind=kind, dims=(1,), boundaries=boundaries, order=order, accepts=accepts,
        solve=lambda p, n, o, m=method: _solve(p, n, o, m), cost=_cost(method), traits=("nonoscillatory",) + traits,
        description=description,
    )


SOLVERS = (
    _spec("burgers_inviscid", "rusanov", 1.0, "Rusanov (local Lax-Friedrichs) finite volume, first order, monotone",
          ("periodic", "dirichlet"), ("dissipative",)),
    _spec("burgers_inviscid", "muscl", 2.0, "MUSCL with minmod limiter and SSP-RK2, second order, no new extrema",
          ("periodic", "dirichlet")),
    _spec("burgers_inviscid", "weno5", 3.0, "WENO5 with Lax-Friedrichs splitting and SSP-RK3 (third order overall)",
          ("periodic", "dirichlet")),
    # for linear advection the limited schemes are only offered when the data jump: on smooth data the limiter
    # clips extrema and the linear schemes (Lax-Wendroff, RK4 + central differences) are cheaper for the same error
    _spec("advection", "muscl", 2.0, "MUSCL with minmod limiter and SSP-RK2; no oscillations at jumps",
          ("periodic",), accepts=_has_jumps),
    _spec("advection", "weno5", 3.0, "WENO5 with SSP-RK3 (fifth order in space, third in time); non-oscillatory at jumps",
          ("periodic",), accepts=_has_jumps),
)
