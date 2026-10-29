"""Methods for ODE systems ``y' = f(t, y)`` (curriculum stages 0 and 1).

Fixed-step methods take ``n`` as the number of time steps. Adaptive methods
(scipy's RK45, Radau and BDF) take it as a tolerance, ``rtol = n^-2``, so
doubling ``n`` is a refinement for them too and their error falls like
``n^-2`` (order 2 in the knob).

Work is counted, not timed: right-hand-side evaluations, Jacobians, LU
factorisations and linear solves, each weighted by ``ODEProblem.costs``.
Every method reports the same counts, so explicit and implicit methods are
compared on equal terms.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from ..ode import NewtonFailed, implicit_step, new_counts
from .base import SolveResult, SolverDiverged, SolverSpec, check_growth, growth_limit

MAX_FEVALS = 2_000_000


def work(problem, counts: dict) -> float:
    c = problem.costs
    return float(sum(counts[k] * c[k] for k in ("f", "jac", "lu", "solve")))


def _result(problem, y: np.ndarray, n: int, counts: dict, **info) -> SolveResult:
    return SolveResult(u=np.asarray(y, dtype=float), grid=(), n=n, t=problem.t_final,
                       steps=info.pop("steps", n), dt=problem.t_final / n, work=work(problem, counts),
                       info={**info, "counts": dict(counts)})


def _check_n(n: int) -> None:
    if n < 1:
        raise ValueError("need at least one time step")


# --- explicit fixed-step -----------------------------------------------------------


def _explicit(name: str, stages: int, step):
    def solve(problem, n: int, options: dict) -> SolveResult:
        _check_n(n)
        h, y, counts = problem.t_final / n, problem.y0.copy(), new_counts()
        limit = growth_limit(y, problem.f(0.0, y))
        with np.errstate(all="ignore"):
            for k in range(n):
                y = step(problem, k * h, y, h)
                if k % 64 == 0:
                    check_growth(y, limit, name, k)
        check_growth(y, limit, name, n)
        counts["f"] = stages * n
        return _result(problem, y, n, counts)

    return solve


def _euler(p, t, y, h):
    return y + h * p.f(t, y)


def _heun(p, t, y, h):
    k1 = p.f(t, y)
    k2 = p.f(t + h, y + h * k1)
    return y + 0.5 * h * (k1 + k2)


def _rk4(p, t, y, h):
    k1 = p.f(t, y)
    k2 = p.f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = p.f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = p.f(t + h, y + h * k3)
    return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_verlet(problem, n: int, options: dict) -> SolveResult:
    """Stormer-Verlet for ``q' = g(p), p' = h(q)``: symplectic, so energy errors stay bounded."""
    _check_n(n)
    if not problem.separable:
        raise ValueError("Verlet needs y = (q, p) with q' depending only on p and p' only on q")
    half, h = problem.size // 2, problem.t_final / n
    y = problem.y0.copy()
    limit = growth_limit(y, problem.f(0.0, y))
    force = problem.f_half(0.0, y, second=True)
    with np.errstate(all="ignore"):
        for k in range(n):
            y[half:] += 0.5 * h * force
            y[:half] += h * problem.f_half(0.0, y, second=False)
            force = problem.f_half(0.0, y, second=True)
            y[half:] += 0.5 * h * force
            if k % 64 == 0:
                check_growth(y, limit, "ode.verlet", k)
    check_growth(y, limit, "ode.verlet", n)
    counts = new_counts()
    counts["f"] = n  # two half evaluations per step, each of half the components
    return _result(problem, y, n, counts)


# --- implicit fixed-step -----------------------------------------------------------


def _implicit(name: str, bdf2: bool, theta: float):
    """theta = 1: backward Euler; theta = 1/2: trapezoid; bdf2 starts with one trapezoid step."""

    def solve(problem, n: int, options: dict) -> SolveResult:
        _check_n(n)
        h, counts = problem.t_final / n, new_counts()
        y = problem.y0.copy()
        y_old = None
        fy = problem.f(0.0, y)
        counts["f"] += 1
        limit = growth_limit(y, fy)
        try:
            for k in range(n):
                t_new = (k + 1) * h
                if bdf2 and y_old is not None:
                    ch, b = 2.0 * h / 3.0, (4.0 * y - y_old) / 3.0
                else:
                    th = 0.5 if bdf2 else theta
                    ch, b = th * h, y + (1.0 - th) * h * fy
                y_new = implicit_step(problem, t_new, y, ch, b, counts)
                fy = (y_new - b) / ch  # f(t_new, y_new), exactly as the step used it
                y_old, y = y, y_new
                if k % 64 == 0:
                    check_growth(y, limit, name, k)
        except NewtonFailed as exc:
            raise SolverDiverged(f"{name}: {exc} at step {k}") from exc
        check_growth(y, limit, name, n)
        return _result(problem, y, n, counts)

    return solve


# --- adaptive (scipy) --------------------------------------------------------------


class _TooManySteps(RuntimeError):
    pass


def rtol_for(n: int) -> float:
    return float(n) ** -2


def _adaptive(name: str, method: str, implicit: bool):
    def solve(problem, n: int, options: dict) -> SolveResult:
        _check_n(n)
        rtol = options.get("rtol", rtol_for(n))
        calls = [0]

        def fun(t, y):
            calls[0] += 1
            if calls[0] > MAX_FEVALS:
                raise _TooManySteps(f"{name} needed more than {MAX_FEVALS} evaluations")
            return problem.f(t, y)

        kwargs = {"jac": problem.jac} if implicit else {}
        try:
            with np.errstate(all="ignore"):
                sol = solve_ivp(fun, (0.0, problem.t_final), problem.y0, method=method, rtol=rtol,
                                atol=rtol * 1e-6, **kwargs)
        except _TooManySteps as exc:
            raise SolverDiverged(str(exc)) from exc
        if sol.status != 0:
            raise SolverDiverged(f"{name}: {sol.message}")
        counts = new_counts()
        counts["f"] = int(sol.nfev)
        if implicit:
            counts["jac"], counts["lu"], counts["solve"] = int(sol.njev), int(sol.nlu), int(sol.nfev)
        return _result(problem, sol.y[:, -1], n, counts, steps=sol.t.size - 1, rtol=rtol)

    return solve


# --- cost models: predicted work before running (see features.predict) -------------


def _fixed_cost(evals_per_step: float):
    def cost(problem, n: int) -> float:
        return n * evals_per_step * problem.costs["f"]

    return cost


def _implicit_cost(problem, n: int) -> float:
    c = problem.costs  # a step is about two Newton iterations with one Jacobian
    return n * (2 * c["f"] + c["jac"] + c["lu"] + 2 * c["solve"])


def _adaptive_steps(problem, n: int, c: float, order: float) -> float:
    """Steps an adaptive method takes at ``rtol = n^-2``: more for more oscillations (fitted on the curriculum)."""
    rate = problem.spectrum()["rate"]
    scale = max(1.0, rate * problem.t_final / (2 * math.pi))
    return 2.0 + c * scale * rtol_for(n) ** (-1.0 / order)


def _cost_rk45(problem, n: int) -> float:
    s = problem.spectrum()
    steps = max(_adaptive_steps(problem, n, 2.0, 6), s["rho"] * problem.t_final / 3.3)  # stability holds the step down
    return steps * 7 * problem.costs["f"]


def _cost_radau(problem, n: int) -> float:
    c = problem.costs
    return _adaptive_steps(problem, n, 3.0, 5) * (6 * c["f"] + 0.5 * c["jac"] + 2 * c["lu"] + 6 * c["solve"])


def _cost_bdf(problem, n: int) -> float:
    c = problem.costs
    return _adaptive_steps(problem, n, 10.0, 7) * (2 * c["f"] + 0.2 * c["jac"] + 0.3 * c["lu"] + 2 * c["solve"])


def _damping_trapezoid(problem, n: int, rate: float) -> float:
    """What the trapezoid rule leaves of a mode decaying at ``rate``: |1 - z/2| / |1 + z/2| per step."""
    z = rate * problem.t_final / n
    return abs((1 - z / 2) / (1 + z / 2)) ** n


def _spec(name, order, solve, description, **kw) -> SolverSpec:
    return SolverSpec(name=name, kind="ode", dims=(0,), boundaries=("ivp",), order=order, solve=solve,
                      description=description, **kw)


SOLVERS = (
    _spec("ode.euler", 1.0, _explicit("ode.euler", 1, _euler), "Explicit Euler, first order",
          cost=_fixed_cost(1), stability=2.0),
    _spec("ode.heun", 2.0, _explicit("ode.heun", 2, _heun), "Heun / explicit RK2, second order",
          cost=_fixed_cost(2), stability=2.0),
    _spec("ode.rk4", 4.0, _explicit("ode.rk4", 4, _rk4), "Classical explicit Runge-Kutta, fourth order",
          cost=_fixed_cost(4), stability=2.785),
    _spec("ode.verlet", 2.0, solve_verlet, "Stormer-Verlet, symplectic, for q' = g(p), p' = h(q)",
          cost=_fixed_cost(1), stability=2.0, accepts=lambda p: p.separable),
    _spec("ode.rk45", 2.0, _adaptive("ode.rk45", "RK45", False),
          "Dormand-Prince RK45, explicit adaptive (scipy); n sets rtol = n^-2",
          cost=_cost_rk45, traits=("adaptive",)),
    _spec("ode.backward_euler", 1.0, _implicit("ode.backward_euler", False, 1.0),
          "Implicit (backward) Euler, L-stable, first order", cost=_implicit_cost),
    _spec("ode.trapezoid", 2.0, _implicit("ode.trapezoid", False, 0.5),
          "Trapezoid rule (Crank-Nicolson), A-stable, second order", cost=_implicit_cost,
          damping=_damping_trapezoid),
    _spec("ode.bdf2", 2.0, _implicit("ode.bdf2", True, 1.0), "BDF2, L-stable, second order",
          cost=_implicit_cost),
    _spec("ode.radau", 2.0, _adaptive("ode.radau", "Radau", True),
          "Radau IIA, implicit adaptive, fifth order (scipy); n sets rtol = n^-2",
          cost=_cost_radau, traits=("adaptive",)),
    _spec("ode.bdf", 2.0, _adaptive("ode.bdf", "BDF", True),
          "Variable-order BDF 1-5, implicit adaptive (scipy); n sets rtol = n^-2",
          cost=_cost_bdf, traits=("adaptive",)),
)

__all__ = ["SOLVERS", "rtol_for", "work"]
