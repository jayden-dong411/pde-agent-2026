"""The result contract: does an answer belong to the problem that was asked?

Refinement estimates and the residual check look at the equation inside the
domain. An answer can satisfy the equation there and still solve a different
problem: the right function plus a constant (wrong boundary values), the
right equation started from the wrong initial data, a state at the wrong
time or on shifted coordinates. These checks look at exactly that:

* **grid**: the axes are ``linspace(a, b, n + 1)`` of the problem's domain,
* **time**: the solver reports the final time it reached (``result.t``, or
  ``steps * dt``) and it is ``t_final``,
* **boundary**: on a Dirichlet problem whose spatial operator is of second
  order (Poisson, heat, waves, reaction-diffusion), every boundary node holds
  the boundary data. First-order hyperbolic equations take data only at the
  inflow, which depends on the sign of the velocity (or of ``u`` itself), so
  their boundary is left to the residual and conservation checks,
* **periodic**: the two ends of every periodic axis agree,
* **initial** (and **initial_velocity** for second order in time): one more
  run up to a tiny time ``d`` has to start from the initial data, and for a
  second-order equation move with the initial velocity.

All checks are required. A check whose run raises is inconclusive, so the
answer stays a candidate instead of being verified.
"""

from __future__ import annotations

import math

import numpy as np

from .crosscheck import Check, CrossCheck, symbolic_operator
from .solvers.base import SolveResult, SolverSpec, evaluate, mesh
from .verify import relative_error

INITIAL_TIME = 1e-6  # the initial-data run goes to t_final * INITIAL_TIME (at most one of the solver's steps): short
# enough that even a stiff mode (decay rate 1e4 / t_final) has barely moved
BOUNDARY_FACTOR = 10.0  # boundary and initial data may be off by this multiple of tol...
BOUNDARY_FLOOR = 1e-3  # ... and never less than this (relative to the size of the answer)
INITIAL_FLOOR = 0.02  # a run of length d moves the answer by about d*|u_t|: the initial check is looser
TIME_RTOL = 1e-9


def _limit(tol: float, floor: float = BOUNDARY_FLOOR) -> float:
    return max(BOUNDARY_FACTOR * tol, floor)


def grid_check(problem, result: SolveResult) -> Check | None:
    if problem.dim == 0:
        return None
    if len(result.grid) != problem.dim:
        return Check("grid", math.inf, 0.0, False, f"返回了 {len(result.grid)} 个坐标轴，题目是 {problem.dim} 维")
    worst = 0.0
    for axis, (a, b) in zip(result.grid, problem.domain):
        axis = np.asarray(axis, dtype=float)
        if axis.shape != (result.n + 1,):
            return Check("grid", math.inf, 0.0, False, f"坐标轴有 {axis.size} 个点，应为 n + 1 = {result.n + 1}")
        worst = max(worst, float(np.max(np.abs(axis - np.linspace(a, b, result.n + 1)))) / (b - a))
    limit = 1e-9
    return Check("grid", worst, limit, worst <= limit, "坐标应为区域上的等距节点")


def time_check(problem, result: SolveResult) -> Check | None:
    if not problem.time_dependent:
        return None
    t_final = problem.t_final
    if result.t is not None:
        reached, how = float(result.t), "求解器报告的末时刻"
    elif result.steps and result.dt:
        reached, how = float(result.steps * result.dt), "步数 × 步长"
    else:
        return Check.inconclusive("time", "求解器没有报告算到了哪个时刻", required=False)
    value = abs(reached - t_final) / t_final
    return Check("time", value, TIME_RTOL, value <= TIME_RTOL, f"{how} {reached:.6g}，应为 {t_final:.6g}")


def _strong_dirichlet(equation, problem) -> bool | None:
    """Is every boundary node a Dirichlet node? ``None`` when the operator cannot be read."""
    if equation is None:
        return None
    sym = symbolic_operator(equation, problem.coefficients, problem.dim) if problem.dim in (1, 2) else None
    if sym is None:
        if problem.dim == 3:  # no sympy view of 3-D operators here: judge by the known kinds
            return problem.kind in ("poisson", "heat", "wave")
        return None
    return max((dx + dy for dx, dy, _ in sym.needs.values()), default=0) >= 2


def _boundary_mask(shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for axis in range(len(shape)):
        index = [slice(None)] * len(shape)
        index[axis] = 0
        mask[tuple(index)] = True
        index[axis] = -1
        mask[tuple(index)] = True
    return mask


def boundary_check(equation, problem, result: SolveResult, tol: float) -> Check | None:
    if problem.dim == 0:
        return None
    u = np.asarray(result.u, dtype=float)
    scale = max(float(np.max(np.abs(u))), 1e-12)
    limit = _limit(tol)
    if problem.boundary == "periodic":
        worst = 0.0
        for axis in range(u.ndim):
            worst = max(worst, float(np.max(np.abs(np.take(u, 0, axis=axis) - np.take(u, -1, axis=axis)))))
        value = worst / scale
        return Check("periodic", value, limit, value <= limit, "两端的值应相同")
    strong = _strong_dirichlet(equation, problem)
    if strong is None:
        return Check.inconclusive("boundary", "算子没法读，不知道哪些边界要满足 Dirichlet 条件", required=False)
    if not strong:
        return None  # first order in space: data only at the inflow (see the module docstring)
    points = mesh(tuple(np.asarray(a, dtype=float) for a in result.grid))
    data = problem.for_solver()
    with np.errstate(all="ignore"):
        g = np.broadcast_to(evaluate(data.fn("boundary_value"), points[:2], problem.t_final or 0.0), u.shape)
    mask = _boundary_mask(u.shape)
    gap = np.abs(u[mask] - g[mask])
    scale = max(scale, float(np.max(np.abs(g[mask]))))
    value = float(np.max(gap)) / scale if np.all(np.isfinite(gap)) else math.inf
    at = int(np.argmax(gap))
    detail = f"边界上最大偏差 {float(gap[at]):.3g}（解 {float(u[mask][at]):.6g}，边界值 {float(g[mask][at]):.6g}）"
    return Check("boundary", value, limit, value <= limit, detail)


def _at_time(problem, t: float):
    data = problem.to_dict()
    data["t_final"] = t
    return type(problem).from_dict(data)


def initial_checks(equation, spec: SolverSpec, problem, result: SolveResult, tol: float,
                   options: dict | None) -> tuple[list[Check], float]:
    """Run to a tiny time and compare with the initial data (and velocity). Returns the checks and their work."""
    if problem.dim == 0 or not problem.time_dependent:
        return [], 0.0
    t_final = problem.t_final
    delta = t_final * INITIAL_TIME
    if result.dt and result.dt > 0:
        delta = min(delta, float(result.dt))
    limit = _limit(tol, INITIAL_FLOOR)
    second = equation is not None and equation.time_order == 2
    try:
        early = spec.run(_at_time(problem, delta), result.n, options)
    except Exception as exc:
        checks = [Check.inconclusive("initial", f"到 t = {delta:.3g} 的短程运行出错：{type(exc).__name__}", limit=limit)]
        return checks, 0.0
    points = mesh(tuple(np.asarray(a, dtype=float) for a in early.grid))
    data = problem.for_solver()
    with np.errstate(all="ignore"):
        u0 = np.broadcast_to(evaluate(data.fn("initial"), points[:2], 0.0), early.u.shape)
    # in L1: data with a jump are smoothed over a width ~ sqrt(d) by the first step, which the max-norm
    # would count as a full jump; a wrong start (an offset, other data) is just as visible in L1
    value = relative_error(early.u, u0, "l1")
    if not math.isfinite(value):
        value = math.inf
    checks = [Check("initial", value, limit, value <= limit, f"t = {delta:.3g} 时与初值的相对差")]
    if second:
        with np.errstate(all="ignore"):
            v0 = np.broadcast_to(evaluate(data.fn("initial_velocity"), points[:2], 0.0), early.u.shape)
        moved = (np.asarray(early.u, dtype=float) - u0) / delta
        size = max(float(np.mean(np.abs(v0))), float(np.mean(np.abs(u0))) / t_final, 1e-12)
        gap = float(np.mean(np.abs(moved - v0))) / size
        if not math.isfinite(gap):
            gap = math.inf
        checks.append(Check("initial_velocity", gap, limit, gap <= limit,
                            f"(u(d) - u0)/d 与初速度的相对差，d = {delta:.3g}"))
    return checks, early.work


def result_contract(equation, spec: SolverSpec, problem, result: SolveResult, tol: float,
                    options: dict | None = None) -> CrossCheck:
    """The checks of the problem's own conditions on one answer. Never raises."""
    out = CrossCheck()
    for make in (lambda: grid_check(problem, result), lambda: time_check(problem, result),
                 lambda: boundary_check(equation, problem, result, tol)):
        try:
            check = make()
        except Exception as exc:
            check = Check.inconclusive("boundary", f"检查出错：{type(exc).__name__}: {exc}"[:200])
        if check is not None:
            out.checks.append(check)
    try:
        checks, work = initial_checks(equation, spec, problem, result, tol, options)
    except Exception as exc:
        checks, work = [Check.inconclusive("initial", f"检查出错：{type(exc).__name__}")], 0.0
    out.checks += checks
    out.work += work
    return out


CONTRACT_CHECKS = ("grid", "time", "boundary", "periodic", "initial", "initial_velocity")
