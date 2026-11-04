"""Cross-checks for answers without an exact solution.

Grid refinement estimates the error of a solution from how it changes when the
grid is refined. That estimate cannot see a solution that converges, but to the
wrong answer (a solver with a wrong sign or coefficient, a boundary handled
wrongly) or one that is not yet in its asymptotic range. Two independent
checks look at the answer itself instead:

* **residual spot check**: put the numerical solution back into the equation.
  A smooth interpolant (quintic spline) through the grid values gives the
  spatial derivatives, two extra runs at ``T - d`` and ``T + d`` give the time
  derivatives, and ``L(u) - f`` is sampled at random points between the
  nodes. It is reported relative to the size of the largest term of the
  equation, so it says "the equation holds to x %".
* **conservation drift**: where the equation conserves something, the answer
  has to as well. The checks are found from the equation itself, by applying
  its operator to test functions, so learned equations get them too:

  - mass ``∫u``: periodic boundaries and a spatial part in divergence form
    (``∫ L(w) = 0`` for every periodic ``w``); the source and, for second
    order equations, the initial velocity are booked in,
  - energy ``½∫u²`` of a first order equation without source, on periodic or
    zero boundaries: conserved when ``∫ w L(w) = 0`` (transport), not allowed
    to grow when ``∫ w L(w) ≥ 0`` (diffusion, viscous Burgers),
  - energy ``½∫u_t² + ½∫u S(u)`` of a linear second order equation
    ``u_tt + S(u) = 0`` (waves) on periodic or zero boundaries.

A check that does not apply (an ODE, 3-D, an operator sympy cannot handle) is
left out, never failed. ``cross_check`` runs them all and says whether the
answer passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline, make_interp_spline

from .problem import PDEProblem, T, X, Y
from .solvers.base import SolveResult, SolverSpec, evaluate, mesh

# how far the answer may be off, as a multiple of the tolerance (and never less than the floor)
RESIDUAL_FACTOR = 20.0
RESIDUAL_FLOOR = 0.01
DRIFT_FACTOR = 10.0
DRIFT_FLOOR = 1e-6
TIME_STEP = 0.02  # the extra runs sit at most T*TIME_STEP before and after T
STEPS_APART = 2  # ... and, for a time-stepping solver, this many of its own steps
PROBES = 400
MARGIN = 0.05  # probes keep this fraction of each side away from the boundary

CHECK_NAMES = {"residual": "残差抽查", "mass": "质量守恒", "energy": "能量守恒", "energy_decay": "能量不增"}


@dataclass
class Check:
    name: str  # residual | mass | energy | energy_decay
    value: float  # relative residual or relative drift
    limit: float
    passed: bool
    detail: str = ""

    def describe(self) -> str:
        mark = "通过" if self.passed else "没通过"
        return f"{CHECK_NAMES.get(self.name, self.name)} {self.value:.1e}（上限 {self.limit:.0e}，{mark}）"

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "limit": self.limit, "passed": self.passed,
                "detail": self.detail}


@dataclass
class CrossCheck:
    checks: list[Check] = field(default_factory=list)
    work: float = 0.0
    skipped: str = ""  # why nothing could be checked

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def describe(self) -> str:
        if not self.checks:
            return f"交叉验证：没有可做的检查（{self.skipped}）" if self.skipped else ""
        return "交叉验证：" + "；".join(c.describe() for c in self.checks)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks], "skipped": self.skipped}


# --- the operator in terms of the unknown function ----------------------------------

def _unknown(dim: int):
    return sp.Function("U")(X, T) if dim == 1 else sp.Function("U")(X, Y, T)


@dataclass
class _Symbolic:
    """``L(U)`` for an unknown function ``U``, its terms and the derivatives they need."""

    expr: sp.Expr
    terms: list[sp.Expr]
    needs: dict[sp.Expr, tuple[int, int, int]]  # U or a derivative of U -> (order in x, y, t)
    linear: bool


def symbolic_operator(equation, coefficients: dict, dim: int) -> _Symbolic | None:
    """The operator applied to an unknown function, or None if sympy cannot do that for this operator."""
    U = _unknown(dim)
    try:
        expr = sp.expand(equation.apply(U, coefficients))
        doubled = sp.expand(equation.apply(2 * U, coefficients))
    except Exception:
        return None
    if not isinstance(expr, sp.Expr) or expr.free_symbols - {X, Y, T}:
        return None
    needs: dict[sp.Expr, tuple[int, int, int]] = {}
    for d in expr.atoms(sp.Derivative):
        if d.expr != U:
            return None
        counts = {X: 0, Y: 0, T: 0}
        for var, k in d.variable_count:
            counts[var] += int(k)
        needs[d] = (counts[X], counts[Y], counts[T])
    if expr.has(U):
        needs[U] = (0, 0, 0)
    if any(order[2] > 2 or order[0] + order[1] > 4 for order in needs.values()):
        return None
    linear = sp.simplify(doubled - 2 * expr) == 0
    return _Symbolic(expr, list(sp.Add.make_args(expr)), needs, linear)


# --- smooth interpolants of a grid solution ----------------------------------------

class _Interpolant:
    def __init__(self, result: SolveResult):
        axes = result.grid
        k = 5 if result.n >= 6 else 3
        if len(axes) == 1:
            self.spline = make_interp_spline(axes[0], result.u, k=k)
        else:
            self.spline = RectBivariateSpline(axes[0], axes[1], result.u, kx=k, ky=k, s=0)
        self.dim = len(axes)

    def __call__(self, points: tuple[np.ndarray, ...], dx: int = 0, dy: int = 0) -> np.ndarray:
        if self.dim == 1:
            return self.spline(points[0], nu=dx) if dy == 0 else np.zeros_like(points[0])
        return self.spline.ev(points[0], points[1], dx=dx, dy=dy)


def _probes(problem: PDEProblem, count: int = PROBES, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    out = []
    for a, b in problem.domain:
        pad = MARGIN * (b - a)
        out.append(rng.uniform(a + pad, b - pad, count))
    return tuple(out)


def _shifted(problem: PDEProblem, t_final: float) -> PDEProblem:
    data = problem.to_dict()
    data["t_final"] = t_final
    return PDEProblem.from_dict(data)


def _values(sym: _Symbolic, needs_values: dict, points, t: float, f: np.ndarray):
    """``L(u)`` term by term at the probes, from the numeric values of U and its derivatives."""
    symbols = {atom: sp.Symbol(f"q{i}") for i, atom in enumerate(sym.needs)}
    args = (X, Y, T, *symbols.values())
    x = points[0]
    y = points[1] if len(points) > 1 else np.zeros_like(x)
    tt = np.full_like(x, t)
    numeric = [needs_values[atom] for atom in symbols]
    terms = []
    for term in sym.terms:
        fn = sp.lambdify(args, term.xreplace(symbols), modules=["scipy", "numpy"])
        terms.append(np.broadcast_to(np.asarray(fn(x, y, tt, *numeric), dtype=float), x.shape))
    return terms


def time_offset(problem: PDEProblem, result: SolveResult) -> float:
    """How far before and after T the extra runs go.

    A few of the solver's own steps when it reports its step: the runs then
    take the same steps and continue the same discrete trajectory, so their
    difference is not drowned by the error of each run.
    """
    t_final = problem.t_final or 0.0
    delta = TIME_STEP * t_final
    if result.dt and result.dt > 0:
        delta = min(delta, STEPS_APART * result.dt)
    return delta


class Neighbours:
    """The answer at T - d and T + d on the same grid, run once when first needed."""

    def __init__(self, spec: SolverSpec, problem: PDEProblem, result: SolveResult, options: dict | None):
        self.spec, self.problem, self.result, self.options = spec, problem, result, options
        self.delta = time_offset(problem, result)
        self.work = 0.0
        self._runs: dict[int, SolveResult] = {}

    def __getitem__(self, sign: int) -> SolveResult:
        if sign not in self._runs:
            shifted = _shifted(self.problem, self.problem.t_final + sign * self.delta)
            self._runs[sign] = self.spec.run(shifted, self.result.n, self.options)
            self.work += self._runs[sign].work
        return self._runs[sign]


def residual_check(problem: PDEProblem, result: SolveResult, sym: _Symbolic, tol: float,
                   neighbours: Neighbours | None) -> Check | None:
    """Relative residual of the answer at random points between the nodes."""
    points = _probes(problem)
    t_final = problem.t_final or 0.0
    now = _Interpolant(result)
    others: dict[int, _Interpolant] = {}
    if any(order[2] for order in sym.needs.values()):
        if not problem.time_dependent or neighbours is None:
            return None
        others = {sign: _Interpolant(neighbours[sign]) for sign in (-1, 1)}
        delta = neighbours.delta
    values = {}
    for atom, (dx, dy, dt) in sym.needs.items():
        if dt == 0:
            values[atom] = now(points, dx, dy)
        elif dt == 1:
            values[atom] = (others[1](points, dx, dy) - others[-1](points, dx, dy)) / (2 * delta)
        else:
            values[atom] = (others[1](points, dx, dy) - 2 * now(points, dx, dy) + others[-1](points, dx, dy)) / delta**2
    with np.errstate(all="ignore"):
        f = evaluate(problem.fn("source"), points, t_final)
        terms = _values(sym, values, points, t_final, f)
    residual = sum(terms) - f
    size = np.max(np.abs(np.stack([*terms, f])), axis=0)
    scale = float(np.sqrt(np.mean(size**2)))
    limit = _residual_limit(tol)
    if not np.all(np.isfinite(residual)) or not math.isfinite(scale):
        return Check("residual", math.inf, limit, False, "残差不是有限数")
    if scale < 1e-12:
        return None  # every term vanishes: nothing to compare with
    value = float(np.sqrt(np.mean(residual**2))) / scale
    return Check("residual", value, limit, value <= limit, f"{PROBES} 个网格点之间的随机点")


def _residual_limit(tol: float) -> float:
    return max(RESIDUAL_FACTOR * tol, RESIDUAL_FLOOR)


def _drift_limit(tol: float) -> float:
    return max(DRIFT_FACTOR * tol, DRIFT_FLOOR)


# --- what the equation conserves --------------------------------------------------

def _test_functions(problem: PDEProblem, periodic: bool, count: int = 4, seed: int = 1) -> list[sp.Expr]:
    """Smooth time-independent functions that are periodic on the domain, or vanish on its boundary."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        w = sp.Integer(1)
        for axis, (a, b) in zip((X, Y), problem.domain):
            s = (axis - a) / (b - a)
            k1, k2 = rng.integers(1, 4, size=2)
            c1, c2, phase = (float(v) for v in rng.uniform(0.3, 1.0, size=3))
            if periodic:
                part = c1 * sp.sin(2 * sp.pi * k1 * s + phase) + c2 * sp.cos(2 * sp.pi * k2 * s) + 0.5
            else:
                part = c1 * sp.sin(sp.pi * k1 * s) + c2 * sp.sin(sp.pi * (k2 + 3) * s)
            w = w * part
        out.append(w)
    return out


def _fine_points(problem: PDEProblem, m: int = 512):
    axes = tuple(np.linspace(a, b, m + 1) for a, b in problem.domain)
    return axes, mesh(axes)


def _integrate(values: np.ndarray, axes: tuple[np.ndarray, ...]) -> float:
    out = values
    for axis in reversed(axes):
        out = simpson(out, x=axis, axis=-1)
    return float(out)


def _field(expr: sp.Expr, points, t: float = 0.0) -> np.ndarray:
    fn = sp.lambdify((X, Y, T), expr, modules=["scipy", "numpy"])
    x = points[0]
    y = points[1] if len(points) > 1 else np.zeros_like(x)
    return np.broadcast_to(np.asarray(fn(x, y, np.full_like(x, t)), dtype=float), x.shape)


def _sign_on_tests(equation, problem: PDEProblem, periodic: bool, weight: bool) -> list[float] | None:
    """``∫ L(w)`` (or ``∫ w L(w)``) over test functions, relative to the size of the integrand."""
    axes, points = _fine_points(problem, 256)
    out = []
    try:
        for w in _test_functions(problem, periodic):
            lw = equation.apply(w, problem.coefficients)
            integrand = _field(w * lw if weight else lw, points)
            scale = _integrate(np.abs(integrand), axes)
            if not math.isfinite(scale):
                return None
            out.append(_integrate(integrand, axes) / scale if scale > 1e-14 else 0.0)
    except Exception:
        return None
    return out


def _zero(expr: sp.Expr | None) -> bool:
    return expr is not None and expr.is_zero is True


def conservation_laws(equation, problem: PDEProblem, sym: _Symbolic | None) -> list[str]:
    """Which of mass, energy and energy_decay the answer to this problem has to respect."""
    if equation is None or problem.dim not in (1, 2) or not problem.time_dependent:
        return []
    periodic = problem.boundary == "periodic"
    zero_wall = problem.boundary == "dirichlet" and _zero(problem.expr("boundary_value"))
    no_source = _zero(problem.expr("source"))
    laws = []
    if periodic and equation.time_order in (1, 2):
        mass = _sign_on_tests(equation, problem, True, weight=False)
        if mass is not None and all(abs(v) < 1e-8 for v in mass):
            laws.append("mass")
    if no_source and (periodic or zero_wall):
        signs = _sign_on_tests(equation, problem, periodic, weight=True)
        if signs is None:
            return laws
        if equation.time_order == 1:
            if all(abs(v) < 1e-8 for v in signs):
                laws.append("energy")
            elif all(v > -1e-8 for v in signs):
                laws.append("energy_decay")
        elif equation.time_order == 2 and sym is not None and sym.linear and _plain_second_order(sym) \
                and all(v > -1e-8 for v in signs):
            laws.append("energy")
    return laws


def _plain_second_order(sym: _Symbolic) -> bool:
    """``L(U) = U_tt + S(U)`` with S free of time derivatives."""
    tt = [atom for atom, order in sym.needs.items() if order == (0, 0, 2)]
    if len(tt) != 1 or any(order[2] == 1 or (order[2] == 2 and order != (0, 0, 2)) for order in sym.needs.values()):
        return False
    return sp.simplify(sp.diff(sym.expr, tt[0]) - 1) == 0


def _grid_integral(u: np.ndarray, result: SolveResult) -> float:
    return _integrate(u, result.grid)


def drift_checks(laws: list[str], equation, problem: PDEProblem, result: SolveResult,
                 sym: _Symbolic | None, tol: float, neighbours: Neighbours) -> list[Check]:
    """Compare what the answer holds at the final time with what the data put in."""
    checks: list[Check] = []
    if not laws:
        return checks
    t_final = problem.t_final
    axes, points = _fine_points(problem)
    volume = float(np.prod([b - a for a, b in problem.domain]))
    initial = problem.expr("initial")
    u0 = _field(initial, points)
    limit = _drift_limit(tol)

    if "mass" in laws:
        mass0 = _integrate(u0, axes)
        expected = mass0
        nodes, weights = np.polynomial.legendre.leggauss(24)
        s = 0.5 * t_final * (nodes + 1)
        w = 0.5 * t_final * weights
        forcing = np.array([_integrate(evaluate(problem.fn("source"), points, si), axes) for si in s])
        if equation.time_order == 1:
            expected += float(np.sum(w * forcing))
        else:
            v0 = _field(problem.expr("initial_velocity"), points)
            expected += t_final * _integrate(v0, axes) + float(np.sum(w * (t_final - s) * forcing))
        mass = _grid_integral(result.u, result)
        size = volume * max(float(np.max(np.abs(u0))), float(np.max(np.abs(result.u))), 1e-300)
        value = abs(mass - expected) / size
        checks.append(Check("mass", value, limit, value <= limit, f"∫u 末时刻 {mass:.6g}，应为 {expected:.6g}"))

    if "energy" in laws and equation.time_order == 1 or "energy_decay" in laws:
        e0 = 0.5 * _integrate(u0**2, axes)
        e1 = 0.5 * _grid_integral(result.u**2, result)
        if e0 > 1e-300:
            drift = (e1 - e0) / e0
            if "energy" in laws:
                value = abs(drift)
                checks.append(Check("energy", value, limit, value <= limit, f"½∫u² 从 {e0:.6g} 变成 {e1:.6g}"))
            else:
                value = max(drift, 0.0)
                checks.append(Check("energy_decay", value, limit, value <= limit,
                                    f"½∫u² 从 {e0:.6g} 变成 {e1:.6g}（只许减少）"))

    if "energy" in laws and equation.time_order == 2 and sym is not None:
        try:
            static = [atom for atom, order in sym.needs.items() if order != (0, 0, 2)]
            tt = next(atom for atom, order in sym.needs.items() if order == (0, 0, 2))
            spatial = _Symbolic(sym.expr.xreplace({tt: 0}), [], {a: sym.needs[a] for a in static}, True)
            spatial.terms = [spatial.expr]
            v0 = _field(problem.expr("initial_velocity"), points)
            s0 = _field(equation.apply(initial, problem.coefficients), points)
            e0 = 0.5 * _integrate(v0**2, axes) + 0.5 * _integrate(u0 * s0, axes)
            ut = (neighbours[1].u - neighbours[-1].u) / (2 * neighbours.delta)
            now = _Interpolant(result)
            gaxes = result.grid
            gpoints = mesh(gaxes)
            values = {atom: now(gpoints, dx, dy) for atom, (dx, dy, _) in spatial.needs.items()}
            s1 = _values(spatial, values, gpoints, t_final, np.zeros_like(result.u))[0]
            e1 = 0.5 * _integrate(ut**2, gaxes) + 0.5 * _integrate(result.u * s1, gaxes)
        except Exception:
            return checks
        if e0 > 1e-300:
            value = abs(e1 - e0) / e0
            checks.append(Check("energy", value, limit, value <= limit, f"能量从 {e0:.6g} 变成 {e1:.6g}"))
    return checks


def cross_check(equation, spec: SolverSpec, problem: PDEProblem, result: SolveResult, tol: float,
                options: dict | None = None) -> CrossCheck:
    """All checks that apply to this answer. Never raises: a check that cannot run is left out."""
    out = CrossCheck()
    if problem.dim == 0:
        out.skipped = "常微分方程"
        return out
    if problem.dim == 3:
        out.skipped = "三维问题"
        return out
    if equation is None:
        out.skipped = "不知道方程的算子"
        return out
    sym = symbolic_operator(equation, problem.coefficients, problem.dim)
    neighbours = Neighbours(spec, problem, result, options) if problem.time_dependent else None
    if sym is not None:
        try:
            check = residual_check(problem, result, sym, tol, neighbours)
            if check is not None:
                out.checks.append(check)
        except Exception:
            pass
    try:
        out.checks += drift_checks(conservation_laws(equation, problem, sym), equation, problem, result, sym, tol,
                                   neighbours)
    except Exception:
        pass
    out.work = neighbours.work if neighbours else 0.0
    if not out.checks:
        out.skipped = "算子太特殊，没法代回检查"
    return out
