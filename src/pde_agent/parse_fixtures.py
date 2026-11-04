"""Chinese problem descriptions with their known correct parse (``pde-agent eval --parse``).

Each fixture is a sentence a user might type and the problem it means. A parse
is graded by what it means for the solver, not by how it is written:

* kind, domain, final time and boundary type must match; for a known
  equation kind the coefficients too (after defaults are filled in),
* source, initial data and initial velocity must agree as functions on the
  domain (sampled numerically), exact solutions too when one is stated,
* Dirichlet data only on the boundary, since that is all a solver reads,
* a new equation (Burgers, reaction-diffusion) must get a new kind, not a
  known one; its source and coefficient names are free.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .problem import PDEProblem


@dataclass
class ParseFixture:
    name: str
    text: str
    expected: PDEProblem
    builtin: bool = True  # False: the equation is not a known kind
    notes: str = ""


def _p(kind, domain, t_final=None, coefficients=None, source="0", boundary="dirichlet", boundary_value=None,
       initial=None, initial_velocity=None, exact=None, equation=""):
    return PDEProblem(kind=kind, domain=domain, t_final=t_final, coefficients=coefficients or {}, source=source,
                      boundary=boundary, boundary_value=boundary_value, initial=initial,
                      initial_velocity=initial_velocity, exact=exact, equation=equation)


UNIT = [(0.0, 1.0)]
SQUARE = [(0.0, 1.0), (0.0, 1.0)]


def parse_fixtures() -> list[ParseFixture]:
    f = ParseFixture
    return [
        f("poisson-1d", "求解 -u'' = π² sin(πx)，x 在 [0, 1] 上，两端 u = 0。",
          _p("poisson", UNIT, source="pi**2*sin(pi*x)")),
        f("poisson-1d-bc", "在区间 [0, 2] 上解 u'' = -1，边界条件 u(0) = 0，u(2) = 1。",
          _p("poisson", [(0.0, 2.0)], source="1", boundary_value="x/2")),
        f("poisson-1d-k", "一维 Poisson 方程 -k u'' = 1，k = 0.5，区间 [0, 1]，两端为零。",
          _p("poisson", UNIT, coefficients={"k": 0.5}, source="1")),
        f("poisson-2d", "在单位正方形 [0,1]×[0,1] 上解 -Δu = 2π² sin(πx) sin(πy)，边界上 u = 0。",
          _p("poisson", SQUARE, source="2*pi**2*sin(pi*x)*sin(pi*y)")),
        f("poisson-steady-rod", "一根长为 1 的杆，内部有均匀热源 f = 2，两端温度保持为 0，求稳态温度分布（导热系数取 1）。",
          _p("poisson", UNIT, coefficients={"k": 1.0}, source="2")),
        f("heat-rod", "一根长为 1 的杆，热扩散系数 0.1，初始温度 sin(πx)，两端保持 0 度，求 t = 0.5 时的温度分布。",
          _p("heat", UNIT, 0.5, {"alpha": 0.1}, initial="sin(pi*x)")),
        f("heat-source", "解 u_t = 0.5 u_xx + x，x ∈ [0, 1]，算到 t = 1，初值为 0，两端为 0。",
          _p("heat", UNIT, 1.0, {"alpha": 0.5}, source="x", initial="0")),
        f("heat-2d", "单位正方形上的二维热方程，扩散系数 1，初始温度 sin(πx)sin(πy)，边界温度为零，算到 t = 0.1。",
          _p("heat", SQUARE, 0.1, {"alpha": 1.0}, initial="sin(pi*x)*sin(pi*y)")),
        f("heat-exact", "热方程 u_t = u_xx，区间 [0, 1]，已知精确解 u = exp(-π² t) sin(πx)，算到 t = 0.1。",
          _p("heat", UNIT, 0.1, {"alpha": 1.0}, exact="exp(-pi**2*t)*sin(pi*x)")),
        f("heat-bc", "杆长 1，扩散系数 0.2，左端保持 1 度，右端保持 0 度，初始温度 1 - x + sin(πx)，求 t = 1 时的温度。",
          _p("heat", UNIT, 1.0, {"alpha": 0.2}, boundary_value="1 - x", initial="1 - x + sin(pi*x)")),
        f("advection-periodic", "周期边界上的对流方程 u_t + 2 u_x = 0，区间 [0, 1]，初值 sin(2πx)，算到 t = 1。",
          _p("advection", UNIT, 1.0, {"c": 2.0}, boundary="periodic", initial="sin(2*pi*x)")),
        f("advection-left", "u_t - 0.5 u_x = 0，x ∈ [0, 1]，周期边界，初始 u = exp(sin(2πx))，求 t = 2 时的解。",
          _p("advection", UNIT, 2.0, {"c": -0.5}, boundary="periodic", initial="exp(sin(2*pi*x))")),
        f("wave-string", "两端固定的弦，长度 1，波速 1，初始位移 sin(πx)，初速度为 0，求 t = 1 时的位移。",
          _p("wave", UNIT, 1.0, {"c": 1.0}, initial="sin(pi*x)", initial_velocity="0")),
        f("wave-velocity", "解 u_tt = 4 u_xx，x ∈ [0, π]，u(x, 0) = 0，u_t(x, 0) = sin(x)，两端为 0，算到 t = 2。",
          _p("wave", [(0.0, math.pi)], 2.0, {"c": 2.0}, initial="0", initial_velocity="sin(x)")),
        f("burgers", "黏性 Burgers 方程 u_t + u u_x = 0.05 u_xx，区间 [0, 1]，初值 sin(πx)，两端为 0，算到 t = 1。",
          _p("burgers", UNIT, 1.0, {"nu": 0.05}, initial="sin(pi*x)", equation="u_t + u*u_x = nu*u_xx + f"),
          builtin=False),
        f("reaction-diffusion", "反应扩散方程 u_t = 0.01 u_xx + u(1 - u)，x ∈ [0, 1]，初值 0.5 + 0.1 sin(πx)，"
                                "两端保持 u = 0.5，算到 t = 1。",
          _p("reaction_diffusion", UNIT, 1.0, {"D": 0.01}, boundary_value="0.5", initial="0.5 + 0.1*sin(pi*x)",
             equation="u_t = D*u_xx + u*(1 - u) + f"),
          builtin=False),
    ]


# --- grading ---------------------------------------------------------------------------

RTOL = 1e-6


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


def _interior_points(problem: PDEProblem, k: int = 5) -> list[tuple[np.ndarray, ...]]:
    axes = [np.linspace(a, b, k + 2)[1:-1] for a, b in problem.domain]
    pts = np.array(list(itertools.product(*axes)))
    return [pts[:, i] for i in range(problem.dim)]


def _boundary_points(problem: PDEProblem, k: int = 5) -> list[np.ndarray]:
    if problem.dim == 1:
        (a, b), = problem.domain
        return [np.array([a, b])]
    pts = []
    for axis, (a, b) in enumerate(problem.domain):
        others = [np.linspace(c, d, k) for j, (c, d) in enumerate(problem.domain) if j != axis]
        for side in (a, b):
            for rest in itertools.product(*others):
                p = list(rest)
                p.insert(axis, side)
                pts.append(p)
    pts = np.array(pts)
    return [pts[:, i] for i in range(problem.dim)]


def _same_field(expected: PDEProblem, got: PDEProblem, name: str, points, times) -> bool:
    try:
        want, have = expected.fn(name), got.fn(name)
    except ValueError:
        return False
    coords = list(points) + [np.zeros_like(points[0])] * (2 - len(points))
    for t in times:
        a = want(coords[0], coords[1], t)
        b = have(coords[0], coords[1], t)
        if not np.all(np.isfinite(b)):
            return False
        if float(np.max(np.abs(a - b))) > 1e-6 * max(1.0, float(np.max(np.abs(a)))):
            return False
    return True


def compare_parse(fixture: ParseFixture, got: PDEProblem, equations) -> list[str]:
    """What the parse ``got`` gets wrong about ``fixture`` (empty when it is right)."""
    expected = fixture.expected
    if fixture.builtin:
        expected = equations.complete(expected)
        try:
            got = equations.complete(got)
        except ValueError as exc:
            return [f"系数：{exc}"]
    wrong: list[str] = []
    if fixture.builtin and got.kind != expected.kind:
        wrong.append(f"方程类型：应为 {expected.kind}，得到 {got.kind}")
    if not fixture.builtin and equations.get(got.kind) is not None:
        wrong.append(f"方程类型：这是新方程，不应归为已知的 {got.kind}")
    if got.dim != expected.dim or not all(_close(a, c) and _close(b, d)
                                          for (a, b), (c, d) in zip(got.domain, expected.domain)):
        wrong.append(f"区域：应为 {expected.domain}，得到 {got.domain}")
        return wrong  # nothing else can be compared on the wrong domain
    if (got.t_final is None) != (expected.t_final is None) or (
            expected.t_final is not None and not _close(got.t_final, expected.t_final)):
        wrong.append(f"终止时间：应为 {expected.t_final}，得到 {got.t_final}")
    if got.boundary != expected.boundary:
        wrong.append(f"边界类型：应为 {expected.boundary}，得到 {got.boundary}")
    if wrong:
        return wrong
    if fixture.builtin:
        for name, value in expected.coefficients.items():
            have = got.coefficients.get(name)
            if have is None or not _close(have, value):
                wrong.append(f"系数 {name}：应为 {value:g}，得到 {have}")
    t_end = expected.t_final or 0.0
    times = np.linspace(0.0, t_end, 4) if t_end else [0.0]
    inside = _interior_points(expected)
    fields = []
    if fixture.builtin:
        fields.append(("source", times))
    if expected.time_dependent:
        fields.append(("initial", [0.0]))
        eq = equations.get(expected.kind)
        if eq is not None and eq.time_order == 2:
            fields.append(("initial_velocity", [0.0]))
    if expected.has_exact():
        if not got.has_exact():
            wrong.append("精确解：题目给了精确解，解析结果没有")
        else:
            fields.append(("exact", times))
    labels = {"source": "源项", "initial": "初值", "initial_velocity": "初速度", "exact": "精确解"}
    for name, ts in fields:
        if not _same_field(expected, got, name, inside, ts):
            wrong.append(f"{labels[name]}：应为 {expected.expr(name)}，得到 {got.expr(name)}")
    if expected.boundary == "dirichlet" and not _same_field(expected, got, "boundary_value",
                                                            _boundary_points(expected), times):
        wrong.append(f"边界值：应为 {expected.expr('boundary_value')}，得到 {got.expr('boundary_value')}")
    return wrong


def expected_reply(fixture: ParseFixture, notes: str = "") -> dict:
    """The fixture's problem as a reply in the parse schema (for tests with a stand-in model)."""
    p = fixture.expected
    return {
        "kind": p.kind, "equation": p.equation, "domain": [{"lower": a, "upper": b} for a, b in p.domain],
        "t_final": p.t_final or 0, "coefficients": [{"name": k, "value": v} for k, v in p.coefficients.items()],
        "source": p.source, "boundary": p.boundary, "boundary_value": p.boundary_value or "",
        "initial": p.initial or "", "initial_velocity": p.initial_velocity or "", "exact": p.exact or "",
        "notes": notes,
    }
