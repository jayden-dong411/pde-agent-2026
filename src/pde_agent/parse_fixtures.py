"""Chinese problem descriptions with their known correct parse (``pde-agent eval --parse``).

Each fixture is a sentence a user might type and the problem it means. A parse
is graded by what it means for the solver, not by how it is written:

* kind, domain, final time and boundary type must match; for a known
  equation kind the coefficients too (after defaults are filled in),
* source, initial data and initial velocity must agree as functions on the
  domain (sampled numerically), exact solutions too when one is stated,
* Dirichlet data only on the boundary, since that is all a solver reads,
* a new equation (Burgers, reaction-diffusion) must get a new kind, not a
  known one, and its operator together with its source must be the right
  one: ``L(w) - f`` is compared on test functions ``w`` (zero included, so a
  term moved between operator and source still counts), with the parse's own
  coefficient names and values; an equation multiplied through by a constant
  is the same equation,
* an ODE system must have the same right-hand side ``f(t, y)``, initial
  values, end time and stated exact solution, up to the order of the unknowns.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from .ode import ODEProblem
from .problem import PDEProblem, T, X, Y


@dataclass
class ParseFixture:
    name: str
    text: str
    expected: PDEProblem | ODEProblem
    builtin: bool = True  # False: the equation is not a known kind
    notes: str = ""


def _p(kind, domain, t_final=None, coefficients=None, source="0", boundary="dirichlet", boundary_value=None,
       initial=None, initial_velocity=None, exact=None, equation=""):
    return PDEProblem(kind=kind, domain=domain, t_final=t_final, coefficients=coefficients or {}, source=source,
                      boundary=boundary, boundary_value=boundary_value, initial=initial,
                      initial_velocity=initial_velocity, exact=exact, equation=equation)


def _o(rhs, initial, t_final, coefficients=None, exact=None, equation=""):
    return ODEProblem(rhs=rhs, initial=initial, t_final=t_final, coefficients=coefficients or {}, exact=exact,
                      equation=equation)


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
        # ordinary differential equations (kind "ode")
        f("ode-logistic", "种群按 Logistic 规律增长：y' = 2y(1 - y)，初始 y(0) = 0.1，求 t 从 0 到 5 的解。",
          _o(["2*y1*(1 - y1)"], ["0.1"], 5.0)),
        f("ode-spring", "弹簧振子 x'' + 4x = 0，x(0) = 1，x'(0) = 0，算到 t = 10。",
          _o(["y2", "-4*y1"], ["1", "0"], 10.0)),
        f("ode-damped", "阻尼振子 x'' + 0.2x' + x = 0，初始位移 1、初速度 0，求 0 ≤ t ≤ 20 的运动。",
          _o(["y2", "-y1 - 0.2*y2"], ["1", "0"], 20.0)),
        f("ode-decay-exact", "解 y' = -50y，y(0) = 1，已知精确解 y = exp(-50t)，算到 t = 0.5。",
          _o(["-50*y1"], ["1"], 0.5, exact=["exp(-50*t)"])),
        f("ode-predator-prey", "捕食者-猎物模型：x' = 1.5x - xy，y' = -3y + xy，x(0) = 10，y(0) = 5，算到 t = 5。",
          _o(["1.5*y1 - y1*y2", "-3*y2 + y1*y2"], ["10", "5"], 5.0)),
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


def compare_parse(fixture: ParseFixture, got, equations) -> list[str]:
    """What the parse ``got`` gets wrong about ``fixture`` (empty when it is right)."""
    if isinstance(fixture.expected, ODEProblem):
        if not isinstance(got, ODEProblem):
            return [f"方程类型：应为常微分方程（ode），得到 {got.kind}"]
        return _compare_ode(fixture.expected, got)
    if isinstance(got, ODEProblem):
        return [f"方程类型：应为偏微分方程 {fixture.expected.kind}，得到常微分方程"]
    return _compare_pde(fixture, got, equations)


def _compare_pde(fixture: ParseFixture, got: PDEProblem, equations) -> list[str]:
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
    t_end = expected.t_final or 0.0
    times = np.linspace(0.0, t_end, 4) if t_end else [0.0]
    inside = _interior_points(expected)
    fields = []
    if fixture.builtin:
        for name, value in expected.coefficients.items():
            have = got.coefficients.get(name)
            if have is None or not _close(have, value):
                wrong.append(f"系数 {name}：应为 {value:g}，得到 {have}")
        fields.append(("source", times))
        time_order = equations.get(expected.kind).time_order
    else:
        # a new equation: its operator and source are compared together, coefficients by their values
        time_order, why = _operator_mismatch(expected, got, equations, inside, times)
        if why:
            wrong.append(why)
    if expected.time_dependent:
        fields.append(("initial", [0.0]))
        if time_order == 2:
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


def _residual(problem: PDEProblem, equations):
    """``(time order, w -> L(w) - f)`` of a problem, for a known kind or its equation text."""
    from .equations import operator_from_text

    eq = equations.get(problem.kind)
    source = problem.expr("source")
    if eq is not None:
        return eq.time_order, lambda w: eq.apply(w, problem.coefficients) - source
    text = operator_from_text(problem.equation, problem.coefficients)  # ValueError if it cannot be read
    return text.time_order, lambda w: text.apply(w, problem.coefficients) - source


def _test_functions(dim: int, steady: bool) -> list:
    """Smooth functions of the task's own variables; 0 and several amplitudes, so nonlinear terms show."""
    ws = [sp.Integer(0)]
    for a, b, shift, rate, amplitude in ((1.3, 0.7, 0.4, 0.9, 1.0), (0.6, 1.9, 1.1, 0.3, 2.5),
                                         (2.2, 1.4, 0.2, 1.6, 0.4)):
        w = amplitude * (sp.sin(a * X + b * Y + shift) * sp.exp(-rate * T) + a * X**2 * T + b * X * Y)
        if dim == 1:
            w = w.subs(Y, 0)
        if steady:
            w = w.subs(T, 0)
        ws.append(w)
    return ws


def _operator_mismatch(expected: PDEProblem, got: PDEProblem, equations, points, times):
    """``(expected time order, None or what is wrong)`` with the operator and source of ``got``.

    ``L(w) - f`` of both is sampled on test functions; the parse may multiply
    the whole equation by a nonzero constant, nothing else.
    """
    want_order, want = _residual(expected, equations)
    try:
        have_order, have = _residual(got, equations)
    except (ValueError, KeyError) as exc:
        return want_order, f"方程：读不出解析结果的算子（{got.equation!r}）：{exc}"
    if have_order != want_order:
        return want_order, f"方程：时间导数阶数应为 {want_order}，得到 {have_order}（{got.equation!r}）"
    coords = list(points) + [np.zeros_like(points[0])] * (2 - len(points))
    a, b = [], []
    with np.errstate(all="ignore"):
        for w in _test_functions(expected.dim, expected.t_final is None):
            fa = sp.lambdify((X, Y, T), want(w), modules=["scipy", "numpy"])
            fb = sp.lambdify((X, Y, T), have(w), modules=["scipy", "numpy"])
            for t in times:
                a.append(np.broadcast_to(np.asarray(fa(coords[0], coords[1], t), dtype=float), coords[0].shape))
                b.append(np.broadcast_to(np.asarray(fb(coords[0], coords[1], t), dtype=float), coords[0].shape))
    a, b = np.concatenate(a), np.concatenate(b)
    if not np.all(np.isfinite(b)):
        return want_order, f"方程：解析结果的算子在区域内不是有限值（{got.equation!r}）"
    scale = float(a @ b) / float(a @ a) if float(a @ a) > 0 else 1.0
    if abs(scale) < 1e-9 or float(np.max(np.abs(b - scale * a))) > 1e-6 * max(1.0, float(np.max(np.abs(b)))):
        return want_order, (f"方程：算子和源项应为 {expected.equation}（系数 {expected.coefficients}，源项 "
                            f"{expected.source}），得到 {got.equation}（系数 {got.coefficients}，源项 {got.source}）")
    return want_order, None


def _compare_ode(expected: ODEProblem, got: ODEProblem) -> list[str]:
    """An ODE parse: same f(t, y), initial values, end time and stated exact solution, unknowns in any order."""
    m = expected.size
    if got.size != m:
        return [f"方程个数：应为 {m} 个一阶方程，得到 {got.size} 个"]
    wrong = []
    if not _close(got.t_final, expected.t_final):
        wrong.append(f"终止时间：应为 {expected.t_final}，得到 {got.t_final}")
    rng = np.random.default_rng(0)
    scale = 1.0 + np.abs(expected.y0)
    samples = [(float(rng.uniform(0, expected.t_final)), expected.y0 + rng.uniform(-1, 1, m) * scale)
               for _ in range(8)]
    times = np.linspace(0, expected.t_final, 5)

    def matches(perm) -> tuple[bool, bool, bool]:
        """(right-hand side, initial values, exact solution) agree when got's y[perm[i]] is expected's y[i]."""
        idx = list(perm)
        y0 = np.allclose(got.y0[idx], expected.y0, rtol=RTOL, atol=RTOL)
        rhs = True
        with np.errstate(all="ignore"):
            for t, y in samples:
                yg = np.empty(m)
                yg[idx] = y
                want, have = expected.f(t, y), np.asarray(got.f(t, yg), dtype=float)[idx]
                if not (np.all(np.isfinite(have)) and np.allclose(have, want, rtol=1e-6, atol=1e-9)):
                    rhs = False
                    break
        exact = True
        if expected.has_exact():
            exact = got.has_exact() and all(
                np.allclose(np.asarray(got.exact_at(t))[idx], expected.exact_at(t), rtol=1e-6, atol=1e-9)
                for t in times)
        return rhs, y0, exact

    results = [matches(perm) for perm in itertools.permutations(range(m))]
    if any(all(r) for r in results):
        return wrong
    rhs, y0, exact = max(results, key=sum)  # the closest ordering says what is wrong
    if not rhs:
        wrong.append(f"右端项：应为 {expected.rhs}（系数 {expected.coefficients}），得到 {got.rhs}（系数 {got.coefficients}）")
    if not y0:
        wrong.append(f"初值：应为 {expected.initial}，得到 {got.initial}")
    if not exact:
        wrong.append("精确解：题目给了精确解，解析结果没有" if not got.has_exact()
                     else f"精确解：应为 {expected.exact}，得到 {got.exact}")
    if rhs and y0 and exact:  # each part matches, but in different orderings of the unknowns
        wrong.append("未知量顺序：右端项和初值对应的未知量顺序不一致")
    return wrong


def expected_reply(fixture: ParseFixture, notes: str = "") -> dict:
    """The fixture's problem as a reply in the parse schema (for tests with a stand-in model)."""
    p = fixture.expected
    if isinstance(p, ODEProblem):
        return {
            "kind": "ode", "equation": p.equation, "domain": [], "t_final": p.t_final,
            "coefficients": [{"name": k, "value": v} for k, v in p.coefficients.items()],
            "source": "0", "boundary": "dirichlet", "boundary_value": "", "initial": "", "initial_velocity": "",
            "exact": "", "ode": {"rhs": list(p.rhs), "initial": list(p.initial), "exact": list(p.exact or [])},
            "notes": notes,
        }
    return {
        "kind": p.kind, "equation": p.equation, "domain": [{"lower": a, "upper": b} for a, b in p.domain],
        "t_final": p.t_final or 0, "coefficients": [{"name": k, "value": v} for k, v in p.coefficients.items()],
        "source": p.source, "boundary": p.boundary, "boundary_value": p.boundary_value or "",
        "initial": p.initial or "", "initial_velocity": p.initial_velocity or "", "exact": p.exact or "",
        "notes": notes,
    }
