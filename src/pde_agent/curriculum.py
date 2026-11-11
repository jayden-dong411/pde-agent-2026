"""A curriculum from the most basic equations upward.

Levels 1-5 can be solved with the built-in methods (level 5 has no exact
solution, so the agent must estimate its own error). The frontier lessons
have no built-in method: the agent has to learn one (with Claude or DeepSeek)
first.

The ODE stages come before all of that in the ODE-to-PDE plan: stage 0 is
non-stiff ODEs (which order pays off at which tolerance), stage 1 stiff ones
(when stability, not accuracy, sets an explicit method's step, and implicit
methods win). Each stage keeps a few problems with other parameters as its
final exam (``ode_exam``), solved without having been practised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

from .equations import Equation, builtin_equations, manufactured_problem
from .ode import ODEProblem
from .problem import PDEProblem


@dataclass
class Lesson:
    name: str
    level: int
    title: str
    problem: PDEProblem | ODEProblem
    tol: float = 1e-3


def _mms(kind: str, exact: str, domain, t_final=None, coefficients=None, boundary="dirichlet", name="", title="") -> PDEProblem:
    return manufactured_problem(
        builtin_equations().get(kind), exact, domain, t_final, coefficients, boundary, name=name, description=title,
    )


def basic_curriculum() -> list[Lesson]:
    unit = [(0.0, 1.0)]
    square = [(0.0, 1.0), (0.0, 1.0)]
    lessons = [
        (1, "poisson-1d", "一维 Poisson：-u'' = π² sin(πx)", "poisson", "sin(pi*x)", unit, None, {}, "dirichlet"),
        (1, "poisson-1d-bc", "一维 Poisson，非零边界", "poisson", "exp(x)*cos(2*x)", unit, None, {"k": 0.5}, "dirichlet"),
        (2, "heat-1d", "一维热方程：自由衰减", "heat", "sin(pi*x)*exp(-0.1*pi**2*t)", unit, 1.0, {"alpha": 0.1}, "dirichlet"),
        (2, "heat-1d-forced", "一维热方程：有源项和非零边界", "heat", "sin(2*pi*x)*cos(t) + x", unit, 1.0, {"alpha": 0.5}, "dirichlet"),
        (3, "advection-1d", "一维对流：周期边界向右平移", "advection", "sin(2*pi*(x - t))", unit, 1.0, {"c": 1.0}, "periodic"),
        (3, "advection-1d-left", "一维对流：向左平移", "advection", "exp(sin(2*pi*(x + 0.5*t)))", unit, 1.0, {"c": -0.5}, "periodic"),
        (3, "wave-1d", "一维波动：驻波", "wave", "sin(pi*x)*cos(pi*t)", unit, 1.0, {"c": 1.0}, "dirichlet"),
        (3, "wave-1d-forced", "一维波动：有源项", "wave", "sin(pi*x)*sin(2*t) + x**2*t", unit, 1.0, {"c": 2.0}, "dirichlet"),
        (4, "poisson-2d", "二维 Poisson：单位正方形", "poisson", "sin(pi*x)*sin(pi*y)", square, None, {}, "dirichlet"),
        (4, "poisson-2d-exp", "二维 Poisson，非零边界", "poisson", "exp(x + 2*y)", square, None, {}, "dirichlet"),
    ]
    out = [
        Lesson(name, level, title, _mms(kind, exact, dom, t, co, bc, name, title))
        for level, name, title, kind, exact, dom, t, co, bc in lessons
    ]
    out += [
        Lesson("poisson-1d-unknown", 5, "一维 Poisson，没有精确解（自己估计误差）", PDEProblem(
            kind="poisson", domain=unit, source="exp(x)*sin(5*x)", name="poisson-1d-unknown",
            description="没有精确解（自己估计误差）")),
        Lesson("heat-1d-unknown", 5, "一维热方程，没有精确解（自己估计误差）", PDEProblem(
            kind="heat", domain=unit, t_final=0.5, coefficients={"alpha": 0.2}, initial="x*(1 - x)*exp(x)",
            name="heat-1d-unknown", description="没有精确解（自己估计误差）")),
    ]
    return out


def _custom(kind: str, form: str, operator, exact: str, domain, t_final, coefficients, boundary, name, title) -> PDEProblem:
    eq = Equation(kind=kind, form=form, time_order=1, operator=operator, defaults=coefficients)
    problem = manufactured_problem(eq, exact, domain, t_final, coefficients, boundary, name=name, description=title)
    problem.equation = form
    return problem


def frontier_lessons() -> list[Lesson]:
    """Problems with no built-in method: the agent must learn to solve them."""
    reaction = _custom(
        "reaction_diffusion", "u_t = D*u_xx - r*u + f",
        lambda u, x, y, t, c: sp.diff(u, t) - c["D"] * sp.diff(u, x, 2) + c["r"] * u,
        "sin(pi*x)*exp(-t) + x/2", [(0.0, 1.0)], 1.0, {"D": 0.5, "r": 2.0}, "dirichlet",
        "reaction-diffusion-1d", "一维反应扩散方程",
    )
    burgers = _custom(
        "burgers", "u_t + u*u_x = nu*u_xx + f",
        lambda u, x, y, t, c: sp.diff(u, t) + u * sp.diff(u, x) - c["nu"] * sp.diff(u, x, 2),
        "1 + 0.5*sin(2*pi*x)*exp(-t)", [(0.0, 1.0)], 0.5, {"nu": 0.05}, "periodic",
        "burgers-1d", "一维黏性 Burgers 方程（非线性）",
    )
    heat_2d = _mms("heat", "sin(pi*x)*sin(pi*y)*exp(-0.2*pi**2*t)", [(0.0, 1.0), (0.0, 1.0)], 0.5, {"alpha": 0.1},
                   "dirichlet", "heat-2d", "二维热方程")
    return [
        Lesson("reaction-diffusion-1d", 6, "一维反应扩散方程（新方程）", reaction),
        Lesson("heat-2d", 6, "二维热方程（已知方程，新维度）", heat_2d),
        Lesson("burgers-1d", 7, "一维黏性 Burgers 方程（非线性）", burgers),
    ]


# --- ODE stages 0 and 1 -----------------------------------------------------------


def _ode(name, rhs, initial, t_final, coefficients=None, exact=None, equation="", title=""):
    return ODEProblem(rhs=rhs, initial=initial, t_final=t_final, coefficients=coefficients or {}, exact=exact,
                      equation=equation, name=name, description=title)


def _linear_stiff(name: str, lam: float, title: str) -> ODEProblem:
    """y' = -lam (y - cos t), y(0) = 0: a fast transient onto the slow solution."""
    a = lam * lam / (lam * lam + 1)
    exact = f"{a!r}*cos(t) + {a / lam!r}*sin(t) - {a!r}*exp(-lam*t)"
    return _ode(name, ["-lam*(y1 - cos(t))"], ["0"], 2.0, {"lam": lam}, [exact], "y' = -lam*(y - cos t)", title)


def _oscillator(name: str, zeta: float, t_final: float, title: str) -> ODEProblem:
    """y'' + 2 zeta y' + y = 0, y(0) = 1, y'(0) = 0, as a first-order system."""
    t = sp.Symbol("t", real=True)
    if zeta == 0:
        y = sp.cos(t)
    else:
        wd = sp.sqrt(1 - sp.Float(zeta) ** 2)
        y = sp.exp(-sp.Float(zeta) * t) * (sp.cos(wd * t) + sp.Float(zeta) / wd * sp.sin(wd * t))
    return _ode(name, ["y2", "-y1 - 2*zeta*y2"], ["1", "0"], t_final, {"zeta": zeta},
                [str(y), str(sp.diff(y, t))], "y'' + 2*zeta*y' + y = 0", title)


def _harmonic(name: str, w: float, t_final: float, title: str) -> ODEProblem:
    return _ode(name, ["y2", "-w**2*y1"], ["1", "0"], t_final, {"w": w},
                ["cos(w*t)", "-w*sin(w*t)"], "y'' = -w^2 y", title)


def _logistic(name: str, r: float, y0: float, t_final: float, title: str) -> ODEProblem:
    c = (1 - y0) / y0
    return _ode(name, ["r*y1*(1 - y1)"], [repr(y0)], t_final, {"r": r},
                [f"1/(1 + {c!r}*exp(-r*t))"], "y' = r*y*(1 - y)", title)


def _van_der_pol(name: str, mu: float, t_final: float, title: str) -> ODEProblem:
    return _ode(name, ["y2", "mu*(1 - y1**2)*y2 - y1"], ["2", "0"], t_final, {"mu": mu},
                equation="y'' = mu*(1 - y^2)*y' - y", title=title)


def _heat_lines(name: str, alpha: float, m: int, t_final: float, title: str) -> ODEProblem:
    """The heat equation u_t = alpha u_xx on (0, 1) after central differences in x (the method of lines).

    Initial data: the slowest and the fastest grid mode, so the exact solution
    of the semi-discrete system is two decaying exponentials per component.
    """
    h = 1.0 / (m + 1)
    rhs = []
    for i in range(1, m + 1):
        terms = ([f"y{i - 1}"] if i > 1 else []) + [f"-2*y{i}"] + ([f"y{i + 1}"] if i < m else [])
        rhs.append(f"alpha*{1 / h**2!r}*({' + '.join(terms)})")
    slow, fast = (4 * alpha / h**2 * math.sin(k * math.pi * h / 2) ** 2 for k in (1, m))
    modes = [(math.sin(math.pi * i * h), math.sin(m * math.pi * i * h)) for i in range(1, m + 1)]
    initial = [repr(a + 0.5 * b) for a, b in modes]
    exact = [f"{a!r}*exp(-{slow!r}*t) + {0.5 * b!r}*exp(-{fast!r}*t)" for a, b in modes]
    return _ode(name, rhs, initial, t_final, {"alpha": alpha}, exact, "u' = alpha*A*u（热方程半离散）", title)


def ode_curriculum() -> list[Lesson]:
    """Stage 0 (non-stiff) and stage 1 (stiff) training problems."""
    stage0 = [
        Lesson("ode-decay", 0, "指数衰减 y' = -y", _ode(
            "ode-decay", ["-y1"], ["1"], 2.0, exact=["exp(-t)"], equation="y' = -y", title="指数衰减")),
        _lesson(0, _logistic("ode-logistic", 2.0, 0.1, 5.0, "Logistic 增长")),
        _lesson(0, _harmonic("ode-harmonic", 2.0, 10.0, "简谐振子")),
        _lesson(0, _harmonic("ode-harmonic-tight", 2.0, 10.0, "简谐振子，严格精度"), tol=1e-6),
        _lesson(0, _oscillator("ode-damped", 0.1, 10.0, "阻尼振子")),
        _lesson(0, _ode("ode-lotka-volterra", ["1.5*y1 - y1*y2", "-3*y2 + y1*y2"], ["10", "5"], 5.0,
                        equation="捕食者-猎物", title="Lotka–Volterra（没有精确解，自己估计误差）")),
        _lesson(0, _ode("ode-kepler", ["y3", "y4", "-y1/(y1**2 + y2**2)**1.5", "-y2/(y1**2 + y2**2)**1.5"],
                        ["1", "0", "0", "1"], 20 * math.pi, exact=["cos(t)", "sin(t)", "-sin(t)", "cos(t)"],
                        equation="q'' = -q/|q|^3", title="Kepler 二体问题，圆轨道转 10 圈")),
    ]
    stage1 = [
        _lesson(1, _linear_stiff("ode-stiff-10", 10.0, "线性刚性，λ = 10（还不刚性）")),
        _lesson(1, _linear_stiff("ode-stiff-1e3", 1e3, "线性刚性，λ = 10³")),
        _lesson(1, _linear_stiff("ode-stiff-1e6", 1e6, "线性刚性，λ = 10⁶")),
        _lesson(1, _ode("ode-robertson", ["-0.04*y1 + 1e4*y2*y3", "0.04*y1 - 1e4*y2*y3 - 3e7*y2**2", "3e7*y2**2"],
                        ["1", "0", "0"], 40.0, equation="Robertson 化学反应",
                        title="Robertson 化学反应（经典刚性题，没有精确解）")),
        _lesson(1, _van_der_pol("ode-vdp-1", 1.0, 10.0, "Van der Pol，μ = 1（不刚性，没有精确解）")),
        _lesson(1, _van_der_pol("ode-vdp-1000", 1000.0, 100.0, "Van der Pol，μ = 1000（刚性，没有精确解）")),
        _lesson(1, _heat_lines("ode-heat-lines", 1.0, 15, 0.5, "热方程的方法线：15 个内点的刚性 ODE")),
    ]
    return stage0 + stage1


def ode_exam() -> list[Lesson]:
    """Final exam of stages 0 and 1: the same kinds of problems with parameters never practised."""
    return [
        _lesson(0, _logistic("exam-logistic", 3.0, 0.2, 4.0, "期末：Logistic，r = 3")),
        _lesson(0, _harmonic("exam-harmonic", 3.0, 8.0, "期末：简谐振子 ω = 3，严格精度"), tol=1e-5),
        _lesson(0, _oscillator("exam-damped", 0.3, 12.0, "期末：阻尼振子 ζ = 0.3")),
        _lesson(1, _linear_stiff("exam-stiff-1e2", 100.0, "期末：线性刚性，λ = 100")),
        _lesson(1, _linear_stiff("exam-stiff-1e4", 1e4, "期末：线性刚性，λ = 10⁴")),
        _lesson(1, _van_der_pol("exam-vdp-500", 500.0, 50.0, "期末：Van der Pol，μ = 500")),
        _lesson(1, _heat_lines("exam-heat-lines", 2.0, 15, 0.25, "期末：热方程方法线，α = 2")),
    ]


# --- PDE stages 2-4 ------------------------------------------------------------------
# Stage 2: parabolic (diffusion gets stiffer as alpha grows and the grid gets finer; jumps in the
# initial data make Crank-Nicolson ring). Stage 3: smooth hyperbolic (long times and tight tolerances
# pay for high order). Stage 4: hyperbolic with jumps and shocks, measured in L1, where a method that
# makes new extrema fails whatever its error.


def _pde(name: str, title: str, kind: str, domain, t_final: float, coefficients: dict, boundary: str = "dirichlet",
         exact: str | None = None, initial: str | None = None, boundary_value: str | None = None) -> PDEProblem:
    return PDEProblem(kind=kind, domain=domain, t_final=t_final, coefficients=coefficients, boundary=boundary,
                      source="0", exact=exact, initial=initial, boundary_value=boundary_value, name=name,
                      description=title)


def _heat_decay(name: str, alpha: float, t_final: float, title: str) -> PDEProblem:
    return _pde(name, title, "heat", [(0.0, 1.0)], t_final, {"alpha": alpha},
                exact=f"sin(pi*x)*exp(-{alpha!r}*pi**2*t) + x")


def _heat_pulse(name: str, alpha: float, left: float, right: float, t_final: float, title: str) -> PDEProblem:
    return _pde(name, title, "heat", [(0.0, 1.0)], t_final, {"alpha": alpha},
                initial=f"Heaviside(x - {left!r}) - Heaviside(x - {right!r})", boundary_value="0")


def _advect(name: str, shape: str, c: float, t_final: float, title: str) -> PDEProblem:
    """A periodic shape ``shape(S)`` (an expression in S, period 1) carried at speed c."""
    s = f"(x - {c!r}*t)"
    return _pde(name, title, "advection", [(0.0, 1.0)], t_final, {"c": c}, "periodic", exact=shape.replace("S", s))


def _square(name: str, c: float, left: float, right: float, t_final: float, title: str) -> PDEProblem:
    s = f"(x - {c!r}*t - floor(x - {c!r}*t))"
    return _pde(name, title, "advection", [(0.0, 1.0)], t_final, {"c": c}, "periodic",
                exact=f"Heaviside({s} - {left!r}) - Heaviside({s} - {right!r})",
                initial=f"Heaviside(x - {left!r}) - Heaviside(x - {right!r})")


def _riemann(name: str, u_left: float, u_right: float, t_final: float, title: str) -> PDEProblem:
    """Inviscid Burgers from a jump at x = 0 on [-1, 1]: a shock (u_left > u_right) or a rarefaction fan."""
    jump = f"{u_left!r} + ({u_right - u_left!r})*Heaviside(x)"
    if u_left > u_right:
        exact = f"{u_left!r} + ({u_right - u_left!r})*Heaviside(x - {(u_left + u_right) / 2!r}*t)"
    else:
        exact = f"Min(Max(x/t, {u_left!r}), {u_right!r})"
    return _pde(name, title, "burgers_inviscid", [(-1.0, 1.0)], t_final, {}, exact=exact, initial=jump,
                boundary_value=jump)


def pde_stages() -> list[Lesson]:
    """Stages 2 (parabolic), 3 (smooth hyperbolic) and 4 (jumps and shocks) of the ODE-to-PDE plan."""
    stage2 = [
        _lesson(2, _heat_decay("heat-alpha-0.01", 0.01, 1.0, "热方程 α = 0.01（扩散很慢）")),
        _lesson(2, _heat_decay("heat-alpha-1", 1.0, 1.0, "热方程 α = 1")),
        _lesson(2, _heat_decay("heat-alpha-10", 10.0, 0.1, "热方程 α = 10（显式方法步长被 dx²/2α 卡死）")),
        _lesson(2, _heat_decay("heat-alpha-tight", 0.1, 1.0, "热方程，严格精度"), tol=1e-6),
        _lesson(2, _heat_pulse("heat-pulse", 0.1, 0.25, 0.75, 0.05,
                               "热方程，初值是方波（有间断，Crank–Nicolson 会振荡；没有精确解）")),
    ]
    stage3 = [
        _lesson(3, _advect("advection-short", "sin(2*pi*S)", 1.0, 0.25, "对流，短时间，宽松精度"), tol=1e-2),
        _lesson(3, _advect("advection-long", "sin(2*pi*S)", 1.0, 10.0, "对流，绕周期区域 10 圈")),
        _lesson(3, _advect("advection-tight", "exp(sin(2*pi*S))", -0.5, 2.0, "对流，严格精度"), tol=1e-6),
        _lesson(3, PDEProblem(kind="wave", domain=[(0.0, 1.0)], t_final=10.0, coefficients={"c": 1.0}, source="0",
                              exact="sin(pi*x)*cos(pi*t)", name="wave-long", description="波动方程，长时间")),
    ]
    stage4 = [
        _lesson(4, _square("advection-square", 1.0, 0.25, 0.75, 1.0, "对流方波转一圈（有间断，L1 误差）"), tol=5e-2),
        _lesson(4, _riemann("burgers-shock", 1.0, 0.0, 0.5, "无黏 Burgers：激波（Riemann 问题）"), tol=1e-2),
        _lesson(4, _riemann("burgers-rarefaction", 0.0, 1.0, 0.5, "无黏 Burgers：稀疏波（Riemann 问题）"), tol=1e-2),
        _lesson(4, _riemann("burgers-shock-tight", 1.0, 0.0, 0.5, "无黏 Burgers：激波，严格精度"), tol=1e-3),
    ]
    return stage2 + stage3 + stage4


def pde_exam() -> list[Lesson]:
    """Final exam of stages 2-4: the same kinds of problems with parameters never practised."""
    return [
        _lesson(2, _heat_decay("exam-heat-alpha-3", 3.0, 0.3, "期末：热方程 α = 3")),
        _lesson(2, _heat_decay("exam-heat-tight", 0.05, 1.0, "期末：热方程 α = 0.05，严格精度"), tol=1e-5),
        _lesson(2, _heat_pulse("exam-heat-pulse", 0.2, 0.375, 0.625, 0.02, "期末：热方程，窄方波初值")),
        _lesson(3, _advect("exam-advection-long", "sin(2*pi*S)", 2.0, 5.0, "期末：对流 c = 2，绕 10 圈"), tol=1e-4),
        _lesson(3, _advect("exam-advection-short", "exp(cos(2*pi*S))", 1.0, 0.5, "期末：对流，短时间，宽松精度"), tol=1e-2),
        _lesson(4, _square("exam-advection-square", -1.0, 0.1, 0.4, 2.0, "期末：方波向左转两圈"), tol=5e-2),
        _lesson(4, _riemann("exam-burgers-shock", 2.0, 0.0, 0.4, "期末：无黏 Burgers 强激波"), tol=1e-2),
        _lesson(4, _riemann("exam-burgers-fan", -1.0, 1.0, 0.5, "期末：无黏 Burgers 对称稀疏波"), tol=1e-2),
    ]


def _lesson(level: int, problem, tol: float = 1e-3) -> Lesson:
    return Lesson(problem.name, level, problem.description, problem, tol)


def best_work(registry, problem, tol: float, max_n: int = 4096) -> tuple[str, float] | None:
    """The cheapest single run that meets ``tol``, over every method that takes the problem, or None.

    Each method runs on doubling grids (from 16) until its exact error meets the tolerance; the work of
    that run is its cost when the right grid is known in advance. Only for problems with an exact solution.
    """
    from .solvers.base import SolverDiverged
    from .verify import OSCILLATION_LIMIT, error_norm, exact_error, new_extrema

    if not problem.has_exact():
        return None
    best: tuple[str, float] | None = None
    for spec in registry.for_problem(problem):
        n = 16
        while n <= max_n:
            try:
                result = spec.run(problem, n)
            except (SolverDiverged, ValueError):
                break
            if best is not None and result.work >= best[1]:
                break  # already dearer than the best, and finer grids only cost more
            if error_norm(problem) == "l1":
                overshoot = new_extrema(problem, result)
                if overshoot is not None and overshoot > OSCILLATION_LIMIT:
                    n *= 2  # a coarse grid can overshoot where a finer one does not
                    continue
            error = exact_error(problem, result)
            if error <= tol:
                best = (spec.name, result.work)
                break
            n *= 2
    return best


def regret(agent_work: float, best: tuple[str, float] | None) -> float | None:
    """Work of the agent's whole episode (exploration included) over the best method's single run."""
    if best is None or not best[1] > 0:
        return None
    return agent_work / best[1]


def run_curriculum(agent, lessons: list[Lesson], rounds: int = 1, learn: bool = True):
    """Solve every lesson ``rounds`` times; yields (round, lesson, report)."""
    for round_ in range(1, rounds + 1):
        for lesson in lessons:
            yield round_, lesson, agent.solve(lesson.problem, lesson.tol, learn=learn)
