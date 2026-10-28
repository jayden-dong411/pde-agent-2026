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


def _lesson(level: int, problem: ODEProblem, tol: float = 1e-3) -> Lesson:
    return Lesson(problem.name, level, problem.description, problem, tol)


def run_curriculum(agent, lessons: list[Lesson], rounds: int = 1, learn: bool = True):
    """Solve every lesson ``rounds`` times; yields (round, lesson, report)."""
    for round_ in range(1, rounds + 1):
        for lesson in lessons:
            yield round_, lesson, agent.solve(lesson.problem, lesson.tol, learn=learn)
