"""A curriculum from the most basic equations upward.

Levels 1-5 can be solved with the built-in methods (level 5 has no exact
solution, so the agent must estimate its own error). The frontier lessons
have no built-in method: the agent has to learn one (with Claude or DeepSeek)
first.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .equations import Equation, builtin_equations, manufactured_problem
from .problem import PDEProblem


@dataclass
class Lesson:
    name: str
    level: int
    title: str
    problem: PDEProblem
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


def run_curriculum(agent, lessons: list[Lesson], rounds: int = 1, learn: bool = True):
    """Solve every lesson ``rounds`` times; yields (round, lesson, report)."""
    for round_ in range(1, rounds + 1):
        for lesson in lessons:
            yield round_, lesson, agent.solve(lesson.problem, lesson.tol, learn=learn)
