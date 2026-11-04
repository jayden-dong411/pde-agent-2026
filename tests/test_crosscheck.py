"""Cross-checks for answers without an exact solution (v1.2.0)."""

from __future__ import annotations

import dataclasses

import pytest

from pde_agent.agent import PDEAgent
from pde_agent.crosscheck import conservation_laws, cross_check, symbolic_operator
from pde_agent.curriculum import frontier_lessons
from pde_agent.equations import builtin_equations
from pde_agent.evaluation import blind, eval_cases
from pde_agent.problem import PDEProblem
from pde_agent.solvers import builtin_registry


def _scaled(spec, name, **factors):
    """A solver with a bug: it solves with some coefficients multiplied by a factor.

    It still converges under refinement, so the refinement estimate cannot see the bug.
    """
    def solve(problem, n, options):
        data = problem.to_dict()
        data["coefficients"] = {k: v * factors.get(k, 1.0) for k, v in problem.coefficients.items()}
        return spec.solve(PDEProblem.from_dict(data), n, options)
    return dataclasses.replace(spec, name=name, solve=solve)


def _heat():
    return PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.5, coefficients={"alpha": 0.2},
                      initial="x*(1 - x)*exp(x)", name="heat-unknown")


def _agent(tmp_path, *specs, only=None):
    agent = PDEAgent(home=tmp_path, remember=False)
    if only:
        for name in list(agent.registry._specs):
            if name.split(".")[0] == only:
                del agent.registry._specs[name]
    for spec in specs:
        agent.registry.register(spec)
    return agent


def test_correct_answers_pass_on_every_blind_eval_problem(tmp_path):
    agent = PDEAgent(home=tmp_path, remember=False)
    for case in eval_cases(("pde",)):
        report = agent.solve(blind(case.problem), case.tol, learn=False)
        assert report.success, (case.name, report.message)
        assert report.crosscheck is not None and report.crosscheck.passed, (case.name, report.crosscheck.describe())
        assert any(c.name == "residual" for c in report.crosscheck.checks), case.name


def test_exact_solution_needs_no_crosscheck(tmp_path):
    agent = PDEAgent(home=tmp_path, remember=False)
    case = next(c for c in eval_cases(("pde",)) if c.name == "train/heat-1d")
    report = agent.solve(case.problem, case.tol, learn=False)
    assert report.success and report.crosscheck is None


def test_wrong_coefficient_is_caught_by_the_residual(tmp_path):
    good = builtin_registry().get("heat.backward_euler")
    bad = _scaled(good, "heat.buggy", alpha=1.3)
    agent = _agent(tmp_path, bad, only="heat")
    report = agent.solve(_heat(), 1e-3, learn=False)
    assert not report.success
    assert report.trials[-1]["outcome"] == "crosscheck"
    assert [c.name for c in report.crosscheck.failed()] == ["residual"]
    assert "交叉验证" in report.summary()


def test_agent_moves_on_to_a_correct_method(tmp_path):
    reg = builtin_registry()
    bad = _scaled(reg.get("heat.backward_euler"), "heat.buggy", alpha=1.3)
    agent = _agent(tmp_path, bad, only="heat")
    agent.registry.register(reg.get("heat.crank_nicolson"))
    report = agent.solve(_heat(), 1e-3, learn=False)
    assert report.success and report.method == "heat.crank_nicolson"
    outcomes = {t["method"]: t["outcome"] for t in report.trials}
    assert outcomes.get("heat.buggy") in (None, "crosscheck")
    assert report.crosscheck.passed


def test_wrong_speed_breaks_the_residual_but_not_mass(tmp_path):
    good = builtin_registry().get("advection.lax_wendroff")
    bad = _scaled(good, "advection.buggy", c=1.1)
    problem = PDEProblem(kind="advection", domain=[(0, 1)], t_final=1.0, coefficients={"c": 1.0},
                         initial="exp(sin(2*pi*x))", boundary="periodic")
    eq = builtin_equations().get("advection")
    result = bad.run(problem, 256)
    checks = {c.name: c for c in cross_check(eq, bad, problem, result, 1e-3).checks}
    assert not checks["residual"].passed
    assert checks["mass"].passed and checks["energy"].passed


def test_energy_drift_catches_extra_dissipation(tmp_path):
    """Upwind on a coarse grid smears the wave: transport must keep ½∫u², so the energy check fails."""
    spec = builtin_registry().get("advection.upwind")
    problem = PDEProblem(kind="advection", domain=[(0, 1)], t_final=1.0, coefficients={"c": 1.0},
                         initial="sin(2*pi*x)", boundary="periodic")
    eq = builtin_equations().get("advection")
    checks = {c.name: c for c in cross_check(eq, spec, problem, spec.run(problem, 32), 1e-3).checks}
    assert not checks["energy"].passed and checks["mass"].passed


def test_mass_budget_books_in_the_source():
    spec = builtin_registry().get("advection.lax_wendroff")
    problem = PDEProblem(kind="advection", domain=[(0, 1)], t_final=1.0, coefficients={"c": 1.0},
                         initial="sin(2*pi*x)", source="1 + cos(t)", boundary="periodic")
    eq = builtin_equations().get("advection")
    cc = cross_check(eq, spec, problem, spec.run(problem, 128), 1e-3)
    mass = next(c for c in cc.checks if c.name == "mass")
    assert mass.passed and mass.value < 1e-6
    assert "energy" not in {c.name for c in cc.checks}  # a source pumps energy in: no energy law


def test_wave_energy_is_conserved_and_a_wrong_speed_breaks_it():
    reg = builtin_registry()
    problem = PDEProblem(kind="wave", domain=[(0, 1)], t_final=1.0, coefficients={"c": 1.0},
                         initial="sin(pi*x) + 0.3*sin(3*pi*x)")
    eq = builtin_equations().get("wave")
    good = reg.get("wave.leapfrog")
    checks = {c.name: c for c in cross_check(eq, good, problem, good.run(problem, 128), 1e-3).checks}
    assert checks["energy"].passed and checks["residual"].passed
    bad = _scaled(good, "wave.buggy", c=1.2)
    checks = {c.name: c for c in cross_check(eq, bad, problem, bad.run(problem, 128), 1e-3).checks}
    assert not checks["energy"].passed and not checks["residual"].passed


@pytest.mark.parametrize("name, laws", [
    ("reaction-diffusion-1d", []),  # -r*u is not a divergence; nonzero boundary and source
    ("burgers-1d", ["mass"]),  # u*u_x = (u²/2)_x; the source rules out the energy law
])
def test_laws_are_read_from_learned_operators(name, laws):
    lesson = next(l for l in frontier_lessons() if l.name == name)
    problem = blind(lesson.problem)
    eq = _lesson_equation(name)
    sym = symbolic_operator(eq, problem.coefficients, problem.dim)
    assert sym is not None
    assert conservation_laws(eq, problem, sym) == laws


def test_viscous_burgers_without_source_may_only_lose_energy():
    eq = _lesson_equation("burgers-1d")
    problem = PDEProblem(kind="burgers", domain=[(0, 1)], t_final=0.5, coefficients={"nu": 0.05},
                         initial="sin(2*pi*x)", boundary="periodic")
    sym = symbolic_operator(eq, problem.coefficients, 1)
    assert not sym.linear
    assert conservation_laws(eq, problem, sym) == ["mass", "energy_decay"]


def test_odes_and_unknown_equations_are_skipped():
    from pde_agent.curriculum import ode_curriculum
    lesson = ode_curriculum()[0]
    spec = builtin_registry().for_problem(lesson.problem)[0]
    cc = cross_check(None, spec, lesson.problem, spec.run(lesson.problem, 32), 1e-3)
    assert cc.passed and not cc.checks and cc.skipped


def _lesson_equation(name):
    import sympy as sp

    from pde_agent.equations import Equation
    if name == "burgers-1d":
        return Equation("burgers", "u_t + u*u_x = nu*u_xx + f", 1,
                        lambda u, x, y, t, c: sp.diff(u, t) + u * sp.diff(u, x) - c["nu"] * sp.diff(u, x, 2),
                        {"nu": 0.05})
    return Equation("reaction_diffusion", "u_t = D*u_xx - r*u + f", 1,
                    lambda u, x, y, t, c: sp.diff(u, t) - c["D"] * sp.diff(u, x, 2) + c["r"] * u,
                    {"D": 0.5, "r": 2.0})
