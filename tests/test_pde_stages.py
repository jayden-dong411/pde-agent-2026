"""v1.6.0: curriculum stages 2-4 (parabolic, smooth hyperbolic, jumps and shocks)."""

import math

import pytest

from pde_agent.agent import PDEAgent
from pde_agent.cli import main
from pde_agent.curriculum import best_work, pde_exam, pde_stages, regret
from pde_agent.features import extract, predict
from pde_agent.problem import PDEProblem
from pde_agent.solvers import builtin_registry
from pde_agent.verify import error_norm, exact_error, new_extrema, relative_error

LESSONS = {lesson.name: lesson for lesson in pde_stages() + pde_exam()}
REGISTRY = builtin_registry()


def run(name, method, n):
    problem = LESSONS[name].problem
    return problem, REGISTRY.get(method).run(problem, n)


def test_lessons_cover_three_stages_and_exam_is_new():
    stages = {lesson.level for lesson in pde_stages()}
    assert stages == {2, 3, 4} and {lesson.level for lesson in pde_exam()} == {2, 3, 4}
    assert not {lesson.name for lesson in pde_stages()} & {lesson.name for lesson in pde_exam()}
    for lesson in pde_stages() + pde_exam():
        assert REGISTRY.for_problem(lesson.problem), lesson.name


def test_error_norm_is_l1_only_for_jumps_and_shocks():
    assert error_norm(LESSONS["advection-square"].problem) == "l1"
    assert error_norm(LESSONS["burgers-rarefaction"].problem) == "l1"
    assert error_norm(LESSONS["advection-long"].problem) == "max"
    assert error_norm(LESSONS["heat-pulse"].problem) == "max"  # diffusion smooths the jump at once
    assert relative_error([1, 0, 0, 0], [0, 0, 0, 0], "l1") == pytest.approx(0.25)
    assert relative_error([2, 1], [1, 1], "l1") == pytest.approx(0.5)


@pytest.mark.parametrize("method", ["burgers_inviscid.rusanov", "burgers_inviscid.muscl", "burgers_inviscid.weno5"])
def test_burgers_shock_converges_without_new_extrema(method):
    errors = []
    for n in (64, 256):
        problem, result = run("burgers-shock", method, n)
        assert new_extrema(problem, result) < 0.01
        errors.append(exact_error(problem, result))
    assert errors[1] < errors[0] / 3  # first order in L1 across a shock
    # the shock moves at (u_l + u_r) / 2 = 0.5: half the jump sits at x = 0.25
    x, u = result.grid[0], result.u
    assert u[x < 0.2].min() > 0.95 and u[x > 0.3].max() < 0.05


def test_linear_schemes_ring_at_a_jump_and_limited_ones_do_not():
    for method, rings in (("advection.lax_wendroff", True), ("advection.rk4_central4", True),
                          ("advection.upwind", False), ("advection.muscl", False), ("advection.weno5", False)):
        problem, result = run("advection-square", method, 128)
        assert (new_extrema(problem, result) > 0.05) == rings, method


def test_features_rule_out_ringing_schemes_on_jumps():
    problem = LESSONS["advection-square"].problem
    feats = extract(problem, 16)
    lw = predict(REGISTRY.get("advection.lax_wendroff"), problem, feats, 0.05, 4096)
    weno = predict(REGISTRY.get("advection.weno5"), problem, feats, 0.05, 4096)
    assert lw.work == math.inf and "振荡" in lw.notes[0]
    assert math.isfinite(weno.work) and weno.order == pytest.approx(0.75)


def test_agent_reports_oscillation_and_moves_on(tmp_path):
    agent = PDEAgent(home=tmp_path)
    problem = LESSONS["advection-square"].problem
    report = agent.solve(problem, 0.05, options=None, learn=False)
    assert report.success and report.method in {"advection.upwind", "advection.muscl", "advection.weno5"}
    assert report.error_kind == "exact"
    # forcing Lax-Wendroff: the run is rejected for new extrema, not accepted on its L1 error
    only = PDEAgent(home=tmp_path / "lw", explore=False)
    only.registry = type(REGISTRY)([REGISTRY.get("advection.lax_wendroff")])
    report = only.solve(problem, 0.5, learn=False)
    assert not report.success and report.trials[0]["outcome"] in {"oscillation", "skipped"}


def test_bdf2_is_second_order_and_damps_a_jump():
    problem = PDEProblem(kind="heat", domain=[(0.0, 1.0)], t_final=0.5, coefficients={"alpha": 0.5}, source="0",
                         exact="sin(pi*x)*exp(-0.5*pi**2*t) + x")
    spec = REGISTRY.get("heat.bdf2")
    e1, e2 = (exact_error(problem, spec.run(problem, n)) for n in (32, 64))
    assert math.log2(e1 / e2) == pytest.approx(2.0, abs=0.3)
    pulse = LESSONS["heat-pulse"].problem
    smooth = spec.run(pulse, 256).u
    cn = REGISTRY.get("heat.crank_nicolson").run(pulse, 256).u
    wiggle = lambda u: abs(sum((u[2:] - 2 * u[1:-1] + u[:-2]) ** 2))  # noqa: E731
    assert wiggle(smooth) < wiggle(cn) / 10


def test_rk4_central4_is_fourth_order():
    problem = LESSONS["advection-long"].problem
    spec = REGISTRY.get("advection.rk4_central4")
    e1, e2 = (exact_error(problem, spec.run(problem, n)) for n in (32, 64))
    assert math.log2(e1 / e2) == pytest.approx(4.0, abs=0.4)


def test_best_work_and_regret():
    lesson = LESSONS["burgers-shock"]
    name, work = best_work(REGISTRY, lesson.problem, lesson.tol)
    assert name == "burgers_inviscid.rusanov" and work > 0
    assert regret(2 * work, (name, work)) == pytest.approx(2.0)
    assert best_work(REGISTRY, LESSONS["heat-pulse"].problem, 1e-3) is None  # no exact solution


def test_cli_runs_stages_and_regret(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "curriculum", "--pde", "--level", "2"]) == 0
    out = capsys.readouterr().out
    assert "heat-alpha-10" in out and "成功 5/5" in out
    assert main(["--home", str(tmp_path), "curriculum", "--pde", "--exam", "--level", "2", "--regret"]) == 0
    out = capsys.readouterr().out
    assert "后悔度" in out and "几何平均" in out
