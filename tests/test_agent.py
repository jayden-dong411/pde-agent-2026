import pytest

from pde_agent.agent import PDEAgent
from pde_agent.curriculum import basic_curriculum, frontier_lessons
from pde_agent.equations import builtin_equations, manufactured_problem
from pde_agent.verify import exact_error, relative_error, restrict


def heat_problem(name="heat"):
    eq = builtin_equations().get("heat")
    return manufactured_problem(eq, "sin(pi*x)*exp(-t) + x*t", [(0, 1)], 0.5, {"alpha": 0.5}, name=name)


@pytest.mark.parametrize("lesson", basic_curriculum(), ids=lambda lesson: lesson.name)
def test_agent_solves_basic_curriculum(tmp_path, lesson):
    agent = PDEAgent(home=tmp_path)
    report = agent.solve(lesson.problem, lesson.tol)
    assert report.success, report.message
    if lesson.problem.has_exact():
        assert exact_error(report.problem, report.result) <= lesson.tol
    else:
        # the self-estimated error must be honest: check against a much finer solution
        spec = agent.registry.get(report.method)
        reference = spec.run(report.problem, report.n * 8)
        true_error = relative_error(report.result.u, restrict(reference, report.n))
        assert true_error <= 2 * lesson.tol


def test_experience_predicts_grid_size(tmp_path):
    agent = PDEAgent(home=tmp_path, explore=False)
    first = agent.solve(heat_problem(), tol=1e-4)
    second = agent.solve(heat_problem(), tol=1e-4)
    assert first.success and second.success
    assert second.solver_runs == 1 < first.solver_runs


def test_exploration_then_exploitation_picks_cheapest_reliable_method(tmp_path):
    agent = PDEAgent(home=tmp_path)
    chosen = [agent.solve(heat_problem(f"heat-{i}"), tol=1e-4).method for i in range(3)]
    assert sorted(chosen) == ["heat.backward_euler", "heat.crank_nicolson", "heat.ftcs"]
    assert agent.solve(heat_problem(), tol=1e-4).method == "heat.crank_nicolson"
    # the lesson survives a restart because experience is stored on disk
    fresh = PDEAgent(home=tmp_path)
    report = fresh.solve(heat_problem(), tol=1e-4)
    assert report.method == "heat.crank_nicolson" and report.solver_runs == 1


def test_unstable_method_is_abandoned_and_remembered(tmp_path):
    agent = PDEAgent(home=tmp_path)
    report = agent.solve(heat_problem(), tol=1e-3, options={"r": 0.7})
    assert report.success
    assert report.trials[0] == {**report.trials[0], "method": "heat.ftcs", "outcome": "unstable"}
    stats = agent.memory.stats("heat/1d/dirichlet")
    assert stats["heat.ftcs"].failures == 1


def test_unknown_equation_without_claude_is_reported(tmp_path):
    agent = PDEAgent(home=tmp_path)
    lesson = frontier_lessons()[0]
    report = agent.solve(lesson.problem)
    assert not report.success
    assert "Claude" in report.message
    assert agent.memory.episodes[-1]["signature"] == "reaction_diffusion/1d/dirichlet"


def test_tolerance_out_of_reach_fails_cleanly(tmp_path):
    agent = PDEAgent(home=tmp_path, max_runs_per_method=2)
    report = agent.solve(heat_problem(), tol=1e-13)
    assert not report.success
    assert report.result is not None and report.error is not None  # best effort is kept
    assert {t["method"] for t in report.trials} == {"heat.ftcs", "heat.backward_euler", "heat.crank_nicolson"}
