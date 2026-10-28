import pytest

from pde_agent.agent import PDEAgent
from pde_agent.curriculum import basic_curriculum, frontier_lessons
from pde_agent.equations import builtin_equations, manufactured_problem
from pde_agent.problem import PDEProblem
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
    # features already land the first try near the right grid; experience keeps it there
    assert second.solver_runs == 1 and second.n <= first.n


def test_features_pick_the_cheapest_method_and_experience_keeps_it(tmp_path):
    agent = PDEAgent(home=tmp_path)
    first = agent.solve(heat_problem("heat-0"), tol=1e-4)
    # explicit steps cost alpha/h^2 and backward Euler is first order: both predicted far dearer, so not explored first
    assert first.method == "heat.crank_nicolson" and first.solver_runs == 1
    assert first.predictions["heat.ftcs"].work > 8 * first.predictions["heat.crank_nicolson"].work
    # the lesson survives a restart because experience is stored on disk
    report = PDEAgent(home=tmp_path).solve(heat_problem(), tol=1e-4)
    assert report.method == "heat.crank_nicolson" and report.solver_runs == 1


def test_unstable_method_is_abandoned_and_remembered(tmp_path):
    # with a tiny alpha explicit steps are cheap, so features put FTCS first
    problem = manufactured_problem(builtin_equations().get("heat"), "sin(pi*x)*exp(-t) + x*t", [(0, 1)], 0.5,
                                   {"alpha": 0.001}, name="slow-heat")
    agent = PDEAgent(home=tmp_path)
    report = agent.solve(problem, tol=1e-3, options={"r": 0.7})
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


R3 = dict(kind="heat", domain=[(0.0, 1.0)], t_final=0.01, coefficients={"alpha": 1e-4})


@pytest.mark.parametrize("exact", [None, "sin(32*pi*x)*exp(-0.0001*(32*pi)**2*t)"], ids=["estimated", "exact"])
def test_aliased_data_is_solved_from_the_floor_up(tmp_path, exact):
    report = PDEAgent(home=tmp_path).solve(PDEProblem(**R3, initial="sin(32*pi*x)", exact=exact), tol=1e-3)
    assert report.min_n == 128
    assert report.success and report.n >= 128
    assert min(a.n for a in report.attempts) >= 128


class NoLearning:
    def skill_session(self, task):
        raise AssertionError("the agent must not try to learn")


def test_unresolvable_data_is_reported_not_solved(tmp_path):
    agent = PDEAgent(home=tmp_path, brain=NoLearning())
    report = agent.solve(PDEProblem(**R3, initial="sin(5000*pi*x)"), tol=1e-3)
    assert not report.success and report.attempts == [] and report.learned == []
    assert report.message == "题目数据在最细网格（n = 4096）上也分辨不清，没有求解"
    assert agent.memory.episodes[-1]["success"] is False


@pytest.mark.parametrize("name", ["poisson-1d-unknown", "heat-1d-unknown"])
def test_smooth_estimated_problems_keep_their_coarse_companion(tmp_path, name):
    lesson = next(item for item in basic_curriculum() if item.name == name)
    report = PDEAgent(home=tmp_path).solve(lesson.problem, lesson.tol)
    # the first grid is the feature prediction's coarse companion, never below the floor
    assert report.min_n == 8 and report.attempts[0].n == max(8, report.predictions[report.method].start // 2)
