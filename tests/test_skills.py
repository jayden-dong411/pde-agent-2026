import pytest

from pde_agent.agent import PDEAgent
from pde_agent.curriculum import frontier_lessons
from pde_agent.equations import builtin_equations
from pde_agent.skills import (
    TEMPLATE,
    SkillCandidate,
    SkillLibrary,
    SkillTask,
    static_check,
    validate_candidate,
)
from pde_agent.solvers import builtin_registry

GOOD = TEMPLATE.read_text(encoding="utf-8")


def reaction_task():
    lesson = frontier_lessons()[0]
    return lesson, SkillTask.from_problem(lesson.problem, builtin_equations(), builtin_registry())


def test_template_passes_static_check():
    assert static_check(GOOD) == []


@pytest.mark.parametrize("snippet", [
    "import os",
    "from subprocess import run",
    "data = open('x').read()",
    "values = np.load('x.npy')",
    "kind = SOLVER.__class__",
    "f = getattr(np, 'sin')",
])
def test_static_check_rejects_dangerous_code(snippet):
    assert static_check(GOOD + "\n" + snippet + "\n")


def test_static_check_requires_the_contract():
    assert any("solve" in p for p in static_check("SOLVER = {}\n"))


def test_task_describes_a_new_equation():
    lesson, task = reaction_task()
    assert task.new_equation and task.kind == "reaction_diffusion"
    assert task.coefficients == {"D": 0.5, "r": 2.0}
    assert task.equation == "u_t = D*u_xx - r*u + f"


def test_correct_skill_passes_the_exam():
    _, task = reaction_task()
    report = validate_candidate(SkillCandidate(code=GOOD), task)
    assert report.ok, report.feedback()
    assert report.solver == "reaction_diffusion.crank_nicolson"
    assert all(case["orders"][-1] > 1.8 for case in report.cases)


def test_wrong_physics_fails_the_exam():
    _, task = reaction_task()
    wrong = GOOD.replace("a = D * laplacian_1d(m, h) - r * eye", "a = D * laplacian_1d(m, h) + r * eye")
    assert wrong != GOOD
    report = validate_candidate(SkillCandidate(code=wrong), task)
    assert not report.ok
    assert "error" in report.message or "order" in report.message


def test_overclaimed_order_fails_the_exam():
    _, task = reaction_task()
    boastful = GOOD.replace('"order": 2,', '"order": 4,')
    report = validate_candidate(SkillCandidate(code=boastful), task)
    assert not report.ok and "order" in report.message


def test_redefining_a_builtin_equation_is_rejected():
    heat_task = SkillTask.from_problem(frontier_lessons()[1].problem, builtin_equations(), builtin_registry())
    code = GOOD.replace('"kind": "reaction_diffusion"', '"kind": "heat"').replace(
        '"name": "reaction_diffusion.crank_nicolson"', '"name": "heat.my_method"')
    report = validate_candidate(SkillCandidate(code=code), heat_task)
    assert not report.ok and "built-in" in report.message


def test_hanging_skill_times_out():
    _, task = reaction_task()
    slow = GOOD.replace("def solve(problem, n, options):", "def solve(problem, n, options):\n    while True:\n        pass\n")
    report = validate_candidate(SkillCandidate(code=slow), task, timeout=5)
    assert not report.ok and "timed out" in report.message


def test_installed_skill_is_used_by_the_agent(tmp_path):
    lesson, task = reaction_task()
    report = validate_candidate(SkillCandidate(code=GOOD, summary="示例"), task)
    SkillLibrary(tmp_path / "skills").install(SkillCandidate(code=GOOD, summary="示例"), report, task)

    agent = PDEAgent(home=tmp_path)
    assert agent.skill_errors == []
    assert "reaction_diffusion" in agent.equations
    assert agent.registry.get("reaction_diffusion.crank_nicolson").origin == "learned"
    result = agent.solve(lesson.problem, lesson.tol)
    assert result.success and result.method == "reaction_diffusion.crank_nicolson"


def test_broken_skill_file_does_not_stop_the_agent(tmp_path):
    _, task = reaction_task()
    report = validate_candidate(SkillCandidate(code=GOOD), task)
    path = SkillLibrary(tmp_path / "skills").install(SkillCandidate(code=GOOD), report, task)
    path.write_text("this is not python", encoding="utf-8")
    agent = PDEAgent(home=tmp_path)
    assert len(agent.skill_errors) == 1
    assert "reaction_diffusion" not in agent.equations
