import dataclasses
import json

import pytest

from pde_agent import skills
from pde_agent.agent import PDEAgent
from pde_agent.curriculum import frontier_lessons
from pde_agent.equations import Equation, builtin_equations
from pde_agent.problem import PDEProblem
from pde_agent.skills import (
    EXAM_VERSION,
    TEMPLATE,
    SkillCandidate,
    SkillLibrary,
    SkillTask,
    ValidationReport,
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


CHEAT = '''
from pde_agent.skill_api import SolveResult, evaluate, grid, mesh
SOLVER = {{"name": "{kind}.peek", "kind": "{kind}", "dims": [1], "boundaries": ["dirichlet"], "order": 2,
          "description": "copies {field} instead of solving"}}
def solve(problem, n, options):
    axes = grid(problem, n)
    u = evaluate(problem.fn("{field}"), mesh(axes), problem.t_final or 0.0)
    return SolveResult(u=u, grid=axes, n=n, t=problem.t_final, steps=1, dt=problem.t_final, work=1.0)
'''


@pytest.mark.parametrize("kind,field,hint", [
    ("heat", "exact", "has no exact"),
    ("heat", "boundary_value", "error on the finest grid"),
    ("poisson", "initial", "has no initial"),
])
def test_skills_cannot_copy_the_answer(kind, field, hint):
    if kind == "heat":
        problem = PDEProblem(kind="heat", domain=[(0.0, 1.0)], t_final=0.5, initial="sin(pi*x)")
    else:
        problem = PDEProblem(kind="poisson", domain=[(0.0, 1.0)], source="1")
    task = SkillTask.from_problem(problem, builtin_equations(), builtin_registry())
    report = validate_candidate(SkillCandidate(code=CHEAT.format(kind=kind, field=field)), task)
    assert not report.ok and hint in report.feedback()


FORM, FLIPPED_FORM = '"u_t = D * u_xx - r * u + f"', '"u_t = D * u_xx + r * u + f"'


def test_skill_must_solve_the_equation_the_task_states():
    _, task = reaction_task()
    flipped = (GOOD.replace('+ c["r"] * u', '- c["r"] * u').replace("- r * eye", "+ r * eye")
               .replace(FORM, FLIPPED_FORM))  # operator, solver and form agree with each other, not with the task
    report = validate_candidate(SkillCandidate(code=flipped), task)
    assert not report.ok and report.message.startswith("operator does not match the task equation")


def test_equation_form_must_match_the_operator():
    _, task = reaction_task()
    report = validate_candidate(SkillCandidate(code=GOOD.replace(FORM, FLIPPED_FORM)), task)
    assert not report.ok and report.message.startswith("EQUATION form does not match operator")


def test_new_equation_needs_equation_text():
    _, task = reaction_task()
    report = validate_candidate(SkillCandidate(code=GOOD), dataclasses.replace(task, equation=""))
    assert not report.ok
    assert report.message == "the task gives no equation text to check EQUATION against"


def test_learned_equations_take_the_problem_text_first():
    lesson, _ = reaction_task()
    book = builtin_equations()
    book.add(Equation(kind="reaction_diffusion", form="u_t = D * u_xx - r * u + f", time_order=1,
                      operator=lambda u, x, y, t, c: u, origin="learned"))
    stated = SkillTask.from_problem(lesson.problem, book, builtin_registry())
    assert stated.equation == "u_t = D*u_xx - r*u + f" and not stated.new_equation
    bare = PDEProblem.from_dict({**lesson.problem.to_dict(), "equation": ""})
    assert SkillTask.from_problem(bare, book, builtin_registry()).equation == "u_t = D * u_xx - r * u + f"


def install_as_v09(root, code, task, solver):
    """A skill as v0.9.0 left it: admitted by the old exam, no exam_version in its manifest."""
    report = ValidationReport(ok=True, message="passed", solver=solver, kind=task.kind, order=2.0)
    path = SkillLibrary(root).install(SkillCandidate(code=code), report, task).with_suffix(".json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["exam_version"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def heat_task():
    problem = PDEProblem(kind="heat", domain=[(0.0, 1.0)], t_final=0.5, initial="sin(pi*x)")
    return SkillTask.from_problem(problem, builtin_equations(), builtin_registry())


def test_builtin_tests_always_come_first():
    _, task = reaction_task()
    extra = ["sin(pi*x)*cos(t)", "cos(pi*x)*exp(-t)", "sin(2*pi*x)*(1 + t)"]
    report = validate_candidate(SkillCandidate(code=GOOD, test_solutions=extra), task)
    assert report.ok, report.feedback()
    assert [case["exact"] for case in report.cases] == ["sin(pi*x)*exp(-t)", "cos(2*x)*cos(t) + x*t", *extra[:2]]


def test_install_records_the_exam_version(tmp_path):
    _, task = reaction_task()
    report = ValidationReport(ok=True, message="passed", solver="reaction_diffusion.crank_nicolson",
                              kind="reaction_diffusion", order=2.0)
    path = SkillLibrary(tmp_path).install(SkillCandidate(code=GOOD), report, task)
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["exam_version"] == EXAM_VERSION == 2


def test_old_skills_are_examined_again_and_upgraded(tmp_path):
    _, task = reaction_task()
    manifest = install_as_v09(tmp_path / "skills", GOOD, task, "reaction_diffusion.crank_nicolson")
    agent = PDEAgent(home=tmp_path)
    assert agent.skill_errors == [] and agent.registry.get("reaction_diffusion.crank_nicolson") is not None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["exam_version"] == 2
    assert data["validation"]["ok"] and len(data["validation"]["cases"]) == 2


def test_old_cheating_skills_are_disabled_once(tmp_path, monkeypatch):
    manifest = install_as_v09(tmp_path / "skills", CHEAT.format(kind="heat", field="exact"), heat_task(), "heat.peek")
    agent = PDEAgent(home=tmp_path)
    reason = "the solver raised an error on a test problem"
    assert agent.skill_errors == [f"技能 heat__peek.py 已停用：{reason}"]
    assert agent.registry.get("heat.peek") is None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["disabled"] == reason and data["disabled_at"] and "exam_version" not in data
    monkeypatch.setattr(skills, "validate_file", lambda *args, **kwargs: pytest.fail("examined a disabled skill again"))
    assert PDEAgent(home=tmp_path).skill_errors == agent.skill_errors


def test_old_new_equation_skills_without_equation_text_are_disabled(tmp_path):
    _, task = reaction_task()
    install_as_v09(tmp_path / "skills", GOOD, dataclasses.replace(task, equation=""), "reaction_diffusion.crank_nicolson")
    assert PDEAgent(home=tmp_path).skill_errors == [
        "技能 reaction_diffusion__crank_nicolson.py 已停用：the task gives no equation text to check EQUATION against"]


def test_a_disabled_skill_comes_back_once_its_reason_is_deleted(tmp_path):
    _, task = reaction_task()
    manifest = install_as_v09(tmp_path / "skills", GOOD, task, "reaction_diffusion.crank_nicolson")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data.update(disabled="observed order 1.1 is below 1.40 (claimed order 2.0)", disabled_at="2026-10-14T00:00:00+00:00")
    manifest.write_text(json.dumps(data), encoding="utf-8")
    assert PDEAgent(home=tmp_path).registry.get("reaction_diffusion.crank_nicolson") is None
    del data["disabled"]  # what the CHANGELOG tells people to do
    manifest.write_text(json.dumps(data), encoding="utf-8")
    agent = PDEAgent(home=tmp_path)
    assert agent.skill_errors == [] and agent.registry.get("reaction_diffusion.crank_nicolson") is not None
    assert json.loads(manifest.read_text(encoding="utf-8"))["exam_version"] == 2
