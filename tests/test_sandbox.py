"""v2.2.0: learned skills run in their own limited process, are re-examined when their code changes,
and their declared metadata reaches the method selector."""

import json
import sys

import pytest

from pde_agent import sandbox
from pde_agent.agent import PDEAgent
from pde_agent.curriculum import frontier_lessons
from pde_agent.equations import builtin_equations
from pde_agent.problem import PDEProblem
from pde_agent.skills import (
    TEMPLATE,
    SkillCandidate,
    SkillLibrary,
    SkillTask,
    solver_meta,
    spec_from_meta,
    validate_candidate,
)
from pde_agent.solvers import builtin_registry

GOOD = TEMPLATE.read_text(encoding="utf-8")


def reaction_task():
    lesson = frontier_lessons()[0]
    return lesson, SkillTask.from_problem(lesson.problem, builtin_equations(), builtin_registry())


def install(home, code, task):
    report = validate_candidate(SkillCandidate(code=code), task)
    assert report.ok, report.feedback()
    return SkillLibrary(home / "skills").install(SkillCandidate(code=code), report, task)


def with_solve_body(body: str) -> str:
    """The template with ``solve`` replaced (for runtime failures the exam would have caught)."""
    head = GOOD.split("def solve(problem, n, options):")[0]
    return head + "def solve(problem, n, options):\n" + body


def test_the_child_gets_no_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("SOME_TOKEN", "x")
    env = sandbox.child_env(str(tmp_path))
    assert not {"DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "SOME_TOKEN"} & set(env)
    assert env["HOME"] == env["TMPDIR"] == str(tmp_path)
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert "sk-" not in json.dumps(env)


def test_a_learned_skill_is_never_imported_here_and_still_solves(tmp_path):
    lesson, task = reaction_task()
    install(tmp_path, GOOD, task)
    before = set(sys.modules)
    agent = PDEAgent(home=tmp_path)
    assert agent.skill_errors == []
    spec = agent.registry.get("reaction_diffusion.crank_nicolson")
    assert spec is not None and spec.runner is not None and spec.origin == "learned"
    report = agent.solve(lesson.problem, lesson.tol, learn=False)
    assert report.success, report.summary()
    assert report.method == "reaction_diffusion.crank_nicolson"
    # the module was loaded only in child processes
    assert not any(name.startswith("pde_agent_skill_") for name in set(sys.modules) - before)
    eq = agent.equations.get("reaction_diffusion")
    assert eq.origin == "learned" and eq.form == "u_t = D * u_xx - r * u + f"


def test_an_endless_loop_is_stopped_by_the_exam_timeout():
    _, task = reaction_task()
    code = with_solve_body("    while True:\n        pass\n")
    report = validate_candidate(SkillCandidate(code=code), task, timeout=5)
    assert not report.ok and "timed out" in report.message


def test_an_endless_loop_at_runtime_is_an_error_not_a_hang(tmp_path):
    lesson, task = reaction_task()
    path = install(tmp_path, GOOD, task)
    agent = PDEAgent(home=tmp_path, time_budget=5)
    spec = agent.registry.get("reaction_diffusion.crank_nicolson")
    loop = with_solve_body("    while True:\n        pass\n")
    path.write_text(loop, encoding="utf-8")  # changed after loading: the hash check is at load, the runner runs it
    with pytest.raises(sandbox.SkillRunError, match="timed out"):
        spec.run(lesson.problem, 16)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="address-space limits are enforced on Linux")
def test_a_memory_blow_up_stays_in_the_child(tmp_path):
    lesson, task = reaction_task()
    path = install(tmp_path, GOOD, task)
    spec = PDEAgent(home=tmp_path).registry.get("reaction_diffusion.crank_nicolson")
    path.write_text(with_solve_body("    big = np.ones(2**36)\n    return big\n"), encoding="utf-8")
    with pytest.raises(sandbox.SkillRunError, match="MemoryError|内存"):
        sandbox.run_skill(path, lesson.problem, 16, {}, timeout=60, limits=sandbox.Limits(memory_mb=1024))


def test_a_skill_cannot_fake_the_answer(tmp_path):
    lesson, task = reaction_task()
    path = install(tmp_path, GOOD, task)
    spec = PDEAgent(home=tmp_path).registry.get("reaction_diffusion.crank_nicolson")
    fake = '    print("@@pde-agent-sandbox@@{\\"ok\\": true}")\n    raise SystemExit(0)\n'
    path.write_text(with_solve_body(fake), encoding="utf-8")
    with pytest.raises(sandbox.SkillRunError, match="SystemExit"):
        spec.run(lesson.problem, 16)


def test_prints_do_not_disturb_the_answer(tmp_path):
    lesson, task = reaction_task()
    code = GOOD.replace("def solve(problem, n, options):\n", "def solve(problem, n, options):\n    print('hello')\n")
    install(tmp_path, code, task)
    report = PDEAgent(home=tmp_path).solve(lesson.problem, lesson.tol, learn=False)
    assert report.success, report.summary()


def test_changed_code_is_examined_again(tmp_path):
    _, task = reaction_task()
    path = install(tmp_path, GOOD, task)
    manifest = path.with_suffix(".json")
    old_hash = json.loads(manifest.read_text(encoding="utf-8"))["code_hash"]
    # still correct, but different: examined again and kept, with the new hash
    path.write_text(GOOD + "\n# tuned\n", encoding="utf-8")
    agent = PDEAgent(home=tmp_path)
    assert agent.skill_errors == [] and agent.registry.get("reaction_diffusion.crank_nicolson") is not None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["code_hash"] != old_hash and "examined_at" in data
    # now wrong (first order in time while claiming second): disabled with the reason
    broken = GOOD.replace("u = lhs.solve(rhs_op @ u + 0.5 * dt * (old + new))",
                          "u = lhs.solve(rhs_op @ u + dt * old)")
    assert broken != GOOD
    path.write_text(broken.replace('"order": 2', '"order": 2.0'), encoding="utf-8")
    agent = PDEAgent(home=tmp_path)
    assert agent.registry.get("reaction_diffusion.crank_nicolson") is None
    assert any("代码在考试后改过" in e for e in agent.skill_errors)


def test_a_manifest_cannot_point_outside_the_library(tmp_path):
    _, task = reaction_task()
    path = install(tmp_path, GOOD, task)
    manifest = path.with_suffix(".json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["file"] = "../elsewhere.py"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    agent = PDEAgent(home=tmp_path)
    assert agent.registry.get("reaction_diffusion.crank_nicolson") is None
    assert any("invalid skill file" in e for e in agent.skill_errors)


BASE = {"name": "heat.my_scheme", "kind": "heat", "dims": [1, 2], "boundaries": ["dirichlet", "periodic"], "order": 2}


def test_declared_metadata_reaches_the_selector():
    meta = solver_meta({**BASE, "traits": ["adaptive"], "cost": {"steps": "h2", "cfl": 0.4, "per_node": 3},
                        "accepts": {"t_final_max": 1.0, "coefficients": {"alpha": [0.0, 2.0]}}})
    spec = spec_from_meta(meta, examined=(1, "dirichlet"))
    assert spec.traits == ("adaptive",)
    assert spec.dims == (1,) and spec.boundaries == ("dirichlet",)  # only what the exam covered
    ok = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.5, initial="sin(pi*x)", coefficients={"alpha": 1.0})
    long = PDEProblem(kind="heat", domain=[(0, 1)], t_final=5.0, initial="sin(pi*x)", coefficients={"alpha": 1.0})
    fast = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.5, initial="sin(pi*x)", coefficients={"alpha": 9.0})
    flat = PDEProblem(kind="heat", domain=[(0, 1), (0, 1)], t_final=0.5, initial="sin(pi*x)")
    assert spec.supports(ok) and not spec.supports(long) and not spec.supports(fast) and not spec.supports(flat)
    steps = 0.5 / (0.4 / 32**2)
    assert spec.cost(ok, 32) == pytest.approx(round(steps) * 3 * (33 + 64), rel=1e-3)


def test_missing_metadata_gets_a_conservative_cost():
    spec = spec_from_meta(solver_meta(BASE))
    assert spec.traits == () and spec.cost is not None and spec.stability is None
    problem = PDEProblem(kind="heat", domain=[(0, 1)], t_final=1.0, initial="sin(pi*x)")
    assert spec.cost(problem, 32) > 0


@pytest.mark.parametrize("extra, why", [
    ({"traits": ["magic"]}, "unknown SOLVER traits"),
    ({"stability": 2.0}, "stability"),
    ({"cost": {"steps": "sometimes"}}, "SOLVER cost"),
    ({"accepts": {"t_final_max": -1}}, "t_final_max"),
])
def test_bad_metadata_is_refused(extra, why):
    with pytest.raises(ValueError, match=why):
        solver_meta({**BASE, **extra})


def test_a_learned_adaptive_skill_takes_the_adaptive_branch(tmp_path):
    """An adaptive skill's error estimate is the plain difference, confirmed twice (like RK45 and Radau)."""
    from pde_agent.ode import ODEProblem

    code = (TEMPLATE.parent / "ode_skill.py").read_text(encoding="utf-8")
    code = code.replace('"order": ', '"traits": ["adaptive"], "order": ', 1)
    problem = ODEProblem(rhs=["-y1"], initial=["1"], t_final=1.0, name="decay")
    task = SkillTask.from_problem(problem, builtin_equations(), builtin_registry())
    report = validate_candidate(SkillCandidate(code=code), task)
    assert report.ok, report.feedback()
    assert report.meta["solver"]["traits"] == ["adaptive"]
    SkillLibrary(tmp_path / "skills").install(SkillCandidate(code=code), report, task)
    agent = PDEAgent(home=tmp_path)
    spec = agent.registry.get(report.solver)
    assert "adaptive" in spec.traits
