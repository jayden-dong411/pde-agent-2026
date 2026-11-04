"""v1.3.0: ODEs in the web page, learned ODE skills, and the ODE cross-check."""

import json
import threading
import urllib.request

import numpy as np
import pytest

from pde_agent.agent import PDEAgent
from pde_agent.app import AppState, RequestError, make_server, preview_json, problem_from_form
from pde_agent.curriculum import ode_curriculum
from pde_agent.equations import builtin_equations
from pde_agent.llm import ClaudeBrain, ODE_SKILL_SYSTEM, problem_from_reply, task_prompt
from pde_agent.ode import ODEProblem
from pde_agent.skills import ODE_TEMPLATE, SkillCandidate, SkillTask, static_check, validate_candidate
from pde_agent.solvers import builtin_registry

from test_llm import ScriptedClient

TEMPLATE_CODE = ODE_TEMPLATE.read_text(encoding="utf-8")
LESSONS = {lesson.name: lesson for lesson in ode_curriculum()}


def task_for(name, reason="test"):
    return SkillTask.from_problem(LESSONS[name].problem, builtin_equations(), builtin_registry(), reason)


# --- the form --------------------------------------------------------------------------------


def test_the_form_builds_an_ode_from_lines():
    problem = problem_from_form({"problem": {
        "kind": "ode", "rhs": "y1' = y2\ndy2/dt = -w**2*y1\n", "initial": "1， 0", "t_final": 3,
        "coefficients": {"w": 2}, "exact": "cos(2*t); -2*sin(2*t)"}})
    assert isinstance(problem, ODEProblem)
    assert problem.rhs == ["y2", "-w**2*y1"] and problem.initial == ["1", "0"] and problem.has_exact()
    assert problem.signature() == "ode/0d/m2"


@pytest.mark.parametrize("data,why", [
    ({"rhs": "", "initial": "1", "t_final": 1}, "请填写方程组"),
    ({"rhs": "y2", "initial": "1", "t_final": 1}, "y2"),  # y2 does not exist in a one-equation system
    ({"rhs": "-y1", "initial": "1, 2", "t_final": 1}, "initial values"),
    ({"rhs": "-y1", "initial": "1", "t_final": 0}, "t_final"),
])
def test_bad_ode_forms_are_the_form_s_fault(data, why):
    with pytest.raises(RequestError) as info:
        problem_from_form({"problem": {"kind": "ode", **data}})
    assert why in str(info.value)


def test_preview_draws_the_exact_solution_of_an_ode():
    problem = LESSONS["ode-harmonic"].problem
    data = preview_json(problem)
    assert data["dim"] == 0 and len(data["curves"]) == 2 and len(data["axes"][0]) == len(data["curves"][0]["values"])
    assert preview_json(problem.for_solver())["curves"] == []


@pytest.fixture
def server(tmp_path):
    state = AppState(PDEAgent(home=tmp_path), "claude")
    srv = make_server(state, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read())


def test_the_page_solves_an_ode_and_gets_a_trajectory(server):
    vdp = LESSONS["ode-vdp-1"].problem  # no exact solution: estimated, then cross-checked
    data = post(server, "/api/solve", {"problem": {"kind": "ode", "rhs": "\n".join(vdp.rhs), "initial": ", ".join(vdp.initial),
                                                   "t_final": vdp.t_final, "coefficients": vdp.coefficients}, "tol": 1e-3})
    assert data["success"] and data["error_kind"] == "estimated"
    assert data["crosscheck"]["passed"] and data["crosscheck"]["checks"][0]["name"] == "reference"
    s = data["solution"]
    assert s["dim"] == 0 and s["names"] == ["y1", "y2"] and len(s["u"]) == 2 and s["exact"] is None
    end = np.array(s["trajectory"]["y"])[:, -1]
    assert np.allclose(end, s["u"], rtol=1e-2, atol=1e-2)  # the drawn curve ends at the agent's answer
    info = json.loads(urllib.request.urlopen(server + "/api/info").read())
    assert any(e["problem"]["kind"] == "ode" for e in info["examples"])


def test_the_page_shows_the_exact_ode_solution(server):
    problem = LESSONS["ode-harmonic"].problem
    data = post(server, "/api/solve", {"problem": problem.to_dict(), "tol": 1e-3})
    assert data["success"] and data["error_kind"] == "exact"
    s = data["solution"]
    assert np.allclose(s["exact"], problem.exact_at(problem.t_final)) and len(s["exact_curves"]["y"]) == 2


# --- text descriptions ----------------------------------------------------------------------------


def test_a_parsed_reply_can_be_an_ode():
    problem = problem_from_reply({
        "kind": "ode", "equation": "θ'' + sin θ = 0", "domain": [], "t_final": 10, "coefficients": [],
        "source": "0", "boundary": "dirichlet", "boundary_value": "", "initial": "", "initial_velocity": "",
        "exact": "", "notes": "θ 记作 y1，角速度记作 y2", "ode": {"rhs": ["y2", "-sin(y1)"], "initial": ["1", "0"], "exact": []}})
    assert isinstance(problem, ODEProblem) and problem.rhs == ["y2", "-sin(y1)"] and not problem.has_exact()
    with pytest.raises(ValueError):
        problem_from_reply({"kind": "ode", "t_final": 1, "ode": {"rhs": [], "initial": [], "exact": []}})


# --- learned ODE skills ---------------------------------------------------------------------------


def test_the_ode_template_passes_its_own_checks():
    assert static_check(TEMPLATE_CODE) == []
    report = validate_candidate(SkillCandidate(code=TEMPLATE_CODE), task_for("ode-vdp-1"))
    assert report.ok, report.feedback()
    assert len(report.cases) == 3 and "task's own ODE" in report.cases[-1]["exact"]


def test_an_explicit_skill_fails_a_stiff_task():
    report = validate_candidate(SkillCandidate(code=TEMPLATE_CODE), task_for("ode-stiff-1e3"))
    assert not report.ok and "task's own ODE" in report.cases[-1]["exact"]


@pytest.mark.parametrize("change,why", [
    (('"dims": [0]', '"dims": [1]'), 'an ODE SOLVER has "dims": [0]'),
    (('"order": 3', '"order": 5'), "observed order"),
    (("SOLVER = {", 'EQUATION = {"kind": "ode", "time_order": 1}\n\ndef operator(u, x, y, t, c):\n    return u\n\nSOLVER = {'),
     "must not define EQUATION"),
])
def test_bad_ode_skills_are_rejected(change, why):
    report = validate_candidate(SkillCandidate(code=TEMPLATE_CODE.replace(*change)), task_for("ode-harmonic"))
    assert not report.ok and why in report.message


def test_the_agent_learns_an_ode_skill_and_keeps_it(tmp_path):
    reply = {"code": TEMPLATE_CODE, "test_solutions": [], "summary": "三阶强稳定 Runge-Kutta"}
    client = ScriptedClient([reply])
    agent = PDEAgent(home=tmp_path, brain=ClaudeBrain(client=client), remember=False)
    outcome = agent.learn(LESSONS["ode-damped"].problem, reason="test", max_rounds=1)
    assert outcome.success and outcome.solver == "ode.ssprk3", outcome.message
    call = client.calls[0]
    assert call["system"].startswith("You write numerical ODE integrators")
    assert "y1' = y2" in call["messages"][0]["content"] and "stiffness" in call["messages"][0]["content"]
    # the next start loads it again, and it competes with the built-in methods
    again = PDEAgent(home=tmp_path, remember=False)
    assert again.skill_errors == [] and again.registry.get("ode.ssprk3").origin == "learned"
    problem = LESSONS["ode-damped"].problem
    assert "ode.ssprk3" in [spec.name for spec in again.registry.for_problem(problem)]
    assert again.solve(problem, 1e-3).success


def test_ode_prompts():
    task = task_for("ode-robertson")
    prompt = task_prompt(task)
    assert "ODE integrator" in prompt and "stiffness: 1.33e+05" in prompt and "ode.bdf2" in prompt
    assert "n = 32, 64 and 128" in ODE_SKILL_SYSTEM and "use an implicit" in ODE_SKILL_SYSTEM
