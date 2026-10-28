"""Claude integration, tested against a scripted stand-in for the API client."""

import json
from types import SimpleNamespace

import pytest

from pde_agent.agent import PDEAgent
from pde_agent.curriculum import frontier_lessons
from pde_agent.equations import builtin_equations
from pde_agent.llm import DEFAULT_MODEL, FALLBACK_BETA, PARSE_SYSTEM, SKILL_SYSTEM, ClaudeBrain, LLMError, task_prompt
from pde_agent.problem import PDEProblem
from pde_agent.skills import TEMPLATE, SkillTask
from pde_agent.solvers import builtin_registry

GOOD = TEMPLATE.read_text(encoding="utf-8")


class ScriptedClient:
    """Mimics ``client.beta.messages.stream(...)`` with canned replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        reply = self.replies.pop(0)
        stop_reason = "end_turn"
        if isinstance(reply, tuple):
            stop_reason, reply = reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        message = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])
        return _Stream(message)


class _Stream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def problem_reply(problem, notes="假设边界为 Dirichlet"):
    data = problem.to_dict()
    return {
        "kind": data["kind"],
        "equation": data["equation"],
        "domain": [{"lower": a, "upper": b} for a, b in data["domain"]],
        "t_final": data["t_final"] or 0,
        "coefficients": [{"name": k, "value": v} for k, v in data["coefficients"].items()],
        "source": data["source"],
        "boundary": data["boundary"],
        "boundary_value": data["boundary_value"] or "",
        "initial": data["initial"] or "",
        "initial_velocity": data["initial_velocity"] or "",
        "exact": data["exact"] or "",
        "notes": notes,
    }


HEAT_REPLY = {
    "kind": "heat", "equation": "u_t = alpha*u_xx",
    "domain": [{"lower": 0, "upper": 1}], "t_final": 0.5,
    "coefficients": [{"name": "alpha", "value": 0.1}],
    "source": "0", "boundary": "dirichlet", "boundary_value": "",
    "initial": "sin(pi*x)", "initial_velocity": "", "exact": "", "notes": "两端温度为零",
}


def test_parse_problem_builds_a_problem_and_uses_structured_output():
    client = ScriptedClient([HEAT_REPLY])
    parsed = ClaudeBrain(client=client).parse_problem("一根长 1 的杆……", builtin_equations())
    p = parsed.problem
    assert (p.kind, p.t_final, p.coefficients, p.initial, p.exact) == ("heat", 0.5, {"alpha": 0.1}, "sin(pi*x)", None)
    assert p.description == "两端温度为零"
    call = client.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "- heat: u_t = alpha * (u_xx + u_yy) + f" in call["system"]


def test_parse_problem_asks_again_after_an_invalid_reply():
    bad = {**HEAT_REPLY, "initial": "foo(x)"}
    client = ScriptedClient([bad, HEAT_REPLY])
    parsed = ClaudeBrain(client=client).parse_problem("……", builtin_equations())
    assert parsed.problem.initial == "sin(pi*x)"
    retry = client.calls[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert "foo" in retry[-1]["content"]


def test_refusal_is_an_error():
    client = ScriptedClient([("refusal", "")])
    with pytest.raises(LLMError):
        ClaudeBrain(client=client).parse_problem("……", builtin_equations())


def test_agent_learns_a_new_equation_from_words(tmp_path):
    lesson = frontier_lessons()[0]
    broken = GOOD.replace("a = D * laplacian_1d(m, h) - r * eye", "a = D * laplacian_1d(m, h) + r * eye")
    client = ScriptedClient([
        problem_reply(lesson.problem),
        {"code": broken, "test_solutions": ["sin(pi*x)*cos(t)"], "summary": "第一版"},
        {"code": GOOD, "test_solutions": [], "summary": "Crank-Nicolson 反应扩散求解器"},
    ])
    agent = PDEAgent(home=tmp_path, brain=ClaudeBrain(client=client))
    report = agent.ask("在 [0,1] 上求解反应扩散方程 u_t = 0.5 u_xx - 2u + f ……", tol=lesson.tol)

    assert report.success, report.summary()
    assert report.method == "reaction_diffusion.crank_nicolson"
    learned = report.learned[0]
    assert learned.success and learned.rounds == 2
    repair = client.calls[2]["messages"]
    assert [m["role"] for m in repair] == ["user", "assistant", "user"]
    assert "did not pass validation" in repair[-1]["content"]
    assert (tmp_path / "skills" / "reaction_diffusion__crank_nicolson.py").exists()

    # a new agent on the same home already knows the equation, no Claude needed
    again = PDEAgent(home=tmp_path).solve(lesson.problem, lesson.tol)
    assert again.success and again.solver_runs == 1


def test_agent_gives_up_after_the_round_limit(tmp_path):
    lesson = frontier_lessons()[0]
    junk = {"code": "SOLVER = {}\n", "test_solutions": [], "summary": ""}
    client = ScriptedClient([junk, junk])
    agent = PDEAgent(home=tmp_path, brain=ClaudeBrain(client=client))
    outcome = agent.learn(lesson.problem, max_rounds=2)
    assert not outcome.success and outcome.rounds == 2
    assert not (tmp_path / "skills").exists()


def test_requests_through_the_real_sdk():
    """The anthropic SDK accepts our request and we echo its content blocks back unchanged."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    replies = [{**HEAT_REPLY, "initial": "foo(x)"}, HEAT_REPLY]
    seen = []

    def sse(text):
        events = [
            {"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": DEFAULT_MODEL, "content": [],
                "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    def handler(request):
        seen.append((request.headers, json.loads(request.content)))
        reply = replies[len(seen) - 1]
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse(json.dumps(reply)))

    client = anthropic.Anthropic(api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    parsed = ClaudeBrain(client=client).parse_problem("……", builtin_equations())
    assert parsed.problem.initial == "sin(pi*x)"
    headers, body = seen[0]
    assert headers["anthropic-beta"] == FALLBACK_BETA
    assert body["fallbacks"] == "default" and body["stream"] is True
    assert body["output_config"]["format"]["schema"]["required"]
    retry = seen[1][1]["messages"]
    assert retry[1]["role"] == "assistant" and retry[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig"}


def test_api_failures_become_a_failed_lesson_not_a_crash(tmp_path):
    class BrokenClient:
        def __init__(self):
            self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

        def _stream(self, **kwargs):
            raise ConnectionError("network unreachable")

    lesson = frontier_lessons()[0]
    agent = PDEAgent(home=tmp_path, brain=ClaudeBrain(client=BrokenClient()))
    report = agent.solve(lesson.problem, lesson.tol)
    assert not report.success
    assert "network unreachable" in report.learned[0].message


def test_skill_prompt_does_not_offer_the_exact_solution():
    assert '"initial_velocity", "exact"' not in SKILL_SYSTEM
    assert "The exact solution is never passed to solve()" in SKILL_SYSTEM


@pytest.mark.parametrize("text,why", [
    ("du/dt = D*d2u/dx2 - r*u", "unknown names in 'du/dt = D*d2u/dx2 - r*u': d2u, dt, du, dx2"),
    ("u_t = D*u_xx - k*u + f", "unknown names in 'u_t = D*u_xx - k*u + f': k"),
])
def test_new_equation_text_is_checked_before_asking_the_model(tmp_path, text, why):
    client = ScriptedClient([{"code": "SOLVER = {}\n", "test_solutions": [], "summary": ""}])
    agent = PDEAgent(home=tmp_path, brain=ClaudeBrain(client=client))
    problem = PDEProblem.from_dict({**frontier_lessons()[0].problem.to_dict(), "equation": text})
    outcome = agent.learn(problem, max_rounds=1)
    assert not outcome.success and client.calls == []
    assert outcome.message == f"新方程的写法没法核对：{why}。请写成 u_t = D*u_xx - r*u + f 这样的形式"


def test_prompts_explain_the_equation_check():
    assert "u is the unknown and f the source term, both without arguments" in PARSE_SYSTEM
    new, known = (SkillTask.from_problem(lesson.problem, builtin_equations(), builtin_registry())
                  for lesson in frontier_lessons()[:2])  # reaction-diffusion (new), 2-D heat (built-in)
    line = "- EQUATION form and operator must match this equation; the exam checks it."
    assert line in task_prompt(new) and line not in task_prompt(known)


def test_skill_prompt_and_session_take_at_most_two_extra_tests():
    assert "up to two extra exact solutions" in SKILL_SYSTEM
    reply = {"code": GOOD, "test_solutions": ["sin(pi*x)*cos(t)", "cos(pi*x)*exp(-t)", "x*t"], "summary": ""}
    task = SkillTask.from_problem(frontier_lessons()[0].problem, builtin_equations(), builtin_registry())
    candidate = ClaudeBrain(client=ScriptedClient([reply])).skill_session(task).propose()
    assert candidate.test_solutions == ["sin(pi*x)*cos(t)", "cos(pi*x)*exp(-t)"]
