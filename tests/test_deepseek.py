"""DeepSeek integration, tested against a scripted stand-in for the OpenAI-compatible client."""

import json
from types import SimpleNamespace

import pytest
from test_llm import GOOD, HEAT_REPLY, problem_reply

from pde_agent.agent import PDEAgent
from pde_agent.curriculum import frontier_lessons
from pde_agent.equations import builtin_equations
from pde_agent.llm import (
    DEEPSEEK_MODEL,
    PROBLEM_SCHEMA,
    DeepSeekBrain,
    LLMError,
    default_provider,
    make_brain,
    schema_errors,
)


class ScriptedOpenAI:
    """Mimics ``client.chat.completions.create(..., stream=True)`` with canned replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        reply = self.replies.pop(0)
        finish = "stop"
        if isinstance(reply, tuple):
            finish, reply = reply
        text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        half = len(text) // 2
        chunks = [_chunk(text[:half]), _chunk(text[half:]), _chunk(None, finish), SimpleNamespace(choices=[])]
        return iter(chunks)


def _chunk(content, finish=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)])


def test_parse_problem_uses_json_mode_and_thinking():
    client = ScriptedOpenAI([HEAT_REPLY])
    parsed = DeepSeekBrain(client=client).parse_problem("一根长 1 的杆……", builtin_equations())
    p = parsed.problem
    assert (p.kind, p.t_final, p.coefficients, p.initial, p.exact) == ("heat", 0.5, {"alpha": 0.1}, "sin(pi*x)", None)
    call = client.calls[0]
    assert call["model"] == DEEPSEEK_MODEL and call["stream"] is True
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"thinking": {"type": "enabled"}}
    assert call["reasoning_effort"] == "high"
    system, user = call["messages"]
    assert system["role"] == "system" and user == {"role": "user", "content": "一根长 1 的杆……"}
    assert "- heat: u_t = alpha * (u_xx + u_yy) + f" in system["content"]
    assert "json object" in system["content"] and '"required"' in system["content"]


def test_fenced_and_empty_replies_are_handled():
    fenced = "```json\n" + json.dumps(HEAT_REPLY) + "\n```"
    client = ScriptedOpenAI(["", fenced])
    parsed = DeepSeekBrain(client=client).parse_problem("……", builtin_equations())
    assert parsed.problem.initial == "sin(pi*x)"
    # an empty reply is asked for again with the same conversation
    assert client.calls[0]["messages"] == client.calls[1]["messages"]


def test_off_schema_reply_is_asked_for_again():
    bad = {**HEAT_REPLY, "t_final": None, "domain": "0..1"}
    client = ScriptedOpenAI([bad, HEAT_REPLY])
    parsed = DeepSeekBrain(client=client).parse_problem("……", builtin_equations())
    assert parsed.problem.t_final == 0.5
    retry = client.calls[1]["messages"]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert "reply.t_final must be a number" in retry[-1]["content"]
    assert "reply.domain must be an array" in retry[-1]["content"]


def test_gives_up_after_three_unusable_replies():
    client = ScriptedOpenAI(["no json here", "{", "[1, 2]"])
    with pytest.raises(LLMError, match="no usable JSON reply"):
        DeepSeekBrain(client=client).parse_problem("……", builtin_equations())


def test_truncated_reply_is_an_error():
    client = ScriptedOpenAI([("length", '{"kind": "he')])
    with pytest.raises(LLMError, match="cut off"):
        DeepSeekBrain(client=client).parse_problem("……", builtin_equations())


def test_schema_errors():
    assert schema_errors(HEAT_REPLY, PROBLEM_SCHEMA) == []
    errors = schema_errors({**HEAT_REPLY, "boundary": "neumann", "coefficients": [{"name": "a"}]}, PROBLEM_SCHEMA)
    assert "reply.boundary must be one of ['dirichlet', 'periodic']" in errors
    assert "reply.coefficients[0].value is missing" in errors
    assert schema_errors({**HEAT_REPLY, "t_final": True}, PROBLEM_SCHEMA) == ["reply.t_final must be a number"]


def test_agent_learns_a_new_equation_with_deepseek(tmp_path):
    lesson = frontier_lessons()[0]
    broken = GOOD.replace("a = D * laplacian_1d(m, h) - r * eye", "a = D * laplacian_1d(m, h) + r * eye")
    client = ScriptedOpenAI([
        problem_reply(lesson.problem),
        {"code": broken, "test_solutions": ["sin(pi*x)*cos(t)"], "summary": "第一版"},
        {"code": GOOD, "test_solutions": [], "summary": "Crank-Nicolson 反应扩散求解器"},
    ])
    agent = PDEAgent(home=tmp_path, brain=DeepSeekBrain(client=client))
    report = agent.ask("在 [0,1] 上求解反应扩散方程 u_t = 0.5 u_xx - 2u + f ……", tol=lesson.tol)

    assert report.success, report.summary()
    assert report.method == "reaction_diffusion.crank_nicolson"
    assert report.learned[0].success and report.learned[0].rounds == 2
    repair = client.calls[2]["messages"]
    assert [m["role"] for m in repair] == ["system", "user", "assistant", "user"]
    assert isinstance(repair[2]["content"], str) and "第一版" in repair[2]["content"]
    assert "did not pass validation" in repair[-1]["content"]
    manifest = json.loads((tmp_path / "skills" / "reaction_diffusion__crank_nicolson.json").read_text(encoding="utf-8"))
    assert manifest["origin"] == "deepseek"


def test_missing_key_is_an_error_and_openai_key_is_never_used(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-deepseek")
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        DeepSeekBrain()


def test_provider_choice(monkeypatch):
    assert default_provider({}) == "claude"
    assert default_provider({"DEEPSEEK_API_KEY": "k"}) == "deepseek"
    assert default_provider({"DEEPSEEK_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}) == "claude"
    with pytest.raises(ValueError):
        make_brain("gpt")
    pytest.importorskip("openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    brain = make_brain("deepseek", "deepseek-flash")
    assert isinstance(brain, DeepSeekBrain) and brain.model == "deepseek-flash"
    assert str(brain.client.base_url).rstrip("/") == "https://api.deepseek.com"


def test_requests_through_the_real_sdk():
    """The openai SDK sends our request to DeepSeek's endpoint and parses its stream."""
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    replies = [{**HEAT_REPLY, "initial": 0}, HEAT_REPLY]
    seen = []

    def sse(text):
        def chunk(delta, finish=None):
            return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": DEEPSEEK_MODEL,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        events = [
            chunk({"role": "assistant", "content": None, "reasoning_content": "先看边界条件……"}),
            chunk({"content": text[:20]}),
            chunk({"content": text[20:]}),
            {**chunk({}, "stop"), "usage": {"prompt_tokens": 9, "completion_tokens": 9, "total_tokens": 18}},
        ]
        return ": keep-alive\n\n" + "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events) + "data: [DONE]\n\n"

    def handler(request):
        seen.append((request, json.loads(request.content)))
        reply = replies[len(seen) - 1]
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse(json.dumps(reply)))

    client = openai.OpenAI(api_key="test-key", base_url="https://api.deepseek.com",
                           http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    parsed = DeepSeekBrain(client=client).parse_problem("……", builtin_equations())
    assert parsed.problem.initial == "sin(pi*x)"
    request, body = seen[0]
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == DEEPSEEK_MODEL and body["stream"] is True
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    retry = seen[1][1]["messages"]
    assert retry[2] == {"role": "assistant", "content": json.dumps(replies[0])}
    assert "reply.initial must be a string" in retry[3]["content"]
