"""v1.5.0: the math chat hands concrete problems to the solver."""

import pytest
from test_app import call, serve  # noqa: F401  (fixture)
from test_deepseek import ScriptedOpenAI
from test_llm import HEAT_REPLY

from pde_agent.app import CHAT_SYSTEM, SOLVE_MARK, solve_request
from pde_agent.llm import DeepSeekBrain


@pytest.mark.parametrize("text, expected", [
    ("@@SOLVE@@ tol=1e-5\n解热方程 u_t = u_xx", ("解热方程 u_t = u_xx", 1e-5)),
    ("  @@SOLVE@@ 求 y' = -y，y(0) = 1，到 t = 1", ("求 y' = -y，y(0) = 1，到 t = 1", None)),
    ("@@SOLVE@@ tol：2e-4\n题目", ("题目", 2e-4)),
    ("@@SOLVE@@ tol=5\n题目", ("题目", None)),  # out of range: the page's tolerance is used
    ("@@SOLVE@@", None),
    ("答案是 2。@@SOLVE@@ 不在开头", None),
])
def test_solve_request(text, expected):
    assert solve_request(text) == expected


def test_system_prompt_explains_the_mark():
    assert SOLVE_MARK in CHAT_SYSTEM and "tol=" in CHAT_SYSTEM


def test_chat_solves_a_problem(serve):  # noqa: F811
    client = ScriptedOpenAI(["@@SOLVE@@ tol=1e-3\n一根长 1 的杆，u_t = 0.1 u_xx，u(x,0) = sin(πx)，两端为 0，算到 t = 0.5",
                             HEAT_REPLY])
    base = serve(make_brain=lambda provider, model, **kw: DeepSeekBrain(client=client))
    status, data = call(base + "/api/chat", {"messages": [{"role": "user", "content": "帮我数值求解这根杆的热传导"}],
                                             "tol": 1e-2})
    assert status == 200, data
    report = data["solve"]
    assert report["success"] and report["tol"] == 1e-3 and report["problem"]["kind"] == "heat"
    assert report["solution"]["t"] == pytest.approx(0.5)
    assert "交给求解器的题目" in data["answer"] and "已验证" in data["answer"]
    parse_messages = client.calls[1]["messages"]
    assert "一根长 1 的杆" in parse_messages[-1]["content"]  # the self-contained text is parsed, not the chat


def test_chat_uses_page_tolerance_and_reports_parse_failure(serve):  # noqa: F811
    client = ScriptedOpenAI(["@@SOLVE@@\n乱七八糟的题", {**HEAT_REPLY, "initial": "foo(x)"}, {**HEAT_REPLY, "initial": "foo(x)"}])
    base = serve(make_brain=lambda provider, model, **kw: DeepSeekBrain(client=client))
    status, data = call(base + "/api/chat", {"messages": [{"role": "user", "content": "解一下"}]})
    assert status == 200 and data["solve"] is None and "没能把它变成求解器能用的题目" in data["answer"]


def test_plain_answers_have_no_solve(serve):  # noqa: F811
    client = ScriptedOpenAI(["√2 是无理数。"])
    base = serve(make_brain=lambda provider, model, **kw: DeepSeekBrain(client=client))
    status, data = call(base + "/api/chat", {"messages": [{"role": "user", "content": "√2 是无理数吗"}]})
    assert status == 200 and "solve" not in data and data["answer"] == "√2 是无理数。"
