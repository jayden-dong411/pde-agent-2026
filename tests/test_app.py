"""The local web app, exercised over real HTTP on a free port."""

import json
import os
import stat
import threading
import urllib.error
import urllib.request

import pytest
from test_deepseek import ScriptedOpenAI
from test_llm import HEAT_REPLY

from pde_agent.agent import PDEAgent
from pde_agent.app import MAX_POINTS, AppState, make_server
from pde_agent.cli import main
from pde_agent.curriculum import basic_curriculum
from pde_agent.llm import DeepSeekBrain, LLMError
from pde_agent.settings import Settings


@pytest.fixture
def serve(tmp_path):
    servers = []

    def start(**kwargs):
        state = AppState(PDEAgent(home=tmp_path), kwargs.pop("provider", "deepseek"), **kwargs)
        server = make_server(state, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def call(url, body=None, content_type="application/json", host=None):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    headers = {"Content-Type": content_type, **({"Host": host} if host else {})}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            status, kind = response.status, response.headers["Content-Type"]
    except urllib.error.HTTPError as exc:
        raw, status, kind = exc.read(), exc.code, exc.headers["Content-Type"]
    return status, (json.loads(raw) if kind.startswith("application/json") else raw.decode())


def test_page_and_info(serve, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    base = serve()
    status, page = call(base + "/")
    assert status == 200 and "<title>PDE Agent</title>" in page and "/api/solve" in page
    assert "http" not in page.replace("http://www.w3.org/2000/svg", "")  # no outside resources
    status, info = call(base + "/api/info")
    assert status == 200
    assert info["llm"] == {"provider": "deepseek", "name": "DeepSeek", "key": "DEEPSEEK_API_KEY",
                           "model": "deepseek-flash", "ready": False}
    assert {"poisson", "heat", "advection", "wave"} <= {eq["kind"] for eq in info["equations"]}
    assert any(m["origin"] == "builtin" for m in info["methods"])
    assert len(info["examples"]) > len(basic_curriculum())  # frontier lessons too
    assert call(base + "/nowhere")[0] == 404


def test_solve_1d_from_a_lesson(serve):
    base = serve()
    poisson = next(lesson for lesson in basic_curriculum() if lesson.problem.dim == 1 and not lesson.problem.time_dependent)
    status, report = call(base + "/api/solve", {"problem": poisson.problem.to_dict(), "tol": poisson.tol})
    assert status == 200, report
    assert report["success"] and report["error"] < poisson.tol and report["error_kind"] == "exact"
    solution = report["solution"]
    assert solution["dim"] == 1 and solution["exact"] is not None
    assert len(solution["axes"][0]) == len(solution["u"]) <= MAX_POINTS
    assert report["trials"][-1]["label"] == "达到精度"
    status, stats = call(base + "/api/stats")
    assert status == 200 and stats["episodes"] >= 1 and stats["rows"][0]["tries"] >= 1


def test_solve_2d_is_thinned_and_form_blanks_are_filled(serve):
    base = serve()
    problem = {"kind": "poisson", "domain": [[0, 1], [0, 2]], "t_final": "", "coefficients": {},
               "source": "", "boundary": "dirichlet", "exact": "sin(pi*x)*sin(pi*y/2)", "initial": "",
               "boundary_value": " ", "initial_velocity": ""}
    status, report = call(base + "/api/solve", {"problem": problem, "tol": 1e-2})
    assert status == 200, report
    solution = report["solution"]
    assert solution["dim"] == 2 and solution["t"] is None
    nx, ny = (len(axis) for axis in solution["axes"])
    assert nx <= MAX_POINTS and ny <= MAX_POINTS
    assert len(solution["u"]) == nx and len(solution["u"][0]) == ny
    assert len(solution["exact"]) == nx


@pytest.mark.parametrize("body, status, message", [
    ({"problem": {"kind": "heat", "domain": [[1, 0]]}}, 400, "题目有误"),
    ({"problem": {"kind": "poisson", "domain": [[0, 1]], "exact": "__import__('os')"}}, 400, "题目有误"),
    ({"problem": {"kind": "poisson", "domain": [[0, 1]]}, "tol": 5}, 400, "0 和 1 之间"),
    ({"problem": {"kind": "poisson", "domain": [[0, 1]]}, "tol": "abc"}, 400, "必须是数字"),
    ([1, 2], 400, "JSON 对象"),
])
def test_bad_requests(serve, body, status, message):
    got, data = call(serve() + "/api/solve", body)
    assert got == status and message in data["error"]


def test_post_needs_json(serve):
    base = serve()
    status, data = call(base + "/api/solve", b"kind=heat", content_type="application/x-www-form-urlencoded")
    assert status == 415
    status, data = call(base + "/api/solve", b"{not json")
    assert status == 400 and "JSON" in data["error"]


def test_ask_without_key_is_a_clear_error(serve, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    base = serve()
    status, data = call(base + "/api/ask", {"text": "求解热方程", "tol": 1e-3})
    assert status == 502 and "DEEPSEEK_API_KEY" in data["error"]
    status, data = call(base + "/api/ask", {"text": "  "})
    assert status == 400


def test_missing_claude_key_is_named(serve, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status, data = call(serve(provider="claude") + "/api/ask", {"text": "求解热方程"})
    assert status == 502 and "还没有 Claude 的 API key" in data["error"] and "ANTHROPIC_API_KEY" in data["error"]


def test_ask_with_a_model(serve):
    client = ScriptedOpenAI([HEAT_REPLY])
    base = serve(make_brain=lambda provider, model, **kw: DeepSeekBrain(client=client))
    status, report = call(base + "/api/ask", {"text": "一根长 1 的杆……", "tol": 1e-3})
    assert status == 200, report
    assert report["problem"]["kind"] == "heat" and report["problem"]["initial"] == "sin(pi*x)"
    assert report["solution"]["t"] == pytest.approx(0.5)


def test_model_failure_is_reported(serve):
    def broken(provider, model, **kw):
        raise LLMError("connection refused")

    status, data = call(serve(make_brain=broken) + "/api/ask", {"text": "……"})
    assert status == 502 and "connection refused" in data["error"]
    assert "检查 DEEPSEEK_API_KEY 和网络" not in data["error"]


def test_truncation_is_reported_without_credentials_advice(serve):
    client = ScriptedOpenAI([("length", "{")] * 2)
    base = serve(make_brain=lambda provider, model, **kw: DeepSeekBrain(client=client))
    status, data = call(base + "/api/ask", {"text": "求解热方程"})
    assert status == 502 and "自动重试一次后仍被截断" in data["error"]
    assert "Key" not in data["error"] and "网络" not in data["error"]


def test_cli_app_reports_a_busy_port(tmp_path, capsys):
    blocker = make_server(AppState(PDEAgent(home=tmp_path), "claude"), port=0)
    try:
        port = blocker.server_address[1]
        assert main(["--home", str(tmp_path), "app", "--port", str(port), "--no-browser"]) == 1
    finally:
        blocker.server_close()
    assert "--port" in capsys.readouterr().err


def test_settings_are_saved_and_used_next_time(serve, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-environment")
    calls = []
    base = serve(make_brain=lambda provider, model, **kw: calls.append((provider, model, kw)) or DeepSeekBrain(
        client=ScriptedOpenAI([HEAT_REPLY])))
    status, settings = call(base + "/api/settings")
    assert status == 200 and settings["provider"] == "deepseek" and settings["model"] == "deepseek-flash"
    deepseek = next(p for p in settings["providers"] if p["id"] == "deepseek")
    assert deepseek["env_key"] and deepseek["saved_key"] is None

    status, settings = call(base + "/api/settings", {"provider": "claude", "model": "claude-sonnet-5-5",
                                                     "api_key": " sk-ant-secret-1234 "})
    assert status == 200 and settings["provider"] == "claude" and settings["model"] == "claude-sonnet-5-5"
    claude = next(p for p in settings["providers"] if p["id"] == "claude")
    assert claude["saved_key"] == "sk-…1234" and "secret" not in json.dumps(settings)
    status, info = call(base + "/api/info")
    assert info["llm"] == {"provider": "claude", "name": "Claude", "key": "ANTHROPIC_API_KEY",
                           "model": "claude-sonnet-5-5", "ready": True}
    assert call(base + "/api/ask", {"text": "求解热方程"})[0] == 200
    assert calls == [("claude", "claude-sonnet-5-5", {"api_key": "sk-ant-secret-1234"})]

    path = tmp_path / "settings.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    # a fresh start (the next visit) picks the saved choice up, whatever the command line said
    again = AppState(PDEAgent(home=tmp_path), "deepseek", "deepseek-v4-pro")
    assert (again.provider, again.model, again.api_key()) == ("claude", "claude-sonnet-5-5", "sk-ant-secret-1234")

    # switching back keeps the Claude key; DeepSeek still uses its environment variable
    status, settings = call(base + "/api/settings", {"provider": "deepseek", "model": ""})
    assert status == 200 and settings["model"] == "deepseek-flash"
    assert Settings(tmp_path).key("claude") == "sk-ant-secret-1234"
    assert call(base + "/api/ask", {"text": "求解热方程"})[0] == 200  # brain rebuilt for the new choice
    assert calls[-1] == ("deepseek", "deepseek-flash", {"api_key": "sk-from-environment"})

    status, settings = call(base + "/api/settings", {"provider": "claude", "clear_key": True})
    assert status == 200 and Settings(tmp_path).key("claude") == ""
    assert call(base + "/api/info")[1]["llm"]["ready"] is False


@pytest.mark.parametrize("body, message", [
    ({"provider": "gpt"}, "不认识的大模型"),
    ({"provider": "deepseek", "model": "bad model;"}, "模型名"),
    ({"provider": "deepseek", "api_key": "has space inside"}, "API key"),
    ({"provider": "deepseek", "api_key": 123}, "格式不对"),
])
def test_bad_settings(serve, tmp_path, body, message):
    status, data = call(serve() + "/api/settings", body)
    assert status == 400 and message in data["error"]
    assert not (tmp_path / "settings.json").exists()


def test_broken_settings_file_is_ignored(tmp_path):
    (tmp_path / "settings.json").write_text("{not json")
    assert Settings(tmp_path).provider is None
    (tmp_path / "settings.json").write_text(json.dumps({"provider": "gpt", "keys": {"gpt": "x", "claude": 5}}))
    settings = Settings(tmp_path)
    assert settings.provider is None and settings.key("claude") == ""


def test_other_hosts_are_refused(serve):
    base = serve()
    port = base.rsplit(":", 1)[1]
    assert call(base + "/api/settings", host="evil.example:" + port)[0] == 403
    assert call(base + "/api/settings", {"provider": "claude"}, host="evil.example")[0] == 403
    assert call(base + "/api/settings", host="localhost:" + port)[0] == 200


def test_cli_uses_saved_settings(tmp_path, monkeypatch):
    Settings(tmp_path).update("claude", "claude-haiku-5-5", "sk-ant-saved-0000")
    seen = []

    def fake_make_brain(provider, model=None, **kw):
        seen.append((provider, model, kw["api_key"]))
        raise LLMError("stop here")

    monkeypatch.setattr("pde_agent.llm.make_brain", fake_make_brain)
    assert main(["--home", str(tmp_path), "ask", "求解热方程"]) == 2
    assert main(["--home", str(tmp_path), "--llm", "deepseek", "ask", "求解热方程"]) == 2
    assert seen == [("claude", "claude-haiku-5-5", "sk-ant-saved-0000"), ("deepseek", None, None)]
