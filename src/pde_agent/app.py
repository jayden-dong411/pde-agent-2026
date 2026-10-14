"""Local web app: ``pde-agent app`` serves a page for solving problems in a browser.

Standard library only (``http.server``). The server listens on 127.0.0.1, so
only this computer can reach it: learned skills run as code on this machine,
and the page must not be a way for others to run things here.
"""

from __future__ import annotations

import json
import math
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import __version__
from .agent import OUTCOMES, PDEAgent, SolveReport
from .curriculum import basic_curriculum, frontier_lessons
from .llm import PROVIDERS, LLMError
from .problem import PDEProblem
from .settings import DEFAULT_MODELS, MODELS, Settings, key_hint
from .verify import exact_values

PAGE = Path(__file__).parent / "templates" / "app.html"
MAX_POINTS = 129  # per axis in the plotted solution


class RequestError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


def _num(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _thin(axis: np.ndarray) -> np.ndarray:
    """Indices of at most MAX_POINTS evenly spread nodes, ends included."""
    return np.unique(np.linspace(0, len(axis) - 1, min(len(axis), MAX_POINTS)).round().astype(int))


def _values(u: np.ndarray, picks: list[np.ndarray]) -> list:
    sub = u[np.ix_(*picks)] if len(picks) > 1 else u[picks[0]]
    return np.where(np.isfinite(sub), sub, 0.0).tolist()


def report_json(report: SolveReport) -> dict:
    """What the page shows about one solve."""
    data: dict[str, Any] = {
        "summary": report.summary(),
        "success": report.success,
        "method": report.method,
        "n": report.n,
        "error": _num(report.error),
        "error_kind": report.error_kind,
        "tol": report.tol,
        "solver_runs": report.solver_runs,
        "work": _num(report.work),
        "runtime": round(report.runtime, 3),
        "message": report.message,
        "trials": [{**t, "label": OUTCOMES.get(t["outcome"], t["outcome"]), "error": _num(t.get("error")),
                    "work": _num(t.get("work"))} for t in report.trials],
        "learned": [{"success": o.success, "solver": o.solver, "rounds": o.rounds, "message": o.message} for o in report.learned],
        "problem": report.problem.to_dict(),
        "solution": None,
    }
    result = report.result
    if result is not None:
        picks = [_thin(axis) for axis in result.grid]
        solution = {
            "dim": len(result.grid),
            "axes": [axis[p].tolist() for axis, p in zip(result.grid, picks)],
            "u": _values(np.asarray(result.u), picks),
            "exact": None,
            "t": result.t,
        }
        if report.problem.has_exact():
            solution["exact"] = _values(exact_values(report.problem, result), picks)
        data["solution"] = solution
    return data


def problem_from_form(payload: dict) -> PDEProblem:
    """Build a problem from the page's form (raises RequestError when invalid)."""
    data = dict(payload.get("problem") or {})
    for key in ("boundary_value", "initial", "initial_velocity", "exact"):
        if not str(data.get(key) or "").strip():
            data[key] = None
    if not str(data.get("source") or "").strip():
        data["source"] = "0"
    if not data.get("t_final"):
        data["t_final"] = None
    try:
        return PDEProblem.from_dict(data)
    except (TypeError, ValueError, KeyError) as exc:
        raise RequestError(f"题目有误：{exc}") from exc


CHAT_SYSTEM = """你是这个偏微分方程求解程序网页里的数学助手。用户在这里问偏微分方程数值求解以外的数学问题：微积分、线性代数、常微分方程、概率统计、数论、证明、解题思路、概念解释等等。

- 用中文回答，先给结论，再给关键步骤；计算题要写出过程并检查结果。
- 页面不能显示 LaTeX。公式用纯文本和 Unicode 写，比如 x² + y² = r²、∫₀¹ f(x) dx、∂u/∂t、√2、≤、→、∑ₖ aₖ；需要对齐的推导可以放进 ``` 代码块。
- 可以用 **加粗**、列表和小标题，不要用表格和图片。
- 如果用户要数值求解一个偏微分方程，告诉他用页面上方的“求解”面板。
- 和数学无关的问题，礼貌地说明这里只回答数学问题。"""
CHAT_TURNS = 20  # messages of history sent to the model
CHAT_CHARS = 8000  # per message
CHAT_TOTAL = 40000

PREVIEW_POINTS = (129, 65)  # per axis for 1-D and 2-D previews


def preview_json(problem: PDEProblem) -> dict:
    """The known fields of a problem on a grid, drawn before solving."""
    dim = len(problem.domain)
    data: dict[str, Any] = {"dim": dim, "axes": [], "curves": []}
    if dim > 2:
        return data
    axes = [np.linspace(a, b, PREVIEW_POINTS[dim - 1]) for a, b in problem.domain]
    points = tuple(np.meshgrid(*axes, indexing="ij"))
    t_final = problem.t_final or 0.0
    wanted = [("exact", t_final, f"精确解 t = {t_final:g}" if problem.t_final else "精确解")] if problem.has_exact() else []
    if problem.t_final:
        wanted.append(("initial", 0.0, "初值 t = 0"))
    if not wanted:
        wanted.append(("source", 0.0, "源项 f"))
    for name, t, label in wanted:
        try:
            values = np.asarray(problem.fn(name)(*(points if dim == 2 else (points[0], 0.0)), t), dtype=float)
        except (ValueError, TypeError, ArithmeticError):
            continue
        if not np.all(np.isfinite(values)):
            continue
        data["curves"].append({"field": name, "label": label, "t": t, "values": values.tolist()})
    data["axes"] = [axis.tolist() for axis in axes]
    return data


class AppState:
    """The agent behind the page; solves run one at a time."""

    def __init__(self, agent: PDEAgent, provider: str, model: str | None = None,
                 make_brain: Callable[..., Any] | None = None):
        self.agent = agent
        self.settings = Settings(agent.home)
        self._start = (provider, model)  # from the command line, until settings are saved on the page
        self._make_brain = make_brain
        self.lock = threading.Lock()

    @property
    def provider(self) -> str:
        return self.settings.provider or self._start[0]

    @property
    def model(self) -> str:
        provider, model = self.provider, self.settings.model(self.provider)
        if model is None and self.settings.provider is None and provider == self._start[0]:
            model = self._start[1]  # nothing saved on the page yet: keep --model
        return model or DEFAULT_MODELS[provider]

    def api_key(self, provider: str | None = None) -> str:
        """The saved key, else the provider's environment variable."""
        provider = provider or self.provider
        return self.settings.key(provider) or os.environ.get(PROVIDERS[provider][1], "")

    def brain(self):
        if self.agent.brain is None:
            make = self._make_brain
            if make is None:
                from .llm import make_brain as make
                self.agent.brain = make(self.provider, self.model, api_key=self.api_key() or None,
                                        log_path=Path(self.agent.home) / "deepseek_requests.jsonl")
            else:
                self.agent.brain = make(self.provider, self.model, api_key=self.api_key() or None)
        return self.agent.brain

    def get_settings(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "file": self._short(self.settings.path),
            "providers": [
                {"id": pid, "name": name, "env": env, "models": MODELS[pid], "default_model": DEFAULT_MODELS[pid],
                 "model": self.settings.model(pid) or DEFAULT_MODELS[pid],
                 "saved_key": key_hint(self.settings.key(pid)) if self.settings.key(pid) else None,
                 "env_key": bool(os.environ.get(env))}
                for pid, (name, env) in PROVIDERS.items()
            ],
        }

    def save_settings(self, payload: dict) -> dict:
        provider = payload.get("provider")
        fields = {name: payload.get(name, "") for name in ("model", "api_key")}
        if not all(isinstance(v, str) for v in fields.values()):
            raise RequestError("设置的格式不对")
        with self.lock:
            try:
                self.settings.update(str(provider), fields["model"], fields["api_key"],
                                     clear_key=bool(payload.get("clear_key")))
            except ValueError as exc:
                raise RequestError(str(exc)) from exc
            except OSError as exc:
                raise RequestError(f"保存设置失败：{exc}", HTTPStatus.INTERNAL_SERVER_ERROR) from exc
            self.agent.brain = None  # rebuilt with the new choice on next use
        return self.get_settings()

    @staticmethod
    def _short(path: Path) -> str:
        try:
            return str(Path("~") / Path(path).relative_to(Path.home()))
        except ValueError:
            return str(path)

    def info(self) -> dict:
        name, key = PROVIDERS[self.provider]
        return {
            "version": __version__,
            "home": self._short(Path(self.agent.home)),
            "llm": {"provider": self.provider, "name": name, "key": key, "model": self.model,
                    "ready": bool(self.api_key())},
            "equations": [
                {"kind": eq.kind, "form": eq.form, "time_order": eq.time_order, "defaults": eq.defaults,
                 "description": eq.description, "origin": eq.origin}
                for eq in self.agent.equations
            ],
            "methods": [
                {"name": s.name, "kind": s.kind, "order": s.order, "dims": list(s.dims),
                 "boundaries": list(s.boundaries), "origin": s.origin, "description": s.description}
                for s in self.agent.registry
            ],
            "examples": [
                {"name": lesson.name, "title": lesson.title, "level": lesson.level, "tol": lesson.tol,
                 "problem": lesson.problem.to_dict()}
                for lesson in basic_curriculum() + frontier_lessons()
            ],
        }

    def solve(self, payload: dict) -> dict:
        problem = problem_from_form(payload)
        tol = self._tol(payload)
        learn = bool(payload.get("learn"))
        with self.lock:
            if learn:
                self._brain_or_error()
            report = self.agent.solve(problem, tol, learn=learn)
        return report_json(report)

    def preview(self, payload: dict) -> dict:
        return preview_json(problem_from_form(payload))

    def chat(self, payload: dict) -> dict:
        """Answer a math question; the page keeps the conversation and sends it each time."""
        history = payload.get("messages")
        if not isinstance(history, list) or not history:
            raise RequestError("请先输入问题")
        messages = []
        for item in history[-CHAT_TURNS:]:
            if not isinstance(item, dict) or item.get("role") not in ("user", "assistant") \
                    or not isinstance(item.get("content"), str) or not item["content"].strip():
                raise RequestError("对话的格式不对")
            if len(item["content"]) > CHAT_CHARS:
                raise RequestError(f"一条消息最多 {CHAT_CHARS} 个字")
            messages.append({"role": item["role"], "content": item["content"].strip()})
        while messages and messages[0]["role"] != "user":
            messages.pop(0)  # the model's turn can't come first
        if not messages or messages[-1]["role"] != "user":
            raise RequestError("最后一条应该是你的问题")
        while sum(len(m["content"]) for m in messages) > CHAT_TOTAL and len(messages) > 1:
            messages = messages[2:] if len(messages) > 2 else messages[1:]
        while messages[0]["role"] != "user":
            messages.pop(0)
        with self.lock:
            brain = self._brain_or_error()
        try:
            text = brain.answer(CHAT_SYSTEM, messages)
        except LLMError as exc:
            raise RequestError(self._llm_message(exc), HTTPStatus.BAD_GATEWAY) from exc
        name, _ = PROVIDERS[self.provider]
        return {"answer": text, "model": f"{name} · {self.model}"}

    def ask(self, payload: dict) -> dict:
        text = str(payload.get("text") or "").strip()
        if not text:
            raise RequestError("请先描述题目")
        tol = self._tol(payload)
        with self.lock:
            self._brain_or_error()
            try:
                report = self.agent.ask(text, tol)
            except LLMError as exc:
                raise RequestError(self._llm_message(exc), HTTPStatus.BAD_GATEWAY) from exc
        return report_json(report)

    def stats(self) -> dict:
        with self.lock:
            return {"rows": [{**r, "mean_work": _num(r["mean_work"])} for r in self.agent.memory.summary()],
                    "episodes": len(self.agent.memory.episodes)}

    def _brain_or_error(self):
        name, key = PROVIDERS[self.provider]
        if self.agent.brain is None and self._make_brain is None and not self.api_key():
            raise RequestError(f"还没有 {name} 的 API key：点右上角的“设置”填一下（或者设置环境变量 {key}）",
                               HTTPStatus.BAD_GATEWAY)
        try:
            return self.brain()
        except LLMError as exc:
            raise RequestError(self._llm_message(exc), HTTPStatus.BAD_GATEWAY) from exc
        except ModuleNotFoundError as exc:
            extra = "deepseek" if exc.name == "openai" else "llm"
            raise RequestError(f'需要先安装 SDK：pip install -e ".[{extra}]"', HTTPStatus.BAD_GATEWAY) from exc

    def _llm_message(self, exc: Exception) -> str:
        name, _ = PROVIDERS[self.provider]
        return f"调用 {name} 失败：{exc}"

    @staticmethod
    def _tol(payload: dict) -> float:
        try:
            tol = float(payload.get("tol", 1e-3))
        except (TypeError, ValueError) as exc:
            raise RequestError("精度要求必须是数字") from exc
        if not (math.isfinite(tol) and 0 < tol < 1):
            raise RequestError("精度要求必须在 0 和 1 之间")
        return tol


def make_handler(state: AppState):
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = f"pde-agent/{__version__}"

        def log_message(self, format, *args):  # keep the terminal quiet
            pass

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, data: dict) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _local_host(self) -> bool:
            """Only requests addressed to this computer: a web page that points its own
            domain at 127.0.0.1 (DNS rebinding) must not reach the key or the models."""
            port = self.server.server_address[1]
            if self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return True
            self._json(HTTPStatus.FORBIDDEN, {"error": "只接受本机地址的请求"})
            return False

        def do_GET(self):
            if not self._local_host():
                return
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(HTTPStatus.OK, state.info())
            elif self.path == "/api/stats":
                self._json(HTTPStatus.OK, state.stats())
            elif self.path == "/api/settings":
                self._json(HTTPStatus.OK, state.get_settings())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            if not self._local_host():
                return
            routes = {"/api/solve": state.solve, "/api/ask": state.ask, "/api/settings": state.save_settings,
                      "/api/preview": state.preview, "/api/chat": state.chat}
            route = routes.get(self.path)
            if route is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            # a page on another site cannot send JSON here without this header
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "需要 JSON 请求"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise RequestError("请求必须是 JSON 对象")
                self._json(HTTPStatus.OK, route(payload))
            except RequestError as exc:
                self._json(exc.status, {"error": str(exc)})
            except json.JSONDecodeError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "请求不是合法的 JSON"})
            except Exception as exc:  # report, keep serving
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def make_server(state: AppState, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))


def serve(state: AppState, port: int = 8765, open_browser: bool = True) -> None:
    server = make_server(state, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"pde-agent {__version__} 网页版已启动：{url}（按 Ctrl+C 退出）")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        server.server_close()
