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


class AppState:
    """The agent behind the page; solves run one at a time."""

    def __init__(self, agent: PDEAgent, provider: str, model: str | None = None,
                 make_brain: Callable[[str, str | None], Any] | None = None):
        self.agent = agent
        self.provider = provider
        self.model = model
        self._make_brain = make_brain
        self.lock = threading.Lock()

    def brain(self):
        if self.agent.brain is None:
            make = self._make_brain
            if make is None:
                from .llm import make_brain as make
            self.agent.brain = make(self.provider, self.model)
        return self.agent.brain

    def info(self) -> dict:
        name, key = PROVIDERS[self.provider]
        home = Path(self.agent.home)
        try:
            home = Path("~") / home.relative_to(Path.home())
        except ValueError:
            pass
        return {
            "version": __version__,
            "home": str(home),
            "llm": {"provider": self.provider, "name": name, "key": key, "ready": bool(os.environ.get(key))},
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
        if self.agent.brain is None and self._make_brain is None and not os.environ.get(key):
            raise RequestError(f"还没有设置 {key}：设置好以后重新运行 pde-agent app（用 {name}）", HTTPStatus.BAD_GATEWAY)
        try:
            return self.brain()
        except LLMError as exc:
            raise RequestError(self._llm_message(exc), HTTPStatus.BAD_GATEWAY) from exc
        except ModuleNotFoundError as exc:
            extra = "deepseek" if exc.name == "openai" else "llm"
            raise RequestError(f'需要先安装 SDK：pip install -e ".[{extra}]"', HTTPStatus.BAD_GATEWAY) from exc

    def _llm_message(self, exc: Exception) -> str:
        name, key = PROVIDERS[self.provider]
        return f"调用 {name} 失败：{exc}（检查 {key} 和网络）"

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

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(HTTPStatus.OK, state.info())
            elif self.path == "/api/stats":
                self._json(HTTPStatus.OK, state.stats())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            routes = {"/api/solve": state.solve, "/api/ask": state.ask}
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
