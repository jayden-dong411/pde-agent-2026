"""Solve jobs for the web app: one at a time, in a child process, with a hard timeout, cancel and progress.

The page starts a job (``POST /api/jobs``), polls it (``GET /api/jobs/<id>``)
and can cancel it (``POST /api/jobs/<id>/cancel``). A job runs the agent in a
fresh process (``multiprocessing`` "spawn", the same on Linux, macOS and
Windows), which sends each step of the episode back as it happens. The
server never waits on the solve itself: other requests (settings, preview,
math questions, the job's own progress) stay answered.

* **timeout**: the job's process group is killed when the job runs longer
  than its limit; the page keeps the steps it saw.
* **cancel**: the same, on request.
* After a job the server reloads the experience file (and the skills, if the
  job learned one), so the next choice of method knows what happened.

``inline`` jobs run in a thread of this process instead, for brains that only
exist here (the tests' scripted models). They cannot be killed: cancel and
timeout stop them at the agent's next step.
"""

from __future__ import annotations

import os
import signal
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .agent import Cancelled

MAX_EVENTS = 300  # steps kept per job
STATES = ("running", "done", "error", "cancelled", "timeout")


@dataclass
class Job:
    id: str
    action: str
    timeout: float
    started: float = field(default_factory=time.time)
    finished: float | None = None
    state: str = "running"
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False
    meta: dict = field(default_factory=dict)  # what the caller needs afterwards (e.g. the chat's problem text)
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def add(self, event: dict) -> None:
        event = {**event, "at": round(self.elapsed, 2)}
        self.events.append(event)
        if len(self.events) > MAX_EVENTS:
            del self.events[1:len(self.events) - MAX_EVENTS + 1]  # keep the first step and the latest ones

    def finish(self, state: str, result: dict | None = None, error: str | None = None) -> None:
        """Record the end; ``done_event`` is set once the manager has also reloaded what the job changed."""
        if self.state != "running":
            return
        self.state, self.result, self.error, self.finished = state, result, error, time.time()

    def to_dict(self) -> dict:
        # finished only once the manager has reloaded what the job changed (see JobManager._done)
        state = self.state if self.done_event.is_set() else "running"
        return {"id": self.id, "action": self.action, "state": state, "elapsed": round(self.elapsed, 2),
                "timeout": self.timeout, "events": list(self.events), "result": self.result, "error": self.error}


class Busy(RuntimeError):
    """Another job is still running."""


class JobManager:
    """Runs one job at a time; finished jobs are kept for a while so the page can read them."""

    KEEP = 20

    def __init__(self, on_finish: Callable[[Job], None] | None = None):
        self.on_finish = on_finish
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._procs: dict[str, Any] = {}

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def current(self) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs.values() if not j.done_event.is_set()), None)

    def _new(self, action: str, timeout: float, meta: dict | None) -> Job:
        with self._lock:
            running = next((j for j in self._jobs.values() if not j.done_event.is_set()), None)
            if running is not None:
                raise Busy(running.id)
            job = Job(id=uuid.uuid4().hex[:12], action=action, timeout=float(timeout), meta=dict(meta or {}))
            self._jobs[job.id] = job
            for old in [j for j in self._jobs.values() if j.done_event.is_set()][:-self.KEEP]:
                self._jobs.pop(old.id, None)
        job.add({"stage": "start", "text": "开始求解"})
        return job

    def _done(self, job: Job) -> None:
        self._procs.pop(job.id, None)
        try:
            if self.on_finish is not None:
                self.on_finish(job)
        except Exception:  # a reload failing must not lose the job's answer
            job.add({"stage": "warning", "text": "求解后重新读取经验和技能失败：" + traceback.format_exc(limit=1)[-300:]})
        finally:
            job.done_event.set()

    # --- in a child process ---------------------------------------------------------

    def start_process(self, action: str, spec: dict, timeout: float, meta: dict | None = None) -> Job:
        """Run ``_job_main(spec)`` in a spawned child; it reports progress and the answer through a pipe."""
        import multiprocessing

        job = self._new(action, timeout, meta)
        ctx = multiprocessing.get_context("spawn")
        receiver, sender = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_job_main, args=(sender, spec), daemon=True, name=f"pde-agent-job-{job.id}")
        try:
            proc.start()
        except Exception as exc:
            job.finish("error", error=f"没能启动求解进程：{exc}")
            self._done(job)
            return job
        sender.close()  # the child holds the only writer: its death closes the pipe
        self._procs[job.id] = proc
        threading.Thread(target=self._watch, args=(job, proc, receiver), daemon=True).start()
        return job

    def _watch(self, job: Job, proc, receiver) -> None:
        def died() -> None:
            if job.cancel_requested:  # cancel() killed it while we were reading
                job.finish("cancelled", error="已取消")
            else:
                proc.join(1)
                job.finish("error", error=f"求解进程意外退出（exit {proc.exitcode}）")

        try:
            while True:
                if job.cancel_requested:
                    _kill(proc)
                    job.finish("cancelled", error="已取消")
                    break
                if job.elapsed > job.timeout:
                    _kill(proc)
                    job.finish("timeout", error=f"求解超过 {job.timeout:g} 秒，已停止（时间上限可以在请求里用 timeout 调整）")
                    break
                try:
                    ready = receiver.poll(0.2)
                except (EOFError, OSError):
                    ready = True
                if not ready:
                    if not proc.is_alive() and not receiver.poll(0):
                        died()
                        break
                    continue
                try:
                    kind, data = receiver.recv()
                except (EOFError, OSError):
                    died()
                    break
                if kind == "progress":
                    job.add(data)
                elif kind == "done":
                    job.finish("done", result=data)
                    break
                elif kind == "error":
                    job.finish("error", error=str(data))
                    break
        finally:
            receiver.close()
            proc.join(2)
            _kill(proc)
            self._done(job)

    # --- in this process ------------------------------------------------------------

    def start_inline(self, action: str, run: Callable[[Callable[[dict], None]], dict], timeout: float,
                     meta: dict | None = None) -> Job:
        """Run ``run(progress)`` in a thread; ``progress`` raises ``Cancelled`` once the job is cancelled or late."""
        job = self._new(action, timeout, meta)

        def progress(event: dict) -> None:
            job.add(event)
            if job.cancel_requested or job.elapsed > job.timeout:
                raise Cancelled()

        def body() -> None:
            try:
                result = run(progress)
            except Cancelled:
                if job.cancel_requested:
                    job.finish("cancelled", error="已取消")
                else:
                    job.finish("timeout", error=f"求解超过 {job.timeout:g} 秒，已停止")
            except Exception as exc:  # the caller turns known errors into messages; this is the rest
                job.finish("error", error=getattr(exc, "public", None) or f"{type(exc).__name__}: {exc}")
                job.meta["exception"] = exc
            else:
                job.finish("done", result=result)
            finally:
                self._done(job)

        threading.Thread(target=body, daemon=True, name=f"pde-agent-job-{job.id}").start()
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is not None and job.state == "running":
            job.cancel_requested = True
            proc = self._procs.get(job_id)
            if proc is not None:
                _kill(proc)
        return job


def _kill(proc) -> None:
    """End a job's process and its process group (the skill and exam processes it started)."""
    if proc.pid is None:
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the child made itself a group leader
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if proc.is_alive():
        proc.kill()


def _job_main(conn, spec: dict) -> None:
    """In the child: build the agent from ``spec``, solve, and send ``("progress"|"done"|"error", data)``."""
    if os.name == "posix":
        try:
            os.setpgrp()  # a cancel kills this group: the agent and every process it started
        except OSError:
            pass

    def send(kind: str, data) -> None:
        try:
            conn.send((kind, data))
        except (BrokenPipeError, OSError):
            raise Cancelled() from None  # nobody listens any more

    try:
        from .agent import PDEAgent
        from .app import report_json
        from .ode import problem_from_dict

        options = spec.get("agent") or {}
        agent = PDEAgent(home=spec["home"], **options)
        agent.progress = lambda event: send("progress", event)
        brain = spec.get("brain")
        if brain:
            from .llm import make_brain

            agent.brain = make_brain(brain["provider"], brain["model"], api_key=brain.get("api_key") or None,
                                     log_path=brain.get("log_path"))
        problem = problem_from_dict(spec["problem"])
        report = agent.solve(problem, float(spec["tol"]), learn=bool(spec.get("learn")))
        send("done", {"report": report_json(report),
                      "learned": [o.solver for o in report.learned if o.success]})
    except Cancelled:
        pass
    except BaseException as exc:
        try:
            send("error", f"{type(exc).__name__}: {exc}"[:1000])
        except Cancelled:
            pass
    finally:
        conn.close()
