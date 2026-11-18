"""Run learned skill code in a separate, limited Python process.

Learned code (a solver an LLM wrote) never runs in the agent's own process:
its exam and every later solve happen in a child process that

* gets a minimal environment: no API keys or other settings of this shell,
  only ``PATH``, the locale, the package path and one-thread math libraries,
* runs in a fresh temporary directory (its ``HOME`` and ``TMPDIR`` too),
* limits itself before the skill is imported: CPU seconds, address space
  (Linux) and the size of files it may write,
* is killed after a wall-clock timeout (it stays in the caller's process
  group, so a web job that is cancelled takes its skill processes along), and
* answers in JSON on a private channel: a line marked with a random token
  the skill never sees, numbers as raw float64, so nothing the skill prints
  or returns is ever executed or unpickled here.

This limits accidents (an endless loop, a memory blow-up, a stray print) and
keeps secrets out of reach; it is not a security sandbox against hostile
code. The static check in ``skills.py`` still rejects files, processes and
reflection before anything runs.
"""

from __future__ import annotations

import base64
import json
import math
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .solvers.base import SolverDiverged, SolveResult

PREFIX = "@@pde-agent-sandbox@@"
CPU_SECONDS = 600.0  # per child process
MEMORY_MB = 4096  # address space, Linux only (macOS does not enforce it)
FILE_BYTES = 0  # a skill writes no files
OUTPUT_FLOATS = 20_000_000  # largest array a child may send back
KEEP_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "WINDIR", "PYTHONPATH")
ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS")


class SkillRunError(RuntimeError):
    """A learned skill failed in its process (raised, timed out, ran out of memory, or answered nonsense)."""


@dataclass(frozen=True)
class Limits:
    cpu_seconds: float = CPU_SECONDS
    memory_mb: int = MEMORY_MB
    file_bytes: int = FILE_BYTES

    def to_dict(self) -> dict:
        return {"cpu_seconds": self.cpu_seconds, "memory_mb": self.memory_mb, "file_bytes": self.file_bytes}


def child_env(scratch: str) -> dict[str, str]:
    """The whole environment a skill process gets: nothing secret, nothing it does not need."""
    env = {k: os.environ[k] for k in KEEP_ENV if os.environ.get(k)}
    package_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH", "")) if p)
    env.update({name: "1" for name in ONE_THREAD})
    env.update(HOME=scratch, TMPDIR=scratch, TEMP=scratch, TMP=scratch, PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    return env


def apply_limits(limits: dict) -> None:
    """Called first thing in the child: the limits hold before any learned code is imported."""
    try:
        import resource
    except ImportError:  # Windows: only the wall-clock timeout applies
        return

    def cap(which: int, value: int) -> None:
        try:
            _, hard = resource.getrlimit(which)
            value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (value, hard if hard != resource.RLIM_INFINITY else value))
        except (ValueError, OSError):
            pass  # not allowed or not supported here: the timeout still holds

    cpu = int(math.ceil(float(limits.get("cpu_seconds", CPU_SECONDS))))
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))  # SIGXCPU, then SIGKILL
    except (ValueError, OSError):
        cap(resource.RLIMIT_CPU, cpu)
    if sys.platform.startswith("linux"):
        cap(resource.RLIMIT_AS, int(limits.get("memory_mb", MEMORY_MB)) * 2**20)
    cap(resource.RLIMIT_FSIZE, int(limits.get("file_bytes", FILE_BYTES)))
    try:  # no core files from a crash
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


def call(entry: str, payload: dict, timeout: float, limits: Limits | None = None) -> dict:
    """Run ``entry`` (``module:function``) in a limited child with ``payload``; returns its JSON answer.

    Raises ``SkillRunError`` when the child times out, dies or does not answer.
    """
    module, function = entry.split(":")
    token = secrets.token_hex(16)
    request = json.dumps({**payload, "token": token, "limits": (limits or Limits()).to_dict()})
    code = f"from {module} import {function}; {function}()"
    with tempfile.TemporaryDirectory(prefix="pde-agent-skill-") as scratch:
        proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                env=child_env(scratch), cwd=scratch)
        try:
            out, err = proc.communicate(request, timeout=timeout)
        except subprocess.TimeoutExpired:
            kill(proc)
            proc.communicate()
            raise SkillRunError(f"超过 {timeout:.0f} 秒没有结束，已终止（timed out）") from None
    marker = f"{PREFIX}{token}"
    for line in reversed(out.splitlines()):
        if line.startswith(marker):
            try:
                answer = json.loads(line[len(marker):])
            except json.JSONDecodeError:
                break
            if isinstance(answer, dict):
                return answer
            break
    tail = (err or "").strip()[-1500:]
    if proc.returncode is not None and proc.returncode < 0:
        name = _signal_name(-proc.returncode)
        why = {"SIGXCPU": "用完了 CPU 时间", "SIGKILL": "被终止（CPU 时间或内存超限）", "SIGSEGV": "崩溃"}.get(name, name)
        raise SkillRunError(f"技能进程{why}（{name}）{(': ' + tail) if tail else ''}")
    if "MemoryError" in tail:
        raise SkillRunError(f"技能进程内存超限：{tail[-300:]}")
    raise SkillRunError(f"技能进程异常退出（exit {proc.returncode}）：{tail}")


def kill(proc: subprocess.Popen) -> None:
    """End a child (a skill cannot start processes of its own: the static check forbids it)."""
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


# --- the child's side -------------------------------------------------------------


def read_request() -> tuple[dict, Any]:
    """In the child: read the request, apply the limits, and move ``print`` off the answer channel.

    Returns the request and a writer for the answer (the real stdout, which the skill can no longer reach).
    """
    payload = json.loads(sys.stdin.read())
    apply_limits(payload.get("limits") or {})
    answer = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)  # whatever the skill prints goes to stderr
    sys.stdout = sys.stderr
    return payload, answer


def write_answer(answer_file, payload: dict, data: dict) -> None:
    answer_file.write(f"{PREFIX}{payload['token']}{json.dumps(data)}\n")
    answer_file.flush()


def encode_array(a) -> dict:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return {"shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode("ascii")}


def decode_array(data: Any) -> np.ndarray:
    """A float64 array from a child's answer; anything malformed is an error, never code."""
    if not isinstance(data, dict) or not isinstance(data.get("data"), str) or not isinstance(data.get("shape"), list):
        raise SkillRunError("技能进程返回的数组格式不对")
    shape = [int(s) for s in data["shape"]]
    if any(s < 0 for s in shape) or math.prod(shape) > OUTPUT_FLOATS:
        raise SkillRunError("技能进程返回的数组太大或形状不对")
    raw = base64.b64decode(data["data"].encode("ascii"), validate=True)
    if len(raw) != 8 * math.prod(shape):
        raise SkillRunError("技能进程返回的数组长度和形状不符")
    return np.frombuffer(raw, dtype=np.float64).reshape(shape).copy()


def _plain(value: Any, depth: int = 0) -> Any:
    """JSON-safe copy of a result's ``info`` (numbers, strings, short lists, dicts); the rest is dropped."""
    if depth > 3:
        return None
    if isinstance(value, (bool, str)) or value is None:
        return value if not isinstance(value, str) else value[:2000]
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value) if not isinstance(value, (bool, int, np.integer)) else int(value)
        return value if not isinstance(value, float) or math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k)[:100]: _plain(v, depth + 1) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple)) and len(value) <= 200:
        return [_plain(v, depth + 1) for v in value]
    return None


def result_to_json(result: SolveResult) -> dict:
    if not isinstance(result, SolveResult):
        raise TypeError(f"solve must return a SolveResult, got {type(result).__name__}")
    return {
        "u": encode_array(result.u), "grid": [encode_array(g) for g in result.grid], "n": int(result.n),
        "t": None if result.t is None else float(result.t), "steps": int(result.steps),
        "dt": None if result.dt is None else float(result.dt), "work": float(result.work),
        "info": _plain(dict(result.info or {})),
    }


def result_from_json(data: dict) -> SolveResult:
    def number(key, default=None):
        value = data.get(key, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SkillRunError(f"技能进程返回的 {key} 不是数字")
        return value

    grid = data.get("grid")
    if not isinstance(grid, list) or len(grid) > 3:
        raise SkillRunError("技能进程返回的网格格式不对")
    info = data.get("info")
    work = float(number("work", 0.0))
    return SolveResult(
        u=decode_array(data.get("u")), grid=tuple(decode_array(g) for g in grid), n=int(number("n", 0)),
        t=number("t"), steps=int(number("steps", 0) or 0), dt=number("dt"),
        work=work if math.isfinite(work) and work >= 0 else 0.0, info=info if isinstance(info, dict) else {},
    )


def run_skill(path: Path, problem, n: int, options: dict, timeout: float, limits: Limits | None = None) -> SolveResult:
    """One solve of the learned skill at ``path``, in its own limited process.

    ``problem`` is the full problem; the child hands the skill ``for_solver()`` as in this process.
    """
    answer = call("pde_agent.skills:_run_main", {
        "path": str(Path(path).resolve()), "problem": problem.to_dict(), "n": int(n), "options": _plain(options) or {},
    }, timeout, limits)
    if not answer.get("ok"):
        message = str(answer.get("message", ""))[:600]
        if answer.get("error") == "SolverDiverged":
            raise SolverDiverged(message)
        raise SkillRunError(f"{answer.get('error', 'Error')}: {message}")
    return result_from_json(answer.get("result") or {})
