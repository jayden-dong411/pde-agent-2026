"""The PDE agent: choose a method, solve, check, refine, remember, evolve.

One call to ``solve`` is an episode:

1. read the problem's features (resolution floor, smoothness, stiffness,
   travel distance; see ``features.py``) and predict, per method, the grid
   and the work needed for the tolerance,
2. rank the methods by past experience with problems of the same class and
   feature tags, cheapest prediction first among equals,
3. solve on a grid predicted from experience (or from the features), but never
   below the resolution floor, the first grid that resolves the problem's data
   (a problem no grid resolves is reported, not solved),
4. measure the error (exact solution) or estimate it (grid refinement); an
   estimate that meets the tolerance is cross-checked (``crosscheck.py``:
   residual spot check, mass and energy drift), and an answer that fails the
   cross-check gets one more refinement before the method is given up,
5. refine until the tolerance is met, or move to the next method when a
   method blows up, stagnates or runs out of budget,
6. append the whole episode to the experience memory.

When no method fits, or all of them fail, and a language model is attached
(``ClaudeBrain`` or ``DeepSeekBrain``), the agent asks it for a new solver,
examines it on manufactured solutions, and keeps it in the skill library
only if it passes.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .contract import CONTRACT_CHECKS, result_contract
from .crosscheck import Check, CrossCheck, cross_check
from .equations import builtin_equations, operator_from_text
from .features import Prediction, ProblemFeatures, calibrate, extract, predict
from .llm import LLMError
from .memory import ExperienceStore
from .problem import PDEProblem
from .skills import SkillLibrary, SkillTask, ValidationReport, validate_candidate
from .solvers import builtin_registry
from .solvers.base import SolverDiverged, SolveResult, SolverSpec
from .solvers.pinn import diagnostics_text
from .verify import (OSCILLATION_LIMIT, error_norm, exact_error, new_extrema, relative_error, resolution_floor, restrict,
                     weak_solution)

ROUGH_SAFETY = 2.0  # an estimated error on non-smooth data must be below tol / ROUGH_SAFETY
# shock problems: methods are compared by trial runs on PILOT_N and 2 PILOT_N (``PDEAgent._pilot``), each
# allowed only if both runs cost at most PILOT_SHARE of the cheapest predicted solve
PILOT_N = 32
PILOT_SHARE = 0.15

# for ODEs (dim 0) n is the number of time steps, or sets the tolerance of an adaptive method
BASE_N = {0: 16, 1: 16, 2: 8, 3: 4}
MAX_N = {0: 2**20, 1: 4096, 2: 256, 3: 64}

OUTCOMES = {
    "success": "达到精度",
    "unstable": "数值发散",
    "error": "运行出错",
    "stagnated": "加密后误差不再下降",
    "max_n": "网格加到上限仍未达到精度",
    "budget": "用完计算预算",
    "skipped": "按题目特征判断达不到精度，没有运行",
    "crosscheck": "误差估计达标，但交叉验证没通过",
    "contract": "结果不满足题目条件（边界、初值、时间或坐标不对）",
    "candidate": "得到候选解，但必要的验证没能完成",
    "oscillation": "间断附近出现数值振荡（新的极值）",
}
CROSSCHECK_RETRIES = 1  # refinements after a failed cross-check before the method is given up


def default_home() -> Path:
    return Path(os.environ.get("PDE_AGENT_HOME") or Path.home() / ".pde_agent")


@dataclass
class Attempt:
    method: str
    n: int
    status: str  # ok | unverified | unstable | error
    error: float | None = None
    work: float = 0.0
    runtime: float = 0.0
    message: str = ""


@dataclass
class LearnOutcome:
    success: bool
    solver: str | None = None
    rounds: int = 0
    report: ValidationReport | None = None
    message: str = ""


STATUS = {"verified": "已验证", "candidate": "候选解（未完成验证）", "failed": "失败"}


@dataclass
class SolveReport:
    problem: PDEProblem
    tol: float
    error_kind: str = "exact"  # exact | estimated
    success: bool = False
    method: str | None = None
    n: int | None = None
    error: float | None = None
    result: SolveResult | None = None
    attempts: list[Attempt] = field(default_factory=list)
    trials: list[dict] = field(default_factory=list)
    learned: list[LearnOutcome] = field(default_factory=list)
    message: str = ""
    min_n: int = 0  # resolution floor: the first grid that resolves the problem's data (0 if no grid does)
    features: ProblemFeatures | None = None
    predictions: dict[str, Prediction] = field(default_factory=dict)
    crosscheck: CrossCheck | None = None  # all verification evidence about the answer (contract + cross-checks)
    check_work: float = 0.0  # work of the extra runs the checks needed
    candidate: bool = False  # an answer met the error target, but a required check could not be done
    # wall-clock seconds by stage (see ``costs``): features, verification, learning and the whole episode
    feature_time: float = 0.0
    check_time: float = 0.0
    learn_time: float = 0.0
    total_time: float = 0.0

    @property
    def status(self) -> str:
        """verified | candidate | failed: how far the answer can be trusted."""
        if self.success:
            return "verified"
        return "candidate" if self.candidate else "failed"

    @property
    def status_text(self) -> str:
        return STATUS[self.status]

    @property
    def solver_runs(self) -> int:
        return len(self.attempts)

    @property
    def work(self) -> float:
        return sum(a.work for a in self.attempts) + self.check_work

    @property
    def runtime(self) -> float:
        return sum(a.runtime for a in self.attempts)

    def _failed_methods(self) -> set[str]:
        return {t["method"] for t in self.trials if t.get("outcome") not in ("success", "candidate")}

    def costs(self) -> dict:
        """Where the episode's time and work went, split by stage.

        ``seconds``: ``total`` (the whole episode, wall clock) = ``features`` (features, resolution floor and
        cost predictions) + ``solver`` (every solver run, of which ``failed`` in methods that gave up or blew up)
        + ``verification`` (contract and cross-checks, including their reference runs) + ``learning`` (the
        language model writing and examining new methods) + ``other``. ``work``: the solver runs' own counts
        (``solver``, of which ``failed``) and the checks' (``verification``); ``total`` is ``work``.
        """
        failed = self._failed_methods()
        failed_time = sum(a.runtime for a in self.attempts if a.method in failed or a.status in ("unstable", "error"))
        failed_work = sum(a.work for a in self.attempts if a.method in failed or a.status in ("unstable", "error"))
        solver_time = self.runtime
        seconds = {"total": self.total_time, "features": self.feature_time, "solver": solver_time,
                   "failed": failed_time, "verification": self.check_time, "learning": self.learn_time}
        seconds["other"] = max(0.0, self.total_time - self.feature_time - solver_time - self.check_time - self.learn_time)
        solver_work = sum(a.work for a in self.attempts)
        work = {"total": solver_work + self.check_work, "solver": solver_work, "failed": failed_work,
                "verification": self.check_work}
        return {"seconds": {k: round(v, 4) for k, v in seconds.items()}, "work": work,
                "solver_runs": self.solver_runs, "learning_rounds": sum(o.rounds for o in self.learned)}

    def cost_text(self) -> str:
        s = self.costs()["seconds"]
        parts = [f"特征 {s['features']:.2f}", f"求解 {s['solver']:.2f}" + (f"（其中没成功的方法 {s['failed']:.2f}）" if s["failed"] else ""),
                 f"验证 {s['verification']:.2f}"]
        if s["learning"]:
            parts.append(f"学习新方法 {s['learning']:.2f}")
        return f"用时 {s['total']:.2f} 秒：" + "，".join(parts) + f"，其他 {s['other']:.2f}"

    def choice(self) -> str:
        """Why the methods were tried in this order: the problem's features and the cheapest prediction."""
        if self.features is None:
            return ""
        text = f"题目特征：{self.features.describe()}"
        ranked = sorted((p for p in self.predictions.values() if p.work is not None), key=lambda p: p.work)
        if ranked:
            best = ranked[0]
            how = "按特征和相近题的实测估计" if best.measured else "按特征估计"
            text += f"。{how} {best.method} 最省（n ≈ {best.n}，工作量约 {best.work:.2g}）"
            notes = [f"{p.method}：{'；'.join(p.notes)}" for p in self.predictions.values() if p.notes]
            if notes:
                text += "。" + "；".join(notes)
        return text

    def summary(self) -> str:
        name = self.problem.name or self.problem.signature()
        learned = [o.solver for o in self.learned if o.success]
        extra = f"，新学会 {', '.join(learned)}" if learned else ""
        how = "对比精确解" if self.error_kind == "exact" else "加密网格估计"
        if self.success:
            checked = f"。{self.crosscheck.describe()}" if self.crosscheck and self.crosscheck.describe() else ""
            return (f"{name}：已验证，方法 {self.method}，n={self.n}，误差 {self.error:.2e}（{how}），"
                    f"求解 {self.solver_runs} 次{extra}{checked}")
        if self.candidate:
            missing = "；".join(c.describe() for c in self.crosscheck.missing()) if self.crosscheck else ""
            return (f"{name}：候选解（未完成验证），方法 {self.method}，n={self.n}，误差 {self.error:.2e}（{how}），"
                    f"求解 {self.solver_runs} 次{extra}。没能完成的必要检查：{missing or '无'}")
        checked = f"。最后一次{self.crosscheck.describe()}" if self.crosscheck and self.crosscheck.checks else ""
        return f"{name}：失败，{self.message}{extra}{checked}"


def observed_rate(d_prev: float, d: float, r_prev: float, r: float) -> float | None:
    """Order p from two successive differences of nested solutions with refinement ratios r_prev, r.

    With error C n^-p, ``d_prev / d = r_prev^p (1 - r_prev^-p) / (1 - r^-p)``; solved for p by
    bisection on [0.1, 12] (equal ratios give the textbook ``log(d_prev / d) / log(r)``).
    """
    if not (d_prev > 0 and d > 0 and r_prev > 1 and r > 1):
        return None
    target = math.log(d_prev / d)

    def gap(p: float) -> float:
        return p * math.log(r_prev) + math.log1p(-r_prev**-p) - math.log1p(-r**-p) - target

    lo, hi = 0.1, 12.0
    if gap(lo) > 0:
        return lo
    if gap(hi) < 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


class Cancelled(BaseException):
    """Raised from a progress callback to stop an episode (a BaseException, so no handler swallows it)."""


class PDEAgent:
    """``progress``: called with a dict for each step of an episode (``stage`` and a Chinese ``text``);
    the web app shows these while a solve runs and may raise ``Cancelled`` from it to stop."""

    progress: Callable[[dict], None] | None = None

    def __init__(
        self,
        home: Path | str | None = None,
        brain=None,
        explore: bool = True,
        remember: bool = True,
        time_budget: float = 120.0,
        max_runs_per_method: int = 8,
        validation_timeout: float = 300.0,
    ):
        self.home = Path(home) if home else default_home()
        self.brain = brain
        self.explore = explore
        self.time_budget = time_budget
        self.max_runs_per_method = max_runs_per_method
        self.validation_timeout = validation_timeout
        self.equations = builtin_equations()
        self.registry = builtin_registry()
        # remember=False: no experience is read, kept or written (not even in memory for the next solve)
        self.memory = ExperienceStore(self.home / "experience.jsonl" if remember else None,
                                      self.home / "shared_experience.jsonl" if remember else None,
                                      update=remember)
        self.skills = SkillLibrary(self.home / "skills", timeout=max(30.0, float(time_budget)))
        self.skill_errors = self.skills.load_into(self.registry, self.equations, timeout=self.validation_timeout)

    # --- public API ---------------------------------------------------------------

    def _emit(self, stage: str, text: str, **fields) -> None:
        if self.progress is not None:
            self.progress({"stage": stage, "text": text, **fields})

    def solve(self, problem: PDEProblem, tol: float = 1e-3, options: dict | None = None, learn: bool = True) -> SolveReport:
        if not tol > 0:
            raise ValueError("tol must be positive")
        started = time.perf_counter()
        report = self._solve(problem, tol, options, learn, started)
        report.total_time = time.perf_counter() - started
        self._remember(report)
        self._emit("done", report.summary(), status=report.status)
        return report

    def _solve(self, problem: PDEProblem, tol: float, options: dict | None, learn: bool, started: float) -> SolveReport:
        problem = self.equations.complete(problem)
        report = SolveReport(problem=problem, tol=tol, error_kind="exact" if problem.has_exact() else "estimated")
        eq = self.equations.get(problem.kind)
        time_order = eq.time_order if eq else int(problem.time_dependent)
        # an estimate's coarse companion is the first grid (BASE_N // 2), so it has to see the data too
        start = max(4, BASE_N[problem.dim] // 2) if report.error_kind == "estimated" else BASE_N[problem.dim]
        # an ODE has no grid data to resolve
        # a jump never resolves on any grid; in L1 it only costs O(h), which the error measures anyway
        floor = start if problem.dim == 0 or error_norm(problem) == "l1" else resolution_floor(
            problem, tol, time_order, start, MAX_N[problem.dim])
        if floor is None:
            report.message = f"题目数据在最细网格（n = {MAX_N[problem.dim]}）上也分辨不清，没有求解"
            report.feature_time = time.perf_counter() - started
            return report
        report.min_n = floor
        report.features = extract(problem, floor)
        report.feature_time = time.perf_counter() - started
        self._emit("features", f"题目特征：{report.features.describe()}")
        candidates = self.registry.for_problem(problem)
        can_learn = learn and self.brain is not None

        if not candidates and can_learn:
            outcome = self._timed_learn(report, problem, "no method covers this problem class yet")
            if outcome.success:
                report.problem = problem = self.equations.complete(problem)
                candidates = self.registry.for_problem(problem)

        if not candidates:
            report.message = "还没有能解这类问题的方法"
            if self.brain is None:
                report.message += "（接入大模型 Claude 或 DeepSeek 后，agent 可以自己学一个）"
            elif report.learned:
                report.message += f"，学习新方法失败：{report.learned[-1].message}"
        else:
            self._run_methods(report, candidates, options)
            if not report.success and can_learn and not report.learned:
                outcome = self._timed_learn(report, problem, f"existing methods failed: {report.message}")
                if outcome.success:
                    self._run_methods(report, [self.registry.get(outcome.solver)], options)
        return report

    def _timed_learn(self, report: SolveReport, problem: PDEProblem, reason: str) -> LearnOutcome:
        start = time.perf_counter()
        try:
            outcome = self.learn(problem, reason=reason)
        finally:
            report.learn_time += time.perf_counter() - start
        report.learned.append(outcome)
        return outcome

    def ask(self, text: str, tol: float = 1e-3) -> SolveReport:
        """Solve a problem described in natural language (needs a language model)."""
        if self.brain is None:
            raise RuntimeError("understanding a problem in words needs a language model (ClaudeBrain or DeepSeekBrain)")
        parsed = self.brain.parse_problem(text, self.equations)
        return self.solve(parsed.problem, tol)

    def learn(self, problem: PDEProblem, reason: str = "", max_rounds: int = 3) -> LearnOutcome:
        """Have the language model write a new method for this problem class and admit it if it passes."""
        if self.brain is None:
            raise RuntimeError("learning new methods needs a language model (ClaudeBrain or DeepSeekBrain)")
        task = SkillTask.from_problem(problem, self.equations, self.registry, reason)
        if task.new_equation:  # the exam checks the skill against this text, so it has to be readable first
            try:
                operator_from_text(task.equation, task.coefficients)
            except ValueError as exc:
                return LearnOutcome(False, message=f"新方程的写法没法核对：{exc}。请写成 u_t = D*u_xx - r*u + f 这样的形式")
        session = self.brain.skill_session(task)
        report = None
        for round_ in range(1, max_rounds + 1):
            self._emit("learn", f"请大模型写新方法（第 {round_} 轮）", round=round_)
            try:
                candidate = session.propose()
            except LLMError as exc:
                return LearnOutcome(False, rounds=round_, report=report, message=str(exc))
            self._emit("exam", "考试中：用构造解检查收敛阶", round=round_)
            report = validate_candidate(candidate, task, timeout=self.validation_timeout)
            self._emit("exam", "考试通过" if report.ok else f"考试没通过：{report.message[:200]}", round=round_,
                       ok=report.ok)
            if report.ok:
                path = self.skills.install(candidate, report, task, origin=getattr(self.brain, "provider", "llm"))
                try:
                    spec = self.skills.load_file(path, self.registry, self.equations)
                except Exception as exc:  # passed the exam process but cannot load here
                    path.unlink(missing_ok=True)
                    path.with_suffix(".json").unlink(missing_ok=True)
                    return LearnOutcome(False, rounds=round_, report=report, message=f"loading the new skill failed: {exc}")
                return LearnOutcome(True, solver=spec.name, rounds=round_, report=report, message=candidate.summary)
            session.feedback(report)
        return LearnOutcome(False, rounds=max_rounds, report=report, message=report.message if report else "")

    # --- one episode ------------------------------------------------------------------

    def _run_methods(self, report: SolveReport, candidates: list[SolverSpec], options: dict | None) -> None:
        deadline = time.monotonic() + self.time_budget
        report.message = ""
        predicting = time.perf_counter()
        if report.features is not None:
            for spec in candidates:
                report.predictions[spec.name] = predict(
                    spec, report.problem, report.features, report.tol, MAX_N[report.problem.dim])
            # the cost model is a guess: correct it by what the same methods measured on similar problems
            measured = self.memory.calibration(report.problem.signature(), report.features.to_dict(), report.tol)
            for name, cal in measured.items():
                if name in report.predictions:
                    calibrate(report.predictions[name], cal.ratio, cal.neighbours)
        report.feature_time += time.perf_counter() - predicting
        if report.features is not None and weak_solution(report.problem):
            self._pilot(report, candidates, options)
        tags = report.features.tags if report.features else None
        predicted = {name: p.work for name, p in report.predictions.items()}
        for spec in self.memory.rank(report.problem.signature(), candidates, self.explore, predicted, tags):
            if time.monotonic() > deadline:
                break
            prediction = report.predictions.get(spec.name)
            if prediction is not None and prediction.work == math.inf:
                # stability or ringing rules it out on every grid up to MAX_N: running it only risks a wrong answer
                report.trials.append({"method": spec.name, "outcome": "skipped", "n": None, "error": None,
                                      "error_kind": report.error_kind, "work": 0.0})
                continue
            self._emit("method", f"试 {spec.name}", method=spec.name)
            outcome, n, error, work = self._try_method(spec, report, options, deadline)
            self._emit("outcome", f"{spec.name}：{OUTCOMES.get(outcome, outcome)}", method=spec.name, outcome=outcome)
            report.trials.append({
                "method": spec.name, "outcome": outcome, "n": n, "error": error,
                "error_kind": report.error_kind, "work": work,
                "predicted": prediction.model_work if prediction is not None else None,
            })
            if outcome == "success":
                return
            if outcome == "candidate":
                # a required check cannot be done for this problem (3-D, an unreadable operator, a reference
                # run that does not finish): another method would meet the same wall, so stop here
                missing = "；".join(c.describe() for c in report.crosscheck.missing()) if report.crosscheck else ""
                report.message = f"得到候选解，但必要的验证没能完成（{missing}）"
                return
        tried = "；".join(f"{t['method']} {OUTCOMES[t['outcome']]}" for t in report.trials)
        report.message = f"没有方法达到精度 {report.tol:g}（{tried or '时间预算用完'}）"

    def _pilot(self, report: SolveReport, candidates: list[SolverSpec], options: dict | None) -> None:
        """Measure each shock-capturing method's error on two coarse grids and predict its grid from that.

        Past a shock every scheme converges at first order in L1, so the cost model's order says nothing
        about which scheme is cheaper: the constant in front decides, and it depends on the problem. Rusanov
        smears smooth parts as well as the shock; on a Riemann problem (flat states around the shock) that
        costs it about a factor 2 in error against MUSCL or WENO5, on smooth data that break into a shock up
        to a factor 10, and its work per grid is about 7 times smaller. So each candidate runs at ``n0`` and
        ``2 n0``; the error (exact) or the difference of the two (estimate) sets the grid for the tolerance,
        and the cost model prices that grid. A method is piloted only when its two runs cost at most
        ``PILOT_SHARE`` of the cheapest predicted solve (on an easy problem the cheapest method is right
        anyway), and only when at least two methods qualify (there is nothing to compare otherwise).
        """
        problem, tol = report.problem, report.tol
        max_n = MAX_N[problem.dim]
        n0 = max(PILOT_N, report.min_n)
        priced = [(spec, report.predictions.get(spec.name)) for spec in candidates]
        priced = [(s, p) for s, p in priced if p is not None and p.work is not None and math.isfinite(p.work)
                  and s.cost is not None and "nonoscillatory" in s.traits and "neural" not in s.traits]
        if len(priced) < 2 or 2 * n0 > max_n:
            return
        cheapest = min(p.work for _, p in priced)
        pilots = [(s, p) for s, p in priced if s.cost(problem, n0) + s.cost(problem, 2 * n0) <= PILOT_SHARE * cheapest]
        if len(pilots) < 2:
            return
        estimated = report.error_kind == "estimated"
        target = tol / ROUGH_SAFETY if estimated and not report.features.smooth else tol
        norm = error_norm(problem)
        for spec, prediction in pilots:
            self._emit("pilot", f"{spec.name} 在 n = {n0}、{2 * n0} 上试算，量出误差常数", method=spec.name, n=n0)
            runs = []
            for n in (n0, 2 * n0):
                ran = time.perf_counter()
                try:
                    result = spec.run(problem, n, options)
                except Exception as exc:  # the method's own trial will meet (and report) the same failure
                    report.attempts.append(Attempt(spec.name, n, "pilot", runtime=time.perf_counter() - ran,
                                                   message=f"试算出错：{type(exc).__name__}"))
                    break
                report.attempts.append(Attempt(spec.name, n, "pilot", None, result.work, result.runtime, message="试算"))
                runs.append(result)
            if len(runs) < 2:
                continue
            overshoot = [new_extrema(problem, r) for r in runs]
            if any(o is not None and o > OSCILLATION_LIMIT for o in overshoot):
                continue  # no clean measurement: the feature prediction stands
            if estimated:
                diff = relative_error(runs[0].u, restrict(runs[1], n0), norm)
                error, order = diff / (2.0**prediction.order - 1.0), prediction.order
            else:
                coarse, error = exact_error(problem, runs[0]), exact_error(problem, runs[1])
                seen = math.log2(coarse / error) if coarse and error and coarse > 0 and error > 0 else prediction.order
                order = min(max(0.5, seen), max(prediction.order, 1.0))
            if error is None or not math.isfinite(error):
                continue
            report.attempts[-1].error = error
            factor = max(1.0, error / target) ** (1.0 / order)
            needed = 2 * n0 * 2 ** max(0, math.ceil(math.log2(factor) - 1e-9))
            n = min(needed, max_n)
            # priced on the continuous grid, not the rounded one: the estimate is good to a factor of about 2,
            # and rounding it up to the next power of two would decide close calls by noise. An explicit 1-D
            # scheme's work grows as n^2 (cells times CFL-limited steps)
            work = spec.cost(problem, 2 * n0) * factor**2
            scale = prediction.work / prediction.model_work if prediction.model_work else 1.0
            # an exact error on two grids is a measurement: start on the grid itself. A difference of two coarse
            # answers overstates the error about twofold (not yet asymptotic): start one grid coarser
            start = n // 2 if estimated else n
            prediction.n, prediction.start, prediction.order = n, max(report.min_n, start), order
            prediction.model_work, prediction.work, prediction.piloted = work, work * scale, True
            prediction.notes.append(f"试算 n = {2 * n0} 误差约 {error:.2g}，按 {order:.2g} 阶需要 n ≈ {needed}")

    def _try_method(self, spec: SolverSpec, report: SolveReport, options: dict | None, deadline: float):
        problem, tol = report.problem, report.tol
        estimated = report.error_kind == "estimated"
        max_n = MAX_N[problem.dim]
        tags = report.features.tags if report.features else None
        prediction = report.predictions.get(spec.name)
        suggested = self.memory.suggest_n(problem.signature(), spec, tol, report.error_kind, tags)
        # experience covers the whole problem class; when this problem's own features predict a coarser
        # start, take it: a start too coarse costs one cheap run, one too fine costs the whole difference
        n = min(suggested, prediction.start) if suggested and prediction else (
            suggested or (prediction.start if prediction else BASE_N[problem.dim]))
        if estimated:
            n //= 2  # the estimate needs a coarser companion solution
        n = min(max(4, report.min_n, n), max_n // 2 if estimated else max_n)

        # an estimate on rough data is least trustworthy (the error is not yet asymptotic): it has to beat tol by a margin
        rough = estimated and report.features is not None and not report.features.smooth
        target = tol / ROUGH_SAFETY if rough else tol
        # an adaptive method's error does not shrink smoothly with its tolerance (it can stall for a step, then
        # drop): its estimate is the plain difference, and it has to hold on two refinements in a row
        adaptive = estimated and "adaptive" in spec.traits
        norm = error_norm(problem)
        confirmed = False
        retries = CROSSCHECK_RETRIES
        prev: SolveResult | None = None
        prev_error = prev_diff = prev_ratio = None
        error, work = None, 0.0
        for _ in range(self.max_runs_per_method):
            if time.monotonic() > deadline:
                return "budget", n, error, work
            self._emit("run", f"{spec.name} 在 n = {n} 上求解", method=spec.name, n=n)
            ran = time.perf_counter()
            try:
                result = spec.run(problem, n, options)
            except SolverDiverged as exc:
                report.attempts.append(Attempt(spec.name, n, "unstable", runtime=time.perf_counter() - ran,
                                               message=str(exc)))
                return "unstable", n, None, work
            except Exception as exc:
                report.attempts.append(Attempt(spec.name, n, "error", runtime=time.perf_counter() - ran,
                                               message=f"{type(exc).__name__}: {exc}"[:300]))
                return "error", n, None, work
            work += result.work
            if norm == "l1":
                overshoot = new_extrema(problem, result)
                if overshoot is not None and overshoot > OSCILLATION_LIMIT:
                    # a wrong shape, not an inaccurate one: refining makes the wiggles narrower, not smaller
                    report.attempts.append(Attempt(spec.name, n, "oscillation", None, result.work, result.runtime,
                                                   message=f"超出初值和边界值范围 {overshoot:.1%}"))
                    return "oscillation", n, None, work

            # rough data cost a method its claimed order: the estimate uses the features' order too
            order = min(spec.order, prediction.order) if prediction else spec.order
            if adaptive:
                order = 1.0
            if estimated and "neural" in spec.traits:
                # each n is a different random network: the error does not shrink at a steady rate, so the
                # difference to the coarser answer is taken as the error itself (no Richardson gain)
                order = 1.0
            if not estimated:
                error = exact_error(problem, result)
            elif prev is not None:
                ratio = n / prev.n
                diff = relative_error(prev.u, restrict(result, prev.n), norm)
                if prev_diff and diff > 0:
                    # trust the observed rate when it is lower than claimed (the refinement ratios may differ)
                    seen = None if adaptive else observed_rate(prev_diff, diff, prev_ratio, ratio)
                    if seen is not None:
                        order = min(order, max(0.5, seen))
                if adaptive or "neural" in spec.traits:
                    # no steady rate to extrapolate with: the difference itself is the estimate, whatever
                    # the refinement ratio (dividing by ratio**order - 1 would shrink it 3x for 4x, 7x for 8x)
                    error = diff
                else:
                    error = diff / (ratio**order - 1.0)
                prev_diff, prev_ratio = diff, ratio
            diagnosis = diagnostics_text(result.info)  # a neural method says how well it fits (empty otherwise)
            report.attempts.append(Attempt(
                spec.name, n, "ok" if error is not None else "unverified", error, result.work, result.runtime,
                message=diagnosis,
            ))
            if error is not None:
                how = "误差" if not estimated else "误差估计"
                self._emit("error", f"{spec.name} n = {n}：{how} {error:.2e}（目标 {target:.2g}）"
                           + (f"；{diagnosis}" if diagnosis else ""), method=spec.name, n=n, error=error)

            met = error is not None and error <= target and (confirmed or not adaptive)
            if adaptive:
                confirmed = error is not None and error <= target
            checked = None
            if met:
                # an error that meets the target says the answer converged, not that it answers this problem:
                # check the problem's own conditions, and (without an exact solution) the answer itself
                equation = self.equations.get(problem.kind)
                self._emit("verify", f"{spec.name} n = {n} 达到精度，检查题目条件" + ("和交叉验证" if estimated else ""),
                           method=spec.name, n=n)
                checking = time.perf_counter()
                checked = result_contract(equation, spec, problem, result, tol, options)
                if estimated:
                    checked = checked.merge(cross_check(equation, spec, problem, result, tol, options))
                else:
                    checked.checks.insert(0, Check("exact", error, target, True, "终止时刻与精确解的相对误差"))
                report.check_time += time.perf_counter() - checking
                report.check_work += checked.work
                work += checked.work
                report.crosscheck = checked
                if checked.status == "fail":
                    met = False
                    report.attempts[-1].message = checked.describe()
                elif checked.status == "inconclusive":
                    report.result, report.error, report.n, report.method = result, error, n, spec.name
                    report.candidate = True
                    report.attempts[-1].message = checked.describe()
                    return "candidate", n, error, work
            if error is not None and (report.error is None or error < report.error or met):
                report.result, report.error, report.n, report.method = result, error, n, spec.name
            if met:
                report.success = True
                return "success", n, error, work
            if checked is not None and not checked.passed:
                if retries <= 0 or n >= max_n:
                    broken = any(c.name in CONTRACT_CHECKS for c in checked.failed())
                    return ("contract" if broken else "crosscheck"), n, error, work
                retries -= 1
                prev, prev_error = result, error
                n = min(2 * n, max_n)
                continue
            if error is not None and prev_error is not None and error > 0.8 * prev_error and not adaptive:
                return "stagnated", n, error, work
            if n >= max_n:
                return "max_n", n, error, work
            prev, prev_error = result, error if error is not None else prev_error
            # a network refines by doubling only: its estimate compares neighbouring sizes (see above)
            n = min(2 * n if "neural" in spec.traits else self._next_n(n, error, target, order), max_n)
        return "budget", n, error, work

    @staticmethod
    def _next_n(n: int, error: float | None, tol: float, order: float) -> int:
        """Refine by a power of two (grids stay nested), sized from the error."""
        if error is None or not math.isfinite(error) or error <= 0:
            return 2 * n
        factor = 1.1 * (error / tol) ** (1.0 / max(order, 0.5))
        # a factor just past a power of two (within 7%) rounds down: the 1.1 margin already covers it
        return n * 2 ** min(3, max(1, math.ceil(math.log2(factor) - 0.1)))

    def _remember(self, report: SolveReport) -> None:
        self.memory.add({
            "signature": report.problem.signature(),
            "name": report.problem.name,
            "problem": report.problem.to_dict(),
            "tol": report.tol,
            "success": report.success,
            "status": report.status,
            "method": report.method if report.success else None,
            "n": report.n,
            "error": report.error,
            "error_kind": report.error_kind,
            "solver_runs": report.solver_runs,
            "work": report.work,
            "runtime": round(report.runtime, 4),
            "methods": report.trials,
            "learned": [o.solver for o in report.learned if o.success],
            "tags": list(report.features.tags) if report.features else [],
            "features": report.features.to_dict() if report.features else None,
            "crosscheck": report.crosscheck.to_dict() if report.crosscheck else None,
            "costs": report.costs(),
        })
