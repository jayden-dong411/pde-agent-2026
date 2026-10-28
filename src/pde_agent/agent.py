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
4. measure the error (exact solution) or estimate it (grid refinement),
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

from .equations import builtin_equations, operator_from_text
from .features import Prediction, ProblemFeatures, extract, predict
from .llm import LLMError
from .memory import ExperienceStore
from .problem import PDEProblem
from .skills import SkillLibrary, SkillTask, ValidationReport, validate_candidate
from .solvers import builtin_registry
from .solvers.base import SolveResult, SolverDiverged, SolverSpec
from .verify import exact_error, relative_error, resolution_floor, restrict

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
}


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

    @property
    def solver_runs(self) -> int:
        return len(self.attempts)

    @property
    def work(self) -> float:
        return sum(a.work for a in self.attempts)

    @property
    def runtime(self) -> float:
        return sum(a.runtime for a in self.attempts)

    def choice(self) -> str:
        """Why the methods were tried in this order: the problem's features and the cheapest prediction."""
        if self.features is None:
            return ""
        text = f"题目特征：{self.features.describe()}"
        ranked = sorted((p for p in self.predictions.values() if p.work is not None), key=lambda p: p.work)
        if ranked:
            best = ranked[0]
            text += f"。按特征估计 {best.method} 最省（n ≈ {best.n}，工作量约 {best.work:.2g}）"
            notes = [f"{p.method}：{'；'.join(p.notes)}" for p in self.predictions.values() if p.notes]
            if notes:
                text += "。" + "；".join(notes)
        return text

    def summary(self) -> str:
        name = self.problem.name or self.problem.signature()
        learned = [o.solver for o in self.learned if o.success]
        extra = f"，新学会 {', '.join(learned)}" if learned else ""
        if self.success:
            how = "对比精确解" if self.error_kind == "exact" else "加密网格估计"
            return (f"{name}：成功，方法 {self.method}，n={self.n}，误差 {self.error:.2e}（{how}），"
                    f"求解 {self.solver_runs} 次{extra}")
        return f"{name}：未成功，{self.message}{extra}"


class PDEAgent:
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
        self.memory = ExperienceStore(self.home / "experience.jsonl" if remember else None,
                                      self.home / "shared_experience.jsonl" if remember else None)
        self.skills = SkillLibrary(self.home / "skills")
        self.skill_errors = self.skills.load_into(self.registry, self.equations, timeout=self.validation_timeout)

    # --- public API ---------------------------------------------------------------

    def solve(self, problem: PDEProblem, tol: float = 1e-3, options: dict | None = None, learn: bool = True) -> SolveReport:
        if not tol > 0:
            raise ValueError("tol must be positive")
        problem = self.equations.complete(problem)
        report = SolveReport(problem=problem, tol=tol, error_kind="exact" if problem.has_exact() else "estimated")
        eq = self.equations.get(problem.kind)
        time_order = eq.time_order if eq else int(problem.time_dependent)
        # an estimate's coarse companion is the first grid (BASE_N // 2), so it has to see the data too
        start = max(4, BASE_N[problem.dim] // 2) if report.error_kind == "estimated" else BASE_N[problem.dim]
        # an ODE has no grid data to resolve
        floor = start if problem.dim == 0 else resolution_floor(problem, tol, time_order, start, MAX_N[problem.dim])
        if floor is None:
            report.message = f"题目数据在最细网格（n = {MAX_N[problem.dim]}）上也分辨不清，没有求解"
            self._remember(report)
            return report
        report.min_n = floor
        report.features = extract(problem, floor)
        candidates = self.registry.for_problem(problem)
        can_learn = learn and self.brain is not None and problem.dim > 0  # skills are written for grids

        if not candidates and can_learn:
            outcome = self.learn(problem, reason="no method covers this problem class yet")
            report.learned.append(outcome)
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
                outcome = self.learn(problem, reason=f"existing methods failed: {report.message}")
                report.learned.append(outcome)
                if outcome.success:
                    self._run_methods(report, [self.registry.get(outcome.solver)], options)
        self._remember(report)
        return report

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
            try:
                candidate = session.propose()
            except LLMError as exc:
                return LearnOutcome(False, rounds=round_, report=report, message=str(exc))
            report = validate_candidate(candidate, task, timeout=self.validation_timeout)
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
        if report.features is not None:
            for spec in candidates:
                report.predictions[spec.name] = predict(
                    spec, report.problem, report.features, report.tol, MAX_N[report.problem.dim])
        tags = report.features.tags if report.features else None
        predicted = {name: p.work for name, p in report.predictions.items()}
        for spec in self.memory.rank(report.problem.signature(), candidates, self.explore, predicted, tags):
            if time.monotonic() > deadline:
                break
            outcome, n, error, work = self._try_method(spec, report, options, deadline)
            report.trials.append({
                "method": spec.name, "outcome": outcome, "n": n, "error": error,
                "error_kind": report.error_kind, "work": work,
            })
            if outcome == "success":
                return
        tried = "；".join(f"{t['method']} {OUTCOMES[t['outcome']]}" for t in report.trials)
        report.message = f"没有方法达到精度 {report.tol:g}（{tried or '时间预算用完'}）"

    def _try_method(self, spec: SolverSpec, report: SolveReport, options: dict | None, deadline: float):
        problem, tol = report.problem, report.tol
        estimated = report.error_kind == "estimated"
        max_n = MAX_N[problem.dim]
        tags = report.features.tags if report.features else None
        prediction = report.predictions.get(spec.name)
        n = (self.memory.suggest_n(problem.signature(), spec, tol, report.error_kind, tags)
             or (prediction.start if prediction else BASE_N[problem.dim]))
        if estimated:
            n //= 2  # the estimate needs a coarser companion solution
        n = min(max(4, report.min_n, n), max_n // 2 if estimated else max_n)

        prev: SolveResult | None = None
        prev_error = prev_diff = prev_ratio = None
        error, work = None, 0.0
        for _ in range(self.max_runs_per_method):
            if time.monotonic() > deadline:
                return "budget", n, error, work
            try:
                result = spec.run(problem, n, options)
            except SolverDiverged as exc:
                report.attempts.append(Attempt(spec.name, n, "unstable", message=str(exc)))
                return "unstable", n, None, work
            except Exception as exc:
                report.attempts.append(Attempt(spec.name, n, "error", message=f"{type(exc).__name__}: {exc}"[:300]))
                return "error", n, None, work
            work += result.work

            order = spec.order
            if not estimated:
                error = exact_error(problem, result)
            elif prev is not None:
                ratio = n / prev.n
                diff = relative_error(prev.u, restrict(result, prev.n))
                if prev_diff and diff > 0 and prev_ratio == ratio:
                    # trust the observed rate when it is lower than claimed
                    order = min(order, max(0.5, math.log(prev_diff / diff) / math.log(ratio)))
                error = diff / (ratio**order - 1.0)
                prev_diff, prev_ratio = diff, ratio
            report.attempts.append(Attempt(
                spec.name, n, "ok" if error is not None else "unverified", error, result.work, result.runtime,
            ))

            if error is not None and (report.error is None or error < report.error or error <= tol):
                report.result, report.error, report.n, report.method = result, error, n, spec.name
            if error is not None and error <= tol:
                report.success = True
                return "success", n, error, work
            if error is not None and prev_error is not None and error > 0.8 * prev_error:
                return "stagnated", n, error, work
            if n >= max_n:
                return "max_n", n, error, work
            prev, prev_error = result, error if error is not None else prev_error
            n = min(self._next_n(n, error, tol, order), max_n)
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
        })
