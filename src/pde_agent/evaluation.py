"""Evaluation: does the agent tell the truth about its answers, and does experience help?

``pde-agent bench`` scores the methods on problems the agent is told the exact
solution of. That cannot catch the failure that matters most in practice: an
answer reported as a success whose true error is above the tolerance (a
*false success*). Here every test problem still has an exact solution, but the
agent gets a blind copy (``blind``) without it, so it must estimate its own
error; the hidden exact solution then grades the answer.

Problems are split into a training set (the curriculum lessons) and a test set
(the benchmark problems and the ODE exam: same problem classes, other
parameters). Two agents solve the test set:

* memory off: no experience at all,
* memory on: experience from solving the training set first (with exact
  solutions, as the curriculum does). Its memory is frozen during the test, so
  the order of the test problems does not matter.

The third part checks how well the language model turns Chinese descriptions
into problems (``parse_fixtures.py``); see ``run_parse_eval``.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .verify import exact_error


@dataclass
class EvalCase:
    name: str
    title: str
    problem: object  # PDEProblem or ODEProblem, with an exact solution
    tol: float
    split: str  # train | test
    part: str  # pde | ode


def eval_cases(parts: tuple[str, ...] = ("pde", "ode")) -> list[EvalCase]:
    """Training set: curriculum lessons; test set: benchmark problems and the ODE exam.

    Only problems with an exact solution are kept (nothing else can be
    graded), and only classes a built-in method covers (Burgers has none).
    """
    from .benchmark import reference_problems
    from .curriculum import basic_curriculum, ode_curriculum, ode_exam
    from .solvers import builtin_registry

    registry = builtin_registry()
    cases: list[EvalCase] = []
    if "pde" in parts:
        cases += [EvalCase(f"train/{l.name}", l.title, l.problem, l.tol, "train", "pde") for l in basic_curriculum()]
        cases += [EvalCase(f"test/{b.name}", b.title, b.problem, b.tol, "test", "pde") for b in reference_problems()]
    if "ode" in parts:
        cases += [EvalCase(f"train/{l.name}", l.title, l.problem, l.tol, "train", "ode") for l in ode_curriculum()]
        cases += [EvalCase(f"test/{l.name}", l.title, l.problem, l.tol, "test", "ode") for l in ode_exam()]
    return [c for c in cases if c.problem.has_exact() and registry.for_problem(c.problem)]


def blind(problem):
    """The problem as a user would pose it: the same data, no exact solution.

    Initial and boundary data that came from the exact solution are written
    out (as ``for_solver`` does), so the problem is unchanged but the agent
    has to estimate its error.
    """
    view = problem.for_solver()
    return type(problem).from_dict(view.to_dict())


@dataclass
class CaseResult:
    case: EvalCase
    arm: str  # train | memory_off | memory_on
    reported: bool  # the agent said it met the tolerance
    method: str | None
    n: int | None
    claimed_error: float | None  # the agent's own (estimated) error
    true_error: float | None  # against the hidden exact solution
    solver_runs: int
    work: float
    runtime: float
    message: str = ""

    @property
    def correct(self) -> bool:
        return self.true_error is not None and self.true_error <= self.case.tol

    @property
    def false_success(self) -> bool:
        return self.reported and not self.correct

    def to_dict(self) -> dict:
        return {
            "case": self.case.name, "split": self.case.split, "part": self.case.part, "arm": self.arm,
            "tol": self.case.tol, "reported": self.reported, "correct": self.correct,
            "false_success": self.false_success, "method": self.method, "n": self.n,
            "claimed_error": self.claimed_error, "true_error": self.true_error,
            "solver_runs": self.solver_runs, "work": self.work, "runtime": round(self.runtime, 4),
            "message": self.message,
        }


@dataclass
class ArmSummary:
    arm: str
    cases: int
    reported: int
    correct: int
    false_success: int
    solver_runs: int
    work: float
    runtime: float

    @property
    def false_success_rate(self) -> float | None:
        """Share of reported successes that are wrong."""
        return self.false_success / self.reported if self.reported else None

    def to_dict(self) -> dict:
        return {**vars(self), "false_success_rate": self.false_success_rate}


def summarize(results: list[CaseResult], arm: str) -> ArmSummary:
    rs = [r for r in results if r.arm == arm]
    return ArmSummary(arm, len(rs), sum(r.reported for r in rs), sum(r.reported and r.correct for r in rs),
                      sum(r.false_success for r in rs), sum(r.solver_runs for r in rs),
                      sum(r.work for r in rs), sum(r.runtime for r in rs))


def _grade(agent, case: EvalCase, arm: str, hide_exact: bool) -> CaseResult:
    problem = blind(case.problem) if hide_exact else case.problem
    report = agent.solve(problem, case.tol, learn=False)
    true_error = exact_error(case.problem, report.result) if report.result is not None else None
    return CaseResult(case, arm, report.success, report.method if report.success else None, report.n,
                      report.error, true_error, report.solver_runs, report.work, report.runtime,
                      "" if report.success else report.message)


def run_eval(cases: list[EvalCase], home: Path | str | None = None, time_budget: float = 60.0, progress=None):
    """Train the memory-on agent, then grade both agents on the blind test set.

    Both agents live in a fresh directory (``home`` or a temporary one) and use
    the built-in methods only, so runs are comparable between computers.
    ``progress(result)`` is called after every solve. Returns all results.
    """
    from .agent import PDEAgent

    with tempfile.TemporaryDirectory(prefix="pde-eval-") as tmp:
        root = Path(home) if home else Path(tmp)
        off = PDEAgent(home=root / "memory_off", remember=False, time_budget=time_budget)
        on = PDEAgent(home=root / "memory_on", remember=True, time_budget=time_budget)
        results: list[CaseResult] = []

        def record(result):
            results.append(result)
            if progress:
                progress(result)

        for case in (c for c in cases if c.split == "train"):
            record(_grade(on, case, "train", hide_exact=False))
        frozen = list(on.memory.episodes)
        on.memory.path = None  # the test does not write to the training memory
        for case in (c for c in cases if c.split == "test"):
            record(_grade(off, case, "memory_off", hide_exact=True))
            record(_grade(on, case, "memory_on", hide_exact=True))
            on.memory.episodes = list(frozen)
        return results


# --- parsing Chinese descriptions ----------------------------------------------------


@dataclass
class ParseResult:
    name: str
    text: str
    ok: bool
    mismatches: list[str] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return vars(self).copy()


def run_parse_eval(brain, fixtures=None, progress=None) -> list[ParseResult]:
    """Ask ``brain`` to parse every fixture and compare with the known correct problem."""
    from .equations import builtin_equations
    from .parse_fixtures import compare_parse, parse_fixtures

    equations = builtin_equations()
    results = []
    for fx in fixtures if fixtures is not None else parse_fixtures():
        start = time.monotonic()
        try:
            parsed = brain.parse_problem(fx.text, equations)
            mismatches = compare_parse(fx, parsed.problem, equations)
            result = ParseResult(fx.name, fx.text, not mismatches, mismatches)
        except Exception as exc:  # an API failure or an invalid reply counts as a wrong parse
            result = ParseResult(fx.name, fx.text, False, error=f"{type(exc).__name__}: {exc}"[:300])
        result.seconds = time.monotonic() - start
        results.append(result)
        if progress:
            progress(result)
    return results
