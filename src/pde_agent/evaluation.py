"""Evaluation: does the agent tell the truth about its answers, and does experience help?

``pde-agent bench`` scores the methods on problems the agent is told the exact
solution of. That cannot catch the failure that matters most in practice: an
answer reported as a success whose true error is above the tolerance (a
*false success*). Here every test problem still has an exact solution, but the
agent gets a blind copy (``blind``) without it, so it must estimate its own
error; the hidden exact solution then grades the answer.

Problems are split into a training set (the curriculum lessons) and a test set
held out from it in three ways (``EvalCase.group``):

* ``param``: the same families as in training, with parameters never practised,
* ``regularity``: data less smooth than anything in training (jumps, shocks,
  spikes, boundary layers),
* ``ood``: out of the training distribution (higher frequencies, stiffer modes).

A test problem that is the same mathematical problem as a training problem (or
an earlier test problem) under another name is dropped (``fingerprint``).

The test set is solved with experience in three modes (``ARMS``):

* ``memory_off``: no experience; each solve reads an empty store and adds nothing,
* ``memory_frozen``: the experience of the training set, read only, so the order
  of the test problems does not matter,
* ``memory_online``: starts from the training experience and keeps learning
  from each test solve, in the order the test set is run (``seed``).

The third part checks how well the language model turns Chinese descriptions
into problems (``parse_fixtures.py``); see ``run_parse_eval``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .verify import exact_error

ARMS = ("memory_off", "memory_frozen", "memory_online")
GROUPS = ("param", "regularity", "ood")


@dataclass
class EvalCase:
    name: str
    title: str
    problem: object  # PDEProblem or ODEProblem, with an exact solution
    tol: float
    split: str  # train | test
    part: str  # pde | ode
    group: str = ""  # test only: param | regularity | ood


# which way each benchmark problem is held out from the curriculum
_BENCH_GROUPS = {"heat-decay": "param", "heat-stiff": "ood", "advection-smooth": "param", "advection-long": "param",
                 "wave-two-modes": "ood", "poisson-2d": "param"}


def heldout_problems() -> list:
    """Test problems written for the evaluation: ``(group, part, Lesson)``."""
    from .curriculum import Lesson, _harmonic, _linear_stiff, _mms

    unit = [(0.0, 1.0)]
    layer = "x - (exp(20*(x - 1)) - exp(-20))/(1 - exp(-20))"  # a boundary layer of width 0.05 at x = 1
    return [
        ("ood", "pde", Lesson("heat-high-mode", 2, "热方程，高频模态 sin(6πx)",
                              _mms("heat", "sin(6*pi*x)*exp(-1.8*pi**2*t)", unit, 0.5, {"alpha": 0.05}, name="heat-high-mode"))),
        ("ood", "pde", Lesson("poisson-2d-high", 4, "二维 Poisson，高频 sin(3πx) sin(4πy)",
                              _mms("poisson", "sin(3*pi*x)*sin(4*pi*y)", [(0.0, 1.0), (0.0, 1.0)],
                                   name="poisson-2d-high"))),
        ("regularity", "pde", Lesson("poisson-boundary-layer", 1, "一维 Poisson，x = 1 处宽 0.05 的边界层",
                                     _mms("poisson", layer, unit, name="poisson-boundary-layer"))),
        ("regularity", "pde", Lesson("advection-spike", 3, "对流，窄尖峰 exp(50(cos 2π(x - 0.7t) - 1))",
                                     _mms("advection", "exp(50*(cos(2*pi*(x - 0.7*t)) - 1))", unit, 1.0, {"c": 0.7},
                                          "periodic", name="advection-spike"))),
        ("param", "ode", Lesson("heldout-stiff-3e3", 1, "线性刚性，λ = 3000（训练过 10、10³、10⁶）",
                                _linear_stiff("heldout-stiff-3e3", 3e3, "线性刚性，λ = 3000"))),
        ("ood", "ode", Lesson("heldout-harmonic-fast", 0, "简谐振子 ω = 7，长时间",
                              _harmonic("heldout-harmonic-fast", 7.0, 30.0, "简谐振子 ω = 7"))),
    ]


def all_cases(parts: tuple[str, ...] = ("pde", "ode")) -> list[EvalCase]:
    """Every case before duplicates are removed (see ``eval_cases``)."""
    from .benchmark import reference_problems
    from .curriculum import basic_curriculum, ode_curriculum, ode_exam, pde_exam, pde_stages

    cases: list[EvalCase] = []
    if "pde" in parts:
        cases += [EvalCase(f"train/{l.name}", l.title, l.problem, l.tol, "train", "pde") for l in basic_curriculum()]
        # stages 2 and 3 (smooth parabolic and hyperbolic) train; stage 4 (jumps, shocks) is held out by regularity
        cases += [EvalCase(f"train/{l.name}", l.title, l.problem, l.tol, "train", "pde")
                  for l in pde_stages() if l.level < 4]
        cases += [EvalCase(f"test/{b.name}", b.title, b.problem, b.tol, "test", "pde", _BENCH_GROUPS[b.name])
                  for b in reference_problems() if b.name in _BENCH_GROUPS]
        cases += [EvalCase(f"test/{l.name}", l.title, l.problem, l.tol, "test", "pde", "regularity")
                  for l in pde_stages() if l.level == 4]
        cases += [EvalCase(f"test/{l.name}", l.title, l.problem, l.tol, "test", "pde",
                           "regularity" if l.level == 4 else "param") for l in pde_exam()]
    if "ode" in parts:
        cases += [EvalCase(f"train/{l.name}", l.title, l.problem, l.tol, "train", "ode") for l in ode_curriculum()]
        cases += [EvalCase(f"test/{l.name}", l.title, l.problem, l.tol, "test", "ode", "param") for l in ode_exam()]
    cases += [EvalCase(f"test/{l.name}", l.title, l.problem, l.tol, "test", part, group)
              for group, part, l in heldout_problems() if part in parts]
    return cases


def eval_cases(parts: tuple[str, ...] = ("pde", "ode")) -> list[EvalCase]:
    """Training set: curriculum lessons; test set: held-out problems (see the module docstring).

    Only problems with an exact solution are kept (nothing else can be
    graded), only classes a built-in method covers, and no test problem that
    repeats a training problem or an earlier test problem (``duplicates``).
    """
    cases, _ = dedupe(all_cases(parts))
    return cases


def dedupe(cases: list[EvalCase]) -> tuple[list[EvalCase], list[tuple[EvalCase, EvalCase]]]:
    """``(kept, [(dropped test case, the case it repeats)])``; ungradable cases are left out silently.

    A test problem that is a training problem, at any tolerance, is dropped:
    it measures recall, not transfer. Two test cases repeat each other only
    with the same tolerance too (a tighter tolerance is a different task).
    """
    from .solvers import builtin_registry

    registry = builtin_registry()
    usable = [c for c in cases if c.problem.has_exact() and registry.for_problem(c.problem)]
    trained: dict[str, EvalCase] = {}
    for c in usable:
        if c.split == "train":
            trained.setdefault(fingerprint(c.problem), c)
    tested: dict[tuple[str, float], EvalCase] = {}
    kept, dropped = [], []
    for c in usable:
        if c.split == "test":
            key = fingerprint(c.problem)
            twin = trained.get(key) or tested.get((key, c.tol))
            if twin is not None:
                dropped.append((c, twin))
                continue
            tested[(key, c.tol)] = c
        kept.append(c)
    return kept, dropped


def _round(values) -> list[float]:
    values = np.broadcast_to(np.asarray(values, dtype=float), np.shape(values) or (1,)).ravel()
    return [float(f"{v:.8g}") for v in values]


def fingerprint(problem) -> str:
    """The problem as a solver sees it, without names, descriptions or the exact solution, as a hash.

    Two problems with equal fingerprints are the same mathematical problem:
    the same equation, domain, end time, coefficients (defaults filled in),
    and the same data sampled on a fixed set of points.
    """
    from .equations import builtin_equations

    view = blind(problem)
    if view.dim == 0:  # an ODE
        rng = np.random.default_rng(0)
        points = [(float(rng.uniform(0, view.t_final)), view.y0 + rng.uniform(-1, 1, view.size) * (1 + abs(view.y0)))
                  for _ in range(6)]
        data = ["ode", view.size, view.t_final, _round(view.y0), [_round(view.f(t, y)) for t, y in points]]
    else:
        view = builtin_equations().complete(view)
        axes = [np.linspace(a, b, 7) for a, b in view.domain]
        grid = np.meshgrid(*axes, indexing="ij")
        x = grid[0].ravel()
        y = grid[1].ravel() if view.dim == 2 else np.zeros_like(x)
        times = np.linspace(0.0, view.t_final, 3) if view.t_final else [0.0]
        fields = ["source"] + (["initial", "initial_velocity"] if view.time_dependent else [])
        if view.boundary == "dirichlet":
            fields.append("boundary_value")
        samples = {}
        for name in fields:
            fn = view.fn(name)
            ts = [0.0] if name.startswith("initial") else times
            samples[name] = [_round(fn(x, y, t)) for t in ts]
        data = [view.kind, view.dim, [_round(d) for d in view.domain], view.t_final, view.boundary,
                sorted((k, float(f"{v:.8g}")) for k, v in view.coefficients.items()),
                " ".join(view.equation.split()), samples]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def blind(problem):
    """The problem as a user would pose it: the same data, no exact solution.

    Initial and boundary data that came from the exact solution are written
    out (as ``for_solver`` does), so the problem is unchanged but the agent
    has to estimate its error.
    """
    from .equations import builtin_equations

    view = problem.for_solver()
    data = view.to_dict()
    eq = builtin_equations().get(data.get("kind"))
    if eq is not None and eq.time_order < 2 and "initial_velocity" in data:
        data["initial_velocity"] = None  # first order in time: unused, and u_t of a jump is no expression
    return type(problem).from_dict(data)


@dataclass
class CaseResult:
    case: EvalCase
    arm: str  # train | memory_off | memory_frozen | memory_online
    reported: bool  # the agent said it met the tolerance
    method: str | None
    n: int | None
    claimed_error: float | None  # the agent's own (estimated) error
    true_error: float | None  # against the hidden exact solution
    solver_runs: int
    work: float
    runtime: float
    message: str = ""
    experience: int = 0  # episodes the agent could read when it started this solve
    order: int = 0  # position in the run (test problems may be shuffled)

    @property
    def correct(self) -> bool:
        return self.true_error is not None and self.true_error <= self.case.tol

    @property
    def false_success(self) -> bool:
        return self.reported and not self.correct

    def to_dict(self) -> dict:
        return {
            "case": self.case.name, "split": self.case.split, "part": self.case.part, "group": self.case.group,
            "arm": self.arm, "order": self.order, "tol": self.case.tol, "reported": self.reported,
            "correct": self.correct, "false_success": self.false_success, "method": self.method, "n": self.n,
            "claimed_error": self.claimed_error, "true_error": self.true_error, "solver_runs": self.solver_runs,
            "work": self.work, "runtime": round(self.runtime, 4), "experience": self.experience,
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
    missed: int = 0  # reported a failure although the answer met the tolerance (cautious, not wrong)

    @property
    def false_success_rate(self) -> float | None:
        """Share of reported successes that are wrong."""
        return self.false_success / self.reported if self.reported else None

    def to_dict(self) -> dict:
        return {**vars(self), "false_success_rate": self.false_success_rate}


def summarize(results: list[CaseResult], arm: str, group: str | None = None) -> ArmSummary:
    rs = [r for r in results if r.arm == arm and (group is None or r.case.group == group)]
    return ArmSummary(arm, len(rs), sum(r.reported for r in rs), sum(r.reported and r.correct for r in rs),
                      sum(r.false_success for r in rs), sum(r.solver_runs for r in rs),
                      sum(r.work for r in rs), sum(r.runtime for r in rs),
                      sum(not r.reported and r.correct for r in rs))


def memory_gain(results: list[CaseResult], arm: str, base: str = "memory_off") -> dict:
    """How ``arm`` compares with ``base`` on the test problems both solved correctly.

    Ratios below 1 mean experience saved solver runs or work. Problems only one
    arm got right are counted separately: a cheap wrong answer is no saving.
    """
    by_case = {}
    for r in results:
        by_case.setdefault(r.case.name, {})[r.arm] = r
    both = [d for d in by_case.values() if arm in d and base in d and d[arm].correct and d[base].correct
            and d[arm].reported and d[base].reported]
    runs = (sum(d[arm].solver_runs for d in both), sum(d[base].solver_runs for d in both))
    work = (sum(d[arm].work for d in both), sum(d[base].work for d in both))
    seconds = (sum(d[arm].runtime for d in both), sum(d[base].runtime for d in both))
    return {
        "arm": arm, "base": base, "both_correct": len(both),
        "only_arm": sum(1 for d in by_case.values() if arm in d and base in d and d[arm].correct and not d[base].correct),
        "only_base": sum(1 for d in by_case.values() if arm in d and base in d and d[base].correct and not d[arm].correct),
        "runs": runs, "work": work, "runtime": seconds,
        "runs_ratio": runs[0] / runs[1] if runs[1] else None,
        "work_ratio": work[0] / work[1] if work[1] else None,
        "runtime_ratio": seconds[0] / seconds[1] if seconds[1] else None,
        "same_method": sum(1 for d in both if d[arm].method == d[base].method),
    }


def environment() -> dict:
    """Version and machine of a run, saved with its results."""
    import scipy
    import sympy

    from . import __version__

    return {"pde_agent": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__}


def _grade(agent, case: EvalCase, arm: str, hide_exact: bool) -> CaseResult:
    problem = blind(case.problem) if hide_exact else case.problem
    memory = getattr(agent, "memory", None)
    experience = len(memory.episodes) + len(memory.shared) if memory is not None else 0
    report = agent.solve(problem, case.tol, learn=False)
    true_error = exact_error(case.problem, report.result) if report.result is not None else None
    return CaseResult(case, arm, report.success, report.method if report.success else None, report.n,
                      report.error, true_error, report.solver_runs, report.work, report.runtime,
                      "" if report.success else report.message, experience)


def test_order(cases: list[EvalCase], seed: int | None = None) -> list[EvalCase]:
    """The test problems in the order they are run: as listed, or shuffled with ``seed``."""
    tests = [c for c in cases if c.split == "test"]
    if seed is not None:
        random.Random(seed).shuffle(tests)
    return tests


def run_eval(cases: list[EvalCase], home: Path | str | None = None, time_budget: float = 60.0, progress=None,
             seed: int | None = None, arms: tuple[str, ...] = ARMS):
    """Train on the training set, then solve the blind test set in each memory mode of ``arms``.

    The agent lives in a fresh directory (``home``, which must be empty, or a
    temporary one) and uses the built-in methods only, so runs are comparable
    between computers. Experience is held in memory only. ``progress(result)``
    is called after every solve. Returns all results.

    The modes are checked as they run: ``memory_off`` must start every solve
    with no experience and ``memory_frozen`` with exactly the training
    experience; anything else raises ``RuntimeError``.
    """
    from .agent import PDEAgent
    from .memory import ExperienceStore

    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown memory modes: {', '.join(sorted(unknown))}")
    if home is not None and Path(home).exists() and any(Path(home).iterdir()):
        raise ValueError(f"评测目录 {home} 不是空的：旧的技能和经验会混进评测")
    with tempfile.TemporaryDirectory(prefix="pde-eval-") as tmp:
        root = Path(home) if home else Path(tmp)
        agent = PDEAgent(home=root, remember=False, time_budget=time_budget)
        results: list[CaseResult] = []

        def record(result):
            result.order = len(results)
            results.append(result)
            if progress:
                progress(result)

        training = ExperienceStore()  # in memory, learns from every training solve
        agent.memory = training
        for case in (c for c in cases if c.split == "train"):
            record(_grade(agent, case, "train", hide_exact=False))
        stores = {"memory_off": ExperienceStore(update=False), "memory_frozen": training.snapshot(update=False),
                  "memory_online": training.snapshot(update=True)}
        trained = stores["memory_frozen"].fingerprint()
        for case in test_order(cases, seed):
            for arm in arms:
                agent.memory = stores[arm]
                record(_grade(agent, case, arm, hide_exact=True))
                if arm == "memory_off" and (stores[arm].episodes or stores[arm].shared):
                    raise RuntimeError("无经验组在测试中积累了经验")
                if arm == "memory_frozen" and stores[arm].fingerprint() != trained:
                    raise RuntimeError("冻结组的经验在测试中变了")
        return results


def gate_failures(results: list[CaseResult]) -> list[str]:
    """Why ``pde-agent eval --gate`` fails: every false success, in any mode (empty: pass)."""
    return [f"{r.case.name}（{r.arm}）自报误差 {r.claimed_error:.3g}，真实误差 {r.true_error:.3g}，容差 {r.case.tol:g}"
            if r.claimed_error is not None and r.true_error is not None else f"{r.case.name}（{r.arm}）"
            for r in results if r.false_success]


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
