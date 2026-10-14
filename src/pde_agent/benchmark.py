"""Benchmark: score solver choices on reference problems with exact solutions.

Every reference problem has a closed-form solution, so errors are measured,
not estimated. Each method that can handle a problem is run on doubling grids
until it reaches the tolerance; its score is the runtime of the first run that
does (the cost of the method when the right grid is known). The agent is then
scored on the same problem from a cold start (no experience): did it pick a
method that reaches the tolerance, and how much longer than the best method
did its whole episode take, exploration included.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import sympy as sp

from .equations import Equation, builtin_equations
from .problem import PDEProblem
from .solvers.base import SolverDiverged, SolverRegistry
from .verify import exact_error, resolution_floor

BASE_N = {1: 16, 2: 8, 3: 4}
MAX_N = {1: 4096, 2: 256, 3: 64}

# Viscous Burgers has no built-in method; its operator is here so the tests
# can check the reference solution, and a learned skill can be benchmarked.
BURGERS = Equation(
    kind="burgers",
    form="u_t + u*u_x = nu*u_xx + f",
    time_order=1,
    operator=lambda u, x, y, t, c: sp.diff(u, t) + u * sp.diff(u, x) - c["nu"] * sp.diff(u, x, 2),
    defaults={"nu": 0.05},
)


@dataclass
class BenchProblem:
    name: str
    title: str
    problem: PDEProblem
    tol: float = 1e-3


def reference_problems() -> list[BenchProblem]:
    """Problems whose exact solutions satisfy the equation with no source term
    (except Poisson, which needs one), chosen to stress different weaknesses."""
    unit = [(0.0, 1.0)]

    def p(name, kind, exact, t_final=None, coefficients=None, boundary="dirichlet", source="0", domain=unit):
        return PDEProblem(kind=kind, domain=domain, t_final=t_final, coefficients=coefficients or {},
                          source=source, boundary=boundary, exact=exact, name=name)

    return [
        BenchProblem("heat-decay", "热方程：单个模态指数衰减",
                     p("heat-decay", "heat", "sin(pi*x)*exp(-0.1*pi**2*t)", 1.0, {"alpha": 0.1})),
        BenchProblem("heat-stiff", "热方程：刚性，高频模态瞬间衰减",
                     p("heat-stiff", "heat", "sin(pi*x)*exp(-pi**2*t) + 0.5*sin(24*pi*x)*exp(-576*pi**2*t)",
                       0.05, {"alpha": 1.0})),
        BenchProblem("advection-smooth", "对流：光滑波平移一个周期",
                     p("advection-smooth", "advection", "sin(2*pi*(x - t))", 1.0, {"c": 1.0}, "periodic")),
        BenchProblem("advection-long", "对流：长时间传播（考验耗散和色散）",
                     p("advection-long", "advection", "exp(sin(2*pi*(x - t)))", 5.0, {"c": 1.0}, "periodic")),
        BenchProblem("wave-two-modes", "波动：两个驻波模态叠加",
                     p("wave-two-modes", "wave", "sin(pi*x)*cos(pi*t) + 0.2*sin(4*pi*x)*cos(4*pi*t)", 1.0, {"c": 1.0})),
        BenchProblem("poisson-2d", "二维 Poisson：-Δu = 5π² sin(πx) sin(2πy)",
                     p("poisson-2d", "poisson", "sin(pi*x)*sin(2*pi*y)", source="5*pi**2*sin(pi*x)*sin(2*pi*y)",
                       coefficients={"k": 1.0}, domain=[(0.0, 1.0), (0.0, 1.0)])),
        BenchProblem("burgers-cole-hopf", "黏性 Burgers：Cole–Hopf 精确解（没有内置方法）",
                     p("burgers-cole-hopf", "burgers",
                       "2*pi*nu*exp(-pi**2*nu*t)*sin(pi*x)/(2 + exp(-pi**2*nu*t)*cos(pi*x))", 1.0, {"nu": 0.05})),
    ]


def equation_for(kind: str) -> Equation | None:
    return builtin_equations().get(kind) or (BURGERS if kind == "burgers" else None)


@dataclass
class Run:
    n: int
    error: float | None = None
    runtime: float = 0.0
    work: float = 0.0
    status: str = "ok"  # ok | unstable | error
    message: str = ""
    hit: bool = False  # reached the tolerance


@dataclass
class MethodScore:
    method: str
    order: float
    runs: list[Run] = field(default_factory=list)

    @property
    def best(self) -> Run | None:
        """The run that reached the tolerance, if any."""
        return next((r for r in self.runs if r.hit), None)

    @property
    def time_to_tol(self) -> float | None:
        return self.best.runtime if self.best else None

    @property
    def observed_order(self) -> float | None:
        ok = [r for r in self.runs if r.status == "ok" and r.error and r.error > 0]
        if len(ok) < 2:
            return None
        a, b = ok[-2], ok[-1]
        return math.log(a.error / b.error) / math.log(b.n / a.n)

    @property
    def outcome(self) -> str:
        if self.best:
            return "success"
        return self.runs[-1].status if self.runs and self.runs[-1].status != "ok" else "max_n"


@dataclass
class AgentScore:
    success: bool
    method: str | None
    n: int | None
    error: float | None
    runtime: float
    solver_runs: int
    message: str = ""


@dataclass
class ProblemScore:
    bench: BenchProblem
    methods: list[MethodScore] = field(default_factory=list)
    agent: AgentScore | None = None

    @property
    def best(self) -> MethodScore | None:
        hits = [m for m in self.methods if m.time_to_tol is not None]
        return min(hits, key=lambda m: m.time_to_tol) if hits else None

    @property
    def agent_overhead(self) -> float | None:
        """Agent's episode runtime divided by the best method's time to tolerance."""
        best = self.best
        if not (self.agent and self.agent.success and best and best.time_to_tol):
            return None
        return self.agent.runtime / best.time_to_tol

    def to_dict(self) -> dict:
        best = self.best
        return {
            "problem": self.bench.name,
            "tol": self.bench.tol,
            "best_method": best.method if best else None,
            "methods": [
                {
                    "method": m.method, "order": m.order, "outcome": m.outcome,
                    "time_to_tol": m.time_to_tol, "n": m.best.n if m.best else None,
                    "observed_order": m.observed_order,
                    "runs": [vars(r) for r in m.runs],
                }
                for m in self.methods
            ],
            "agent": None if self.agent is None else {**vars(self.agent), "overhead": self.agent_overhead},
        }


def score_method(spec, bench: BenchProblem, time_budget: float = 20.0, repeat: int = 3) -> MethodScore:
    """Run ``spec`` on doubling grids until it reaches the tolerance, fails or runs out of time.

    The first grid is the resolution floor of the problem's data (no runs at
    all when no grid resolves them). The run that reaches the tolerance is
    repeated and its fastest time kept, since millisecond timings are noisy.
    """
    problem = bench.problem
    score = MethodScore(spec.name, spec.order)
    eq = equation_for(problem.kind)
    time_order = eq.time_order if eq else int(problem.time_dependent)
    max_n = MAX_N[problem.dim]
    n = resolution_floor(problem, bench.tol, time_order, BASE_N[problem.dim], max_n)
    if n is None:  # no grid up to max_n resolves the data, so there is nothing a run could be scored on
        return score
    deadline = time.monotonic() + time_budget
    while n <= max_n and time.monotonic() < deadline:
        try:
            result = spec.run(problem, n)
        except SolverDiverged as exc:
            score.runs.append(Run(n, status="unstable", message=str(exc)))
            break
        except Exception as exc:
            score.runs.append(Run(n, status="error", message=f"{type(exc).__name__}: {exc}"[:200]))
            break
        error = exact_error(problem, result)
        run = Run(n, error, result.runtime, result.work, hit=error <= bench.tol)
        score.runs.append(run)
        if run.hit:
            for _ in range(repeat - 1):
                run.runtime = min(run.runtime, spec.run(problem, n).runtime)
            break
        n *= 2
    return score


def score_agent(bench: BenchProblem, home=None, time_budget: float = 60.0) -> AgentScore:
    """Solve with a fresh agent: no experience is read or written, learned skills are loaded."""
    from .agent import PDEAgent

    agent = PDEAgent(home=home, remember=False, time_budget=time_budget)
    report = agent.solve(bench.problem, bench.tol, learn=False)
    return AgentScore(report.success, report.method, report.n, report.error, report.runtime,
                      report.solver_runs, "" if report.success else report.message)


def run_benchmark(
    problems: list[BenchProblem],
    registry: SolverRegistry,
    methods: list[str] | None = None,
    agent_home=None,
    with_agent: bool = True,
    time_budget: float = 20.0,
    repeat: int = 3,
):
    """Score every applicable method (and the agent) on each problem; yields ProblemScore."""
    for bench in problems:
        candidates = [s for s in registry.for_problem(bench.problem) if not methods or s.name in methods]
        score = ProblemScore(bench, [score_method(spec, bench, time_budget, repeat) for spec in candidates])
        if with_agent:
            score.agent = score_agent(bench, agent_home, time_budget=3 * time_budget)
        yield score
