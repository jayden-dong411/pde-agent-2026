"""Skill library: solver code the agent learned, admitted only after checks.

A skill is one Python module (format: ``templates/reaction_diffusion_skill.py``)
that defines ``SOLVER`` + ``solve`` and, for a new kind of equation,
``EQUATION`` + ``operator``. Before a skill is admitted it must pass:

1. a static check (imports limited to numpy/scipy/sympy/math/pde_agent, no
   file, process or reflection builtins), and
2. a numerical exam in a separate process: on manufactured problems with
   known solutions the error has to shrink at the order the skill claims.

ODE skills (``kind`` "ode", for ``y' = f(t, y)``) take the same exam on ODE
systems with 32, 64 and 128 steps: two standard ones with exact solutions
(logistic growth, an oscillator) and the task's own system on at most
``[0, 2]``, against a tightly toleranced Radau run, so the test keeps the
task's stiffness and nonlinearity.

Learned code never runs in the agent's own process (``sandbox.py``): the exam
and every later solve run in a child with a minimal environment (no API keys),
CPU, memory and file-size limits and a wall-clock timeout. The agent itself
only uses what the exam reported about a skill (its ``SOLVER`` metadata, and
for a new equation the operator read from its checked text), recorded in the
manifest together with a hash of the code: a file whose code changed since is
examined again before it is used.

The static check and the limited child process limit accidents; they are not
a security sandbox. Learned code runs on this machine, so review skills you
keep and run the agent where that is acceptable.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np

from . import sandbox
from .equations import Equation, EquationBook, builtin_equations, check_operator, manufactured_problem, operator_from_text
from .problem import BOUNDARY_KINDS, PDEProblem, parse_expr
from .solvers.base import STEP_OVERHEAD, SolverDiverged, SolverRegistry, SolverSpec
from .verify import convergence_study

TEMPLATE = Path(__file__).parent / "templates" / "reaction_diffusion_skill.py"
ODE_TEMPLATE = Path(__file__).parent / "templates" / "ode_skill.py"

ALLOWED_IMPORTS = {"numpy", "scipy", "sympy", "math", "functools", "itertools", "typing", "__future__"}
# of the package itself only the helper surface: other modules import os and subprocess, which a skill
# could otherwise pick up with "from pde_agent.skills import subprocess"
PACKAGE_IMPORTS = {"pde_agent.skill_api"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
}
FORBIDDEN_ATTRIBUTES = {
    "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt",
    "fromfile", "tofile", "memmap", "ctypeslib", "system", "popen", "DataSource",
}

STUDY_GRIDS = {0: [32, 64, 128], 1: [16, 32, 64], 2: [8, 16, 32], 3: [4, 8, 16]}  # dim 0: ODE time steps
# raise when the exam gets stricter: installed skills with an older version are examined again
# 3: the exam runs in the limited child and reports the metadata the agent uses (v2.2.0)
EXAM_VERSION = 3
# what a SOLVER may declare for the method selector (see SolverSpec): traits it has, the step limit of an
# explicit ODE method, a cost model and which problems of its class it accepts
SKILL_TRAITS = {"adaptive", "dissipative", "nonoscillatory", "neural"}
COST_STEPS = {"none", "fixed", "h", "h2"}  # time steps ~ t / (cfl * h**p), or n steps (ODE), or none (steady)
MAX_FINAL_ERROR = 5e-2
_SOLVER_NAME = re.compile(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*")


@dataclass
class SkillCandidate:
    code: str
    test_solutions: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class SkillTask:
    """What the agent needs a new method for."""

    kind: str
    dim: int
    boundary: str
    equation: str
    coefficients: dict[str, float]
    domain: list[tuple[float, float]]
    t_final: float | None
    time_order: int | None
    new_equation: bool
    existing_methods: list[dict] = field(default_factory=list)
    reason: str = ""
    rhs: list[str] | None = None  # ODE tasks: the right-hand side and initial values the exam builds tests from
    initial: list[str] | None = None
    stiffness: float | None = None

    @classmethod
    def from_problem(
        cls, problem: PDEProblem, equations: EquationBook, registry: SolverRegistry, reason: str = ""
    ) -> SkillTask:
        if problem.kind == "ode":
            try:
                stiffness = round(float(problem.stiffness()), 3)
            except Exception:
                stiffness = None
            return cls(
                kind="ode", dim=0, boundary="ivp",
                equation="; ".join(f"y{i + 1}' = {e}" for i, e in enumerate(problem.rhs)),
                coefficients=dict(problem.coefficients), domain=[], t_final=problem.t_final, time_order=1,
                new_equation=False,
                existing_methods=[
                    {"name": s.name, "order": s.order, "dims": list(s.dims), "boundaries": list(s.boundaries),
                     "description": s.description}
                    for s in registry if s.kind == "ode"
                ],
                reason=reason, rhs=list(problem.rhs), initial=list(problem.initial), stiffness=stiffness,
            )
        eq = equations.get(problem.kind)
        if eq is not None:
            time_order = eq.time_order
        elif problem.t_final is None:
            time_order = 0
        else:
            time_order = None  # the skill declares it (1 or 2)
        return cls(
            kind=problem.kind,
            dim=problem.dim,
            boundary=problem.boundary,
            equation=eq.form if eq and eq.origin == "builtin" else problem.equation or (eq.form if eq else ""),
            coefficients={**(eq.defaults if eq else {}), **problem.coefficients},
            domain=[tuple(d) for d in problem.domain],
            t_final=problem.t_final,
            time_order=time_order,
            new_equation=eq is None,
            existing_methods=[
                {"name": s.name, "order": s.order, "dims": list(s.dims), "boundaries": list(s.boundaries),
                 "description": s.description}
                for s in registry if s.kind == problem.kind
            ],
            reason=reason,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["domain"] = [list(d) for d in self.domain]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SkillTask:
        data = dict(data)
        data["domain"] = [tuple(d) for d in data["domain"]]
        return cls(**data)


@dataclass
class ValidationReport:
    ok: bool
    message: str
    solver: str | None = None
    kind: str | None = None
    order: float | None = None
    cases: list[dict] = field(default_factory=list)
    meta: dict | None = None  # what the agent may use about the skill: {"solver": ..., "equation": ...}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ValidationReport:
        known = {"ok", "message", "solver", "kind", "order", "cases", "meta"}
        return cls(**{k: v for k, v in data.items() if k in known})

    def feedback(self) -> str:
        """Plain-text account of the exam, for the model to repair its code."""
        lines = [self.message]
        for case in self.cases:
            errors = ", ".join(f"{e:.3e}" for e in case.get("errors", []))
            orders = ", ".join("n/a" if o is None else f"{o:.2f}" for o in case.get("orders", []))
            lines.append(f"- exact u = {case['exact']}: n = {case.get('ns')}, errors = [{errors}], observed orders = [{orders}]")
            if case.get("error"):
                lines.append(f"  failure: {case['error']}")
        return "\n".join(lines)


# --- static check ---------------------------------------------------------------


def static_check(code: str) -> list[str]:
    """Problems that disqualify the code before it is ever run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            for alias in node.names:
                if alias.name.startswith("pde_agent") and alias.asname is None:
                    # "import pde_agent.skill_api" binds the package, and through it every loaded module
                    problems.append(f"line {node.lineno}: write 'from pde_agent.skill_api import ...' instead")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                modules = ["<relative>"]
            elif node.module == "pde_agent":  # from pde_agent import skill_api
                modules = [f"pde_agent.{alias.name}" for alias in node.names]
            else:
                modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            root = module.split(".")[0]
            allowed = module in PACKAGE_IMPORTS if root == "pde_agent" else root in ALLOWED_IMPORTS
            if not allowed:
                problems.append(f"line {node.lineno}: import of {module!r} is not allowed")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: use of {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES):
            problems.append(f"line {node.lineno}: attribute {node.attr!r} is not allowed")
    for name in ("SOLVER", "solve"):
        if not any(
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == name)
            for n in tree.body
        ):
            problems.append(f"module must define {name}")
    return problems


# --- loading --------------------------------------------------------------------


def load_module(path: Path) -> ModuleType:
    name = f"pde_agent_skill_{Path(path).stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling look modules up here
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def equation_from_module(module: ModuleType) -> Equation | None:
    data = getattr(module, "EQUATION", None)
    if data is None:
        return None
    operator = getattr(module, "operator", None)
    if not callable(operator):
        raise ValueError("EQUATION is defined but operator(u, x, y, t, c) is missing")
    kind = data.get("kind", "")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", kind):
        raise ValueError(f"EQUATION kind must be snake_case, got {kind!r}")
    time_order = int(data.get("time_order", -1))
    if time_order not in (0, 1, 2):
        raise ValueError("EQUATION time_order must be 0, 1 or 2")
    defaults = {str(k): float(v) for k, v in dict(data.get("defaults", {})).items()}
    return Equation(
        kind=kind,
        form=str(data.get("form", "")),
        time_order=time_order,
        operator=operator,
        defaults=defaults,
        description=str(data.get("description", "")),
        origin="learned",
    )


def solver_meta(data) -> dict:
    """The checked, JSON-plain form of a ``SOLVER`` dict; raises ``ValueError`` for anything malformed.

    Besides the required name, kind, dims, boundaries and order a skill may declare, for the method selector:

    * ``traits``: any of ``adaptive`` (n sets a tolerance), ``dissipative``, ``nonoscillatory``, ``neural``;
    * ``stability``: for an explicit ODE method, the largest stable |lambda| dt;
    * ``cost``: ``{"steps": "h" | "h2" | "fixed" | "none", "cfl": c, "per_node": w}``, the work model: time
      steps t / (cfl h) or t / (cfl h^2) (PDE), n steps (ODE), none (steady); w units of work per node and step;
    * ``accepts``: ``{"t_final_max": T, "coefficients": {"D": [lo, hi]}}``, the problems it may be given.

    Without ``cost`` a conservative explicit model (``steps`` "h", cfl 1) is used, so an untried skill is not
    ranked as free.
    """
    if not isinstance(data, dict):
        raise ValueError("module must define SOLVER (a dict) and solve(problem, n, options)")
    name = str(data.get("name", ""))
    if not _SOLVER_NAME.fullmatch(name):
        raise ValueError(f"SOLVER name must look like 'kind.method', got {name!r}")
    kind = str(data.get("kind", ""))
    if name.split(".")[0] != kind:
        raise ValueError("SOLVER name must start with its kind")
    dims = [int(d) for d in data.get("dims", ())]
    boundaries = [str(b) for b in data.get("boundaries", ())]
    if kind == "ode":  # y' = f(t, y): no space, an initial value problem
        if dims != [0] or boundaries != ["ivp"]:
            raise ValueError('an ODE SOLVER has "dims": [0] and "boundaries": ["ivp"]')
    else:
        if not dims or any(d not in (1, 2, 3) for d in dims):
            raise ValueError("SOLVER dims must be a non-empty list drawn from 1, 2, 3")
        if not boundaries or any(b not in BOUNDARY_KINDS for b in boundaries):
            raise ValueError(f"SOLVER boundaries must be drawn from {BOUNDARY_KINDS}")
    order = float(data.get("order", 0))
    if not (math.isfinite(order) and order > 0):
        raise ValueError("SOLVER order must be positive")
    traits = sorted({str(t) for t in data.get("traits", ())})
    unknown = set(traits) - SKILL_TRAITS
    if unknown:
        raise ValueError(f"unknown SOLVER traits {sorted(unknown)}; allowed: {sorted(SKILL_TRAITS)}")
    stability = data.get("stability")
    if stability is not None:
        stability = float(stability)
        if not (math.isfinite(stability) and stability > 0) or kind != "ode":
            raise ValueError("SOLVER stability must be a positive number, and only an ODE method has one")
    cost = dict(data.get("cost") or {})
    steps = str(cost.get("steps", "fixed" if kind == "ode" else "h"))
    cfl, per_node = float(cost.get("cfl", 1.0)), float(cost.get("per_node", 1.0))
    if steps not in COST_STEPS or not (math.isfinite(cfl) and cfl > 0) or not (math.isfinite(per_node) and per_node > 0):
        raise ValueError(f'SOLVER cost must be {{"steps": one of {sorted(COST_STEPS)}, "cfl" > 0, "per_node" > 0}}')
    accepts = dict(data.get("accepts") or {})
    t_max = accepts.get("t_final_max")
    if t_max is not None:
        t_max = float(t_max)
        if not (math.isfinite(t_max) and t_max > 0):
            raise ValueError("SOLVER accepts.t_final_max must be positive")
    ranges = {}
    for coefficient, bounds in dict(accepts.get("coefficients") or {}).items():
        lo, hi = (float(b) for b in bounds)
        if not (lo <= hi):
            raise ValueError(f"SOLVER accepts.coefficients.{coefficient} must be [low, high]")
        ranges[str(coefficient)] = [lo, hi]
    return {
        "name": name, "kind": kind, "dims": dims, "boundaries": boundaries, "order": order,
        "description": str(data.get("description", ""))[:500], "traits": traits, "stability": stability,
        "cost": {"steps": steps, "cfl": cfl, "per_node": per_node},
        "accepts": {"t_final_max": t_max, "coefficients": ranges},
    }


def equation_meta(equation: Equation | None) -> dict | None:
    if equation is None:
        return None
    return {"kind": equation.kind, "form": equation.form, "time_order": equation.time_order,
            "defaults": dict(equation.defaults), "description": equation.description}


def solver_from_module(module: ModuleType) -> SolverSpec:
    """The skill as a SolverSpec that runs in this process (only inside the exam's child process)."""
    meta = solver_meta(getattr(module, "SOLVER", None))
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise ValueError("module must define SOLVER (a dict) and solve(problem, n, options)")
    return replace(spec_from_meta(meta), solve=solve)


def _cost_model(meta: dict):
    model = meta["cost"]
    steps, cfl, per_node = model["steps"], model["cfl"], model["per_node"]

    def cost(problem, n: int) -> float:
        if problem.dim == 0:  # an ODE: n steps of the whole state
            return float(n * per_node * (problem.size + STEP_OVERHEAD))
        nodes = (n + 1) ** problem.dim
        if steps == "none" or problem.t_final is None:
            return float(per_node * nodes)
        if steps == "fixed":
            count = n
        else:
            h = min(b - a for a, b in problem.domain) / n
            count = math.ceil(problem.t_final / (cfl * (h if steps == "h" else h * h)) - 1e-9)
        return float(max(1, count) * per_node * (nodes + STEP_OVERHEAD))

    return cost


def _accepts(meta: dict):
    accepts = meta["accepts"]
    t_max, ranges = accepts.get("t_final_max"), accepts.get("coefficients") or {}
    if t_max is None and not ranges:
        return None

    def check(problem) -> bool:
        if t_max is not None and problem.t_final is not None and problem.t_final > t_max * (1 + 1e-12):
            return False
        for name, (lo, hi) in ranges.items():
            value = problem.coefficients.get(name)
            if value is not None and not (lo <= value <= hi):
                return False
        return True

    return check


def _not_here(problem, n, options):
    raise RuntimeError("a learned skill runs only in its own process (sandbox.run_skill)")


def spec_from_meta(meta: dict, path: Path | None = None, examined: tuple[int, str] | None = None,
                   timeout: float = 300.0) -> SolverSpec:
    """A SolverSpec from checked metadata. With ``path`` it runs the skill in a limited child process.

    ``examined`` (dim, boundary): the only case the exam covered; the method is offered for that case only.
    """
    meta = solver_meta(meta)
    dims, boundaries = tuple(meta["dims"]), tuple(meta["boundaries"])
    if examined is not None:
        dim, boundary = examined
        if dim in dims and boundary in boundaries:
            dims, boundaries = (dim,), (boundary,)
    runner = None
    if path is not None:
        path = Path(path)

        def runner(problem, n, options, path=path):
            return sandbox.run_skill(path, problem, n, options, timeout, sandbox.Limits(cpu_seconds=timeout))
    return SolverSpec(
        name=meta["name"], kind=meta["kind"], dims=dims, boundaries=boundaries, order=meta["order"],
        solve=_not_here, description=meta["description"], origin="learned", cost=_cost_model(meta),
        traits=tuple(meta["traits"]), stability=meta["stability"], accepts=_accepts(meta), runner=runner,
    )


def equation_from_meta(meta: dict, coefficient_names) -> Equation:
    """A learned equation for this process: its operator is read from its form, the text the exam checked
    against the skill's own operator, so the skill's Python never runs here."""
    defaults = {str(k): float(v) for k, v in dict(meta.get("defaults") or {}).items()}
    text = str(meta.get("form", ""))
    reader = operator_from_text(text, set(defaults) | set(coefficient_names))
    if reader.time_order != int(meta.get("time_order", -1)):
        raise ValueError(f"equation form {text!r} has time order {reader.time_order}, not {meta.get('time_order')}")
    return Equation(kind=str(meta["kind"]), form=text, time_order=int(meta["time_order"]),
                    operator=lambda u, x, y, t, c, reader=reader: reader.apply(u, c), defaults=defaults,
                    description=str(meta.get("description", ""))[:500], origin="learned")


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


# --- the numerical exam -----------------------------------------------------------


def default_test_solutions(time_order: int, boundary: str, domain: list[tuple[float, float]]) -> list[str]:
    """Smooth exact solutions that exercise a scheme without being too easy."""
    dim = len(domain)
    if boundary == "periodic":
        waves = [f"sin(2*pi*{v}/{b - a!r})" for v, (a, b) in zip("xyz", domain)]
        product = "*".join(waves)
        cos_mix = "*".join(f"cos(4*pi*{v}/{b - a!r})" for v, (a, b) in zip("xyz", domain))
        if time_order == 0:
            return [product, f"{product} + 0.5*{cos_mix}"]
        return [f"{product}*cos(t)", f"exp({product})*(1 + t/2)"]
    if dim == 1:
        if time_order == 0:
            return ["sin(pi*x)*exp(x)", "cos(3*x) + x**2"]
        return ["sin(pi*x)*exp(-t)", "cos(2*x)*cos(t) + x*t"]
    if time_order == 0:
        return ["sin(pi*x)*sin(pi*y)*exp(x)", "exp(x*y) + cos(2*y)"]
    return ["sin(pi*x)*sin(pi*y)*exp(-t)", "exp(x + y)*cos(t)"]


def run_exam(module: ModuleType, task: SkillTask | None, test_solutions: list[str]) -> ValidationReport:
    """Validate a loaded skill module in this process; a passing report carries the skill's metadata."""
    report = _run_exam(module, task, test_solutions)
    if report.ok:
        spec = solver_from_module(module)
        if spec.kind == "ode":
            examined = [0, "ivp"]
        else:
            examined = [task.dim, task.boundary] if task is not None else [spec.dims[0], spec.boundaries[0]]
        report.meta = {"solver": solver_meta(module.SOLVER), "equation": equation_meta(equation_from_module(module)),
                       "examined": examined}
    return report


def _run_exam(module: ModuleType, task: SkillTask | None, test_solutions: list[str]) -> ValidationReport:
    try:
        equation = equation_from_module(module)
        spec = solver_from_module(module)
    except Exception as exc:  # malformed module metadata
        return ValidationReport(ok=False, message=f"invalid skill definition: {exc}")
    if spec.kind == "ode" or (task is not None and task.kind == "ode"):
        return run_ode_exam(spec, equation, task)

    book = builtin_equations()
    if equation is not None:
        if equation.kind in book:
            return ValidationReport(ok=False, message=f"{equation.kind!r} is a built-in equation: drop EQUATION and reuse it")
        if equation.kind != spec.kind:
            return ValidationReport(ok=False, message="EQUATION kind and SOLVER kind differ")
        book.add(equation)
    eq = book.get(spec.kind)
    if eq is None:
        return ValidationReport(ok=False, message=f"unknown equation kind {spec.kind!r}: define EQUATION and operator")

    report = ValidationReport(ok=False, message="", solver=spec.name, kind=spec.kind, order=spec.order)
    if task is not None:
        if spec.kind != task.kind:
            report.message = f"skill solves {spec.kind!r} but the task is {task.kind!r}"
            return report
        if task.dim not in spec.dims or task.boundary not in spec.boundaries:
            report.message = f"skill does not cover the task ({task.dim}-D, {task.boundary} boundaries)"
            return report
        if any(m["name"] == spec.name for m in task.existing_methods):
            report.message = f"a solver named {spec.name!r} already exists; pick a new name"
            return report
        if task.time_order is not None and eq.time_order != task.time_order:
            report.message = f"equation time order {eq.time_order} does not match the task ({task.time_order})"
            return report

    dim = task.dim if task else spec.dims[0]
    boundary = task.boundary if task else spec.boundaries[0]
    domain = task.domain if task else [(0.0, 1.0)] * dim
    t_final = (task.t_final if task and task.t_final else 0.5) if eq.time_order else None
    coefficients = task.coefficients if task else {}
    if equation is not None:  # a new equation must be the one the task states, and its form must say what it does
        if task is None or not task.equation.strip():
            report.message = "the task gives no equation text to check EQUATION against"
            return report
        why = check_operator(equation, task.equation, coefficients, dim, domain, t_final)
        if why:
            report.message = f"operator does not match the task equation: {why}"
            return report
        why = check_operator(equation, equation.form, coefficients, dim, domain, t_final)
        if why:
            report.message = f"EQUATION form does not match operator: {why}"
            return report
    exacts = []
    for text in default_test_solutions(eq.time_order, boundary, domain) + list(test_solutions):  # built-in tests always
        if text not in exacts:
            exacts.append(text)

    ns = STUDY_GRIDS[dim]
    for exact in exacts[:4]:
        case: dict = {"exact": exact, "ns": ns}
        report.cases.append(case)
        try:
            parse_expr(exact)
            problem = manufactured_problem(eq, exact, domain, t_final, coefficients, boundary)
            study = convergence_study(spec, problem, ns)
        except Exception as exc:
            return _crashed(report, case, exc)
        if not _graded(report, case, study, spec):
            return report
    report.ok = True
    report.message = f"passed {len(report.cases)} manufactured-solution tests"
    return report


def _crashed(report: ValidationReport, case: dict, exc: Exception) -> ValidationReport:
    case["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()[-600:]
    report.message = "the solver raised an error on a test problem"
    return report


def _graded(report: ValidationReport, case: dict, study, spec: SolverSpec, overall: bool = False) -> bool:
    """Record one convergence study; False (with the reason in ``report.message``) if it fails.

    ``overall``: judge the order from the first to the last run instead of the last pair (ODE methods
    with adaptive steps do not converge evenly in their tolerance knob).
    """
    case["errors"], case["orders"] = study.errors, study.orders
    final_error, final_order = study.errors[-1], study.final_order
    if overall and len(study.errors) > 1 and study.errors[0] > 0 and final_error > 0:
        final_order = math.log(study.errors[0] / final_error) / math.log(study.ns[-1] / study.ns[0])
    if not math.isfinite(final_error) or final_error > MAX_FINAL_ERROR:
        report.message = f"error on the finest grid is {final_error:.3e} (needs <= {MAX_FINAL_ERROR})"
        return False
    if final_error > 1e-10:  # below this the scheme is exact up to round-off
        needed = max(0.8, 0.7 * spec.order)
        if final_order is None or final_order < needed:
            report.message = f"observed order {final_order} is below {needed:.2f} (claimed order {spec.order})"
            return False
    return True


# --- the ODE exam -----------------------------------------------------------------

ODE_STANDARD_TESTS = [  # (rhs, initial, t_final, exact)
    (["y1*(1 - y1)"], ["0.5"], 2.0, ["1/(1 + exp(-t))"]),
    (["y2", "-y1"], ["1", "0"], 2.0, ["cos(t)", "-sin(t)"]),
]


ODE_TEST_SPAN = 2.0
ODE_REFERENCE_RTOL = 1e-10


def ode_test_problems(task: SkillTask | None) -> list[tuple]:
    """``(problem, reference)`` pairs: the standard tests (exact solutions, reference None) and the task's own
    problem on at most ``[0, 2]`` with a tight Radau run as the reference (when the task has one)."""
    from .ode import ODEProblem

    tests = [(ODEProblem(rhs=r, initial=i, t_final=tf, exact=e, name="standard"), None)
             for r, i, tf, e in ODE_STANDARD_TESTS]
    if task is None or not task.rhs or not task.initial or not task.t_final:
        return tests
    try:
        # a long horizon would need more steps than the exam takes
        problem = ODEProblem(rhs=task.rhs, initial=task.initial, t_final=min(task.t_final, ODE_TEST_SPAN),
                             coefficients=task.coefficients, name="task")
        reference = _reference(problem)
    except Exception:
        return tests
    if reference is not None:
        tests.append((problem, reference))
    return tests


def _reference(problem) -> np.ndarray | None:
    from scipy.integrate import solve_ivp

    scale = max(1.0, float(np.max(np.abs(problem.y0))))
    with np.errstate(all="ignore"):
        sol = solve_ivp(problem.f, (0.0, problem.t_final), problem.y0, method="Radau", jac=problem.jac,
                        rtol=ODE_REFERENCE_RTOL, atol=ODE_REFERENCE_RTOL * scale * 1e-4)
    if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
        return None
    return sol.y[:, -1]


def _ode_study(spec: SolverSpec, problem, reference: np.ndarray | None, ns: list[int]):
    from .verify import ConvergenceStudy, observed_order, relative_error

    if reference is None:
        return convergence_study(spec, problem, ns)
    errors = [relative_error(spec.run(problem, n).u, reference) for n in ns]
    study = ConvergenceStudy(method=spec.name, ns=list(ns), errors=errors)
    study.orders = [observed_order(e0, e1, n1 / n0) for e0, e1, n0, n1 in zip(errors, errors[1:], ns, ns[1:])]
    return study


def run_ode_exam(spec: SolverSpec, equation: Equation | None, task: SkillTask | None) -> ValidationReport:
    report = ValidationReport(ok=False, message="", solver=spec.name, kind=spec.kind, order=spec.order)
    if spec.kind != "ode":
        report.message = f"skill solves {spec.kind!r} but the task is an ODE (kind 'ode')"
        return report
    if equation is not None:
        report.message = "an ODE skill must not define EQUATION or operator"
        return report
    if task is not None and any(m["name"] == spec.name for m in task.existing_methods):
        report.message = f"a solver named {spec.name!r} already exists; pick a new name"
        return report
    ns = STUDY_GRIDS[0]
    for problem, reference in ode_test_problems(task):
        label = "; ".join(problem.exact) if reference is None else \
            f"the task's own ODE on [0, {problem.t_final:g}], reference = tight Radau run"
        case: dict = {"exact": label, "ns": ns}
        report.cases.append(case)
        try:
            study = _ode_study(spec, problem, reference, ns)
        except Exception as exc:
            return _crashed(report, case, exc)
        if not _graded(report, case, study, spec, overall=True):
            return report
    report.ok = True
    report.message = f"passed {len(report.cases)} ODE tests with known solutions"
    return report


def validate_file(path: Path, task: SkillTask | None, test_solutions: list[str], timeout: float = 300.0) -> ValidationReport:
    """Run the exam on a skill file in a separate, limited Python process (see ``sandbox.py``)."""
    payload = {"path": str(Path(path).resolve()), "task": task.to_dict() if task else None,
               "test_solutions": list(test_solutions)}
    try:
        answer = sandbox.call("pde_agent.skills:_child_main", payload, timeout, sandbox.Limits(cpu_seconds=timeout))
    except sandbox.SkillRunError as exc:
        return ValidationReport(ok=False, message=f"validation process failed: {exc}")
    try:
        return ValidationReport.from_dict(answer)
    except TypeError as exc:
        return ValidationReport(ok=False, message=f"validation process answered nonsense: {exc}")


def validate_candidate(candidate: SkillCandidate, task: SkillTask | None, timeout: float = 300.0) -> ValidationReport:
    problems = static_check(candidate.code)
    if problems:
        return ValidationReport(ok=False, message="static check failed:\n" + "\n".join(problems))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate_skill.py"
        path.write_text(candidate.code, encoding="utf-8")
        return validate_file(path, task, candidate.test_solutions, timeout)


def _child_main() -> None:
    """The exam, in the limited child process."""
    payload, answer = sandbox.read_request()
    try:
        module = load_module(Path(payload["path"]))
    except BaseException:
        report = ValidationReport(ok=False, message="importing the module failed:\n" + traceback.format_exc(limit=3)[-1500:])
    else:
        task = SkillTask.from_dict(payload["task"]) if payload.get("task") else None
        try:
            report = run_exam(module, task, payload.get("test_solutions", []))
        except BaseException as exc:  # SystemExit and friends from the skill: still an answer
            report = ValidationReport(ok=False, message=f"the exam stopped: {type(exc).__name__}: {exc}"[:1500])
    sandbox.write_answer(answer, payload, report.to_dict())


def _run_main() -> None:
    """One solve of a learned skill, in the limited child process."""
    from .ode import problem_from_dict

    payload, answer = sandbox.read_request()
    try:
        module = load_module(Path(payload["path"]))
        solve = getattr(module, "solve")
        problem = problem_from_dict(payload["problem"]).for_solver()
        result = solve(problem, int(payload["n"]), dict(payload.get("options") or {}))
        data = {"ok": True, "result": sandbox.result_to_json(result)}
    except SolverDiverged as exc:
        data = {"ok": False, "error": "SolverDiverged", "message": str(exc)[:600]}
    except BaseException as exc:
        data = {"ok": False, "error": type(exc).__name__, "message": str(exc)[:600]}
    sandbox.write_answer(answer, payload, data)


# --- the library on disk --------------------------------------------------------------


class SkillLibrary:
    """Installed skills: ``<name>.py`` with a ``<name>.json`` manifest (exam result, metadata, code hash).

    ``timeout``: the wall-clock limit of one solve by a learned skill (its own process).
    """

    def __init__(self, root: Path | str, timeout: float = 300.0):
        self.root = Path(root)
        self.timeout = timeout

    def manifests(self) -> list[dict]:
        if not self.root.exists():
            return []
        items = []
        for path in sorted(self.root.glob("*.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(manifest, dict):
                manifest["_manifest"] = path.name
                items.append(manifest)
        return sorted(items, key=lambda m: m.get("created", ""))

    def _path(self, manifest: dict) -> Path:
        """The skill file a manifest names, which has to sit in this library (no ``../`` or absolute paths)."""
        name = str(manifest.get("file", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.py", name) or name.startswith("."):
            raise ValueError(f"manifest names an invalid skill file {name!r}")
        return self.root / name

    def install(self, candidate: SkillCandidate, report: ValidationReport, task: SkillTask | None, origin: str = "llm") -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        stem = (report.solver or "skill").replace(".", "__")
        path, version = self.root / f"{stem}.py", 1
        while path.exists():
            version += 1
            path = self.root / f"{stem}_v{version}.py"
        path.write_text(candidate.code, encoding="utf-8")
        manifest = {
            "solver": report.solver,
            "kind": report.kind,
            "file": path.name,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "origin": origin,
            "summary": candidate.summary,
            "task": task.to_dict() if task else None,
            "validation": report.to_dict(),
            "exam_version": EXAM_VERSION,
            "code_hash": code_hash(candidate.code),
        }
        path.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_file(self, path: Path, registry: SolverRegistry, equations: EquationBook,
                  manifest: dict | None = None) -> SolverSpec:
        """Register an examined skill from its manifest; its code is not imported here (it runs in a child)."""
        path = Path(path)
        if manifest is None:
            manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        meta = (manifest.get("validation") or {}).get("meta")
        if not isinstance(meta, dict) or not isinstance(meta.get("solver"), dict):
            raise ValueError(f"{path.name}: the manifest has no exam metadata")
        if manifest.get("code_hash") != code_hash(path.read_text(encoding="utf-8")):
            raise ValueError(f"{path.name}: the code changed since its exam")
        examined = meta.get("examined")
        examined = (int(examined[0]), str(examined[1])) if isinstance(examined, list) and len(examined) == 2 else None
        spec = spec_from_meta(meta["solver"], path, examined, self.timeout)
        if meta.get("equation"):
            task = manifest.get("task") or {}
            equation = equation_from_meta(meta["equation"], dict(task.get("coefficients") or {}))
            if equation.kind != spec.kind:
                raise ValueError(f"{path.name}: EQUATION kind and SOLVER kind differ")
            if equation.kind not in equations:
                equations.add(equation)
        if spec.kind != "ode" and spec.kind not in equations:
            raise ValueError(f"{path.name}: unknown equation kind {spec.kind!r}")
        registry.register(spec)
        return spec

    def load_into(self, registry: SolverRegistry, equations: EquationBook, timeout: float = 300.0) -> list[str]:
        """Load every installed skill; returns a sentence for each one that is disabled or failed to load.

        A skill admitted by an older exam, or whose code changed since its exam
        (the manifest keeps a hash of the code), is examined again first. If it
        fails, its manifest gets ``disabled`` (the reason) and it is not loaded
        again until someone deletes that entry.
        """
        errors = []
        for manifest in self.manifests():
            manifest_name = manifest.pop("_manifest")
            try:
                path = self._path(manifest)
            except ValueError as exc:
                errors.append(f"技能清单 {manifest_name} 加载失败：{exc}")
                continue
            if manifest.get("disabled"):
                errors.append(f"技能 {path.name} 已停用：{manifest['disabled']}")
                continue
            try:
                code = path.read_text(encoding="utf-8")
                problems = static_check(code)
                if problems:  # admitted under an older, looser check, or edited since: checked again every start
                    errors.append(f"技能 {path.name} 加载失败：静态检查不通过：{'；'.join(problems[:3])}")
                    continue
                changed = manifest.get("code_hash") not in (None, code_hash(code))
                old = int(manifest.get("exam_version", 1)) < EXAM_VERSION or not manifest.get("code_hash")
                if old or changed:
                    task = SkillTask.from_dict(manifest["task"]) if manifest.get("task") else None
                    tests = [case["exact"] for case in manifest.get("validation", {}).get("cases", [])]
                    report = validate_file(path, task, tests, timeout)
                    if not report.ok:
                        why = f"代码在考试后改过，重新考试没通过：{report.message}" if changed else report.message
                        manifest.update(disabled=why,
                                        disabled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                        self._save_manifest(self.root / manifest_name, manifest)
                        errors.append(f"技能 {path.name} 已停用：{why}")
                        continue
                    manifest.update(exam_version=EXAM_VERSION, validation=report.to_dict(), code_hash=code_hash(code),
                                    examined_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                    manifest.pop("disabled_at", None)
                    self._save_manifest(self.root / manifest_name, manifest)
                self.load_file(path, registry, equations, manifest)
            except Exception as exc:
                errors.append(f"技能 {path.name} 加载失败：{exc}")
        return errors

    @staticmethod
    def _save_manifest(path: Path, manifest: dict) -> None:
        """Replace a manifest in one step; if that fails, the skill is simply examined again next start."""
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass
