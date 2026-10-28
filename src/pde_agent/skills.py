"""Skill library: solver code the agent learned, admitted only after checks.

A skill is one Python module (format: ``templates/reaction_diffusion_skill.py``)
that defines ``SOLVER`` + ``solve`` and, for a new kind of equation,
``EQUATION`` + ``operator``. Before a skill is admitted it must pass:

1. a static check (imports limited to numpy/scipy/sympy/math/pde_agent, no
   file, process or reflection builtins), and
2. a numerical exam in a separate process: on manufactured problems with
   known solutions the error has to shrink at the order the skill claims.

The static check and the separate process limit accidents; they are not a
security sandbox. Learned code runs on this machine, so review skills you
keep and run the agent where that is acceptable.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

from .equations import Equation, EquationBook, builtin_equations, check_operator, manufactured_problem
from .problem import BOUNDARY_KINDS, PDEProblem, parse_expr
from .solvers.base import SolverRegistry, SolverSpec
from .verify import convergence_study

TEMPLATE = Path(__file__).parent / "templates" / "reaction_diffusion_skill.py"

ALLOWED_IMPORTS = {"numpy", "scipy", "sympy", "math", "pde_agent", "functools", "itertools", "typing", "__future__"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
}
FORBIDDEN_ATTRIBUTES = {
    "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt",
    "fromfile", "tofile", "memmap", "ctypeslib", "system", "popen", "DataSource",
}

STUDY_GRIDS = {1: [16, 32, 64], 2: [8, 16, 32], 3: [4, 8, 16]}
EXAM_VERSION = 2  # raise when the exam gets stricter: installed skills with an older version are examined again
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

    @classmethod
    def from_problem(
        cls, problem: PDEProblem, equations: EquationBook, registry: SolverRegistry, reason: str = ""
    ) -> SkillTask:
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ValidationReport:
        return cls(**data)

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
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]] if node.level == 0 else ["<relative>"]
        else:
            roots = []
        for root in roots:
            if root not in ALLOWED_IMPORTS:
                problems.append(f"line {node.lineno}: import of {root!r} is not allowed")
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


def solver_from_module(module: ModuleType) -> SolverSpec:
    data = getattr(module, "SOLVER", None)
    solve = getattr(module, "solve", None)
    if not isinstance(data, dict) or not callable(solve):
        raise ValueError("module must define SOLVER (a dict) and solve(problem, n, options)")
    name = str(data.get("name", ""))
    if not _SOLVER_NAME.fullmatch(name):
        raise ValueError(f"SOLVER name must look like 'kind.method', got {name!r}")
    kind = str(data.get("kind", ""))
    if name.split(".")[0] != kind:
        raise ValueError("SOLVER name must start with its kind")
    dims = tuple(int(d) for d in data.get("dims", ()))
    if not dims or any(d not in (1, 2, 3) for d in dims):
        raise ValueError("SOLVER dims must be a non-empty list drawn from 1, 2, 3")
    boundaries = tuple(str(b) for b in data.get("boundaries", ()))
    if not boundaries or any(b not in BOUNDARY_KINDS for b in boundaries):
        raise ValueError(f"SOLVER boundaries must be drawn from {BOUNDARY_KINDS}")
    order = float(data.get("order", 0))
    if not (math.isfinite(order) and order > 0):
        raise ValueError("SOLVER order must be positive")
    return SolverSpec(
        name=name, kind=kind, dims=dims, boundaries=boundaries, order=order, solve=solve,
        description=str(data.get("description", "")), origin="learned",
    )


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
    """Validate a loaded skill module in this process."""
    try:
        equation = equation_from_module(module)
        spec = solver_from_module(module)
    except Exception as exc:  # malformed module metadata
        return ValidationReport(ok=False, message=f"invalid skill definition: {exc}")

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
            case["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()[-600:]
            report.message = "the solver raised an error on a test problem"
            return report
        case["errors"], case["orders"] = study.errors, study.orders
        final_error, final_order = study.errors[-1], study.final_order
        if not math.isfinite(final_error) or final_error > MAX_FINAL_ERROR:
            report.message = f"error on the finest grid is {final_error:.3e} (needs <= {MAX_FINAL_ERROR})"
            return report
        if final_error > 1e-10:  # below this the scheme is exact up to round-off
            needed = max(0.8, 0.7 * spec.order)
            if final_order is None or final_order < needed:
                report.message = f"observed order {final_order} is below {needed:.2f} (claimed order {spec.order})"
                return report
    report.ok = True
    report.message = f"passed {len(report.cases)} manufactured-solution tests"
    return report


def validate_file(path: Path, task: SkillTask | None, test_solutions: list[str], timeout: float = 300.0) -> ValidationReport:
    """Run the exam on a skill file in a separate Python process."""
    payload = json.dumps({
        "path": str(Path(path).resolve()),
        "task": task.to_dict() if task else None,
        "test_solutions": list(test_solutions),
    })
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH", "")) if p)
    with tempfile.TemporaryDirectory() as scratch:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", "from pde_agent.skills import _child_main; _child_main()"],
                input=payload, capture_output=True, text=True, timeout=timeout, env=env, cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            return ValidationReport(ok=False, message=f"validation timed out after {timeout:.0f} s (too slow?)")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_REPORT_PREFIX):
            return ValidationReport.from_dict(json.loads(line[len(_REPORT_PREFIX):]))
    tail = (proc.stderr or proc.stdout).strip()[-1500:]
    return ValidationReport(ok=False, message=f"validation process failed (exit {proc.returncode}): {tail}")


def validate_candidate(candidate: SkillCandidate, task: SkillTask | None, timeout: float = 300.0) -> ValidationReport:
    problems = static_check(candidate.code)
    if problems:
        return ValidationReport(ok=False, message="static check failed:\n" + "\n".join(problems))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate_skill.py"
        path.write_text(candidate.code, encoding="utf-8")
        return validate_file(path, task, candidate.test_solutions, timeout)


_REPORT_PREFIX = "@@pde-agent-report@@"


def _child_main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        module = load_module(Path(payload["path"]))
    except Exception:
        report = ValidationReport(ok=False, message="importing the module failed:\n" + traceback.format_exc(limit=3)[-1500:])
    else:
        task = SkillTask.from_dict(payload["task"]) if payload.get("task") else None
        report = run_exam(module, task, payload.get("test_solutions", []))
    print(_REPORT_PREFIX + json.dumps(report.to_dict()), flush=True)


# --- the library on disk --------------------------------------------------------------


class SkillLibrary:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def manifests(self) -> list[dict]:
        if not self.root.exists():
            return []
        items = []
        for path in sorted(self.root.glob("*.json")):
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(items, key=lambda m: m.get("created", ""))

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
        }
        path.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_file(self, path: Path, registry: SolverRegistry, equations: EquationBook) -> SolverSpec:
        module = load_module(path)
        equation = equation_from_module(module)
        spec = solver_from_module(module)
        if equation is not None and equation.kind not in equations:
            equations.add(equation)
        if spec.kind not in equations:
            raise ValueError(f"{path.name}: unknown equation kind {spec.kind!r}")
        registry.register(spec)
        return spec

    def load_into(self, registry: SolverRegistry, equations: EquationBook, timeout: float = 300.0) -> list[str]:
        """Load every installed skill; returns a sentence for each one that is disabled or failed to load.

        A skill admitted by an older exam is examined again first. If it fails,
        its manifest gets ``disabled`` (the reason) and it is not loaded again
        until someone deletes that entry.
        """
        errors = []
        for manifest in self.manifests():
            path = self.root / manifest.get("file", "")
            if manifest.get("disabled"):
                errors.append(f"技能 {path.name} 已停用：{manifest['disabled']}")
                continue
            try:
                if int(manifest.get("exam_version", 1)) < EXAM_VERSION:
                    task = SkillTask.from_dict(manifest["task"]) if manifest.get("task") else None
                    tests = [case["exact"] for case in manifest.get("validation", {}).get("cases", [])]
                    report = validate_file(path, task, tests, timeout)
                    if not report.ok:
                        manifest.update(disabled=report.message,
                                        disabled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                        self._save_manifest(path.with_suffix(".json"), manifest)
                        errors.append(f"技能 {path.name} 已停用：{report.message}")
                        continue
                    manifest.update(exam_version=EXAM_VERSION, validation=report.to_dict())
                    manifest.pop("disabled_at", None)
                    self._save_manifest(path.with_suffix(".json"), manifest)
                self.load_file(path, registry, equations)
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

