"""Claude as the agent's language brain.

Two jobs need language understanding or code writing:

* ``parse_problem``: turn a problem described in words into a ``PDEProblem``;
* ``skill_session``: write a new solver module for a problem class the agent
  cannot handle yet, and repair it from the validation report.

Everything numerical (solving, checking, choosing methods) stays in plain
Python, so the agent works without Claude for the equations it already knows.
Requests use structured outputs, so replies always match a JSON schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .equations import EquationBook
from .problem import PDEProblem
from .skills import TEMPLATE, SkillCandidate, SkillTask, ValidationReport

DEFAULT_MODEL = "claude-opus-5-5"
# Server-side fallback: if the model declines a request, the API retries it
# on a fallback model within the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


PROBLEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string"},
        "equation": {"type": "string"},
        "domain": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"lower": {"type": "number"}, "upper": {"type": "number"}},
                "required": ["lower", "upper"],
                "additionalProperties": False,
            },
        },
        "t_final": {"type": "number"},
        "coefficients": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                "required": ["name", "value"],
                "additionalProperties": False,
            },
        },
        "source": {"type": "string"},
        "boundary": {"type": "string", "enum": ["dirichlet", "periodic"]},
        "boundary_value": {"type": "string"},
        "initial": {"type": "string"},
        "initial_velocity": {"type": "string"},
        "exact": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": [
        "kind", "equation", "domain", "t_final", "coefficients", "source", "boundary",
        "boundary_value", "initial", "initial_velocity", "exact", "notes",
    ],
    "additionalProperties": False,
}

SKILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "test_solutions": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["code", "test_solutions", "summary"],
    "additionalProperties": False,
}

PARSE_SYSTEM = """You translate a PDE problem described in natural language (often Chinese) into the structured problem format of pde-agent.

Known equation kinds (the PDE is written as L(u) = f):
{kinds}

Rules:
- Use a known kind when the equation matches it, and rename coefficients to that kind's names (for example thermal diffusivity -> alpha). Otherwise invent a short snake_case kind and write the equation in `equation`, e.g. "u_t = D*u_xx - r*u + f".
- Expressions use x, y, t, pi and sin, cos, tan, exp, log, sqrt, sinh, cosh, tanh, Abs; write powers with **. Use "" for a field that is not given or does not apply.
- domain: one interval per spatial dimension. t_final: 0 for steady problems.
- boundary: "dirichlet" (values prescribed on the boundary: g(x, y, t) goes in boundary_value; "" means zero, or the exact solution when one is given) or "periodic".
- source: the right-hand side f of L(u) = f, "0" if there is none. Move every term that does not involve u to the source.
- exact: only when the user states the exact solution.
- notes: the assumptions you made, in Chinese."""

SKILL_SYSTEM = """You write numerical PDE solver modules ("skills") for pde-agent, a PDE-solving agent that grows its own library of methods.

Each module is checked statically and then examined on manufactured solutions: an exact u is chosen, f = L(u) is computed symbolically, and the problem is solved on grids n = 16, 32, 64 (1-D), 8, 16, 32 (2-D) or 4, 8, 16 (3-D). The skill is admitted only if the relative max-norm error on the finest grid is at most 5e-2 and the observed convergence order reaches max(0.8, 0.7 * the order you claim). Prefer robust, accurate and efficient schemes, and claim the order you can actually deliver.

Module contract:
- Imports: only numpy, scipy, sympy, math and pde_agent.skill_api. No file, network or process access; no eval, exec or getattr; no attributes starting with an underscore.
- SOLVER = {{"name": "<kind>.<method>", "kind": ..., "dims": [...], "boundaries": ["dirichlet" and/or "periodic"], "order": <expected convergence order in n>, "description": ...}}
- def solve(problem, n, options) -> SolveResult
- For a new equation kind only: EQUATION = {{"kind": ..., "form": ..., "time_order": 0 (steady), 1 (u_t) or 2 (u_tt), "defaults": {{coefficient: value}}, "description": ...}} and def operator(u, x, y, t, c) returning the sympy expression L(u) such that the PDE reads L(u) = f (time derivatives included; c is the coefficient dict).

The problem object (read only):
- problem.domain (list of (a, b) per spatial dimension), problem.dim, problem.t_final (None when steady), problem.boundary, problem.coefficients (dict; read with .get(name, default)).
- problem.fn(name) returns f(x, y=0.0, t=0.0) that broadcasts like numpy, for name in "source", "boundary_value" (Dirichlet data g), "initial", "initial_velocity", "exact".

Grid and result:
- axes = grid(problem, n): n equal intervals, n + 1 nodes per dimension including both ends. For periodic problems the last node repeats the first, so return u[-1] equal to u[0] along each periodic axis.
- Return SolveResult(u=..., grid=axes, n=n, t=problem.t_final, steps=steps, dt=dt, work=nodes*steps) with u of shape (n + 1,) * dim, u[i, j] = u(x_i, y_j) at the final time (meshgrid indexing="ij").
- Choose the time step from stability and accuracy with time_grid(t_final, dt_target) -> (steps, dt); when dt is tied to h, the time error must shrink at least as fast as the claimed order.

Helpers in pde_agent.skill_api: grid, mesh(axes) (meshgrid "ij"), evaluate(fn, points, t), time_grid, laplacian_1d(m, h) (sparse second difference on m interior nodes), laplacian_matrix(m, hs) (interior Laplacian in 1-D/2-D, C ordering), interior_laplacian(u, hs), check_growth(u, limit, name, step), growth_limit(*arrays), SolverDiverged, SolveResult.

A complete example skill:
```python
{template}
```

Reply with the full module source in `code`, up to three extra exact solutions for the exam in `test_solutions` (expressions in x, y, t that respect the boundary type, e.g. periodic in x for periodic problems), and a one-paragraph `summary` of the method in Chinese."""


def _blank(value: str) -> str | None:
    value = (value or "").strip()
    return value or None


def problem_from_reply(data: dict[str, Any]) -> PDEProblem:
    """Build a problem from the parse schema (raises ValueError if invalid)."""
    t_final = float(data.get("t_final") or 0.0)
    return PDEProblem(
        kind=data["kind"].strip(),
        domain=[(d["lower"], d["upper"]) for d in data["domain"]],
        t_final=t_final if t_final > 0 else None,
        coefficients={c["name"]: c["value"] for c in data.get("coefficients", [])},
        source=_blank(data.get("source", "")) or "0",
        boundary=data.get("boundary", "dirichlet"),
        boundary_value=_blank(data.get("boundary_value", "")),
        initial=_blank(data.get("initial", "")),
        initial_velocity=_blank(data.get("initial_velocity", "")),
        exact=_blank(data.get("exact", "")),
        equation=data.get("equation", "").strip(),
        description=data.get("notes", "").strip(),
    )


@dataclass
class ParsedProblem:
    problem: PDEProblem
    notes: str
    raw: dict[str, Any]


class ClaudeBrain:
    def __init__(self, client: Any = None, model: str = DEFAULT_MODEL, effort: str = "high", max_tokens: int = 64000):
        if client is None:
            import anthropic  # optional dependency: pip install "pde-agent[llm]"

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete_json(self, system: str, messages: list[dict], schema: dict, effort: str | None = None) -> tuple[dict, list]:
        """One structured-output request; returns the JSON and the raw content blocks."""
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": effort or self.effort, "format": {"type": "json_schema", "schema": schema}},
                cache_control={"type": "ephemeral"},  # repair rounds resend the same prefix
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:  # credentials, network, rate limits after the SDK's retries
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        if message.stop_reason == "refusal":
            raise LLMError("the model declined this request")
        if message.stop_reason == "max_tokens":
            raise LLMError("the reply was cut off at max_tokens")
        # the last text block: after a server-side fallback, earlier blocks can
        # hold the declining model's partial output
        texts = [b.text for b in message.content if b.type == "text"]
        text = texts[-1] if texts else None
        if text is None:
            raise LLMError("the reply had no text block")
        try:
            return json.loads(text), message.content
        except json.JSONDecodeError as exc:
            raise LLMError(f"the reply was not valid JSON: {exc}") from exc

    def parse_problem(self, text: str, equations: EquationBook, attempts: int = 2) -> ParsedProblem:
        kinds = "\n".join(
            f"- {eq.kind}: {eq.form} (time order {eq.time_order}; coefficients "
            + (", ".join(f"{k}={v:g}" for k, v in eq.defaults.items()) or "none")
            + ")"
            for eq in equations
        )
        system = PARSE_SYSTEM.format(kinds=kinds)
        messages: list[dict] = [{"role": "user", "content": text}]
        for attempt in range(attempts):
            data, content = self.complete_json(system, messages, PROBLEM_SCHEMA, effort="medium")
            try:
                return ParsedProblem(problem=problem_from_reply(data), notes=data.get("notes", ""), raw=data)
            except (ValueError, KeyError, TypeError) as exc:
                if attempt + 1 == attempts:
                    raise LLMError(f"could not build a valid problem: {exc}") from exc
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"That problem is invalid: {exc}. Please correct it."})
        raise AssertionError("unreachable")

    def skill_session(self, task: SkillTask) -> SkillSession:
        return SkillSession(self, task)


class SkillSession:
    """A conversation in which Claude writes a skill and repairs it."""

    def __init__(self, brain: ClaudeBrain, task: SkillTask):
        self.brain = brain
        self.task = task
        self.system = SKILL_SYSTEM.format(template=TEMPLATE.read_text(encoding="utf-8").strip())
        self.messages: list[dict] = [{"role": "user", "content": task_prompt(task)}]

    def propose(self) -> SkillCandidate:
        data, content = self.brain.complete_json(self.system, self.messages, SKILL_SCHEMA)
        self.messages.append({"role": "assistant", "content": content})
        return SkillCandidate(
            code=data["code"],
            test_solutions=[s for s in data.get("test_solutions", []) if isinstance(s, str)][:3],
            summary=data.get("summary", ""),
        )

    def feedback(self, report: ValidationReport) -> None:
        self.messages.append({
            "role": "user",
            "content": "Your module did not pass validation:\n"
            + report.feedback()
            + "\n\nFix the problem and reply with the complete corrected module.",
        })


def task_prompt(task: SkillTask) -> str:
    if task.new_equation:
        kind_line = f"- kind: {task.kind} (a NEW equation: also define EQUATION and operator)"
    else:
        kind_line = f"- kind: {task.kind} (known equation {task.equation!r}: do not define EQUATION)"
    methods = "\n".join(
        f"  - {m['name']} (order {m['order']}, dims {m['dims']}, boundaries {m['boundaries']}): {m['description']}"
        for m in task.existing_methods
    ) or "  - none"
    return "\n".join([
        "Write a solver skill for this problem class.",
        kind_line,
        f"- equation: {task.equation or '(see kind)'}",
        f"- spatial dimension: {task.dim}; boundary: {task.boundary}; domain: {task.domain}",
        f"- final time: {task.t_final if task.t_final else 'steady problem'}",
        f"- time order: {task.time_order if task.time_order is not None else 'decide from the equation'}",
        f"- coefficients: {task.coefficients or 'none'}",
        f"- why a new method is needed: {task.reason or 'no method covers this problem class yet'}",
        "- existing methods for this kind:",
        methods,
        "Choose a solver name (kind.method) that is not used yet.",
    ])
