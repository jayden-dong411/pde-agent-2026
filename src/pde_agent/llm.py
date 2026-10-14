"""A language model as the agent's brain: Claude or DeepSeek.

Two jobs need language understanding or code writing:

* ``parse_problem``: turn a problem described in words into a ``PDEProblem``;
* ``skill_session``: write a new solver module for a problem class the agent
  cannot handle yet, and repair it from the validation report.

Everything numerical (solving, checking, choosing methods) stays in plain
Python, so the agent works without a language model for the equations it
already knows. ``Brain`` holds the prompts and the conversations; a subclass
only makes the API call. Claude uses structured outputs, so its replies always
match the JSON schema; DeepSeek uses JSON output mode, and its replies are
checked against the schema here and asked for again when they do not match.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .equations import EquationBook
from .problem import PDEProblem
from .skills import TEMPLATE, SkillCandidate, SkillTask, ValidationReport

DEFAULT_MODEL = "claude-opus-5-5"
# Server-side fallback: if the model declines a request, the API retries it
# on a fallback model within the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MAX_OUTPUT_TOKENS = 393216  # official Chat Completions output ceiling
DEEPSEEK_PARSE_TOKENS = 16384
ANSWER_TOKENS = 32768  # free-form answers in the web page's math Q&A
# DeepSeek's thinking effort has the levels low, high and max
# Brain.parse_problem uses our internal medium level; parsing needs less reasoning.
_DEEPSEEK_EFFORT = {"low": "low", "medium": "low", "high": "high", "max": "max"}

# provider -> (display name, API key variable)
PROVIDERS = {"claude": ("Claude", "ANTHROPIC_API_KEY"), "deepseek": ("DeepSeek", "DEEPSEEK_API_KEY")}


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

# Shown to models that only get the schema in the prompt (DeepSeek)
PROBLEM_EXAMPLE: dict[str, Any] = {
    "kind": "heat", "equation": "u_t = alpha*u_xx", "domain": [{"lower": 0, "upper": 1}], "t_final": 0.5,
    "coefficients": [{"name": "alpha", "value": 0.1}], "source": "0", "boundary": "dirichlet",
    "boundary_value": "", "initial": "sin(pi*x)", "initial_velocity": "", "exact": "", "notes": "两端温度为零",
}
SKILL_EXAMPLE: dict[str, Any] = {
    "code": "<the complete module source>", "test_solutions": ["sin(pi*x)*exp(-t)"], "summary": "<方法简介>",
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


class Brain:
    """What the agent needs from a language model; subclasses make the API call."""

    provider = "llm"

    def complete_json(
        self, system: str, messages: list[dict], schema: dict, effort: str | None = None, example: dict | None = None,
    ) -> tuple[dict, Any]:
        """One request whose reply is a JSON object matching ``schema``.

        Returns the JSON and the assistant content to append to ``messages``
        for the next turn of the conversation.
        """
        raise NotImplementedError

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
            data, content = self.complete_json(system, messages, PROBLEM_SCHEMA, effort="medium", example=PROBLEM_EXAMPLE)
            try:
                return ParsedProblem(problem=problem_from_reply(data), notes=data.get("notes", ""), raw=data)
            except (ValueError, KeyError, TypeError) as exc:
                if attempt + 1 == attempts:
                    raise LLMError(f"could not build a valid problem: {exc}") from exc
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"That problem is invalid: {exc}. Please correct it."})
        raise AssertionError("unreachable")

    def answer(self, system: str, messages: list[dict]) -> str:
        """A free-text reply to a conversation (the web page's math Q&A)."""
        raise NotImplementedError

    def skill_session(self, task: SkillTask) -> SkillSession:
        return SkillSession(self, task)


class ClaudeBrain(Brain):
    provider = "claude"

    def __init__(self, client: Any = None, model: str = DEFAULT_MODEL, effort: str = "high", max_tokens: int = 64000,
                 api_key: str | None = None):
        if client is None:
            import anthropic  # optional dependency: pip install "pde-agent[llm]"

            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete_json(
        self, system: str, messages: list[dict], schema: dict, effort: str | None = None, example: dict | None = None,
    ) -> tuple[dict, list]:
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

    def answer(self, system: str, messages: list[dict]) -> str:
        try:
            with self.client.messages.stream(
                model=self.model, max_tokens=ANSWER_TOKENS, system=system, messages=messages,
                thinking={"type": "adaptive"},
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:  # credentials, network, rate limits after the SDK's retries
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        if message.stop_reason == "refusal":
            raise LLMError("the model declined this request")
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        if not text:
            raise LLMError("the reply had no text")
        if message.stop_reason == "max_tokens":
            text += "\n\n（回答太长，被截断了。）"
        return text


class DeepSeekBrain(Brain):
    """DeepSeek through its OpenAI-compatible API, with JSON output and thinking."""

    provider = "deepseek"

    def __init__(
        self,
        client: Any = None,
        model: str = DEEPSEEK_MODEL,
        effort: str = "high",
        max_tokens: int = 131072,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        log_path: Path | str | None = None,
    ):
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= DEEPSEEK_MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_tokens must be between 1 and {DEEPSEEK_MAX_OUTPUT_TOKENS}")
        if client is None:
            api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                # checked here because the OpenAI SDK would fall back to
                # OPENAI_API_KEY, and that key must not be sent to DeepSeek
                raise LLMError("DEEPSEEK_API_KEY is not set")
            import openai  # optional dependency: pip install "pde-agent[deepseek]"

            # no brotli: httpx2 needs brotli>=1.2 for it, and an older brotli
            # (common in Anaconda) fails every reply with "process() takes no
            # keyword arguments"
            client = openai.OpenAI(api_key=api_key, base_url=base_url,
                                   default_headers={"Accept-Encoding": "gzip, deflate"})
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.log_path = Path(log_path) if log_path is not None else None
        self.last_request: dict[str, Any] | None = None

    def complete_json(
        self, system: str, messages: list[dict], schema: dict, effort: str | None = None, example: dict | None = None,
    ) -> tuple[dict, str]:
        """JSON-mode request; replies that are empty or off-schema are asked for again."""
        conversation = [{"role": "system", "content": f"{system}\n\n{json_instructions(schema, example)}"}, *messages]
        problem = ""
        budget = min(self.max_tokens, DEEPSEEK_PARSE_TOKENS) if schema is PROBLEM_SCHEMA else self.max_tokens
        request_effort = effort or self.effort
        length_retried = False
        for _ in range(3):
            text, finish = self._chat(conversation, request_effort, budget)
            if finish == "length":
                if not length_retried:
                    # Start over from the intact conversation; partial JSON is never
                    # executable and must not contaminate the next response.
                    length_retried = True
                    budget = min(budget * 2, DEEPSEEK_MAX_OUTPUT_TOKENS)
                    request_effort = "low"
                    text, finish = self._chat(conversation, request_effort, budget)
                if finish == "length":
                    usage = (self.last_request or {}).get("usage") or {}
                    tokens = usage.get("completion_tokens")
                    detail = f"，实际生成 {tokens} tokens" if tokens is not None else ""
                    raise LLMError(f"DeepSeek 回复达到长度限制，自动重试一次后仍被截断"
                                   f"（本次 max_tokens={budget}{detail}）。"
                                   "请缩小任务范围，或检查用量日志中的生成与思考 token 统计。")
            if finish == "content_filter":
                raise LLMError("DeepSeek's content filter stopped the reply")
            if finish == "insufficient_system_resource":
                raise LLMError("DeepSeek is short of capacity right now; try again later")
            if not text.strip():
                problem = "the reply was empty"  # an occasional DeepSeek hiccup: ask again
                continue
            try:
                data = json_object(text)
                errors = schema_errors(data, schema)
            except ValueError as exc:
                errors = [str(exc)]
            if not errors:
                return data, text
            problem = "; ".join(errors[:5])
            conversation = [
                *conversation,
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That reply cannot be used: {problem}. "
                 "Reply again with one complete json object that matches the schema."},
            ]
        raise LLMError(f"no usable JSON reply after 3 attempts: {problem}")

    def answer(self, system: str, messages: list[dict]) -> str:
        text, finish = self._chat([{"role": "system", "content": system}, *messages], "high",
                                  min(self.max_tokens, ANSWER_TOKENS), json_mode=False)
        if finish == "content_filter":
            raise LLMError("DeepSeek's content filter stopped the reply")
        if finish == "insufficient_system_resource":
            raise LLMError("DeepSeek is short of capacity right now; try again later")
        text = text.strip()
        if not text:
            raise LLMError("DeepSeek 的回复是空的，请再问一次。")
        if finish == "length":
            text += "\n\n（回答太长，被截断了。）"
        return text

    def _chat(self, messages: list[dict], effort: str, max_tokens: int | None = None,
              json_mode: bool = True) -> tuple[str, str | None]:
        """Stream one chat completion; returns the reply text and the finish reason."""
        parts, finish = [], None
        budget = self.max_tokens if max_tokens is None else max_tokens
        level = _DEEPSEEK_EFFORT.get(effort, "high")
        started = time.monotonic()
        stream = None
        record: dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": self.model, "max_tokens": budget, "reasoning_effort": level,
            "request_id": None, "finish_reason": None, "usage": None,
            "content_chars": 0, "reasoning_chars": 0,
        }
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=budget,
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
                reasoning_effort=level,
                extra_body={"thinking": {"type": "enabled"}},
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                record["request_id"] = getattr(chunk, "id", None) or record["request_id"]
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    record["usage"] = usage if isinstance(usage, dict) else usage.model_dump(mode="json")
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta is not None and choice.delta.content:
                    parts.append(choice.delta.content)
                    record["content_chars"] += len(choice.delta.content)
                if choice.delta is not None:
                    record["reasoning_chars"] += len(getattr(choice.delta, "reasoning_content", None) or "")
                finish = choice.finish_reason or finish
        except Exception as exc:  # credentials, balance, network, rate limits after the SDK's retries
            record["error_type"] = type(exc).__name__
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            record["finish_reason"] = finish
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            self.last_request = record
            self._record_request(record)
            if stream is not None and hasattr(stream, "close"):
                stream.close()
        if finish is None:
            raise LLMError("DeepSeek 回复流提前结束，未收到结束标记；请重试。")
        return "".join(parts), finish

    def _record_request(self, record: dict[str, Any]) -> None:
        """Persist metadata only: no credentials, prompt, reply or reasoning text."""
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # A diagnostic file must never turn a successful API call into a failure.
            logging.getLogger(__name__).warning("Could not write DeepSeek usage log: %s", self.log_path)


def json_instructions(schema: dict, example: dict | None = None) -> str:
    """The reply format, spelled out for models that do not enforce a schema."""
    lines = [
        "Reply with one json object and nothing else: no Markdown fences and no text around it.",
        "It must match this JSON schema; every listed property is required (use \"\" for a text field that does not apply):",
        json.dumps(schema, ensure_ascii=False),
    ]
    if example is not None:
        lines += ["An example of the format:", json.dumps(example, ensure_ascii=False)]
    return "\n".join(lines)


def json_object(text: str) -> dict:
    """The JSON object in a reply, tolerating Markdown fences or text around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the reply contains no JSON object")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"the reply is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    return data


def schema_errors(value: Any, schema: dict, path: str = "reply") -> list[str]:
    """Where ``value`` breaks the schema (types, required keys, enums)."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{path} must be an object"]
        errors = [f"{path}.{key} is missing" for key in schema.get("required", []) if key not in value]
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors += schema_errors(value[key], sub, f"{path}.{key}")
        return errors
    if kind == "array":
        if not isinstance(value, list):
            return [f"{path} must be an array"]
        return [e for i, item in enumerate(value) for e in schema_errors(item, schema.get("items", {}), f"{path}[{i}]")]
    if kind == "string" and not isinstance(value, str):
        return [f"{path} must be a string"]
    if kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return [f"{path} must be a number"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} must be one of {schema['enum']}"]
    return []


def default_provider(env: dict | None = None) -> str:
    """DeepSeek when only its API key is set, otherwise Claude."""
    env = os.environ if env is None else env
    if env.get("DEEPSEEK_API_KEY") and not env.get("ANTHROPIC_API_KEY"):
        return "deepseek"
    return "claude"


def make_brain(provider: str, model: str | None = None, *, log_path: Path | str | None = None,
               api_key: str | None = None) -> Brain:
    """A brain for ``provider``; ``api_key`` overrides the provider's environment variable."""
    if provider == "claude":
        return ClaudeBrain(model=model or DEFAULT_MODEL, api_key=api_key)
    if provider == "deepseek":
        return DeepSeekBrain(model=model or DEEPSEEK_MODEL, log_path=log_path, api_key=api_key)
    raise ValueError(f"unknown language model provider {provider!r} (use one of {', '.join(PROVIDERS)})")


class SkillSession:
    """A conversation in which the language model writes a skill and repairs it."""

    def __init__(self, brain: Brain, task: SkillTask):
        self.brain = brain
        self.task = task
        self.system = SKILL_SYSTEM.format(template=TEMPLATE.read_text(encoding="utf-8").strip())
        self.messages: list[dict] = [{"role": "user", "content": task_prompt(task)}]

    def propose(self) -> SkillCandidate:
        data, content = self.brain.complete_json(self.system, self.messages, SKILL_SCHEMA, example=SKILL_EXAMPLE)
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
