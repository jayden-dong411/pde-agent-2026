"""Problem specification: what the agent is asked to solve.

A problem is plain data (JSON-serialisable) so it can be stored in the
experience memory, produced by the language model, or written by hand.
Mathematical fields are expression strings in the variables ``x``, ``y``
and ``t`` such as ``"sin(pi*x)*exp(-t)"``.
"""

from __future__ import annotations

import copy
import io
import math
import re
import tokenize
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np
import sympy as sp

X, Y, T = sp.symbols("x y t", real=True)

BOUNDARY_KINDS = ("dirichlet", "periodic")

# sympify is an eval, so expressions are restricted to arithmetic on numbers
# and a whitelist of names: no quotes, underscores, attributes or builtins.
_SAFE_CHARS = re.compile(r"^[0-9A-Za-z\s.+\-*/^(),]*$")
_FUNCTIONS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log", "sqrt",
    "Abs", "abs", "sign", "Min", "Max", "floor", "ceiling", "erf", "erfc", "Heaviside",
}
_CONSTANTS = {"x": X, "y": Y, "t": T, "pi": sp.pi, "E": sp.E, "e": sp.E}

_EXPR_FIELDS = ("source", "boundary_value", "initial", "initial_velocity", "exact")

Field = Callable[..., np.ndarray]


def _names(text: str) -> set[str]:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        raise ValueError(f"cannot parse expression {text!r}: {exc}") from exc
    return {tok.string for tok in tokens if tok.type == tokenize.NAME}


def parse_expr(text: str, coefficients: dict[str, float] | None = None) -> sp.Expr:
    """Parse an expression in x, y, t; coefficient names are substituted."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"empty expression: {text!r}")
    if not _SAFE_CHARS.match(text):
        raise ValueError(f"unsupported characters in expression: {text!r}")
    coefficients = coefficients or {}
    unknown = _names(text) - _FUNCTIONS - set(_CONSTANTS) - set(coefficients)
    if unknown:
        raise ValueError(f"unknown names in {text!r}: {', '.join(sorted(unknown))}")
    local: dict[str, Any] = dict(_CONSTANTS)
    for name, value in coefficients.items():
        local[name] = sp.Float(value)
    try:
        expr = sp.sympify(text, locals=local, convert_xor=True)
    except (sp.SympifyError, SyntaxError, TypeError) as exc:
        raise ValueError(f"cannot parse expression {text!r}: {exc}") from exc
    if not isinstance(expr, sp.Expr):
        raise ValueError(f"not a scalar expression: {text!r}")
    unknown = expr.free_symbols - {X, Y, T}
    if unknown:
        names = ", ".join(sorted(str(s) for s in unknown))
        raise ValueError(f"unknown symbols in {text!r}: {names}")
    return expr


def lambdify_field(expr: sp.Expr) -> Field:
    """Turn an expression into ``f(x, y=0, t=0)`` that broadcasts like numpy."""
    raw = sp.lambdify((X, Y, T), expr, modules=["scipy", "numpy"])

    def evaluate(x, y=0.0, t=0.0):
        x, y, t = (np.asarray(v, dtype=float) for v in (x, y, t))
        shape = np.broadcast_shapes(x.shape, y.shape, t.shape)
        return np.broadcast_to(np.asarray(raw(x, y, t), dtype=float), shape).copy()

    return evaluate


def _boundary_interpolant(exact: sp.Expr, domain: list[tuple[float, float]]) -> sp.Expr:
    """Dirichlet data that match ``exact`` on the boundary and ignore its interior.

    Linear interpolation between the two end values in 1-D, a Coons patch over
    the four edges in 2-D. 3-D has no such blend and returns ``exact`` itself
    (a known limitation).
    """
    if len(domain) == 3:
        return exact
    a, b = domain[0]
    s = (X - a) / (b - a)
    if len(domain) == 1:
        return (1 - s) * exact.subs(X, a) + s * exact.subs(X, b)
    c, d = domain[1]
    r = (Y - c) / (d - c)

    def e(p, q):
        return exact.subs({X: p, Y: q})

    return ((1 - s) * e(a, Y) + s * e(b, Y) + (1 - r) * e(X, c) + r * e(X, d)
            - ((1 - s) * (1 - r) * e(a, c) + s * (1 - r) * e(b, c) + (1 - s) * r * e(a, d) + s * r * e(b, d)))


@dataclass
class PDEProblem:
    """A PDE problem ``L(u) = source`` on a box, optionally up to ``t_final``.

    ``kind`` names the equation (``poisson``, ``heat``, ``advection``,
    ``wave`` or a kind the agent learned later); ``equation`` is an optional
    human-readable form such as ``"u_t = D*u_xx - r*u + f"``. Missing initial
    or boundary data fall back to ``exact`` when it is given. Solvers get
    ``for_solver()``, a copy without ``exact`` whose fallbacks keep only
    boundary information.
    """

    kind: str
    domain: list[tuple[float, float]]
    t_final: float | None = None
    coefficients: dict[str, float] = field(default_factory=dict)
    source: str = "0"
    boundary: str = "dirichlet"
    boundary_value: str | None = None
    initial: str | None = None
    initial_velocity: str | None = None
    exact: str | None = None
    equation: str = ""
    name: str = ""
    description: str = ""
    _exprs: dict[str, sp.Expr] = field(default_factory=dict, init=False, repr=False, compare=False)
    _fields: dict[str, Field] = field(default_factory=dict, init=False, repr=False, compare=False)
    _view: PDEProblem | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.kind or not re.fullmatch(r"[a-z][a-z0-9_]*", self.kind):
            raise ValueError(f"kind must be a snake_case name, got {self.kind!r}")
        self.domain = [(float(a), float(b)) for a, b in self.domain]
        if not 1 <= len(self.domain) <= 3:
            raise ValueError("domain must have 1 to 3 spatial dimensions")
        for a, b in self.domain:
            if not (math.isfinite(a) and math.isfinite(b) and a < b):
                raise ValueError(f"invalid interval [{a}, {b}]")
        if self.t_final is not None:
            self.t_final = float(self.t_final)
            if not (math.isfinite(self.t_final) and self.t_final > 0):
                raise ValueError("t_final must be positive")
        if self.boundary not in BOUNDARY_KINDS:
            raise ValueError(f"boundary must be one of {BOUNDARY_KINDS}")
        self.coefficients = {str(k): float(v) for k, v in self.coefficients.items()}
        for name, value in self.coefficients.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name) or name in _CONSTANTS or name in _FUNCTIONS:
                raise ValueError(f"invalid coefficient name {name!r}")
            if not math.isfinite(value):
                raise ValueError(f"coefficient {name} must be finite")
        for name in _EXPR_FIELDS:
            text = getattr(self, name)
            if text is not None:
                self._exprs[name] = parse_expr(text, self.coefficients)
        exact = self._exprs.get("exact")
        if exact is not None and self.t_final is None and T in exact.free_symbols:
            raise ValueError("a steady problem cannot have a time-dependent exact solution")

    @property
    def dim(self) -> int:
        return len(self.domain)

    @property
    def time_dependent(self) -> bool:
        return self.t_final is not None

    def signature(self) -> str:
        """Coarse problem class used to share experience between problems."""
        return f"{self.kind}/{self.dim}d/{self.boundary}"

    def has_exact(self) -> bool:
        return "exact" in self._exprs

    def expr(self, name: str) -> sp.Expr | None:
        """Parsed expression for a field, applying the documented fallbacks."""
        if name in self._exprs:
            return self._exprs[name]
        exact = self._exprs.get("exact")
        if name == "boundary_value":
            return exact if exact is not None else sp.Integer(0)
        if name == "initial":
            return exact.subs(T, 0) if exact is not None else None
        if name == "initial_velocity":
            return sp.diff(exact, T).subs(T, 0) if exact is not None else sp.Integer(0)
        if name == "source":
            return sp.Integer(0)
        return None

    def for_solver(self) -> PDEProblem:
        """The problem as a solver gets it: a copy without ``exact``, made once and cached.

        A problem without an exact solution is returned as it is. The copy
        shares the parsed expressions of the fields the problem gives (they are
        not parsed again) and holds no reference to the original. Missing
        fields are rebuilt from ``exact`` as far as a solver may know them:
        initial data at ``t = 0`` and Dirichlet data through the boundary
        values alone. Periodic problems get no boundary data and read zero.
        """
        exact = self._exprs.get("exact")
        if exact is None:
            return self
        if self._view is None:
            view = copy.copy(self)
            view.exact = None
            view.coefficients, view.domain = dict(self.coefficients), list(self.domain)
            view._exprs = {name: expr for name, expr in self._exprs.items() if name != "exact"}
            view._fields, view._view = {}, None
            filled: dict[str, sp.Expr] = {}
            if self.time_dependent:
                if "initial" not in view._exprs:
                    filled["initial"] = exact.subs(T, 0)
                if "initial_velocity" not in view._exprs:
                    filled["initial_velocity"] = sp.diff(exact, T).subs(T, 0)
            if self.boundary == "dirichlet" and "boundary_value" not in view._exprs:
                filled["boundary_value"] = _boundary_interpolant(exact, self.domain)
            for name, expr in filled.items():
                view._exprs[name] = expr
                setattr(view, name, str(expr))  # the strings only serve debugging
            self._view = view
        return self._view

    def fn(self, name: str) -> Field:
        """Numpy-callable ``f(x, y=0, t=0)`` for a field (cached)."""
        if name not in self._fields:
            expr = self.expr(name)
            if expr is None:
                raise ValueError(f"problem {self.name or self.kind!r} has no {name.replace('_', ' ')}")
            self._fields[name] = lambdify_field(expr)
        return self._fields[name]

    def with_coefficients(self, defaults: dict[str, float]) -> PDEProblem:
        """Copy with missing coefficients filled in from ``defaults``."""
        merged = {**defaults, **self.coefficients}
        if merged == self.coefficients:
            return self
        data = self.to_dict()
        data["coefficients"] = merged
        return PDEProblem.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        data["domain"] = [list(d) for d in self.domain]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PDEProblem:
        known = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in data.items() if k in known})
