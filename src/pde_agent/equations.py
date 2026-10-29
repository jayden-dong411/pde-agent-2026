"""The equations the agent knows, written as operators ``L(u) = f``.

Knowing the operator symbolically lets the agent build test problems with
known answers for any equation (the method of manufactured solutions): pick
an exact ``u``, compute ``f = L(u)``, solve, and compare.

An equation can also be stated as text (``"u_t = D*u_xx - r*u + f"``).
``check_operator`` reads such a text and compares it with an operator, so a
stated equation is known to say what the operator computes.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import sympy as sp

from .problem import _CONSTANTS, _FUNCTIONS, PDEProblem, T, X, Y, _names, parse_expr

# operator(u, x, y, t, coefficients) -> L(u) as a sympy expression
Operator = Callable[[sp.Expr, sp.Symbol, sp.Symbol, sp.Symbol, dict], sp.Expr]


@dataclass(frozen=True)
class Equation:
    kind: str
    form: str
    time_order: int
    operator: Operator
    defaults: dict[str, float] = field(default_factory=dict)
    description: str = ""
    origin: str = "builtin"
    # coefficients -> why they make no sense for this equation, or None
    check: Callable[[dict], str | None] | None = None

    def apply(self, u: sp.Expr, coefficients: dict[str, float]) -> sp.Expr:
        return self.operator(u, X, Y, T, {**self.defaults, **coefficients})


def _laplacian(u, x, y):
    return sp.diff(u, x, 2) + sp.diff(u, y, 2)


BUILTIN_EQUATIONS = (
    Equation(
        kind="poisson",
        form="-k * (u_xx + u_yy) = f",
        time_order=0,
        operator=lambda u, x, y, t, c: -c["k"] * _laplacian(u, x, y),
        defaults={"k": 1.0},
        description="Poisson equation (elliptic, steady diffusion)",
        check=lambda c: "系数 k 不能是 0" if c["k"] == 0 else None,
    ),
    Equation(
        kind="heat",
        form="u_t = alpha * (u_xx + u_yy) + f",
        time_order=1,
        operator=lambda u, x, y, t, c: sp.diff(u, t) - c["alpha"] * _laplacian(u, x, y),
        defaults={"alpha": 1.0},
        description="Heat / diffusion equation (parabolic)",
        check=lambda c: "系数 alpha 必须大于 0（alpha ≤ 0 时热方程是不适定的）" if c["alpha"] <= 0 else None,
    ),
    Equation(
        kind="advection",
        form="u_t + c * u_x = f",
        time_order=1,
        operator=lambda u, x, y, t, c: sp.diff(u, t) + c["c"] * sp.diff(u, x),
        defaults={"c": 1.0},
        description="Linear advection / transport equation (hyperbolic, first order)",
    ),
    Equation(
        kind="wave",
        form="u_tt = c^2 * (u_xx + u_yy) + f",
        time_order=2,
        operator=lambda u, x, y, t, c: sp.diff(u, t, 2) - c["c"] ** 2 * _laplacian(u, x, y),
        defaults={"c": 1.0},
        description="Wave equation (hyperbolic, second order)",
    ),
)


class EquationBook:
    """Registry of known equations; learned skills can add new ones."""

    def __init__(self, equations: Iterable[Equation] = ()):
        self._equations: dict[str, Equation] = {}
        for eq in equations:
            self.add(eq)

    def add(self, equation: Equation, replace: bool = False) -> None:
        if equation.kind in self._equations and not replace:
            raise ValueError(f"equation {equation.kind!r} already registered")
        self._equations[equation.kind] = equation

    def get(self, kind: str) -> Equation | None:
        return self._equations.get(kind)

    def __contains__(self, kind: str) -> bool:
        return kind in self._equations

    def __iter__(self):
        return iter(self._equations.values())

    def complete(self, problem: PDEProblem) -> PDEProblem:
        """Fill in default coefficients for a known equation kind.

        Raises ``ValueError`` for a coefficient the equation does not have and no
        expression of the problem uses: it is almost always a misspelled name
        (``D`` for ``alpha``), and solving with the default instead gives a wrong answer.
        """
        eq = self.get(problem.kind)
        if eq is None:
            return problem
        used: set[str] = set()
        for name in ("source", "boundary_value", "initial", "initial_velocity", "exact"):
            text = getattr(problem, name, None)
            if text:
                used |= _names(text)
        stray = sorted(set(problem.coefficients) - set(eq.defaults) - used)
        if stray:
            known = ", ".join(eq.defaults) or "没有系数"
            raise ValueError(f"{eq.kind} 方程没有系数 {', '.join(stray)}（它的系数是：{known}）")
        problem = problem.with_coefficients(eq.defaults)
        why = eq.check(problem.coefficients) if eq.check else None
        if why:
            raise ValueError(f"{eq.kind} 方程：{why}")
        return problem


def builtin_equations() -> EquationBook:
    return EquationBook(BUILTIN_EQUATIONS)


def manufactured_problem(
    equation: Equation,
    exact: str,
    domain: list[tuple[float, float]],
    t_final: float | None = None,
    coefficients: dict[str, float] | None = None,
    boundary: str = "dirichlet",
    name: str = "",
    description: str = "",
) -> PDEProblem:
    """Problem whose exact solution is ``exact``: the source is ``L(exact)``."""
    coeffs = {**equation.defaults, **(coefficients or {})}
    u = parse_expr(exact, coeffs)
    if equation.time_order == 0:
        if T in u.free_symbols:
            raise ValueError("steady equation needs a time-independent exact solution")
        t_final = None
    elif t_final is None:
        raise ValueError(f"{equation.kind} is time dependent; give t_final")
    source = equation.apply(u, coeffs)
    return PDEProblem(
        kind=equation.kind,
        domain=domain,
        t_final=t_final,
        coefficients=coeffs,
        source=str(source),
        boundary=boundary,
        exact=exact,
        name=name,
        description=description,
    )


# --- equations stated as text ----------------------------------------------------

# the characters parse_expr allows, plus "="
_EQUATION_CHARS = re.compile(r"^[0-9A-Za-z\s.+\-*/^(),=]*$")
_DERIVATIVE = re.compile(r"\bu_([txy]+)\b")
_CALLED = re.compile(r"\b[uf]\s*\(")  # u(x, t), f(x, t)
_AXES = {"x": X, "y": Y, "t": T}


def _inner(name: str) -> sp.Symbol:
    """The symbol for ``u``, ``f`` or a derivative while the text is solved for ``f``.

    A text cannot hold an underscore, so these never clash with a coefficient.
    """
    return sp.Symbol(f"{name}_")


@dataclass(frozen=True)
class TextOperator:
    """The operator ``L`` of an equation stated as text, with ``L(u) = f``.

    ``expr`` is ``L`` in symbols: one for the unknown, one for each derivative
    that occurs (``derivatives`` maps its name to the axes it differentiates by,
    e.g. ``"Dqxx": "xx"``) and one for each coefficient, by name. ``apply``
    puts a concrete ``u`` and the coefficient values in their place.
    """

    expr: sp.Expr
    derivatives: dict[str, str]
    time_order: int

    def apply(self, u: sp.Expr, coefficients: dict[str, float]) -> sp.Expr:
        """``L(u)`` for a given ``u``; every coefficient the text uses must be in ``coefficients``."""
        swap = {_inner("u"): u}
        for name, axes in self.derivatives.items():
            swap[_inner(name)] = sp.diff(u, *(_AXES[axis] for axis in axes))
        for symbol in self.expr.free_symbols - {X, Y, T} - set(swap):
            swap[symbol] = sp.Float(coefficients[symbol.name])
        return self.expr.xreplace(swap)


def operator_from_text(text: str, coefficient_names: Iterable[str]) -> TextOperator:
    """Read an equation such as ``"u_t = D*u_xx - r*u + f"`` as ``L(u) = f``.

    ``u`` is the unknown and ``f`` the source term, both without arguments;
    derivatives are ``u_t``, ``u_xx``, ``u_xy`` and so on. Coefficients are the
    given names or plain numbers. A text without ``f`` gets ``+ f`` on its
    right side. Terms without ``u`` belong to the source, so ``L(0) = 0``.
    Raises ``ValueError`` for anything that cannot be read.
    """
    try:
        return _read_operator(text, set(coefficient_names))
    except Exception as exc:  # sympify evaluates the text, and sympy fails in many ways on odd input
        if type(exc) is ValueError:  # one of ours (SympifyError is a ValueError too, but not ours)
            raise
        cause = getattr(exc, "base_exc", None) or exc  # SympifyError wraps the real error
        raise ValueError(f"cannot parse {text!r}: {type(cause).__name__}: {cause}") from exc


def _read_operator(text: str, names: set[str]) -> TextOperator:
    """The work of ``operator_from_text``.

    A plain ``ValueError`` says what is wrong with the text; anything else is
    sympy giving up on it and is turned into a ``ValueError`` by the caller.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("no equation text")
    if names & {"u", "f"}:
        raise ValueError("u and f stand for the unknown and the source term, not for coefficients")
    if text.count("=") != 1:
        raise ValueError(f"{text!r} must contain exactly one '='")

    # u_xt and u_tx are the same derivative; sympify gets a plain name for each
    derivatives: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        axes = "".join(sorted(match.group(1)))
        derivatives[f"Dq{axes}"] = axes
        return f"Dq{axes}"

    # sympify drops newlines, which would glue two checked names into a third: parse one line
    plain = _DERIVATIVE.sub(rename, " ".join(text.split()))
    for name, axes in derivatives.items():
        if name in names:
            raise ValueError(f"coefficient name {name!r} is taken by the derivative u_{axes}")
    if _CALLED.search(plain):
        raise ValueError("write u and f without arguments, e.g. u_t = D*u_xx + f")
    if not _EQUATION_CHARS.match(plain):  # sympify is an eval: nothing else reaches it
        raise ValueError(f"unsupported characters in {text!r}")
    try:
        used = _names(plain)
    except ValueError as exc:  # its message quotes `plain`, with the derivatives renamed
        raise ValueError(f"cannot parse {text!r}: {exc.__cause__ or exc}") from exc
    unknown = used - names - set(derivatives) - {"u", "f"} - set(_CONSTANTS) - _FUNCTIONS
    if unknown:
        raise ValueError(f"unknown names in {text!r}: {', '.join(sorted(unknown))}")

    local = dict(_CONSTANTS)
    local.update((name, sp.Symbol(name)) for name in names)
    local.update((name, _inner(name)) for name in ("u", "f", *derivatives))
    sides = [sp.sympify(side, locals=local, convert_xor=True) for side in plain.split("=")]
    if not all(isinstance(side, sp.Expr) for side in sides):
        raise ValueError(f"not a scalar equation: {text!r}")

    source = _inner("f")
    if "f" not in used:
        sides[1] = sides[1] + source  # a text without f: the source term is added on the right
    difference = sides[0] - sides[1]
    linear = not sp.diff(difference, source).has(source)  # solve is slow, or gives up, unless f is linear
    solutions = sp.solve(difference, source, simplify=False) if linear else []  # simplifying (x + 1)^1800 takes seconds
    if len(solutions) != 1 or source in solutions[0].free_symbols:
        raise ValueError("f must appear once, to the first power")
    operator = solutions[0]
    operator -= operator.xreplace({_inner(name): sp.Integer(0) for name in ("u", *derivatives)})
    time_order = max((axes.count("t") for axes in derivatives.values()), default=0)
    return TextOperator(operator, derivatives, time_order)


def _real(value) -> float:
    """``value`` as a float; a complex number never matches a real one, so it becomes nan."""
    value = complex(value)
    return value.real if value.imag == 0 else math.nan


def check_operator(
    equation: Equation,
    text: str,
    coefficients: dict[str, float],
    dim: int,
    domain: list[tuple[float, float]],
    t_final: float | None,
) -> str | None:
    """Does ``text`` say what ``equation.operator`` computes? ``None`` if so, else one sentence.

    Both are applied to smooth test functions of the task's own variables (no
    ``y`` in 1-D, no ``t`` when steady) and compared at random points of the
    domain, so a flipped sign or a wrong coefficient shows up as a number.
    """
    values = {**equation.defaults, **coefficients}
    try:
        text_operator = operator_from_text(text, values)
    except ValueError as exc:
        return f"cannot read {text!r}: {exc}"
    if text_operator.time_order != equation.time_order:
        return f"{text!r} has time order {text_operator.time_order}, EQUATION has {equation.time_order}"
    rng = random.Random(0)
    try:
        with np.errstate(all="ignore"):  # a text may leave the range of a function
            for _ in range(4):
                a, b, shift, rate = (rng.uniform(0.5, 2.0) for _ in range(4))
                w = sp.sin(a * X + b * Y + shift) * sp.exp(-rate * T) + a * X**2 * T + b * X * Y
                if dim == 1:
                    w = w.subs(Y, 0)
                if t_final is None:
                    w = w.subs(T, 0)
                of_text = sp.lambdify((X, Y, T), text_operator.apply(w, values), modules=["scipy", "numpy"])
                of_operator = sp.lambdify((X, Y, T), equation.apply(w, values), modules=["scipy", "numpy"])
                for _ in range(5):
                    px = rng.uniform(*domain[0])
                    py = rng.uniform(*domain[1]) if dim > 1 else 0.0
                    pt = rng.uniform(0.0, t_final) if t_final is not None else 0.0
                    # numpy scalars: an overflow gives inf, where a python float raises
                    point = np.float64(px), np.float64(py), np.float64(pt)
                    left, right = _real(of_text(*point)), _real(of_operator(*point))
                    finite = math.isfinite(left) and math.isfinite(right)
                    if not finite or abs(left - right) > 1e-9 * (1 + abs(left) + abs(right)):
                        return (f"{text!r} and operator differ at x={px:.4g}, y={py:.4g}, t={pt:.4g}: "
                                f"{left:.6g} vs {right:.6g}")
    except Exception as exc:
        return f"cannot evaluate {text!r} against operator: {type(exc).__name__}: {exc}"
    return None
