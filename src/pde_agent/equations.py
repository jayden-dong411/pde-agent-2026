"""The equations the agent knows, written as operators ``L(u) = f``.

Knowing the operator symbolically lets the agent build test problems with
known answers for any equation (the method of manufactured solutions): pick
an exact ``u``, compute ``f = L(u)``, solve, and compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import sympy as sp

from .problem import PDEProblem, T, X, Y, parse_expr

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
    ),
    Equation(
        kind="heat",
        form="u_t = alpha * (u_xx + u_yy) + f",
        time_order=1,
        operator=lambda u, x, y, t, c: sp.diff(u, t) - c["alpha"] * _laplacian(u, x, y),
        defaults={"alpha": 1.0},
        description="Heat / diffusion equation (parabolic)",
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
        """Fill in default coefficients for a known equation kind."""
        eq = self.get(problem.kind)
        return problem.with_coefficients(eq.defaults) if eq else problem


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
