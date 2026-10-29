"""Ordinary differential equations ``y' = f(t, y)``: stages 0 and 1 of the ODE-to-PDE curriculum.

An ODE problem is plain data like a PDE problem: the right-hand side is a list
of expressions in ``t`` and the unknowns ``y1, y2, ...``, the initial values
and the optional exact solution are lists of expressions (constants, and
functions of ``t``). It quacks like ``PDEProblem`` where the agent needs it
(``kind``, ``dim`` = 0, ``signature``, ``for_solver``, ``to_dict``), so the
same agent loop chooses, solves, checks and remembers.

The resolution knob ``n`` of a fixed-step method is its number of time steps;
an adaptive method maps it to a tolerance (see ``solvers/ode.py``). A solver
returns the state at ``t_final``, a vector of length ``size``.

Stiffness, the feature stage 1 is about, comes from the Jacobian: the
eigenvalues along a cheap backward-Euler probe of the trajectory say how fast
the fastest decaying mode dies compared with the time span. That is known
before any method runs, so the agent can tell an explicit method will be held
back by stability instead of finding out by trying.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import scipy.linalg as la
import sympy as sp

from .problem import _CONSTANTS, _FUNCTIONS, PDEProblem, T, parse_expr

MAX_SIZE = 64
PROBE_STEPS = 64
_UNKNOWN = re.compile(r"y[0-9]+")


class NewtonFailed(RuntimeError):
    """The nonlinear system of an implicit step did not converge."""


@dataclass
class ODEProblem:
    """An initial value problem ``y' = f(t, y)``, ``y(0) = y0`` on ``[0, t_final]``."""

    rhs: list[str]
    initial: list[str]
    t_final: float
    coefficients: dict[str, float] = field(default_factory=dict)
    exact: list[str] | None = None
    equation: str = ""
    name: str = ""
    description: str = ""
    kind: str = "ode"
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.kind != "ode":
            raise ValueError("an ODE problem has kind 'ode'")
        if isinstance(self.rhs, str) or isinstance(self.initial, str):
            raise ValueError("rhs and initial are lists, one entry per unknown")
        self.rhs, self.initial = [str(e) for e in self.rhs], [str(e) for e in self.initial]
        m = len(self.rhs)
        if not 1 <= m <= MAX_SIZE:
            raise ValueError(f"an ODE system needs 1 to {MAX_SIZE} equations")
        if len(self.initial) != m:
            raise ValueError(f"{m} equations need {m} initial values")
        if self.exact is not None:
            self.exact = [str(e) for e in self.exact]
            if len(self.exact) != m:
                raise ValueError(f"{m} equations need {m} exact solution components")
        self.t_final = float(self.t_final)
        if not (math.isfinite(self.t_final) and self.t_final > 0):
            raise ValueError("t_final must be positive")
        self.coefficients = {str(k): float(v) for k, v in self.coefficients.items()}
        for name, value in self.coefficients.items():
            if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name) or name in _CONSTANTS or name in _FUNCTIONS
                    or _UNKNOWN.fullmatch(name)):
                raise ValueError(f"invalid coefficient name {name!r}")
            if not math.isfinite(value):
                raise ValueError(f"coefficient {name} must be finite")
        self.symbols = [sp.Symbol(f"y{i + 1}", real=True) for i in range(m)]
        names = {str(s): s for s in self.symbols}
        self._rhs = [self._parse(e, names, set(self.symbols) | {T}, "rhs") for e in self.rhs]
        y0 = [self._parse(e, {}, set(), "initial value") for e in self.initial]
        self.y0 = np.array([float(e) for e in y0])
        if not np.all(np.isfinite(self.y0)):
            raise ValueError("initial values must be finite")
        self._exact = None
        if self.exact is not None:
            self._exact = [self._parse(e, {}, {T}, "exact solution") for e in self.exact]
            start = np.array([float(e.subs(T, 0)) for e in self._exact])
            if not np.allclose(start, self.y0, rtol=1e-8, atol=1e-10):
                raise ValueError("the exact solution does not start at the initial values")

    def _parse(self, text: str, names: dict, allowed: set, what: str) -> sp.Expr:
        expr = parse_expr(text, self.coefficients, names)
        extra = expr.free_symbols - allowed
        if extra:
            raise ValueError(f"{what} {text!r} may not use {', '.join(sorted(map(str, extra)))}")
        return expr

    # --- the parts of the PDEProblem interface the agent uses -------------------------

    @property
    def size(self) -> int:
        return len(self.rhs)

    @property
    def dim(self) -> int:
        return 0

    @property
    def domain(self) -> list:
        return []

    @property
    def boundary(self) -> str:
        return "ivp"

    @property
    def time_dependent(self) -> bool:
        return True

    def signature(self) -> str:
        """Problem class for experience: ODE systems of the same size."""
        return f"ode/0d/m{self.size}"

    def has_exact(self) -> bool:
        return self._exact is not None

    def for_solver(self) -> ODEProblem:
        """The problem as a solver gets it: a copy without ``exact``, made once and cached."""
        if self._exact is None:
            return self
        if "view" not in self._cache:
            view = copy.copy(self)
            view.exact, view._exact, view._cache = None, None, {}
            view.coefficients = dict(self.coefficients)
            self._cache["view"] = view
        return self._cache["view"]

    def to_dict(self) -> dict[str, Any]:
        return {f.name: copy.copy(getattr(self, f.name)) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ODEProblem:
        known = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in data.items() if k in known})

    # --- numerics ---------------------------------------------------------------------

    def _lambdified(self, key: str, exprs):
        if key not in self._cache:
            self._cache[key] = sp.lambdify([T, *self.symbols], exprs, modules=["numpy"])
        return self._cache[key]

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side at one state."""
        raw = self._lambdified("f", self._rhs)
        return np.array(raw(t, *y), dtype=float).reshape(self.size)

    def f_half(self, t: float, y: np.ndarray, second: bool) -> np.ndarray:
        """One half of the right-hand side: the first ``size // 2`` components, or the rest (for Verlet)."""
        half = self.size // 2
        part = self._rhs[half:] if second else self._rhs[:half]
        raw = self._lambdified("f2" if second else "f1", part)
        return np.array(raw(t, *y), dtype=float).reshape(len(part))

    def jac(self, t: float, y: np.ndarray) -> np.ndarray:
        """Jacobian df/dy at one state, from the symbolic right-hand side."""
        raw = self._lambdified("jac", self._jacobian())
        return np.array(raw(t, *y), dtype=float).reshape(self.size, self.size)

    def exact_at(self, t: float) -> np.ndarray:
        if self._exact is None:
            raise ValueError(f"problem {self.name or 'ode'!r} has no exact solution")
        raw = self._cache.setdefault("exact", sp.lambdify([T], self._exact, modules=["numpy"]))
        return np.array(raw(float(t)), dtype=float).reshape(self.size)

    def _jacobian(self) -> sp.Matrix:
        if "jac_expr" not in self._cache:
            self._cache["jac_expr"] = sp.Matrix(self._rhs).jacobian(self.symbols)
        return self._cache["jac_expr"]

    @property
    def costs(self) -> dict[str, float]:
        """Work of the basic operations, in right-hand-side component evaluations.

        A Jacobian costs its nonzeros, an LU factorisation and a solve cost what a
        banded matrix with the Jacobian's bandwidth costs.
        """
        if "costs" not in self._cache:
            jac = self._jacobian()
            nonzero = [(i, j) for i in range(self.size) for j in range(self.size) if jac[i, j] != 0]
            band = max((abs(i - j) for i, j in nonzero), default=0)
            m = self.size
            self._cache["costs"] = {"f": float(m), "jac": float(max(1, len(nonzero))),
                                    "lu": float(m * (band + 1) ** 2), "solve": float(m * (2 * band + 1))}
        return self._cache["costs"]

    @property
    def separable(self) -> bool:
        """``y = (q, p)`` with ``q' = g(p)`` and ``p' = h(q)``: a separable Hamiltonian-like system."""
        if "separable" not in self._cache:
            m, half = self.size, self.size // 2
            q, p = set(self.symbols[:half]), set(self.symbols[half:])
            self._cache["separable"] = m % 2 == 0 and all(
                not (e.free_symbols & (q | {T})) for e in self._rhs[:half]) and all(
                not (e.free_symbols & (p | {T})) for e in self._rhs[half:])
        return self._cache["separable"]

    def spectrum(self) -> dict[str, float]:
        """Rates from the Jacobian's eigenvalues along a coarse backward-Euler probe (geometric steps).

        ``decay``: fastest decaying rate, ``growth``: fastest growing rate,
        ``rho``: largest |eigenvalue| (what limits explicit steps),
        ``rate``: fastest oscillation or growth at the start (what sets the steps for accuracy).
        Stiffness is ``t_final * (decay - growth)``: Hamiltonian saddles
        (decay = growth) are not stiff.
        """
        if "spectrum" not in self._cache:
            out = {"decay": 0.0, "growth": 0.0, "rho": 0.0, "rate": 0.0}
            # geometric times: stiff transients live at the start, and small first steps keep Newton on the right root
            times = np.concatenate([[0.0], self.t_final * np.logspace(-6, 0, PROBE_STEPS)])
            y, counts = self.y0.copy(), new_counts()
            for k, t in enumerate(times):
                try:
                    lam = np.linalg.eigvals(self.jac(t, y))
                except (ValueError, la.LinAlgError, ZeroDivisionError, FloatingPointError):
                    break
                if not np.all(np.isfinite(lam)):
                    break
                out["decay"] = max(out["decay"], float(np.max(-lam.real)))
                out["growth"] = max(out["growth"], float(np.max(lam.real)))
                out["rho"] = max(out["rho"], float(np.max(np.abs(lam))))
                if k == 0:  # late probe states are rough (backward Euler damps orbits), so the rate is read at the start
                    out["rate"] = max(float(np.max(np.abs(lam.imag))), float(np.max(lam.real)), 0.0)
                if k == PROBE_STEPS:
                    break
                try:
                    y = self._probe_step(t, y, times[k + 1] - t, counts)
                except NewtonFailed:
                    break
            self._cache["spectrum"] = out
        return self._cache["spectrum"]

    def _probe_step(self, t: float, y: np.ndarray, h: float, counts: dict, depth: int = 0) -> np.ndarray:
        """One backward-Euler step, split in halves where Newton does not converge."""
        try:
            return implicit_step(self, t + h, y, h, y, counts)
        except NewtonFailed:
            if depth >= 12:
                raise
            y = self._probe_step(t, y, h / 2, counts, depth + 1)
            return self._probe_step(t + h / 2, y, h / 2, counts, depth + 1)

    def stiffness(self) -> float:
        s = self.spectrum()
        return self.t_final * max(0.0, s["decay"] - s["growth"])


def new_counts() -> dict[str, int]:
    return {"f": 0, "jac": 0, "lu": 0, "solve": 0}


def implicit_step(problem: ODEProblem, t: float, guess: np.ndarray, ch: float, b: np.ndarray, counts: dict,
                  tol: float = 1e-10, max_iter: int = 20) -> np.ndarray:
    """Solve ``y - ch * f(t, y) = b`` by Newton's method.

    The Jacobian is kept while Newton contracts fast (simplified Newton, cheap
    for nearly linear problems) and refreshed when it does not.
    """
    y = np.array(guess, dtype=float)
    eye = np.eye(problem.size)
    lu, last = None, math.inf
    for _ in range(max_iter):
        if lu is None:
            with np.errstate(all="ignore"):
                matrix = eye - ch * problem.jac(t, y)
            if not np.all(np.isfinite(matrix)):
                raise NewtonFailed("Jacobian is not finite")
            lu = la.lu_factor(matrix, check_finite=False)
            counts["jac"] += 1
            counts["lu"] += 1
        with np.errstate(all="ignore"):
            residual = y - ch * problem.f(t, y) - b
        counts["f"] += 1
        dy = la.lu_solve(lu, -residual, check_finite=False)
        counts["solve"] += 1
        if not np.all(np.isfinite(dy)):
            raise NewtonFailed("Newton step is not finite")
        y = y + dy
        size = float(np.max(np.abs(dy)))
        if size <= tol * (1.0 + np.max(np.abs(y))):
            return y
        if size > 0.25 * last:
            lu = None  # slow contraction: the Jacobian is stale
        last = size
    raise NewtonFailed(f"Newton did not converge in {max_iter} iterations")


def problem_from_dict(data: dict[str, Any]) -> PDEProblem | ODEProblem:
    """A PDE or ODE problem from its dict form (``kind == "ode"`` picks the ODE)."""
    if data.get("kind") == "ode":
        return ODEProblem.from_dict(data)
    return PDEProblem.from_dict(data)
