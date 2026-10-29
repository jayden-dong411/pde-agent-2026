"""Problem features: choose a method and a grid from what the problem looks like.

Experience is kept per problem class (``kind/dim/boundary``), which lumps a
gentle heat problem together with a stiff one. Features describe the single
problem in front of the agent:

* ``floor``: the first grid that resolves the data (``verify.resolution_floor``),
* ``smooth``: no kinks or jumps in the data (``Abs``, ``Heaviside``, ...),
* ``stiffness``: for diffusion, how many times faster the finest mode in the
  initial data decays than the time span (``alpha * T * k_max^2``),
* ``travel``: for waves and transport, how many domain lengths a signal crosses.

ODE systems (``dim`` 0, see ``ode.py``) have no grid data. Their features
come from the Jacobian: ``stiffness`` is the time span times how much faster
the fastest mode decays than anything grows, ``wavenumber`` holds the
largest |eigenvalue| (the step limit of explicit methods) and ``scale`` how
many oscillations or growth times the span holds (fixed-step methods need
proportionally more steps).

From these and each method's traits and cost model the agent predicts, per
method, the grid that reaches the tolerance and the work to get there. The
cheapest prediction is tried first; experience still decides which methods
are reliable, and only within the same feature tags (``stiff``, ``rough``,
``long``) once there is any.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import sympy as sp

from .problem import PDEProblem
from .solvers.base import SolverSpec
from .verify import RESOLUTION_LIMIT, interpolation_mismatch

# a method of order p is assumed to make an error of FLOOR_RESOLUTION^p on the floor grid
# (as if the floor were 10 intervals of a unit problem); finer grids gain (n_floor/n)^p
FLOOR_RESOLUTION = 0.1
STIFF = 50.0  # stiffness above which the problem counts as stiff
LONG = 4.0  # domain lengths crossed above which dissipation and dispersion pile up
_ROUGH = {"Abs", "sign", "Heaviside", "floor", "ceiling", "Min", "Max"}
_JUMPS = {"sign", "Heaviside", "floor", "ceiling"}  # discontinuous: every mode a grid carries is in the data
_DIFFUSION = {"heat": "alpha"}
_SPEED = {"advection": "c", "wave": "c"}


@dataclass
class ProblemFeatures:
    floor: int
    smooth: bool = True
    jumps: bool = False  # the data jump (not just kink): a grid sees modes up to its own resolution
    stiffness: float = 0.0
    wavenumber: float = 0.0  # finest wavenumber in the initial data
    travel: float = 0.0
    tags: tuple[str, ...] = ()
    ode: bool = False
    scale: float = 1.0  # ODE: oscillations or growth times in the time span (at least 1)

    def to_dict(self) -> dict:
        data = {"floor": self.floor, "smooth": self.smooth, "stiffness": round(self.stiffness, 3),
                "travel": round(self.travel, 3), "tags": list(self.tags)}
        if self.ode:
            data.update(ode=True, scale=round(self.scale, 3), rate=round(self.wavenumber, 3))
        return data

    def describe(self) -> str:
        if self.ode:
            if "stiff" in self.tags:
                text = f"刚性（最快的模态比时间跨度快 {self.stiffness:.3g} 倍衰减，显式方法会被稳定性卡住）"
            else:
                text = f"不刚性（刚性比 {self.stiffness:.3g}）"
            return text + (f"；时间跨度内约有 {self.scale:.1f} 个振荡周期或增长时间" if self.scale > 1 else "")
        parts = [f"数据要 n ≥ {self.floor} 才看得清"]
        if not self.smooth:
            parts.append("数据不光滑（高阶方法只剩一阶）")
        if "stiff" in self.tags:
            parts.append(f"刚性（初值里的高频衰减比时间跨度快 {self.stiffness:.0f} 倍）")
        if "long" in self.tags:
            parts.append(f"长时间传播（跨过区域 {self.travel:.1f} 次）")
        return "；".join(parts)


@dataclass
class Prediction:
    method: str
    n: int  # grid predicted to reach the tolerance
    start: int  # first grid to try: one step coarser than n, but never below what stiffness demands
    work: float | None  # None: the method has no cost model
    order: float
    notes: list[str] = field(default_factory=list)


def _finest_wavenumber(problem: PDEProblem) -> float:
    """Largest wavenumber the initial data needs, from the coarsest grid that resolves it."""
    if problem.expr("initial") is None:
        return 0.0
    fn = problem.fn("initial")
    n = 4
    while n < 4096 and interpolation_mismatch(fn, problem.domain, n, 0.0) > RESOLUTION_LIMIT:
        n *= 2
    length = problem.domain[0][1] - problem.domain[0][0]
    # a cubic interpolant resolves a sine to 1% with about 8 intervals per wavelength
    return 2 * math.pi * (n / 8) / length


def _extract_ode(problem, floor: int) -> ProblemFeatures:
    spectrum = problem.spectrum()
    feats = ProblemFeatures(floor=floor, ode=True, stiffness=problem.stiffness(), wavenumber=spectrum["rho"])
    feats.scale = max(1.0, spectrum["rate"] * problem.t_final / (2 * math.pi))
    feats.tags = ("stiff",) if feats.stiffness > STIFF else ()
    return feats


def extract(problem: PDEProblem, floor: int) -> ProblemFeatures:
    if problem.dim == 0:
        return _extract_ode(problem, floor)
    feats = ProblemFeatures(floor=floor)
    names = set()
    for name in ("initial", "initial_velocity", "source", "boundary_value", "exact"):
        expr = problem.expr(name)
        if expr is not None:
            names |= {node.func.__name__ for node in sp.preorder_traversal(expr)}
    feats.smooth = not (names & _ROUGH)
    feats.jumps = bool(names & _JUMPS)
    tags = []
    if not feats.smooth:
        tags.append("rough")
    if problem.time_dependent:
        length = problem.domain[0][1] - problem.domain[0][0]
        coeff = _DIFFUSION.get(problem.kind)
        if coeff is not None:
            feats.wavenumber = k = _finest_wavenumber(problem)
            feats.stiffness = abs(problem.coefficients.get(coeff, 1.0)) * problem.t_final * k**2
            if feats.stiffness > STIFF:
                tags.append("stiff")
        coeff = _SPEED.get(problem.kind)
        if coeff is not None:
            feats.travel = abs(problem.coefficients.get(coeff, 1.0)) * problem.t_final / length
            if feats.travel > LONG:
                tags.append("long")
    feats.tags = tuple(tags)
    return feats


def predict(spec: SolverSpec, problem: PDEProblem, feats: ProblemFeatures, tol: float, max_n: int) -> Prediction:
    """The grid ``spec`` should need for ``tol`` and the work to run it there."""
    order, notes = max(spec.order, 0.5), []
    if not feats.smooth and order > 1:
        order, notes = 1.0, notes + ["数据不光滑，按一阶估计"]
    if "long" in feats.tags and "dissipative" in spec.traits:
        order, notes = 0.5, notes + ["长时间传播，耗散会累积，按半阶估计"]
    floor = feats.floor
    if feats.scale > 1 and "adaptive" not in spec.traits:  # more oscillations need proportionally more steps
        floor = min(max_n, floor * 2 ** math.ceil(math.log2(feats.scale) - 1e-9))
    floor_error = FLOOR_RESOLUTION**order * max(1.0, feats.travel)
    factor = max(1.0, floor_error / tol) ** (1.0 / order)
    n = min(max_n, floor * 2 ** max(0, math.ceil(math.log2(factor) - 1e-9)))
    # the guess of the error on the floor is rough: start one grid coarser and let measured errors size the rest
    start = max(floor, n // 2)
    if spec.stability and feats.ode and feats.wavenumber > 0:
        need = feats.wavenumber * problem.t_final / spec.stability
        if need > n:  # an explicit method below this many steps blows up: stability, not accuracy, decides
            stable = 2 ** math.ceil(math.log2(need) - 1e-9)
            notes.append(f"稳定性限制：显式方法至少要 {math.ceil(need)} 步")
            if stable > max_n:
                return Prediction(spec.name, max_n, max_n, math.inf, order, notes)
            n = start = stable
    if ("stiff" in feats.tags or (feats.jumps and problem.time_dependent)) and spec.damping is not None:
        length = problem.domain[0][1] - problem.domain[0][0] if problem.dim else 1.0

        def mode(m: int) -> float:
            # jumping data hold every mode the grid carries, up to k = pi * m / L, not just the resolved ones
            return max(feats.wavenumber, math.pi * m / length) if feats.jumps and problem.dim else feats.wavenumber

        accurate = n
        while n < max_n and spec.damping(problem, n, mode(n)) > tol:
            n *= 2
        if spec.damping(problem, n, mode(n)) > tol:
            # the fast modes ring on every grid up to max_n: the method cannot meet tol, whatever its error estimate says
            notes.append("高频模态在最细的网格上也衰减不掉，这个方法不适用")
            return Prediction(spec.name, max_n, max_n, math.inf, order, notes)
        if n > accurate:
            start = n  # below it the fast modes ring; that is not an estimate
            notes.append(f"刚性：最快的模态要 n ≥ {n} 步才衰减干净" if feats.ode else f"刚性：初值里的高频要 n ≥ {n} 才衰减干净")
    work = spec.cost(problem, n) if spec.cost else None
    return Prediction(spec.name, n, start, work, order, notes)
