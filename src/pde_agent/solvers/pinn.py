"""Physics-informed neural networks (PINN) in pure numpy: a neural method the agent can weigh against grids.

The network has one hidden layer of ``tanh`` units over the normalised inputs
(``x``, ``y`` and, for evolution problems, ``t`` mapped to [-1, 1]):

    u(z) = sum_i a_i tanh(w_i . z + b_i) + a_0.

Its derivatives are closed-form (``tanh' = 1 - tanh^2``), so the PDE residual
at collocation points, the boundary values, the initial values (and for
periodic problems the matching of both ends) are all linear in the output
weights ``a``. Training is the physics-informed loss

    sum |L(u) - f|^2 at interior points + lambda^2 (sum |u - g|^2 on the boundary + |u - u0|^2 at t = 0),

minimised exactly for ``a`` by least squares, with the hidden weights drawn at
random (a physics-informed extreme learning machine, PIELM). That needs no
autograd library and no stochastic optimiser, is deterministic (the seed is
fixed per ``n``), and on smooth problems the error falls faster than any power
of the network size. Space-time collocation means the solution at ``t_final``
comes without time steps.

``n`` sets the network size: 4n hidden units for one input, 8n for two
(``x, t`` or ``x, y``), collocation points twice that. ``work`` is the
least-squares cost in flops divided by 5 (about one finite-difference node
update), so the agent compares methods on the same scale.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import lstsq

from ..problem import PDEProblem
from ..verify import RESOLUTION_LIMIT, interpolation_mismatch
from .base import SolveResult, SolverSpec, evaluate, grid, mesh

MAX_UNITS = 2048  # beyond this the least-squares solve takes minutes
UNITS = {1: 4, 2: 8}  # hidden units per unit of n, by number of inputs
COLLOCATION = 2  # collocation points per hidden unit
FLOPS_PER_WORK = 5.0


def _inputs(problem: PDEProblem) -> list[tuple[float, float]]:
    spans = [tuple(map(float, d)) for d in problem.domain]
    if problem.time_dependent:
        spans.append((0.0, float(problem.t_final)))
    return spans


def _units(problem: PDEProblem, n: int) -> int:
    return UNITS[len(_inputs(problem))] * n


def _wavenumber(fn, span) -> float:
    """Finest wavenumber of ``fn`` along one line, in normalised coordinates (the span mapped to [-1, 1])."""
    n = 4
    while n < 1024 and interpolation_mismatch(fn, [span], n, 0.0) > RESOLUTION_LIMIT:
        n *= 2
    return 2 * math.pi * (n / 8) / 2  # a cubic interpolant needs about 8 intervals per wavelength


def _scale(problem: PDEProblem) -> np.ndarray:
    """Spread of the hidden weights per input: wide enough for the finest wavenumber of the data along it.

    The data are what a solver sees (initial values for evolution problems, else the source and the boundary
    values), never the exact solution. Time gets the default spread: the data say nothing about it.
    """
    spans = _inputs(problem)
    out = np.full(len(spans), 2.0)
    names = ["initial"] if problem.time_dependent else ["source", "boundary_value"]
    rng = np.random.default_rng(7)
    probes = [[lo + (hi - lo) * rng.uniform(0.1, 0.9) for lo, hi in spans] for _ in range(3)]
    for name in names:
        if problem.expr(name) is None:
            continue
        fn = problem.fn(name)
        for axis in range(problem.dim):
            for probe in probes:
                def line(v, _y=0.0, _t=0.0, axis=axis, probe=probe, fn=fn):
                    args = [np.full(np.shape(v), p) for p in probe]
                    args[axis] = v
                    y = args[1] if problem.dim == 2 else 0.0 * args[0]
                    return fn(args[0], y, 0.0)
                out[axis] = max(out[axis], 0.5 * _wavenumber(line, spans[axis]))
    return out


def _chebyshev(m: int) -> np.ndarray:
    return np.cos(np.pi * (np.arange(m) + 0.5) / m)[::-1]


class _Net:
    def __init__(self, spans, units: int, scale, seed: int):
        rng = np.random.default_rng(seed)
        d = len(spans)
        directions = rng.normal(size=(d, units))
        directions /= np.linalg.norm(directions, axis=0)
        self.w = directions * rng.uniform(0.2, 1.0, units) * np.reshape(scale, (-1, 1)) * 2
        centres = rng.uniform(-1.0, 1.0, (d, units))
        self.b = -(self.w * centres).sum(axis=0)
        self.spans = spans
        self.chain = np.array([2.0 / (hi - lo) for lo, hi in spans])  # d xi / d x per input

    def normalise(self, *coords) -> np.ndarray:
        return np.column_stack([2.0 * (np.ravel(c) - lo) / (hi - lo) - 1.0 for c, (lo, hi) in zip(coords, self.spans)])

    def rows(self, z: np.ndarray, derivative: dict[int, int] | None = None) -> np.ndarray:
        """Feature rows (one column per unit plus the constant) of u or of a derivative {input: order}."""
        s = np.tanh(z @ self.w + self.b)
        order = sum((derivative or {}).values())
        if order == 0:
            return np.hstack([s, np.ones((len(z), 1))])
        d1 = 1.0 - s * s
        act = d1 if order == 1 else -2.0 * s * d1
        factor = np.ones(self.w.shape[1])
        for axis, k in derivative.items():
            factor = factor * (self.w[axis] * self.chain[axis]) ** k
        return np.hstack([act * factor, np.zeros((len(z), 1))])


def _operator_rows(problem: PDEProblem, net: _Net, z: np.ndarray) -> np.ndarray:
    c = problem.coefficients
    dim = problem.dim
    t_axis = dim  # time is the input after the space dimensions
    lap = sum(net.rows(z, {axis: 2}) for axis in range(dim))
    if problem.kind == "poisson":
        return -c.get("k", 1.0) * lap
    if problem.kind == "heat":
        return net.rows(z, {t_axis: 1}) - c.get("alpha", 1.0) * lap
    if problem.kind == "advection":
        return net.rows(z, {t_axis: 1}) + c.get("c", 1.0) * net.rows(z, {0: 1})
    if problem.kind == "wave":
        return net.rows(z, {t_axis: 2}) - c.get("c", 1.0) ** 2 * lap
    raise ValueError(f"no PINN operator for {problem.kind}")


def _points(spans, count: int) -> list[np.ndarray]:
    """Interior collocation points: a Chebyshev tensor grid with about ``count`` points."""
    per_axis = max(4, math.ceil(count ** (1.0 / len(spans))))
    nodes = [lo + (hi - lo) * (_chebyshev(per_axis) + 1) / 2 for lo, hi in spans]
    return [p.ravel() for p in np.meshgrid(*nodes, indexing="ij")]


def solve_pinn(problem: PDEProblem, n: int, options: dict) -> SolveResult:
    spans = _inputs(problem)
    units = _units(problem, n)
    if units > MAX_UNITS:
        raise ValueError(f"PINN with {units} hidden units (limit {MAX_UNITS}); this n is too large")
    net = _Net(spans, units, options.get("scale", _scale(problem)), seed=1000 + n)
    time_dep = problem.time_dependent
    dim = problem.dim

    def field(name, pts):
        x, y = pts[0], pts[1] if dim == 2 else 0.0 * pts[0]
        t = pts[dim] if time_dep else 0.0 * pts[0]
        return np.asarray(problem.fn(name)(x, y, t), dtype=float) * np.ones_like(pts[0])

    blocks, targets, weights = [], [], []
    interior = _points(spans, COLLOCATION * units)
    z = net.normalise(*interior)
    a_int = _operator_rows(problem, net, z)
    blocks.append(a_int)
    targets.append(field("source", interior))
    weights.append(1.0)
    scale = float(np.sqrt(np.mean(a_int**2))) or 1.0
    per_edge = max(8, math.ceil(math.sqrt(COLLOCATION * units)) * 2)

    def edge_points(axis: int, value: float) -> list[np.ndarray]:
        others = [s for k, s in enumerate(spans) if k != axis]
        free = _points(others, per_edge ** len(others)) if others else []
        pts, k = [], 0
        for i in range(len(spans)):
            if i == axis:
                pts.append(np.full(free[0].shape if free else (1,), value))
            else:
                pts.append(free[k])
                k += 1
        return pts

    lam = scale * math.sqrt(len(z) / max(1, per_edge))
    for axis in range(dim):
        lo, hi = spans[axis]
        if problem.boundary == "periodic":
            left, right = edge_points(axis, lo), edge_points(axis, hi)
            zl, zr = net.normalise(*left), net.normalise(*right)
            for der in (None, {axis: 1}):
                blocks.append(net.rows(zl, der) - net.rows(zr, der))
                targets.append(np.zeros(len(zl)))
                weights.append(lam if der is None else lam / net.chain[axis])
        else:
            for value in (lo, hi):
                pts = edge_points(axis, value)
                blocks.append(net.rows(net.normalise(*pts)))
                targets.append(field("boundary_value", pts))
                weights.append(lam)
    if time_dep:
        pts = edge_points(dim, 0.0)
        z0 = net.normalise(*pts)
        blocks.append(net.rows(z0))
        targets.append(field("initial", pts))
        weights.append(lam)
        if problem.kind == "wave":
            blocks.append(net.rows(z0, {dim: 1}))
            targets.append(field("initial_velocity", pts))
            weights.append(lam / net.chain[dim])

    matrix = np.vstack([w * b for w, b in zip(weights, blocks)])
    rhs = np.concatenate([w * t for w, t in zip(weights, targets)])
    coef, _, rank, _ = lstsq(matrix, rhs, lapack_driver="gelsd")

    axes = grid(problem, n)
    nodes = mesh(axes)
    out_pts = [p.ravel() for p in nodes]
    if time_dep:
        out_pts.append(np.full(out_pts[0].shape, float(problem.t_final)))
    u = (net.rows(net.normalise(*out_pts)) @ coef).reshape(nodes[0].shape)
    if problem.boundary == "dirichlet":  # the grid's boundary nodes take the given values, as for the grid methods
        g = evaluate(problem.fn("boundary_value"), nodes, problem.t_final or 0.0)
        edge = np.zeros(u.shape, dtype=bool)
        for axis in range(dim):
            index = [slice(None)] * dim
            for end in (0, -1):
                index[axis] = end
                edge[tuple(index)] = True
        u[edge] = g[edge]
    else:
        u[..., -1] = u[..., 0] if dim == 1 else u[..., -1]
    flops = matrix.shape[0] * matrix.shape[1] ** 2
    return SolveResult(u=u, grid=axes, n=n, t=problem.t_final, work=flops / FLOPS_PER_WORK,
                       info={"units": units, "collocation": len(z), "rank": int(rank)})


def cost(problem: PDEProblem, n: int) -> float:
    units = _units(problem, n) + 1
    rows = COLLOCATION * units * 1.3  # interior plus boundary and initial rows
    return rows * units**2 / FLOPS_PER_WORK


def _spec(kind: str, dims, boundaries) -> SolverSpec:
    return SolverSpec(
        name=f"{kind}.pinn", kind=kind, dims=dims, boundaries=boundaries, order=4.0, solve=solve_pinn, cost=cost,
        traits=("neural",),
        description="PINN: one-hidden-layer tanh network, physics-informed loss solved by least squares "
                    "(random hidden weights); space-time, no grid",
    )


SOLVERS = (
    _spec("poisson", (1, 2), ("dirichlet",)),
    _spec("heat", (1,), ("dirichlet",)),
    _spec("advection", (1,), ("periodic",)),
    _spec("wave", (1,), ("dirichlet",)),
)
