"""Poisson equation ``-k * laplace(u) = f`` with Dirichlet boundaries."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

from ..problem import PDEProblem
from .base import SolveResult, SolverSpec, evaluate, grid, laplacian_1d, mesh


def interior_laplacian(u: np.ndarray, hs: tuple[float, ...]) -> np.ndarray:
    """Standard (2*dim + 1)-point Laplacian of ``u`` at interior nodes."""
    core = (slice(1, -1),) * u.ndim
    out = np.zeros(tuple(s - 2 for s in u.shape))
    for axis, h in enumerate(hs):
        lo, hi = list(core), list(core)
        lo[axis], hi[axis] = slice(0, -2), slice(2, None)
        out += (u[tuple(lo)] - 2.0 * u[core] + u[tuple(hi)]) / h**2
    return out


def laplacian_matrix(m: int, hs: tuple[float, ...]) -> sparse.csc_matrix:
    """Interior Laplacian on an m^dim block, C ordering (last axis fastest)."""
    blocks = [laplacian_1d(m, h) for h in hs]
    if len(blocks) == 1:
        return blocks[0]
    eye = sparse.identity(m, format="csc")
    return (sparse.kron(blocks[0], eye) + sparse.kron(eye, blocks[1])).tocsc()


def solve_fd(problem: PDEProblem, n: int, options: dict) -> SolveResult:
    k = problem.coefficients.get("k", 1.0)
    axes = grid(problem, n)
    points = mesh(axes)
    hs = tuple(float(ax[1] - ax[0]) for ax in axes)
    core = (slice(1, -1),) * problem.dim

    u = evaluate(problem.fn("boundary_value"), points)
    u[core] = 0.0
    rhs = -evaluate(problem.fn("source"), tuple(p[core] for p in points)) / k
    rhs -= interior_laplacian(u, hs)
    m = n - 1
    u[core] = spsolve(laplacian_matrix(m, hs), rhs.ravel()).reshape((m,) * problem.dim)
    return SolveResult(u=u, grid=axes, n=n, work=float(m**problem.dim), info={"unknowns": m**problem.dim})


def cost(problem: PDEProblem, n: int) -> float:
    return float((n - 1) ** problem.dim)


SOLVERS = (
    SolverSpec(
        name="poisson.fd2",
        kind="poisson",
        dims=(1, 2),
        boundaries=("dirichlet",),
        order=2.0,
        solve=solve_fd,
        cost=cost,
        description="Second-order central differences, sparse direct solve",
    ),
)
