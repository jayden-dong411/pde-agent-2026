import math

import pytest

from pde_agent.agent import BASE_N, MAX_N
from pde_agent.benchmark import equation_for, reference_problems
from pde_agent.curriculum import basic_curriculum, frontier_lessons
from pde_agent.problem import PDEProblem
from pde_agent.verify import interpolation_mismatch, resolution_fields, resolution_floor

R3, R3_EXACT = "sin(32*pi*x)", "sin(32*pi*x)*exp(-0.0001*(32*pi)**2*t)"


def heat(initial, exact=None):
    return PDEProblem(kind="heat", domain=[(0.0, 1.0)], t_final=0.01, coefficients={"alpha": 1e-4},
                      initial=initial, exact=exact)


def floor(problem, tol=1e-3):
    eq = equation_for(problem.kind)
    order = eq.time_order if eq else int(problem.time_dependent)
    return resolution_floor(problem, tol, order, BASE_N[problem.dim], MAX_N[problem.dim])


def test_mismatch_sees_what_the_nodes_miss():
    p = PDEProblem(kind="poisson", domain=[(0.0, 1.0)], source="sin(16*pi*x)")
    # every node of n = 16 is a zero of sin(16*pi*x); the probes never sit on a node
    assert interpolation_mismatch(p.fn("source"), p.domain, 16, 0.0) == pytest.approx(2 / math.pi, rel=1e-3)
    assert interpolation_mismatch(p.fn("source"), p.domain, 256, 0.0) < 1e-4
    assert interpolation_mismatch(heat("0").fn("initial"), [(0.0, 1.0)], 16, 0.0) == 0.0


@pytest.mark.parametrize("item", basic_curriculum() + frontier_lessons() + reference_problems(), ids=lambda i: i.name)
def test_existing_problems_keep_their_starting_grid(item):
    assert floor(item.problem, item.tol) == BASE_N[item.problem.dim]


@pytest.mark.parametrize("problem,expected", [
    (heat(R3), 128),
    (heat(R3, R3_EXACT), 128),
    (heat("sin(pi*x) + 0.05*sin(64*pi*x)"), 128),
    (heat("0"), 16),
    (heat("sin(5000*pi*x)"), None),
    (PDEProblem(kind="poisson", domain=[(0, 1), (0, 1)], source="sin(16*pi*x)*sin(16*pi*y)"), 64),  # not the 1-D numbers
    (PDEProblem(kind="poisson", domain=[(0, 1)] * 3, source="sin(pi*x)*cos(y)"), 4),
    (PDEProblem(kind="poisson", domain=[(0, 1)] * 3, source="sin(16*pi*x)"), 64),
], ids=["aliased", "aliased-exact", "hidden-5%", "zero", "unresolvable", "2d", "3d-smooth", "3d-aliased"])
def test_resolution_floor(problem, expected):
    assert floor(problem) == expected


def test_non_finite_data_is_left_to_the_solver():  # log(x) is -inf at the node x = 0
    p = PDEProblem(kind="heat", domain=[(0.0, 1.0)], t_final=0.1, initial="log(x)")
    assert math.isnan(interpolation_mismatch(p.fn("initial"), p.domain, 16, 0.0))
    assert floor(p) == 16


def test_fields_checked_for_resolution():
    wave = PDEProblem(kind="wave", domain=[(0.0, 1.0)], t_final=1.0, initial="sin(pi*x)", source="x*t")
    assert resolution_fields(wave, 2) == [("initial", 0.0), ("initial_velocity", 0.0),
                                          ("source", 0.0), ("source", 0.5), ("source", 1.0)]
    known = PDEProblem(kind="wave", domain=[(0.0, 1.0)], t_final=1.0, exact="sin(pi*x)*cos(t)")
    assert resolution_fields(known, 2) == [("exact", 1.0)]
    assert resolution_fields(PDEProblem(kind="poisson", domain=[(0.0, 1.0)], source="1"), 0) == [("source", 0.0)]
    bare = PDEProblem(kind="heat", domain=[(0.0, 1.0)], t_final=1.0)
    assert resolution_fields(bare, 1) == [("source", 0.0), ("source", 0.5), ("source", 1.0)]
