import numpy as np
import pytest
import sympy as sp

from pde_agent.equations import builtin_equations, manufactured_problem
from pde_agent.problem import PDEProblem, T, X, parse_expr


def test_parse_expr_accepts_math():
    expr = parse_expr("sin(pi*x)*exp(-alpha*t) + x^2", {"alpha": 2.0})
    assert sp.simplify(expr - (sp.sin(sp.pi * X) * sp.exp(-2 * T) + X**2)) == 0


@pytest.mark.parametrize("text", [
    "__import__('os')",
    "eval(chr(95))",
    "open(x)",
    "x.__class__",
    "z + 1",
    "[x]",
    "lambda: x",
])
def test_parse_expr_rejects_code_and_unknown_names(text):
    with pytest.raises(ValueError):
        parse_expr(text)


def test_fields_fall_back_to_exact_solution():
    p = PDEProblem(kind="wave", domain=[(0, 1)], t_final=1.0, exact="sin(pi*x)*cos(t) + x*t")
    x = np.linspace(0, 1, 5)
    np.testing.assert_allclose(p.fn("initial")(x), np.sin(np.pi * x))
    np.testing.assert_allclose(p.fn("initial_velocity")(x), x)  # d/dt at t = 0
    np.testing.assert_allclose(p.fn("boundary_value")(1.0, 0.0, 0.5), 0.5, atol=1e-12)


def test_constant_fields_broadcast():
    p = PDEProblem(kind="poisson", domain=[(0, 1), (0, 2)], source="3")
    xx, yy = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 2, 4), indexing="ij")
    assert p.fn("source")(xx, yy).shape == (3, 4)
    assert p.fn("boundary_value")(xx, yy).shape == (3, 4)


def test_validation_errors():
    with pytest.raises(ValueError):
        PDEProblem(kind="heat", domain=[(1, 0)], t_final=1)
    with pytest.raises(ValueError):
        PDEProblem(kind="heat", domain=[(0, 1)], t_final=-1)
    with pytest.raises(ValueError):
        PDEProblem(kind="Heat Equation", domain=[(0, 1)])
    with pytest.raises(ValueError):
        PDEProblem(kind="poisson", domain=[(0, 1)], exact="x*t")  # steady but time dependent
    with pytest.raises(ValueError):
        PDEProblem(kind="heat", domain=[(0, 1)], t_final=1, coefficients={"x": 1.0})
    p = PDEProblem(kind="heat", domain=[(0, 1)], t_final=1)
    with pytest.raises(ValueError):
        p.fn("initial")  # no initial condition and no exact solution


def test_round_trip_through_dict():
    p = PDEProblem(kind="heat", domain=[(0, 2)], t_final=0.5, coefficients={"alpha": 0.3},
                   initial="sin(pi*x/2)", equation="u_t = alpha*u_xx", name="demo")
    q = PDEProblem.from_dict(p.to_dict())
    assert q == p
    assert q.signature() == "heat/1d/dirichlet"


def test_manufactured_source_vanishes_for_true_solution():
    heat = builtin_equations().get("heat")
    p = manufactured_problem(heat, "sin(pi*x)*exp(-0.1*pi**2*t)", [(0, 1)], 1.0, {"alpha": 0.1})
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(p.fn("source")(x, 0.0, 0.3), 0.0, atol=1e-12)


def test_complete_fills_default_coefficients():
    book = builtin_equations()
    p = book.complete(PDEProblem(kind="wave", domain=[(0, 1)], t_final=1.0, initial="sin(pi*x)"))
    assert p.coefficients == {"c": 1.0}
