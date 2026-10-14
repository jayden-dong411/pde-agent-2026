import time

import pytest
import sympy as sp

from pde_agent.benchmark import BURGERS
from pde_agent.curriculum import frontier_lessons
from pde_agent.equations import Equation, builtin_equations, check_operator, operator_from_text
from pde_agent.problem import T, X

UNIT, SQUARE = [(0.0, 1.0)], [(0.0, 1.0), (0.0, 1.0)]
RD = {"D": 0.5, "r": 2.0}


def reaction(sign=1):
    return Equation(kind="reaction_diffusion", form="u_t = D*u_xx - r*u + f", time_order=1,
                    operator=lambda u, x, y, t, c: sp.diff(u, t) - c["D"] * sp.diff(u, x, 2) + sign * c["r"] * u)


@pytest.mark.parametrize("dim,domain", [(1, UNIT), (2, SQUARE)])
@pytest.mark.parametrize("eq", list(builtin_equations()), ids=lambda e: e.kind)
def test_builtin_forms_match_their_operators(eq, dim, domain):
    assert check_operator(eq, eq.form, {}, dim, domain, 0.5 if eq.time_order else None) is None


def test_frontier_equations_match_their_operators():
    rd, _, burgers = (lesson.problem for lesson in frontier_lessons())
    assert check_operator(reaction(), rd.equation, rd.coefficients, 1, UNIT, 1.0) is None
    assert check_operator(BURGERS, burgers.equation, burgers.coefficients, 1, UNIT, 0.5) is None
    assert check_operator(BURGERS, "u_t + u*u_x = nu*u_xx", {"nu": 0.05}, 1, UNIT, 0.5) is None


def test_numbers_may_stand_in_for_coefficients():  # a text often has 0.5 where a name would be
    assert operator_from_text("u_t = 0.5*u_xx - 2*u", []).time_order == 1
    assert check_operator(reaction(), "u_t = 0.5*u_xx - 2*u", RD, 1, UNIT, 1.0) is None


def test_flipped_sign_is_caught():
    problem = check_operator(reaction(sign=-1), "u_t = D*u_xx - r*u + f", RD, 1, UNIT, 1.0)
    assert problem and "differ" in problem and "u_t = D*u_xx - r*u + f" in problem


def test_time_order_mismatch_is_caught():
    assert "time order" in check_operator(reaction(), "u_tt = D*u_xx - r*u + f", RD, 1, UNIT, 1.0)


def test_only_the_task_dimensions_are_compared():
    one_d = Equation(kind="line", form="", time_order=0, defaults={"k": 1.0},
                     operator=lambda u, x, y, t, c: -c["k"] * sp.diff(u, x, 2))
    assert check_operator(one_d, "-k * (u_xx + u_yy) = f", {}, 1, UNIT, None) is None
    assert check_operator(one_d, "-k * (u_xx + u_yy) = f", {}, 2, SQUARE, None)


def test_terms_without_u_belong_to_the_source():
    op = operator_from_text("u_t = D*u_xx - r*u + sin(x)", ["D", "r"])
    w = sp.sin(2 * X) * sp.exp(-T)
    assert sp.simplify(op.apply(w, RD) - (sp.diff(w, T) - 0.5 * sp.diff(w, X, 2) + 2.0 * w)) == 0


@pytest.mark.parametrize("text,order", [("-k*u_xx = f", 0), ("u_t = k*u_xx", 1), ("u_tt = k^2*u_xx + f", 2)])
def test_time_order_is_the_highest_time_derivative(text, order):
    assert operator_from_text(text, ["k"]).time_order == order


@pytest.mark.parametrize("text", [
    "",
    "u_t = D*u_xx = f",
    "u_t = D*u_xx + g",
    "u_t = D*Δu + f",
    "du/dt = D*u_xx",
    "u_t = D*u_xx + f**2",
    "u_t = D*u_xx + f(x, t)",  # u and f take no arguments
    "u_t = __import__('os')",
])
def test_unreadable_equations_are_rejected(text):
    with pytest.raises(ValueError):
        operator_from_text(text, ["D"])


@pytest.mark.parametrize("name", ["u", "f"])
def test_coefficients_named_u_or_f_are_rejected(name):
    with pytest.raises(ValueError):
        operator_from_text("u_t = u_xx + f", [name])


def test_mixed_derivatives_are_the_same_in_either_order():
    xt, tx = operator_from_text("u_xt = f", []), operator_from_text("u_tx = f", [])
    assert xt.derivatives == tx.derivatives == {"Dqtx": "tx"} and xt.expr == tx.expr


@pytest.mark.parametrize("text,message", [
    ("u_t = D*u_xx + f(x, t)", "write u and f without arguments, e.g. u_t = D*u_xx + f"),
    ("u_t = D*u_xx - r*u(x, t) + f", "write u and f without arguments, e.g. u_t = D*u_xx + f"),
    ("du/dt = D*d2u/dx2 - r*u", "unknown names in 'du/dt = D*d2u/dx2 - r*u': d2u, dt, du, dx2"),
    ("u_t = D*u_xx - k*u + f", "unknown names in 'u_t = D*u_xx - k*u + f': k"),
    ("u_t = D*u_xx = f", "'u_t = D*u_xx = f' must contain exactly one '='"),
    ("u_t - D*u_xx + f", "'u_t - D*u_xx + f' must contain exactly one '='"),
    ("u_t = D*Δu + f", "unsupported characters in 'u_t = D*Δu + f'"),
    ("u_t = __import__('os')", "unsupported characters in \"u_t = __import__('os')\""),
])
def test_the_reason_for_a_rejection_is_given(text, message):
    with pytest.raises(ValueError) as caught:
        operator_from_text(text, ["D", "r"])
    assert str(caught.value) == message


def test_an_equation_may_span_lines():
    assert operator_from_text("u_t =\n  D*u_xx\n  - r*u + f", ["D", "r"]).time_order == 1


def test_newlines_cannot_glue_checked_names_into_a_new_one():
    # sympify deletes newlines before it evaluates: l, e and n pass the name check, and `len` is a builtin
    with pytest.raises(ValueError):
        operator_from_text("u_t = l\ne\nn((1, 2)) * u_xx", ["l", "n"])
    with pytest.raises(ValueError):
        operator_from_text("u_t = D\nu_xx", ["D"])  # not D*u_xx: one unknown name


@pytest.mark.parametrize("text,names", [
    ("u_t = (D*u_xx", ["D"]),
    ("u_t = D*u_xx +", ["D"]),
    ("u_t = D*u_xx + 2(x + 1)", ["D"]),  # an implicit product: sympy calls a number
    ("u_t = D(x)*u_xx", ["D"]),  # a coefficient is not a function
    ("u_t = x.sin*u_xx", ["D"]),  # sympy raises AttributeError
    ("u_t = (D, 1)", ["D"]),  # not a scalar
    ("u_t = atan2(u, 0.5/u_x)", ["D"]),  # sympy raises TypeError once u_x is 0
    ("u_t = D*u_xx + exp(f)", ["D"]),  # one solution for f, but not to the first power
    ("u_t = Dqxx*u_xx", ["Dqxx"]),  # the coefficient would be taken for the derivative
])
def test_whatever_sympy_makes_of_a_text_the_answer_is_a_value_error(text, names):
    with pytest.raises(ValueError) as caught:
        operator_from_text(text, names)
    assert type(caught.value) is ValueError  # not sympy's SympifyError, which prints the whole failed call


@pytest.mark.parametrize("text", ["u_t = D*(u_xx + u_yy", "u_t = D*u_xx + u_yy)", "u_t = D)*(u_xx"])
def test_unbalanced_parentheses_are_rejected_and_the_text_is_quoted_as_written(text):
    with pytest.raises(ValueError) as caught:
        operator_from_text(text, ["D"])
    assert repr(text) in str(caught.value) and "Dq" not in str(caught.value)


def test_a_small_difference_is_caught():
    assert check_operator(reaction(), "u_t = 0.5001*u_xx - 2*u", RD, 1, UNIT, 1.0)  # D is 0.5: one part in 5000


def test_the_comparison_is_repeatable():
    args = reaction(sign=-1), "u_t = D*u_xx - r*u + f", RD, 1, UNIT, 1.0
    assert check_operator(*args) == check_operator(*args)


def test_check_operator_reports_instead_of_raising():
    unreadable = check_operator(reaction(), "u_t = D*u_xx - k*u + f", RD, 1, UNIT, 1.0)
    assert unreadable == "cannot read 'u_t = D*u_xx - k*u + f': unknown names in 'u_t = D*u_xx - k*u + f': k"
    failing = check_operator(reaction(), "u_t = D*u_xx + f", {"D": 0.5}, 1, UNIT, 1.0)  # the operator needs r
    assert failing == "cannot evaluate 'u_t = D*u_xx + f' against operator: KeyError: 'r'"


@pytest.mark.filterwarnings("error")
def test_values_that_are_not_finite_never_match():
    root = Equation(kind="root", form="", time_order=1, operator=lambda u, x, y, t, c: sp.diff(u, t) + sp.sqrt(-u))
    problem = check_operator(root, "u_t + sqrt(-u) = f", {}, 1, UNIT, 1.0)  # u > 0 at most points: nan on both sides
    assert problem and "differ" in problem and "nan vs nan" in problem


def test_a_large_power_does_not_slow_the_reading_down():
    start = time.perf_counter()
    operator_from_text("u_t = D*u_xx + (x + 1)^4000*u", ["D"])  # simplifying the solution for f takes tens of seconds
    assert time.perf_counter() - start < 5


@pytest.mark.filterwarnings("error")
def test_a_value_beyond_the_float_range_is_a_difference_not_an_error():
    # (x + 1)^10000 overflows a python float, which raises, but not a numpy scalar, which becomes inf
    problem = check_operator(reaction(), "u_t = D*u_xx - r*u + (x + 1)^10000*u + f", RD, 1, UNIT, 1.0)
    assert problem and "differ" in problem and "inf" in problem


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("term", ["sqrt(-1)*u", "(-1)^(1/3)*u"])
def test_a_complex_text_never_matches_a_real_operator(term):
    problem = check_operator(reaction(), f"u_t = D*u_xx - r*u + {term} + f", RD, 1, UNIT, 1.0)
    assert problem and "differ" in problem and "nan" in problem
