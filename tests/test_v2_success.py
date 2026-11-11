"""v2.0.0: the success judgement. Reproductions from the 2026-11-11 review, turned into regression tests."""

import numpy as np
import pytest

from pde_agent import PDEAgent, PDEProblem
from pde_agent.agent import SolveReport
from pde_agent.crosscheck import Check, CrossCheck, cross_check, ode_check
from pde_agent.equations import builtin_equations
from pde_agent.ode import ODEProblem
from pde_agent.solvers.base import SolveResult, SolverRegistry, SolverSpec


def _agent(tmp_path, *specs):
    agent = PDEAgent(home=tmp_path, remember=False)
    agent.registry = SolverRegistry(specs)
    return agent


POISSON = dict(kind="poisson", domain=[(0, 1)], source="1", boundary_value="0")


def _poisson_solver(offset=0.0, shift=0.0, t=None):
    def solve(problem, n, options):
        x = np.linspace(0, 1, n + 1)
        return SolveResult(u=x * (1 - x) / 2 + offset, grid=(x + shift,), n=n, t=t)
    return solve


def _spec(name, kind, solve, **kw):
    kw.setdefault("dims", (1,))
    kw.setdefault("boundaries", ("dirichlet",))
    kw.setdefault("order", 2)
    return SolverSpec(name=name, kind=kind, solve=solve, **kw)


# --- 4.1 the result contract ---------------------------------------------------------

def test_correct_solution_plus_one_is_rejected(tmp_path):
    """Review 4.1: u = x(1-x)/2 + 1 satisfies -u_xx = 1 inside, but not u(0) = u(1) = 0."""
    agent = _agent(tmp_path, _spec("poisson.offset", "poisson", _poisson_solver(offset=1.0)))
    report = agent.solve(PDEProblem(**POISSON), tol=1e-3, learn=False)
    assert not report.success
    assert report.status == "failed"
    assert any(c.name == "boundary" and c.status == "fail" for c in report.crosscheck.checks)
    assert report.trials[-1]["outcome"] == "contract"


def test_correct_solution_is_verified(tmp_path):
    agent = _agent(tmp_path, _spec("poisson.right", "poisson", _poisson_solver()))
    report = agent.solve(PDEProblem(**POISSON), tol=1e-3, learn=False)
    assert report.success and report.status == "verified"
    names = {c.name for c in report.crosscheck.checks}
    assert {"grid", "boundary", "residual"} <= names


def test_correct_solution_plus_one_rejected_even_with_exact(tmp_path):
    """With an exact solution only the final values are compared; the boundary check still runs."""
    agent = _agent(tmp_path, _spec("poisson.right", "poisson", _poisson_solver()))
    report = agent.solve(PDEProblem(**POISSON, exact="x*(1-x)/2"), tol=1e-3, learn=False)
    assert report.success
    assert report.crosscheck.checks[0].name == "exact"


def test_shifted_coordinates_are_rejected(tmp_path):
    agent = _agent(tmp_path, _spec("poisson.shift", "poisson", _poisson_solver(shift=0.1)))
    report = agent.solve(PDEProblem(**POISSON), tol=1e-3, learn=False)
    assert not report.success
    assert any(c.name == "grid" and c.status == "fail" for c in report.crosscheck.checks)


HEAT = dict(kind="heat", domain=[(0, 1)], t_final=0.1, initial="sin(pi*x)", boundary_value="0")
HEAT_EXACT = "exp(-pi**2*t)*sin(pi*x)"


def _heat_solver(wrong_initial=False, wrong_time=False):
    def solve(problem, n, options):
        x = np.linspace(0, 1, n + 1)
        t = problem.t_final
        u = np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
        if wrong_initial:  # right at t_final = 0.1 by construction, wrong at t -> 0
            u = u + np.sin(np.pi * x) * (0.1 - t) * 5.0
        return SolveResult(u=u, grid=(x,), n=n, t=t + 0.05 if wrong_time else t, steps=1, dt=t)
    return solve


def test_heat_right_answer_verified(tmp_path):
    agent = _agent(tmp_path, _spec("heat.exact", "heat", _heat_solver()))
    report = agent.solve(PDEProblem(**HEAT, exact=HEAT_EXACT), tol=1e-3, learn=False)
    assert report.success, report.summary()
    assert {"time", "boundary", "initial"} <= {c.name for c in report.crosscheck.checks}


def test_wrong_initial_value_is_rejected(tmp_path):
    agent = _agent(tmp_path, _spec("heat.bad_start", "heat", _heat_solver(wrong_initial=True)))
    report = agent.solve(PDEProblem(**HEAT, exact=HEAT_EXACT), tol=1e-3, learn=False)
    assert not report.success
    assert any(c.name == "initial" and c.status == "fail" for c in report.crosscheck.checks)


def test_wrong_return_time_is_rejected(tmp_path):
    agent = _agent(tmp_path, _spec("heat.late", "heat", _heat_solver(wrong_time=True)))
    report = agent.solve(PDEProblem(**HEAT, exact=HEAT_EXACT), tol=1e-3, learn=False)
    assert not report.success
    assert any(c.name == "time" and c.status == "fail" for c in report.crosscheck.checks)


WAVE = dict(kind="wave", domain=[(0, 1)], t_final=0.5, initial="sin(pi*x)", initial_velocity="0",
            boundary_value="0", exact="sin(pi*x)*cos(pi*t)")


def test_wrong_initial_velocity_is_rejected(tmp_path):
    """Starts from u0 but with velocity sin(pi x) instead of 0; lands on the exact answer at T = 0.5 by design."""
    def solve(problem, n, options):
        x = np.linspace(0, 1, n + 1)
        t = problem.t_final
        u = np.sin(np.pi * x) * np.cos(np.pi * t) + np.sin(np.pi * x) * t * (0.5 - t) * 2
        return SolveResult(u=u, grid=(x,), n=n, t=t)
    agent = _agent(tmp_path, _spec("wave.kick", "wave", solve))
    report = agent.solve(PDEProblem(**WAVE), tol=1e-3, learn=False)
    assert not report.success
    assert any(c.name == "initial_velocity" and c.status == "fail" for c in report.crosscheck.checks)


def test_builtin_solvers_still_verified(tmp_path):
    agent = PDEAgent(home=tmp_path, remember=False)
    for problem in (PDEProblem(**POISSON), PDEProblem(**HEAT), PDEProblem(**{**WAVE, "exact": None, "t_final": 0.3})):
        report = agent.solve(problem, tol=1e-3, learn=False)
        assert report.status == "verified", report.summary()


# --- 4.2 cross-check: pass / fail / inconclusive -------------------------------------

def test_empty_crosscheck_is_not_a_pass():
    assert CrossCheck().status == "inconclusive"
    assert not CrossCheck().passed


def test_failed_required_and_optional_checks():
    ok = Check("residual", 0.0, 0.01, True)
    missing = Check.inconclusive("residual", "boom")
    optional = Check.inconclusive("mass", "boom", required=False)
    assert CrossCheck([ok]).status == "pass"
    assert CrossCheck([ok, optional]).status == "pass"
    assert CrossCheck([missing, Check("mass", 0.0, 0.01, True, required=False)]).status == "inconclusive"
    assert CrossCheck([ok, Check("mass", 1.0, 0.01, False, required=False)]).status == "fail"


def test_reference_integration_error_is_inconclusive(monkeypatch):
    """Review 4.2: solve_ivp raising used to give passed=True with no checks."""
    import scipy.integrate

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(scipy.integrate, "solve_ivp", broken)
    problem = ODEProblem(rhs=["-y1"], initial=["1"], t_final=1.0)
    spec = SolverSpec(name="ode.x", kind="ode", dims=(0,), boundaries=("ivp",), order=1, solve=None)
    result = SolveResult(u=np.array([np.exp(-1.0)]), grid=(), n=10, t=1.0)
    out = ode_check(spec, problem, result, 1e-3)
    assert out.status == "inconclusive" and not out.passed
    assert out.checks and out.checks[0].status == "inconclusive"


def test_residual_error_is_inconclusive(monkeypatch):
    import pde_agent.crosscheck as cc

    def broken(*args, **kwargs):
        raise ValueError("interpolation failed")

    monkeypatch.setattr(cc, "residual_check", broken)
    problem = PDEProblem(**POISSON)
    spec = _spec("poisson.right", "poisson", _poisson_solver())
    result = spec.run(problem, 16)
    out = cross_check(builtin_equations().get("poisson"), spec, problem, result, 1e-3)
    assert out.status == "inconclusive"


def test_three_d_is_a_candidate_not_verified(tmp_path):
    """No residual check in 3-D yet: an answer that meets the estimate is a candidate, not verified."""
    def solve(problem, n, options):
        axes = tuple(np.linspace(0, 1, n + 1) for _ in range(3))
        x, y, z = np.meshgrid(*axes, indexing="ij")
        return SolveResult(u=np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z), grid=axes, n=n)

    agent = _agent(tmp_path, _spec("poisson.box", "poisson", solve, dims=(3,)))
    problem = PDEProblem(kind="poisson", domain=[(0, 1)] * 3, source="3*pi**2*sin(pi*x)*sin(pi*y)", boundary_value="0")
    report = agent.solve(problem, tol=1e-2, learn=False)
    assert report.status == "candidate", report.summary()
    assert not report.success and report.result is not None
    assert "候选解" in report.summary()


# --- 4.3 equation text vs kind ---------------------------------------------------------

def test_equation_text_conflicting_with_kind_is_rejected(tmp_path):
    """Review 4.3: -2*u_xx = f under poisson (k = 1) used to be solved with k = 1."""
    agent = PDEAgent(home=tmp_path, remember=False)
    with pytest.raises(ValueError, match="不一致"):
        agent.solve(PDEProblem(**POISSON, equation="-2*u_xx=f"), tol=1e-3, learn=False)


@pytest.mark.parametrize("text", ["-k*u_xx = f", "-u_xx = f", "f = -k*u_xx", "-k*u_xx - f = 0"])
def test_equivalent_equation_text_is_accepted(text):
    builtin_equations().complete(PDEProblem(**POISSON, equation=text))


@pytest.mark.parametrize("text", ["u_t = -alpha*u_xx + f", "u_t = 2*alpha*u_xx + f", "u_t = alpha*u_xx - u + f"])
def test_wrong_sign_coefficient_or_reaction_is_rejected(text):
    with pytest.raises(ValueError):
        builtin_equations().complete(PDEProblem(**HEAT, equation=text))


def test_matching_coefficient_in_text_is_accepted():
    builtin_equations().complete(PDEProblem(**POISSON, coefficients={"k": 2}, equation="-2*u_xx = f"))


# --- 4.4 adaptive ODE error -------------------------------------------------------------

@pytest.mark.parametrize("ratio", [2, 4, 8])
def test_adaptive_error_is_the_plain_difference(tmp_path, ratio):
    """Review 4.4: the estimate of an adaptive method is the difference itself, for any refinement ratio."""
    seen = []

    def solve(problem, n, options):
        seen.append(n)
        value = np.exp(-1.0) + (0.0 if len(seen) > 1 else 1e-2)
        return SolveResult(u=np.array([value]), grid=(), n=n, t=problem.t_final)

    spec = SolverSpec(name="ode.fake", kind="ode", dims=(0,), boundaries=("ivp",), order=5, solve=solve,
                      traits=("adaptive",))
    agent = _agent(tmp_path, spec)
    agent.max_runs_per_method = 2
    agent._next_n = staticmethod(lambda n, error, tol, order: n * ratio)
    report = agent.solve(ODEProblem(rhs=["-y1"], initial=["1"], t_final=1.0), tol=1e-6, learn=False)
    estimate = [a.error for a in report.attempts if a.error is not None][0]
    expected = 1e-2 / np.exp(-1.0)  # the plain relative difference, not divided by ratio**p - 1
    assert estimate == pytest.approx(expected, rel=1e-9)


# --- 4.5 status in reports, CLI and web ---------------------------------------------------

def test_status_in_report_json(tmp_path):
    from pde_agent.app import report_json

    agent = _agent(tmp_path, _spec("poisson.offset", "poisson", _poisson_solver(offset=1.0)))
    data = report_json(agent.solve(PDEProblem(**POISSON), tol=1e-3, learn=False))
    assert data["status"] == "failed" and data["status_text"] == "失败"
    assert data["crosscheck"]["status"] == "fail"


def test_status_text():
    report = SolveReport(problem=PDEProblem(**POISSON), tol=1e-3)
    assert report.status == "failed"
    report.candidate = True
    assert report.status == "candidate" and "未完成验证" in report.status_text
