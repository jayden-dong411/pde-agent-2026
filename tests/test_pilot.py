"""v2.4.0: on shock problems the methods are compared by trial runs on two coarse grids."""

import math

from pde_agent.agent import PILOT_N, PDEAgent
from pde_agent.problem import PDEProblem
from pde_agent.solvers import builtin_registry
from pde_agent.verify import relative_error, restrict


def burgers(initial, t_final, boundary="periodic", domain=(0.0, 1.0), **kw):
    return PDEProblem(kind="burgers_inviscid", domain=[domain], t_final=t_final, initial=initial,
                      boundary=boundary, **kw)


def riemann(u_left, u_right, t_final=0.5):
    jump = f"{u_left!r} + ({u_right - u_left!r})*Heaviside(x)"
    exact = f"{u_left!r} + ({u_right - u_left!r})*Heaviside(x - {(u_left + u_right) / 2!r}*t)"
    return burgers(jump, t_final, boundary="dirichlet", domain=(-1.0, 1.0), boundary_value=jump, exact=exact)


def pilots(report):
    return [a for a in report.attempts if a.status == "pilot"]


def test_smooth_data_breaking_into_a_shock_pick_muscl_over_rusanov(tmp_path):
    # Rusanov's error here is about 10 times MUSCL's on the same grid: it needed n = 4096 in v2.3.0
    problem = burgers("sin(2*pi*x)", 0.3)
    report = PDEAgent(home=tmp_path, time_budget=300).solve(problem, 1e-3, learn=False)
    assert report.success
    assert report.method in ("burgers_inviscid.muscl", "burgers_inviscid.weno5")
    assert report.n <= 1024
    assert {a.method for a in pilots(report)} >= {"burgers_inviscid.rusanov", "burgers_inviscid.muscl"}
    assert {a.n for a in pilots(report)} == {PILOT_N, 2 * PILOT_N}
    assert all(a.work > 0 for a in pilots(report))  # the trial runs count in the episode's work
    piloted = [p for p in report.predictions.values() if p.piloted]
    assert len(piloted) >= 2 and all("试算" in p.notes[-1] for p in piloted)
    # the answer is right: compare with a fine WENO5 run
    reference = builtin_registry().get("burgers_inviscid.weno5").run(problem, 4096)
    assert relative_error(report.result.u, restrict(reference, report.n), "l1") < 2e-3


def test_riemann_shock_keeps_rusanov_and_uses_the_measured_grid(tmp_path):
    # flat states around the shock: MUSCL is only ~2x more accurate and ~7x dearer per grid, Rusanov is right
    report = PDEAgent(home=tmp_path, time_budget=300).solve(riemann(1.0, 0.0), 1e-3, learn=False)
    assert report.success and report.method == "burgers_inviscid.rusanov"
    assert report.n == 512  # v2.3.0 went to 1024 from the feature guess; the exact pilot errors say 512 suffices
    assert pilots(report)


def test_easy_problem_skips_the_pilot(tmp_path):
    # the trial runs would cost more than PILOT_SHARE of the cheapest solve: no pilot
    report = PDEAgent(home=tmp_path, time_budget=120).solve(riemann(1.0, 0.0), 1e-2, learn=False)
    assert report.success and not pilots(report)
    assert not any(p.piloted for p in report.predictions.values())


def test_smooth_problem_has_no_pilot(tmp_path):
    problem = PDEProblem(kind="advection", domain=[(0.0, 1.0)], t_final=0.5, coefficients={"c": 1.0},
                         boundary="periodic", exact="sin(2*pi*(x - t))")
    report = PDEAgent(home=tmp_path, time_budget=120).solve(problem, 1e-4, learn=False)
    assert report.success and not pilots(report)


def test_pilot_prediction_prices_the_continuous_grid(tmp_path):
    report = PDEAgent(home=tmp_path, time_budget=300).solve(burgers("sin(2*pi*x)", 0.3), 1e-3, learn=False)
    for name, p in report.predictions.items():
        if p.piloted:
            assert math.isfinite(p.work) and p.work > 0
            assert p.n & (p.n - 1) == 0  # the grid to run stays a power of two
