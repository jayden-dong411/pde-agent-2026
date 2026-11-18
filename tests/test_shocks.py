"""v2.3.0: shocks are verified in weak form with the entropy condition; per-step CFL; heat LU reuse; PINN diagnostics."""

import math

import numpy as np
import pytest

from pde_agent.agent import PDEAgent
from pde_agent.crosscheck import Neighbours, weak_checks
from pde_agent.features import extract, predict
from pde_agent.problem import PDEProblem
from pde_agent.solvers import builtin_registry, heat
from pde_agent.solvers.base import SolveResult, SolverSpec, grid
from pde_agent.solvers.pinn import diagnostics_text, solve_pinn
from pde_agent.verify import relative_error, restrict, shock_time, weak_solution


def burgers(initial, t_final, boundary="periodic", domain=(0.0, 1.0), **kw):
    return PDEProblem(kind="burgers_inviscid", domain=[domain], t_final=t_final, initial=initial,
                      boundary=boundary, **kw)


def riemann(u_left, u_right, t_final=0.5):
    jump = f"{u_left!r} + ({u_right - u_left!r})*Heaviside(x)"
    return burgers(jump, t_final, boundary="dirichlet", domain=(-1.0, 1.0), boundary_value=jump)


# --- screening -----------------------------------------------------------------------------------------


def test_breaking_time_of_smooth_and_jumping_data():
    smooth = burgers("sin(2*pi*x)", 0.5)
    assert shock_time(smooth) == pytest.approx(1 / (2 * math.pi), rel=1e-3)
    assert shock_time(riemann(1.0, 0.0)) == 0.0
    assert shock_time(burgers("x", 0.5, boundary="dirichlet", boundary_value="x/(1+t)")) == math.inf
    advection = PDEProblem(kind="advection", domain=[(0, 1)], t_final=1, initial="sin(2*pi*x)", boundary="periodic")
    assert shock_time(advection) is None


def test_weak_solution_only_past_the_breaking_time():
    assert not weak_solution(burgers("sin(2*pi*x)", 0.1))  # breaks at 0.159
    assert weak_solution(burgers("sin(2*pi*x)", 0.5))
    assert weak_solution(riemann(1.0, 0.0))
    smooth_heat = PDEProblem(kind="heat", domain=[(0, 1)], t_final=1, initial="sin(pi*x)")
    assert not weak_solution(smooth_heat)


def test_a_shock_is_tagged_and_caps_the_order():
    registry = builtin_registry()
    late = burgers("sin(2*pi*x)", 0.5)
    early = burgers("sin(2*pi*x)", 0.05)
    feats = extract(late, 16)
    assert "shock" in feats.tags and "激波" in feats.describe()
    assert "shock" not in extract(early, 16).tags
    weno = registry.get("burgers_inviscid.weno5")
    assert predict(weno, late, feats, 1e-3, 4096).order == 1.0
    assert predict(weno, early, extract(early, 16), 1e-3, 4096).order > 1.0


# --- weak form and entropy -----------------------------------------------------------------------------


def moving_jump(u_left, u_right, speed):
    """A solver that returns a jump moving at ``speed``, as cell averages (so it moves smoothly in time)."""

    def solve(problem, n, options):
        (x,) = grid(problem, n)
        h = float(x[1] - x[0])
        share = np.clip((x + h / 2 - speed * problem.t_final) / h, 0.0, 1.0)  # of the cell right of the jump
        return SolveResult(u=u_left + (u_right - u_left) * share, grid=(x,), n=n, t=problem.t_final)

    return SolverSpec(name="burgers_inviscid.fake", kind="burgers_inviscid", dims=(1,), boundaries=("dirichlet",),
                      order=1.0, solve=solve)


def checks_of(spec, problem, n=2000, tol=1e-2):
    result = spec.run(problem, n)
    return {c.name: c for c in weak_checks(problem, result, tol, Neighbours(spec, problem, result, {}))}


def test_the_right_shock_passes_both():
    checks = checks_of(moving_jump(1.0, 0.0, 0.5), riemann(1.0, 0.0))
    assert checks["weak"].passed and checks["entropy"].passed


def test_a_shock_at_the_wrong_speed_fails_the_weak_form():
    # the limit is 20 x tol (the integrals carry an O(h) error at the jump), so this needs a tight tolerance
    checks = checks_of(moving_jump(1.0, 0.0, 0.8), riemann(1.0, 0.0), tol=1e-3)
    assert not checks["weak"].passed and checks["weak"].value > 0.05
    assert checks_of(moving_jump(1.0, 0.0, 0.5), riemann(1.0, 0.0), tol=1e-3)["weak"].passed


def test_an_expansion_shock_satisfies_the_weak_form_but_not_entropy():
    # 0 | 1 moving at the Rankine–Hugoniot speed 1/2: a weak solution, but the physical one is a rarefaction fan
    checks = checks_of(moving_jump(0.0, 1.0, 0.5), riemann(0.0, 1.0))
    assert checks["weak"].passed
    assert not checks["entropy"].passed


def test_the_agent_verifies_a_shock_from_smooth_data(tmp_path):
    problem = burgers("sin(2*pi*x)", 0.5)
    report = PDEAgent(home=tmp_path, time_budget=120).solve(problem, 1e-2, learn=False)
    assert report.success and report.status == "verified", report.summary()
    names = {c.name for c in report.crosscheck.checks}
    assert {"weak", "entropy"} <= names and "residual" not in names
    # the estimate holds against a much finer run of a higher order scheme
    reference = builtin_registry().get("burgers_inviscid.weno5").run(problem, 8192)
    assert relative_error(report.result.u, restrict(reference, report.n), "l1") < 1e-2


# --- per-step CFL --------------------------------------------------------------------------------------


def test_nonlinear_steps_follow_the_wave_speed():
    spec = builtin_registry().get("burgers_inviscid.rusanov")
    problem = riemann(1.0, 0.0)
    result = spec.run(problem, 200)
    h = float(result.grid[0][1] - result.grid[0][0])
    assert result.info["largest_step"] <= result.info["cfl"] * h / 1.0 * (1 + 1e-9)  # max |u| = 1
    # slower data take larger steps, and the run still lands on t_final
    slow = spec.run(riemann(0.25, 0.0), 200)
    assert slow.steps < result.steps and slow.t == problem.t_final
    assert np.isclose(slow.dt * slow.steps, problem.t_final)


# --- heat: LU reuse ------------------------------------------------------------------------------------


def test_heat_reuses_its_factorization_and_matches_a_dense_solve():
    heat._FACTORS.clear()
    lu = heat.factor(50, 0.7)
    rhs = np.random.default_rng(0).normal(size=50)
    dense = np.eye(50) * (1 + 1.4) - 0.7 * (np.eye(50, k=1) + np.eye(50, k=-1))
    assert np.allclose(lu.solve(rhs), np.linalg.solve(dense, rhs))
    hits = heat.CACHE_STATS["hits"]
    assert heat.factor(50, 0.7) is lu and heat.CACHE_STATS["hits"] == hits + 1
    for m in range(heat.CACHE_SIZE + 5):
        heat.factor(10 + m, 0.5)
    assert len(heat._FACTORS) == heat.CACHE_SIZE


@pytest.mark.parametrize("method", ["heat.crank_nicolson", "heat.backward_euler", "heat.bdf2"])
def test_implicit_heat_still_converges(method):
    spec = builtin_registry().get(method)
    if spec is None:
        pytest.skip(f"{method} not registered")
    problem = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.1, initial="sin(pi*x)", source="x*(1-x)",
                         exact=None, boundary_value="0")
    errs = []
    reference = builtin_registry().get("heat.crank_nicolson").run(problem, 1024).u[::32]
    for n in (32, 64):
        u = spec.run(problem, n).u
        errs.append(float(np.max(np.abs(u[:: n // 32] - reference))))
    assert errs[1] < errs[0]


# --- PINN diagnostics ----------------------------------------------------------------------------------


def test_pinn_reports_how_well_it_fits():
    problem = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.1, initial="sin(pi*x)",
                         exact="exp(-pi**2*t)*sin(pi*x)")
    result = solve_pinn(problem, 32, {})
    d = result.info["diagnostics"]
    assert {"condition", "residual", "fresh_residual", "boundary_error", "initial_error", "seed", "rank"} <= set(d)
    assert d["residual"] < 1e-3 and d["fresh_residual"] < 1e-3 and d["condition"] > 1
    text = diagnostics_text(result.info)
    assert text.startswith("PINN 残差") and "条件数" in text and "过拟合" not in text
    assert diagnostics_text({"diagnostics": {"residual": 1e-8, "fresh_residual": 0.5, "rank": 3, "units": 9}}) \
        .endswith("网络可能过拟合了训练点")
    assert diagnostics_text({"rank": 1}) == ""


def test_pinn_diagnostics_reach_the_attempts(tmp_path):
    from pde_agent.solvers.base import SolverRegistry

    problem = PDEProblem(kind="poisson", domain=[(0, 1)], source="pi**2*sin(pi*x)", exact="sin(pi*x)")
    agent = PDEAgent(home=tmp_path)
    agent.registry = SolverRegistry([builtin_registry().get("poisson.pinn")])  # the only method it may pick
    events = []
    agent.progress = events.append
    report = agent.solve(problem, 1e-4, learn=False)
    assert report.attempts and all(a.method == "poisson.pinn" for a in report.attempts)
    assert any(a.message.startswith("PINN 残差") for a in report.attempts)
    assert any("PINN 残差" in e.get("text", "") for e in events if e.get("stage") == "error")
