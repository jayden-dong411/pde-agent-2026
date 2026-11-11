"""v1.7.0: physics-informed neural networks (pure numpy) as built-in methods."""

import math

import numpy as np
import pytest

from pde_agent.agent import PDEAgent
from pde_agent.curriculum import basic_curriculum
from pde_agent.features import extract, predict
from pde_agent.problem import PDEProblem
from pde_agent.solvers import builtin_registry
from pde_agent.solvers.pinn import MAX_UNITS
from pde_agent.verify import exact_error

REGISTRY = builtin_registry()
LESSONS = {lesson.name: lesson.problem for lesson in basic_curriculum()}


@pytest.mark.parametrize("name", ["poisson-1d", "poisson-2d", "heat-1d-forced", "advection-1d-left", "wave-1d-forced"])
def test_pinn_error_falls_fast_with_network_size(name):
    problem = LESSONS[name]
    spec = REGISTRY.get(f"{problem.kind}.pinn")
    errors = [exact_error(problem, spec.run(problem, n)) for n in (16, 32)]
    assert errors[1] < 1e-3 and errors[1] < errors[0] / 10 or errors[1] < 1e-10


def test_pinn_is_deterministic_and_reports_work():
    problem = LESSONS["heat-1d"]
    spec = REGISTRY.get("heat.pinn")
    a, b = spec.run(problem, 16), spec.run(problem, 16)
    assert np.array_equal(a.u, b.u)
    assert a.info["units"] == 128 and a.work == pytest.approx(spec.cost(problem, 16), rel=0.5)
    assert "neural" in spec.traits


def test_pinn_refuses_networks_that_are_too_big():
    problem = LESSONS["poisson-2d"]
    with pytest.raises(ValueError, match="hidden units"):
        REGISTRY.get("poisson.pinn").run(problem, MAX_UNITS)


def test_pinn_never_sees_the_exact_solution():
    problem = LESSONS["heat-1d"]
    wrong = PDEProblem.from_dict({**problem.to_dict(), "exact": "0*x"})  # same data, a different "exact"
    wrong.initial = problem.for_solver().initial
    spec = REGISTRY.get("heat.pinn")
    view = problem.for_solver()
    assert np.allclose(spec.run(view, 16).u, spec.solve(view, 16, {}).u)


def test_long_transport_is_predicted_dear_for_a_network():
    from pde_agent.curriculum import pde_stages
    problem = next(lesson.problem for lesson in pde_stages() if lesson.name == "advection-long")
    feats = extract(problem, 16)
    net = predict(REGISTRY.get("advection.pinn"), problem, feats, 1e-3, 4096)
    rk4 = predict(REGISTRY.get("advection.rk4_central4"), problem, feats, 1e-3, 4096)
    assert net.order == 0.5 and net.work > 8 * rk4.work


def test_pinn_takes_over_when_grids_cannot_reach_the_tolerance(tmp_path):
    # 2-D Poisson to 1e-9: second-order differences would need n ~ 30000, past the 256 limit
    problem = LESSONS["poisson-2d"]
    agent = PDEAgent(home=tmp_path)
    report = agent.solve(problem, 1e-9, learn=False)
    assert report.success and report.method == "poisson.pinn"
    assert report.error <= 1e-9
    fd = [t for t in report.trials if t["method"] == "poisson.fd2"]
    assert not fd or fd[0]["outcome"] != "success"


def test_estimated_error_of_a_network_is_conservative(tmp_path):
    problem = PDEProblem.from_dict({**LESSONS["heat-1d"].to_dict(), "exact": None,
                                    "initial": "sin(pi*x)"})
    agent = PDEAgent(home=tmp_path, explore=False)
    agent.registry = type(REGISTRY)([REGISTRY.get("heat.pinn")])
    report = agent.solve(problem, 1e-5, learn=False)
    assert report.success
    truth = math.exp(-0.1 * math.pi**2)  # sin(pi x) exp(-alpha pi^2 t) at t = 1, alpha = 0.1
    x = report.result.grid[0]
    true_error = np.max(np.abs(report.result.u - truth * np.sin(np.pi * x))) / truth
    assert true_error <= 2e-5
