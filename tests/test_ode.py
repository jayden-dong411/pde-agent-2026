import json
import math

import numpy as np
import pytest

from pde_agent.agent import PDEAgent
from pde_agent.benchmark import ode_reference_problems, score_method
from pde_agent.cli import main
from pde_agent.curriculum import ode_curriculum, ode_exam
from pde_agent.features import extract, predict
from pde_agent.memory import clean_episode
from pde_agent.ode import ODEProblem, problem_from_dict
from pde_agent.problem import PDEProblem
from pde_agent.solvers import builtin_registry
from pde_agent.verify import exact_error

LESSONS = {lesson.name: lesson for lesson in ode_curriculum() + ode_exam()}


def lesson(name):
    return LESSONS[name].problem


def test_problem_checks_its_data():
    decay = ODEProblem(rhs=["-k*y1"], initial=["1"], t_final=1, coefficients={"k": 2}, exact=["exp(-2*t)"])
    assert decay.signature() == "ode/0d/m1" and decay.dim == 0 and decay.has_exact()
    assert np.allclose(decay.f(0.0, np.array([3.0])), [-6.0])
    with pytest.raises(ValueError, match="initial values"):
        ODEProblem(rhs=["y2", "-y1"], initial=["1"], t_final=1)
    with pytest.raises(ValueError, match="does not start"):
        ODEProblem(rhs=["-y1"], initial=["1"], t_final=1, exact=["2*exp(-t)"])
    with pytest.raises(ValueError, match="may not use"):
        ODEProblem(rhs=["-y1*x"], initial=["1"], t_final=1)
    with pytest.raises(ValueError, match="unknown names"):
        ODEProblem(rhs=["-y1 - y2"], initial=["1"], t_final=1)
    with pytest.raises(ValueError, match="coefficient"):
        ODEProblem(rhs=["-y1"], initial=["1"], t_final=1, coefficients={"y1": 1.0})
    with pytest.raises(ValueError, match="unsupported"):
        ODEProblem(rhs=["__import__('os')"], initial=["1"], t_final=1)


def test_dict_round_trip_and_solver_view():
    p = lesson("ode-harmonic")
    data = json.loads(json.dumps(p.to_dict()))
    q = problem_from_dict(data)
    assert isinstance(q, ODEProblem) and q == p
    assert isinstance(problem_from_dict({"kind": "heat", "domain": [[0, 1]], "t_final": 1, "initial": "x"}), PDEProblem)
    view = p.for_solver()
    assert not view.has_exact() and p.has_exact() and view.rhs == p.rhs


@pytest.mark.parametrize("method,order", [
    ("ode.euler", 1), ("ode.heun", 2), ("ode.rk4", 4), ("ode.verlet", 2),
    ("ode.backward_euler", 1), ("ode.trapezoid", 2), ("ode.bdf2", 2),
])
def test_fixed_step_methods_converge_at_their_order(method, order):
    p = lesson("ode-harmonic")
    spec = builtin_registry().get(method)
    e1, e2 = (exact_error(p, spec.run(p, n)) for n in (512, 1024))
    assert math.log2(e1 / e2) == pytest.approx(order, abs=0.3)


@pytest.mark.parametrize("method", ["ode.rk45", "ode.radau", "ode.bdf"])
def test_adaptive_methods_tighten_with_n(method):
    p = lesson("ode-damped")
    spec = builtin_registry().get(method)
    coarse, fine = spec.run(p, 32), spec.run(p, 512)
    assert exact_error(p, fine) < exact_error(p, coarse) / 20
    assert fine.info["rtol"] == pytest.approx(512.0 ** -2)


def test_work_counts_linear_solves_for_implicit_methods():
    p = lesson("ode-heat-lines")  # tridiagonal Jacobian: banded LU, not dense
    assert p.costs == {"f": 15.0, "jac": 43.0, "lu": 60.0, "solve": 45.0}
    result = builtin_registry().get("ode.backward_euler").run(p, 64)
    counts = result.info["counts"]
    assert counts["lu"] >= 64 and result.work == sum(counts[k] * p.costs[k] for k in counts)


def test_verlet_only_takes_separable_systems():
    verlet = builtin_registry().get("ode.verlet")
    assert verlet.supports(lesson("ode-kepler")) and verlet.supports(lesson("ode-harmonic"))
    assert not verlet.supports(lesson("ode-damped")) and not verlet.supports(lesson("ode-lotka-volterra"))


def test_stiffness_is_read_from_the_jacobian():
    stiff = {name for name in LESSONS if "stiff" in extract(lesson(name), 16).tags}
    assert {"ode-stiff-1e3", "ode-stiff-1e6", "ode-robertson", "ode-vdp-1000", "ode-heat-lines"} <= stiff
    # Hamiltonian saddles (Kepler) and oscillators decay as fast as they grow: not stiff
    assert not stiff & {"ode-decay", "ode-harmonic", "ode-kepler", "ode-stiff-10", "ode-vdp-1", "ode-lotka-volterra"}
    assert extract(lesson("ode-stiff-1e6"), 16).stiffness == pytest.approx(2e6, rel=1e-6)


def test_stability_decides_explicit_steps_on_stiff_problems():
    reg = builtin_registry()
    p = lesson("ode-stiff-1e6")
    feats = extract(p, 16)
    rk4 = predict(reg.get("ode.rk4"), p, feats, 1e-3, 2**20)
    radau = predict(reg.get("ode.radau"), p, feats, 1e-3, 2**20)
    assert rk4.start >= 2e6 / 2.785 and any("稳定性" in note for note in rk4.notes)
    assert rk4.work > 1000 * radau.work
    gentle = lesson("ode-stiff-10")
    feats = extract(gentle, 16)
    assert predict(reg.get("ode.rk4"), gentle, feats, 1e-3, 2**20).work < predict(
        reg.get("ode.radau"), gentle, feats, 1e-3, 2**20).work


def test_agent_switches_to_implicit_methods_from_stiffness_alone(tmp_path):
    agent = PDEAgent(home=tmp_path, remember=False)
    implicit = {"ode.backward_euler", "ode.trapezoid", "ode.bdf2", "ode.radau", "ode.bdf"}
    for name in ("ode-stiff-1e3", "ode-stiff-1e6", "ode-vdp-1000", "exam-stiff-1e4"):
        report = agent.solve(lesson(name), LESSONS[name].tol, learn=False)
        assert report.success, report.message
        # no explicit method was even tried: the stiffness ratio predicted they would be held back
        assert {t["method"] for t in report.trials} <= implicit, report.trials
    report = agent.solve(lesson("ode-stiff-10"), 1e-3, learn=False)
    assert report.success and report.trials[0]["method"] not in implicit


@pytest.mark.parametrize("name", sorted(LESSONS), ids=str)
def test_agent_solves_ode_stages(tmp_path, name):
    problem, tol = lesson(name), LESSONS[name].tol
    agent = PDEAgent(home=tmp_path)
    report = agent.solve(problem, tol, learn=False)
    assert report.success, report.message
    if problem.has_exact():
        assert exact_error(problem, report.result) <= tol
    else:
        # the self-estimated error must be honest: compare with a much tighter Radau solution
        reference = agent.registry.get("ode.radau").run(problem, 1 << 14).u
        true_error = np.max(np.abs(report.result.u - reference)) / np.max(np.abs(reference))
        assert true_error <= 3 * tol


def test_ode_episodes_can_be_shared(tmp_path):
    agent = PDEAgent(home=tmp_path)
    agent.solve(lesson("ode-decay"), 1e-3)
    episode = clean_episode(agent.memory.episodes[-1])
    assert episode["signature"] == "ode/0d/m1" and problem_from_dict(episode["problem"]) == lesson("ode-decay")


def test_cli_runs_ode_stages_and_exam(tmp_path, capsys):
    home = str(tmp_path)
    assert main(["--home", home, "curriculum", "--ode", "--level", "0"]) == 0
    out = capsys.readouterr().out
    assert "ode-kepler" in out and "成功 7/7" in out and "不刚性" in out
    assert main(["--home", home, "curriculum", "--exam"]) == 0
    out = capsys.readouterr().out
    assert "exam-stiff-1e4" in out and "成功 7/7" in out and "刚性（" in out
    path = tmp_path / "stiff.json"
    path.write_text(json.dumps(lesson("ode-stiff-1e6").to_dict()), encoding="utf-8")
    assert main(["--home", home, "solve", str(path)]) == 0
    assert "成功" in capsys.readouterr().out


def test_bench_scores_ode_methods():
    bench = next(b for b in ode_reference_problems() if b.name == "ode-decay")
    score = score_method(builtin_registry().get("ode.rk4"), bench, repeat=1)
    assert score.best is not None and score.best.n == 16
