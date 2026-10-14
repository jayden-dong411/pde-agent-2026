import json

import numpy as np
import pytest

from pde_agent.benchmark import equation_for, reference_problems, run_benchmark, score_method
from pde_agent.cli import main
from pde_agent.problem import lambdify_field
from pde_agent.solvers import builtin_registry


@pytest.mark.parametrize("bench", reference_problems(), ids=lambda b: b.name)
def test_reference_solutions_satisfy_their_equations(bench):
    problem = bench.problem
    eq = equation_for(problem.kind)
    residual = eq.apply(problem.expr("exact"), problem.coefficients) - problem.expr("source")
    x = np.linspace(*problem.domain[0], 23)
    y = np.linspace(*problem.domain[-1], 19)[:, None] if problem.dim == 2 else 0.0
    for t in (0.0, 0.3 * (problem.t_final or 1.0), problem.t_final or 0.0):
        assert np.max(np.abs(lambdify_field(residual)(x, y, t))) < 1e-9


def test_methods_are_scored_by_time_to_tolerance():
    bench = next(b for b in reference_problems() if b.name == "advection-smooth")
    registry = builtin_registry()
    upwind = score_method(registry.get("advection.upwind"), bench, repeat=1)
    lw = score_method(registry.get("advection.lax_wendroff"), bench, repeat=1)
    assert lw.best.n == 128 and lw.best.error <= bench.tol
    assert lw.observed_order == pytest.approx(2.0, abs=0.1)
    assert upwind.best.n > lw.best.n


def test_benchmark_scores_the_agent_without_touching_its_memory(tmp_path):
    problems = [b for b in reference_problems() if b.name in ("wave-two-modes", "burgers-cole-hopf")]
    scores = list(run_benchmark(problems, builtin_registry(), agent_home=tmp_path, repeat=1))
    wave, burgers = scores
    assert wave.best.method == "wave.leapfrog" and wave.agent.success
    assert wave.agent.method == "wave.leapfrog" and wave.agent_overhead > 0
    assert burgers.methods == [] and not burgers.agent.success
    assert not (tmp_path / "experience.jsonl").exists()
    json.dumps([s.to_dict() for s in scores])


def test_bench_command(tmp_path, capsys):
    out_json = tmp_path / "bench.json"
    assert main(["--home", str(tmp_path), "bench", "--problem", "poisson-2d", "--json", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "poisson.fd2 ★" in out and "agent 解出 1/1 道，其中 1/1 道选中了最快的方法" in out
    assert json.loads(out_json.read_text(encoding="utf-8"))[0]["best_method"] == "poisson.fd2"
    assert main(["--home", str(tmp_path), "bench", "--problem", "nope"]) == 1
