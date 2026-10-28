from pde_agent.agent import PDEAgent
from pde_agent.benchmark import reference_problems
from pde_agent.features import extract, predict
from pde_agent.memory import ExperienceStore, clean_episode
from pde_agent.problem import PDEProblem
from pde_agent.solvers import builtin_registry

BENCH = {b.name: b for b in reference_problems()}


def features(name):
    return extract(BENCH[name].problem, 16)


def test_tags_follow_the_problem_not_its_class():
    assert features("heat-decay").tags == ()
    assert features("heat-stiff").tags == ("stiff",)
    assert features("advection-smooth").tags == ()
    assert features("advection-long").tags == ("long",)
    step = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.1, initial="Heaviside(x - 0.5)")
    assert not extract(step, 64).smooth and "rough" in extract(step, 64).tags


def test_crank_nicolson_needs_a_fine_grid_to_damp_stiff_modes():
    bench = BENCH["heat-stiff"]
    spec = builtin_registry().get("heat.crank_nicolson")
    stiff = predict(spec, bench.problem, features("heat-stiff"), bench.tol, 4096)
    assert stiff.n == 512  # the benchmark measures 512 as the first grid where CN reaches the tolerance
    gentle = predict(spec, BENCH["heat-decay"].problem, features("heat-decay"), bench.tol, 4096)
    assert gentle.n < 512


def test_upwind_is_predicted_dear_on_long_transport():
    bench = BENCH["advection-long"]
    registry, feats = builtin_registry(), features("advection-long")
    upwind = predict(registry.get("advection.upwind"), bench.problem, feats, bench.tol, 4096)
    lw = predict(registry.get("advection.lax_wendroff"), bench.problem, feats, bench.tol, 4096)
    assert upwind.order == 0.5 and upwind.work > 8 * lw.work


def test_cold_agent_picks_the_right_heat_method_in_one_run(tmp_path):
    for name in ("heat-decay", "heat-stiff"):
        bench = BENCH[name]
        report = PDEAgent(home=tmp_path, remember=False).solve(bench.problem, bench.tol, learn=False)
        assert report.success and report.method == "heat.crank_nicolson" and report.solver_runs == 1


def test_experience_is_kept_apart_by_feature_tags(tmp_path):
    store = ExperienceStore(tmp_path / "e.jsonl")
    trial = {"method": "heat.ftcs", "outcome": "success", "n": 32, "work": 1.0}
    store.add({"signature": "heat/1d/dirichlet", "tol": 1e-3, "methods": [trial], "tags": ["stiff"]})
    store.add({"signature": "heat/1d/dirichlet", "tol": 1e-3, "methods": [{**trial, "outcome": "unstable"}], "tags": []})
    assert store.stats("heat/1d/dirichlet", ["stiff"])["heat.ftcs"].successes == 1
    assert store.stats("heat/1d/dirichlet", [])["heat.ftcs"].successes == 0
    assert store.stats("heat/1d/dirichlet", ["long"])["heat.ftcs"].tries == 2  # no such episodes: the whole class
    assert store.stats("heat/1d/dirichlet")["heat.ftcs"].tries == 2


def test_shared_episodes_keep_only_clean_tags():
    base = {"signature": "heat/1d/dirichlet", "tol": 1e-3, "methods": []}
    assert clean_episode({**base, "tags": ["stiff", "stiff"]})["tags"] == ["stiff"]
    for bad in (["Stiff!"], "stiff", [1]):
        try:
            clean_episode({**base, "tags": bad})
        except ValueError:
            continue
        raise AssertionError(f"accepted tags {bad!r}")
