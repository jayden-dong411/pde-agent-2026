"""v1.4.0: the cost model is corrected by what similar past problems measured."""

import math

from pde_agent.agent import PDEAgent
from pde_agent.curriculum import ode_curriculum
from pde_agent.features import calibrate, extract, predict
from pde_agent.memory import (
    CALIB_RADIUS,
    ExperienceStore,
    clean_episode,
    feature_distance,
)
from pde_agent.solvers import builtin_registry

SIG = "ode/0d/m1"
STIFF = {"floor": 16, "smooth": True, "stiffness": 2000.0, "travel": 0.0, "tags": ["stiff"], "scale": 1.0, "rate": 1000.0}


def episode(features, trials, tol=1e-3, signature=SIG):
    return {"signature": signature, "tol": tol, "success": True, "features": features, "tags": features["tags"],
            "methods": [{"method": m, "outcome": "success", "n": 64, "work": w, "predicted": p} for m, w, p in trials]}


def test_ratio_comes_from_near_problems_and_is_shrunk_toward_the_model(tmp_path):
    store = ExperienceStore(tmp_path / "e.jsonl")
    for _ in range(3):
        store.add(episode(STIFF, [("ode.bdf", 400.0, 200.0), ("ode.bdf2", 100.0, 400.0)]))
    cal = store.calibration(SIG, STIFF, 1e-3)
    assert cal["ode.bdf"].neighbours == 3 and cal["ode.bdf"].distance == 0.0
    assert 1.5 < cal["ode.bdf"].ratio < 2.0  # measured 2x, pulled a little toward 1
    assert 0.25 < cal["ode.bdf2"].ratio < 0.4


def test_far_problems_and_old_episodes_without_predictions_do_not_count(tmp_path):
    store = ExperienceStore(tmp_path / "e.jsonl")
    gentle = dict(STIFF, stiffness=0.5, rate=0.5, tags=[])
    assert feature_distance(STIFF, gentle, 1e-3, 1e-3) > CALIB_RADIUS
    store.add(episode(gentle, [("ode.bdf", 1000.0, 100.0)]))
    store.add(episode(STIFF, [("ode.bdf", 1000.0, None)]))  # recorded before v1.4.0
    assert store.calibration(SIG, STIFF, 1e-3) == {}


def test_another_class_of_the_same_kind_counts_further_away(tmp_path):
    store = ExperienceStore(tmp_path / "e.jsonl")
    store.add(episode(STIFF, [("ode.bdf2", 100.0, 400.0)], signature="ode/0d/m2"))
    store.add(episode(STIFF, [("ode.bdf2", 100.0, 400.0)], signature="heat/1d/dirichlet"))
    cal = store.calibration(SIG, STIFF, 1e-3)
    assert cal["ode.bdf2"].neighbours == 1 and cal["ode.bdf2"].distance == 2.0


def test_shared_episodes_keep_predictions_and_features():
    ep = clean_episode(episode(STIFF, [("ode.bdf", 400.0, 200.0)]))
    assert ep["methods"][0]["predicted"] == 200.0
    assert ep["features"]["stiffness"] == 2000.0 and ep["features"]["tags"] == ["stiff"]


def test_calibrated_prediction_reorders_implicit_methods():
    problem = next(l.problem for l in ode_curriculum() if l.problem.name == "ode-stiff-1e3")
    registry = builtin_registry()
    feats = extract(problem, 16)
    bdf = predict(registry.get("ode.bdf"), problem, feats, 1e-3, 2**20)
    bdf2 = predict(registry.get("ode.bdf2"), problem, feats, 1e-3, 2**20)
    assert bdf.work < bdf2.work  # the model alone prefers BDF
    calibrate(bdf, 2.0, 3)
    calibrate(bdf2, 0.5, 3)
    assert bdf2.work < bdf.work and bdf.model_work < bdf.work and bdf.measured == 3
    assert any("实测" in note for note in bdf.notes)


def test_rank_uses_expected_work_not_a_lucky_success_count(tmp_path):
    store = ExperienceStore(tmp_path / "e.jsonl")
    store.add(episode(STIFF, [("ode.bdf", 300.0, 150.0)]))
    store.add(episode(STIFF, [("ode.bdf", 300.0, 150.0)]))
    store.add(episode(STIFF, [("ode.bdf2", 200.0, 400.0)]))
    specs = [builtin_registry().get(n) for n in ("ode.bdf", "ode.bdf2")]
    # bdf has one more success (0.75 vs 0.67) but costs 1.5x as much: bdf2 goes first
    order = store.rank(SIG, specs, explore=False, predicted={"ode.bdf": 300.0, "ode.bdf2": 200.0}, tags=["stiff"])
    assert [s.name for s in order] == ["ode.bdf2", "ode.bdf"]


def test_agent_records_predictions_and_uses_them_next_time(tmp_path):
    lesson = next(l for l in ode_curriculum() if l.problem.name == "ode-stiff-1e3")
    agent = PDEAgent(home=tmp_path)
    first = agent.solve(lesson.problem, lesson.tol, learn=False)
    tried = [t for t in first.trials if t["outcome"] == "success"][0]
    assert tried["predicted"] == first.predictions[tried["method"]].model_work
    second = PDEAgent(home=tmp_path).solve(lesson.problem, lesson.tol, learn=False)
    p = second.predictions[tried["method"]]
    assert p.measured == 1 and not math.isclose(p.work, p.model_work)
    assert "相近题的实测" in second.choice() or second.predictions[second.method].measured == 0
