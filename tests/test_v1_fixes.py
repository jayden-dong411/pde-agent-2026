"""Regressions for the errors found in the v0.13.0 review (see docs/BUGFIXES-v1.0.0.md)."""

import math

import numpy as np
import pytest

from pde_agent.agent import PDEAgent, observed_rate
from pde_agent.app import PAGE, AppState, problem_from_form
from pde_agent.cloud import SyncState, sync
from pde_agent.curriculum import frontier_lessons, ode_curriculum
from pde_agent.equations import builtin_equations
from pde_agent.memory import SHARED_WEIGHT, ExperienceStore, clean_episode
from pde_agent.ode import ODEProblem
from pde_agent.problem import PDEProblem
from pde_agent.skills import static_check
from pde_agent.solvers import builtin_registry


def fourier_heat(f0, t, x, terms=2000):
    """u_t = u_xx on (0, 1), zero ends, from the sine series of the initial data."""
    k = np.arange(1, terms + 1)
    xs = np.linspace(0.0, 1.0, 100001)
    w = np.full(xs.size, 1.0 / (xs.size - 1))
    w[[0, -1]] *= 0.5
    b = np.array([2 * np.sum(w * f0(xs) * np.sin(kk * np.pi * xs)) for kk in k])
    return (b * np.exp(-(k * np.pi) ** 2 * t)) @ np.sin(np.outer(k, np.pi * x))


def true_error(report, f0, t):
    exact = fourier_heat(f0, t, report.result.grid[0])
    return float(np.max(np.abs(report.result.u - exact)) / np.max(np.abs(exact)))


# 1. estimated errors on rough data -------------------------------------------------------


def test_observed_rate_with_unequal_ratios():
    e = lambda n: n ** -1.5  # noqa: E731
    assert observed_rate(e(32) - e(64), e(64) - e(256), 2, 4) == pytest.approx(1.5, abs=1e-6)
    assert observed_rate(e(32) - e(64), e(64) - e(128), 2, 2) == pytest.approx(1.5, abs=1e-6)
    assert observed_rate(0.0, 1.0, 2, 2) is None


def test_kinked_initial_data_is_not_a_false_success(tmp_path):
    problem = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.02, coefficients={"alpha": 1.0},
                         initial="Abs(x - 0.5) - 0.5")
    report = PDEAgent(home=tmp_path, remember=False).solve(problem, 1e-2)
    assert report.success and report.error_kind == "estimated"
    assert true_error(report, lambda x: np.abs(x - 0.5) - 0.5, 0.02) <= 1e-2  # was 1.85e-2 in v0.13.0


def test_jumping_data_rule_out_crank_nicolson(tmp_path):
    problem = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.01, coefficients={"alpha": 1.0},
                         initial="Heaviside(x - 0.3) - Heaviside(x - 0.7)")
    report = PDEAgent(home=tmp_path, remember=False).solve(problem, 1e-2)
    assert report.predictions["heat.crank_nicolson"].work == math.inf
    assert report.success and report.method != "heat.crank_nicolson"
    step = lambda x: ((x > 0.3) & (x < 0.7)).astype(float)  # noqa: E731
    assert true_error(report, step, 0.01) <= 1e-2  # CN reported 6.9e-3 with a true 0.525 in v0.13.0


def test_ruled_out_method_is_skipped_not_run(tmp_path):
    from pde_agent.solvers.base import SolverRegistry

    problem = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.01, coefficients={"alpha": 1.0},
                         initial="Heaviside(x - 0.3) - Heaviside(x - 0.7)")
    agent = PDEAgent(home=tmp_path, remember=False)
    agent.registry = SolverRegistry([agent.registry.get("heat.crank_nicolson")])
    report = agent.solve(problem, 1e-2)
    assert not report.success and report.attempts == []
    assert report.trials == [{"method": "heat.crank_nicolson", "outcome": "skipped", "n": None, "error": None,
                              "error_kind": "estimated", "work": 0.0}]


def test_skipped_trials_do_not_count_as_failures():
    store = ExperienceStore()
    store.add({"signature": "heat/1d/dirichlet", "methods": [{"method": "heat.crank_nicolson", "outcome": "skipped"}]})
    assert "heat.crank_nicolson" not in store.stats("heat/1d/dirichlet")


def test_adaptive_estimate_holds_on_a_stiff_problem(tmp_path):
    from scipy.integrate import solve_ivp

    from pde_agent.solvers.base import SolverRegistry

    problem = next(l.problem for l in ode_curriculum() if l.name == "ode-vdp-1000")
    agent = PDEAgent(home=tmp_path, remember=False)
    agent.registry = SolverRegistry([agent.registry.get("ode.bdf")])
    report = agent.solve(problem, 1e-4)
    ref = solve_ivp(problem.f, (0, problem.t_final), problem.y0, method="Radau", jac=problem.jac,
                    rtol=1e-12, atol=1e-14).y[:, -1]
    assert report.success
    assert np.max(np.abs(report.result.u - ref)) / np.max(np.abs(ref)) <= 1e-4  # was 1.3e-4 with 1.8e-5 claimed


# 2. coefficient names --------------------------------------------------------------------


def test_misspelled_coefficient_is_an_error(tmp_path):
    problem = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.5, coefficients={"D": 0.01}, initial="sin(pi*x)")
    with pytest.raises(ValueError, match="没有系数 D"):
        PDEAgent(home=tmp_path, remember=False).solve(problem)


def test_coefficient_used_in_an_expression_is_fine(tmp_path):
    problem = PDEProblem(kind="heat", domain=[(0, 1)], t_final=0.1, coefficients={"alpha": 1.0, "a": 2.0},
                         initial="a*sin(pi*x)")
    assert PDEAgent(home=tmp_path, remember=False).solve(problem).success


@pytest.mark.parametrize("kind, coefficients, t_final", [("poisson", {"k": 0.0}, None), ("heat", {"alpha": -1.0}, 1.0)])
def test_meaningless_coefficients_are_errors(tmp_path, kind, coefficients, t_final):
    problem = PDEProblem(kind=kind, domain=[(0, 1)], t_final=t_final, coefficients=coefficients, initial="x*(1-x)")
    with pytest.raises(ValueError, match="方程："):
        PDEAgent(home=tmp_path, remember=False).solve(problem)


def test_app_reports_a_misspelled_coefficient_as_a_form_error(tmp_path):
    state = AppState(PDEAgent(home=tmp_path, remember=False), "claude")
    form = {"kind": "heat", "domain": [[0, 1]], "t_final": 0.5, "coefficients": {"D": 0.1}, "initial": "sin(pi*x)"}
    with pytest.raises(Exception, match="题目有误"):
        state.solve({"problem": form})


# 3. the web form sends the equation ------------------------------------------------------


def test_web_form_has_an_equation_field():
    page = PAGE.read_text(encoding="utf-8")
    assert 'id="equation"' in page and '"equation"]' in page  # an input, and part of FIELDS
    lesson = frontier_lessons()[0].problem.to_dict()
    assert problem_from_form({"problem": lesson}).equation == "u_t = D*u_xx - r*u + f"


def test_learning_from_the_page_reaches_the_language_model(tmp_path):
    asked = []

    class Brain:
        provider = "fake"

        def skill_session(self, task):
            asked.append(task.equation)
            raise RuntimeError("stop here")

    state = AppState(PDEAgent(home=tmp_path, brain=Brain(), remember=False), "claude", make_brain=lambda *a, **k: Brain())
    form = frontier_lessons()[0].problem.to_dict()
    with pytest.raises(RuntimeError):
        state.solve({"problem": form, "learn": True})
    assert asked == ["u_t = D*u_xx - r*u + f"]


# 4. static check -------------------------------------------------------------------------

HEADER = 'SOLVER = {"name": "heat.x"}\ndef solve(problem, n, options):\n    pass\n'


@pytest.mark.parametrize("line", [
    "from pde_agent.skills import subprocess",
    "from pde_agent.settings import Settings",
    "from pde_agent import skills",
    "import pde_agent.skill_api",
    "import pde_agent",
])
def test_package_modules_other_than_skill_api_are_refused(line):
    assert static_check(line + "\n" + HEADER)


@pytest.mark.parametrize("line", [
    "from pde_agent.skill_api import grid, SolveResult",
    "from pde_agent import skill_api",
    "import pde_agent.skill_api as api",
])
def test_skill_api_is_still_allowed(line):
    assert static_check(line + "\n" + HEADER) == []


def test_installed_skill_that_fails_the_static_check_is_not_loaded(tmp_path):
    from pde_agent.skills import SkillLibrary

    root = tmp_path / "skills"
    root.mkdir()
    (root / "heat__x.py").write_text("from pde_agent.skills import os\n" + HEADER, encoding="utf-8")
    (root / "heat__x.json").write_text('{"solver": "heat.x", "file": "heat__x.py", "exam_version": 2}', encoding="utf-8")
    errors = SkillLibrary(root).load_into(builtin_registry(), builtin_equations())
    assert len(errors) == 1 and "静态检查不通过" in errors[0]


# 6. ODEs and learning --------------------------------------------------------------------


def test_learning_an_ode_method_says_it_is_not_supported(tmp_path):
    class Brain:
        provider = "fake"

    outcome = PDEAgent(home=tmp_path, brain=Brain(), remember=False).learn(
        ODEProblem(rhs=["-y1"], initial=["1"], t_final=1.0, equation="y' = -y"))
    assert not outcome.success and "常微分方程" in outcome.message


# speculative findings that were confirmed ------------------------------------------------


def shared(member, n, i):
    return clean_episode({"id": f"{i:032x}", "signature": "heat/1d/dirichlet", "tol": 1e-3, "success": True,
                          "member": member, "methods": [{"method": "heat.ftcs", "outcome": "success", "n": n}]})


def test_one_member_cannot_outvote_everyone():
    store = ExperienceStore()
    store.add_shared([shared("mallory", 4, i) for i in range(100)])
    st = store.stats("heat/1d/dirichlet")["heat.ftcs"]
    assert st.tries == pytest.approx(20 * SHARED_WEIGHT)  # the latest 20, at half weight


def test_own_grids_come_before_shared_ones():
    store = ExperienceStore()
    store.add_shared([shared("mallory", 4, 1)])
    store.add({"signature": "heat/1d/dirichlet", "tol": 1e-3,
               "methods": [{"method": "heat.ftcs", "outcome": "success", "n": 256, "error_kind": "exact"}]})
    spec = builtin_registry().get("heat.ftcs")
    assert store.suggest_n("heat/1d/dirichlet", spec, 1e-3, "exact") == 256


def test_sync_finds_new_episodes_after_the_file_was_cleared(tmp_path):
    class Client:
        url = "http://hub.example"

        def __init__(self):
            self.sent = []

        def upload(self, episodes):
            self.sent += episodes
            return {"accepted": len(episodes)}

        def download(self, after):
            return {"episodes": [], "next": after, "more": False}

    def episode():
        return {"signature": "heat/1d/dirichlet", "tol": 1e-3, "methods": []}

    memory, client = ExperienceStore(), Client()
    for _ in range(3):
        memory.add(episode())
    sync(memory, tmp_path, client)
    memory.episodes.clear()  # the experience file was deleted
    memory.add(episode())
    sync(memory, tmp_path, client)
    assert len(client.sent) == 4  # v0.13.0 skipped the new one until there were more than 3 again
    assert SyncState(tmp_path).for_hub(client.url)["sent_id"] == memory.episodes[-1]["id"]


def test_verlet_evaluates_each_half_once_per_step(monkeypatch):
    problem = next(l.problem for l in ode_curriculum() if l.name == "ode-harmonic")
    calls = []
    original = ODEProblem.f_half
    monkeypatch.setattr(ODEProblem, "f_half",
                        lambda self, t, y, second: calls.append(second) or original(self, t, y, second))
    result = builtin_registry().get("ode.verlet").run(problem, 8)
    assert calls == [True] + [False, True] * 8 and result.info["counts"]["f"] == 8
