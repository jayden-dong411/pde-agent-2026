"""pde-agent eval: false successes, train/test split, memory on/off, Chinese parse fixtures."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent.cli import main
from pde_agent.equations import builtin_equations
from pde_agent.evaluation import CaseResult, EvalCase, _grade, blind, eval_cases, run_eval, run_parse_eval, summarize
from pde_agent.llm import Brain, LLMError
from pde_agent.parse_fixtures import compare_parse, expected_reply, parse_fixtures


def test_cases_split_into_train_and_test_with_exact_solutions():
    cases = eval_cases()
    names = [c.name for c in cases]
    assert len(names) == len(set(names))
    for part in ("pde", "ode"):
        splits = {c.split for c in cases if c.part == part}
        assert splits == {"train", "test"}
    assert all(c.problem.has_exact() for c in cases)
    assert not any("burgers" in c.name for c in cases)  # no built-in method: nothing to grade
    assert {c.part for c in eval_cases(("ode",))} == {"ode"}


@pytest.mark.parametrize("case", eval_cases(), ids=lambda c: c.name)
def test_blind_copy_hides_the_exact_solution_but_keeps_the_data(case):
    problem, hidden = case.problem, blind(case.problem)
    assert not hidden.has_exact()
    if problem.dim == 0:
        assert np.allclose(hidden.y0, problem.y0)
        return
    x = np.linspace(*problem.domain[0], 7)
    y = np.linspace(*problem.domain[-1], 5)[:, None] if problem.dim == 2 else 0.0
    if problem.time_dependent:
        assert np.allclose(hidden.fn("initial")(x, y, 0.0), problem.fn("initial")(x, y, 0.0))
    if problem.boundary == "dirichlet":
        for t in (0.0, problem.t_final or 0.0):
            for side in problem.domain[0]:
                yy = np.linspace(*problem.domain[-1], 5) if problem.dim == 2 else 0.0
                assert np.allclose(hidden.fn("boundary_value")(side, yy, t), problem.fn("exact")(side, yy, t))


def test_a_wrong_answer_reported_as_success_is_a_false_success():
    case = next(c for c in eval_cases(("ode",)) if c.split == "test")
    wrong = case.problem.exact_at(case.problem.t_final) + 1.0
    report = SimpleNamespace(success=True, method="ode.liar", n=8, error=1e-9, result=SimpleNamespace(u=wrong, grid=[]),
                             solver_runs=1, work=1.0, runtime=0.0, message="")
    seen = {}

    class LyingAgent:
        def solve(self, problem, tol, learn):
            seen["exact"] = problem.has_exact()
            return report

    result = _grade(LyingAgent(), case, "memory_off", hide_exact=True)
    assert seen["exact"] is False  # the agent never sees the exact solution
    assert result.reported and not result.correct and result.false_success
    honest = CaseResult(case, "memory_off", True, "m", 8, 1e-4, 1e-4, 1, 1.0, 0.0)
    failed = CaseResult(case, "memory_off", False, None, 8, 1e-2, 1e-2, 1, 1.0, 0.0)
    s = summarize([result, honest, failed], "memory_off")
    assert (s.cases, s.reported, s.correct, s.false_success) == (3, 2, 1, 1)
    assert s.false_success_rate == 0.5


def test_memory_on_trains_first_and_is_frozen_during_the_test(tmp_path):
    cases = [c for c in eval_cases(("ode",)) if c.name in ("train/ode-stiff-1e3", "test/exam-stiff-1e4",
                                                          "test/exam-stiff-1e2")]
    results = run_eval(cases, home=tmp_path, time_budget=30)
    assert [r.arm for r in results] == ["train", "memory_off", "memory_on", "memory_off", "memory_on"]
    assert all(r.correct and not r.false_success for r in results)
    lines = (tmp_path / "memory_on" / "experience.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # only the training episode was written
    assert not (tmp_path / "memory_off" / "experience.jsonl").exists()


# --- Chinese parse fixtures --------------------------------------------------------------


class FixtureBrain(Brain):
    """A stand-in model that answers each fixture with ``replies[name]`` (default: the right parse)."""

    def __init__(self, replies=None):
        self.by_text = {fx.text: (fx, (replies or {}).get(fx.name)) for fx in parse_fixtures()}

    def complete_json(self, system, messages, schema, effort=None, example=None):
        fx, reply = self.by_text[messages[0]["content"]]
        if isinstance(reply, Exception):
            raise reply
        data = reply(expected_reply(fx)) if reply else expected_reply(fx)
        return data, json.dumps(data)


def test_there_are_10_to_20_chinese_fixtures_and_their_own_parse_passes():
    fixtures = parse_fixtures()
    assert 10 <= len(fixtures) <= 20
    assert all(any("一" <= ch <= "鿿" for ch in fx.text) for fx in fixtures)
    assert {fx.expected.kind for fx in fixtures} >= {"poisson", "heat", "advection", "wave"}
    assert any(not fx.builtin for fx in fixtures)
    results = run_parse_eval(FixtureBrain())
    assert all(r.ok for r in results), [(r.name, r.mismatches, r.error) for r in results]


def _set(**changes):
    return lambda data: {**data, **changes}


def test_wrong_parses_are_caught():
    def coefficient(name, value):
        return lambda d: {**d, "coefficients": [{"name": name, "value": value}]}

    replies = {
        "heat-rod": coefficient("alpha", 1.0),  # forgot the diffusivity
        "poisson-1d-bc": _set(boundary_value=""),  # dropped u(2) = 1
        "advection-left": coefficient("c", 0.5),  # wrong sign of the speed
        "wave-velocity": _set(initial_velocity=""),  # lost u_t(x, 0)
        "heat-exact": _set(exact=""),  # dropped the stated exact solution
        "burgers": _set(kind="heat", equation="", coefficients=[]),  # forced into a known kind
        "advection-periodic": _set(boundary="dirichlet"),
        "heat-2d": lambda d: {**d, "domain": d["domain"][:1]},
        "heat-source": _set(t_final=0),
        "poisson-2d": _set(source="pi**2*sin(pi*x)*sin(pi*y)"),
        "poisson-1d": LLMError("API down"),
    }
    results = {r.name: r for r in run_parse_eval(FixtureBrain(replies))}
    for name in replies:
        assert not results[name].ok, name
    assert results["poisson-1d"].error.startswith("LLMError")
    assert any("系数 alpha" in m for m in results["heat-rod"].mismatches)
    assert any("边界值" in m for m in results["poisson-1d-bc"].mismatches)
    assert any("方程类型" in m for m in results["burgers"].mismatches)
    assert any("源项" in m for m in results["poisson-2d"].mismatches)
    assert sum(r.ok for r in results.values()) == len(parse_fixtures()) - len(replies)


def test_equivalent_parses_written_differently_pass():
    eq = builtin_equations()
    from pde_agent.llm import problem_from_reply

    def check(name, change):
        fx = next(f for f in parse_fixtures() if f.name == name)
        return compare_parse(fx, problem_from_reply(change(expected_reply(fx))), eq)

    # Dirichlet data only matter on the boundary; exact solution instead of initial data; other spelling
    assert check("heat-bc", _set(boundary_value="1 - x**3")) == []
    assert check("heat-rod", _set(source="", initial="sin(x*pi)")) == []
    assert check("poisson-1d", _set(source="9.8696044010893586*sin(3.14159265358979*x)")) == []
    assert check("reaction-diffusion", lambda d: {**d, "kind": "fisher_kpp", "coefficients": [{"name": "eps", "value": 0.01}],
                                                    "equation": "u_t = eps*u_xx + u*(1 - u) + f"}) == []


# --- command line ------------------------------------------------------------------------


def test_eval_command(tmp_path, capsys):
    out_json = tmp_path / "eval.json"
    assert main(["--home", str(tmp_path), "eval", "--only", "ode", "--json", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "假成功率" in out and "无经验" in out and "有经验" in out and "--parse" in out
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(data["solve"]["summary"]) == {"memory_off", "memory_on"}
    assert all(r["part"] == "ode" for r in data["solve"]["results"])


def test_eval_parse_only_uses_the_configured_model(tmp_path, capsys, monkeypatch):
    import pde_agent.llm

    monkeypatch.setattr(pde_agent.llm, "make_brain", lambda *a, **k: FixtureBrain())
    assert main(["--home", str(tmp_path), "--llm", "deepseek", "eval", "--parse-only"]) == 0
    out = capsys.readouterr().out
    n = len(parse_fixtures())
    assert f"解析正确 {n}/{n}" in out and "无经验" not in out
