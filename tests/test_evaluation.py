"""pde-agent eval: false successes, train/test split, memory on/off, Chinese parse fixtures."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent.cli import main
from pde_agent.equations import builtin_equations
from pde_agent.evaluation import (ARMS, GROUPS, CaseResult, EvalCase, _grade, all_cases, blind, dedupe, eval_cases,
                                   fingerprint, gate_failures, memory_gain, run_eval, run_parse_eval, summarize)
from pde_agent.llm import Brain, LLMError
from pde_agent.parse_fixtures import compare_parse, expected_reply, parse_fixtures


def test_cases_split_into_train_and_held_out_test_groups_with_exact_solutions():
    cases = eval_cases()
    names = [c.name for c in cases]
    assert len(names) == len(set(names))
    for part in ("pde", "ode"):
        splits = {c.split for c in cases if c.part == part}
        assert splits == {"train", "test"}
    assert all(c.problem.has_exact() for c in cases)
    assert not any("burgers-cole-hopf" in c.name for c in cases)  # no built-in method: nothing to grade
    assert {c.part for c in eval_cases(("ode",))} == {"ode"}
    assert all(c.group == "" for c in cases if c.split == "train")
    assert {c.group for c in cases if c.split == "test"} == set(GROUPS)
    # stage 4 (jumps and shocks) is held out by regularity: nothing like it is trained on
    assert not any(c.problem.kind == "burgers_inviscid" for c in cases if c.split == "train")
    assert any(c.problem.kind == "burgers_inviscid" and c.group == "regularity" for c in cases)


def test_a_test_problem_that_repeats_a_training_problem_is_dropped():
    kept, dropped = dedupe(all_cases())
    pairs = {(c.name, twin.name) for c, twin in dropped}
    assert ("test/heat-decay", "train/heat-1d") in pairs  # the same problem under another name
    assert ("test/advection-smooth", "train/advection-1d") in pairs
    trained = {fingerprint(c.problem) for c in kept if c.split == "train"}
    assert not any(fingerprint(c.problem) in trained for c in kept if c.split == "test")


def test_fingerprint_is_the_math_not_the_spelling():
    from pde_agent.problem import PDEProblem

    def heat(**kw):
        base = dict(kind="heat", domain=[(0.0, 1.0)], t_final=1.0, coefficients={"alpha": 0.1},
                    exact="sin(pi*x)*exp(-0.1*pi**2*t)")
        return PDEProblem(**{**base, **kw})

    same = heat(exact="exp(-0.1*pi**2*t)*sin(x*pi)", name="other", description="另一个名字")
    assert fingerprint(heat()) == fingerprint(same)
    assert fingerprint(heat()) == fingerprint(heat(exact=None, initial="sin(pi*x)"))  # the data a solver gets
    assert fingerprint(heat()) != fingerprint(heat(coefficients={"alpha": 0.2}))
    assert fingerprint(heat()) != fingerprint(heat(t_final=2.0))
    unit_rate = "sin(pi*x)*exp(-pi**2*t)"
    assert fingerprint(heat(coefficients={}, exact=unit_rate)) == fingerprint(heat(coefficients={"alpha": 1.0},
                                                                                   exact=unit_rate))  # default filled
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


ODE_SMALL = ("train/ode-stiff-1e3", "train/ode-logistic", "test/exam-stiff-1e4", "test/exam-stiff-1e2",
             "test/exam-logistic")


def _small():
    return [c for c in eval_cases(("ode",)) if c.name in ODE_SMALL]


def test_memory_modes_off_frozen_online(tmp_path):
    results = run_eval(_small(), home=tmp_path / "run", time_budget=30)
    assert [r.arm for r in results] == ["train"] * 2 + list(ARMS) * 3
    assert all(r.correct and not r.false_success for r in results)
    by_arm = {arm: [r.experience for r in results if r.arm == arm] for arm in ARMS}
    assert by_arm["memory_off"] == [0, 0, 0]  # every solve starts with no experience
    assert by_arm["memory_frozen"] == [2, 2, 2]  # the training experience, never more
    assert by_arm["memory_online"] == [2, 3, 4]  # keeps learning from each test solve
    assert not any((tmp_path / "run").rglob("experience.jsonl"))  # nothing written to disk


def test_frozen_and_off_do_not_depend_on_the_order_of_the_test_problems():
    def picks(seed):
        rs = run_eval(_small(), time_budget=30, seed=seed, arms=("memory_off", "memory_frozen"))
        return {(r.case.name, r.arm): (r.method, r.n, r.claimed_error) for r in rs if r.arm != "train"}, \
            [r.case.name for r in rs if r.arm == "memory_off"]

    listed, order = picks(None)
    shuffled, other_order = picks(3)
    assert order != other_order  # the seed did change the order
    assert listed == shuffled


def test_an_agent_that_does_not_remember_keeps_nothing_between_solves(tmp_path):
    from pde_agent.agent import PDEAgent

    agent = PDEAgent(home=tmp_path, remember=False)
    case = _small()[0]
    agent.solve(case.problem, case.tol)
    agent.solve(case.problem, case.tol)
    assert agent.memory.episodes == []  # in v2.0 the second solve read the first one's episode


def test_the_run_directory_must_be_empty(tmp_path):
    (tmp_path / "experience.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不是空的"):
        run_eval(_small(), home=tmp_path)


def test_memory_gain_counts_only_problems_both_got_right():
    case = next(c for c in eval_cases(("ode",)) if c.split == "test")
    other = EvalCase("test/other", "", case.problem, case.tol, "test", "ode", "param")
    rs = [CaseResult(case, "memory_off", True, "a", 8, 1e-4, 1e-4, 4, 100.0, 1.0),
          CaseResult(case, "memory_frozen", True, "b", 8, 1e-4, 1e-4, 2, 50.0, 0.5),
          CaseResult(other, "memory_off", True, "a", 8, 1e-4, 1e-4, 4, 100.0, 1.0),
          CaseResult(other, "memory_frozen", True, "b", 8, 1e-4, 1.0, 1, 1.0, 0.1)]  # cheap but wrong
    g = memory_gain(rs, "memory_frozen")
    assert (g["both_correct"], g["only_base"], g["runs_ratio"], g["work_ratio"]) == (1, 1, 0.5, 0.5)
    assert len(gate_failures(rs)) == 1 and "test/other" in gate_failures(rs)[0]


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


def test_chinese_fixtures_cover_pdes_new_equations_and_odes_and_their_own_parse_passes():
    fixtures = parse_fixtures()
    assert 15 <= len(fixtures) <= 30
    assert len({fx.name for fx in fixtures}) == len(fixtures)
    assert all(any("一" <= ch <= "鿿" for ch in fx.text) for fx in fixtures)
    assert {fx.expected.kind for fx in fixtures} >= {"poisson", "heat", "advection", "wave", "ode"}
    assert sum(fx.expected.kind == "ode" for fx in fixtures) >= 4
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


def _fixture(name):
    return next(f for f in parse_fixtures() if f.name == name)


def _grade_reply(name, change):
    from pde_agent.llm import problem_from_reply

    fx = _fixture(name)
    return compare_parse(fx, problem_from_reply(change(expected_reply(fx))), builtin_equations())


def test_a_wrong_operator_of_a_new_equation_is_caught():
    # the case from the review: the reaction term replaced, everything else unchanged, used to grade as right
    wrong = _grade_reply("reaction-diffusion", _set(equation="u_t = D*u_xx - 99*u + f"))
    assert wrong and wrong[0].startswith("方程")
    assert _grade_reply("burgers", _set(equation="u_t + u_x = nu*u_xx + f"))  # nonlinearity dropped
    assert _grade_reply("burgers", _set(equation="u_t + u*u_x = nu*u_xx")) == []  # "+ f" is implied
    assert _grade_reply("burgers", _set(equation="2*u_t + 2*u*u_x = 2*nu*u_xx + 2*f")) == []  # scaled through
    assert _grade_reply("reaction-diffusion", _set(source="0.1"))  # a source the text does not have
    assert _grade_reply("reaction-diffusion", lambda d: {**d, "coefficients": [{"name": "D", "value": 0.1}]})
    assert _grade_reply("burgers", _set(equation="u_tt + u*u_x = nu*u_xx + f"))  # wrong time order
    assert _grade_reply("burgers", _set(equation="u_t + u*u_x = nu*u_xx +"))  # unreadable


def test_every_mutated_field_is_rejected():
    """Change each mathematical field of each fixture's right answer: the grader must notice every one."""
    caught, missed = 0, []
    for fx in parse_fixtures():
        p = fx.expected
        mutations = {}
        if p.kind == "ode":
            mutations["rhs"] = lambda d: {**d, "ode": {**d["ode"], "rhs": [d["ode"]["rhs"][0] + " + 0.5",
                                                                           *d["ode"]["rhs"][1:]]}}
            mutations["initial"] = lambda d: {**d, "ode": {**d["ode"], "initial": [d["ode"]["initial"][0] + " + 1",
                                                                                   *d["ode"]["initial"][1:]]}}
            mutations["t_final"] = lambda d: {**d, "t_final": 2 * d["t_final"]}
            if p.exact:
                mutations["exact"] = lambda d: {**d, "ode": {**d["ode"], "exact": []}}
        else:
            mutations["domain"] = lambda d: {**d, "domain": [{**d["domain"][0], "upper": d["domain"][0]["upper"] + 0.5},
                                                              *d["domain"][1:]]}
            mutations["source"] = lambda d: {**d, "source": f"({d['source'] or '0'}) + 1"}
            if p.coefficients:
                mutations["coefficient"] = lambda d: {**d, "coefficients": [
                    {**c, "value": c["value"] + 0.37 * (1 + abs(c["value"]))} for c in d["coefficients"]]}
            if p.time_dependent:
                mutations["t_final"] = lambda d: {**d, "t_final": 2 * d["t_final"]}
                mutations["initial"] = lambda d: {**d, "initial": f"({d['initial'] or d['exact'].replace('t', '0')}) "
                                                                  "+ x*(1 - x)"}
            if p.boundary == "dirichlet":
                mutations["boundary_value"] = lambda d: {**d, "boundary_value": f"({d['boundary_value'] or '0'}) + 1"}
                mutations["boundary"] = lambda d: {**d, "boundary": "periodic"}
            if p.kind == "wave":
                mutations["initial_velocity"] = lambda d: {**d, "initial_velocity": f"({d['initial_velocity'] or '0'}) + x"}
            if p.exact:
                mutations["exact"] = lambda d: {**d, "exact": ""}
        for field, change in mutations.items():
            try:
                wrong = _grade_reply(fx.name, change)
            except ValueError:
                wrong = ["invalid problem"]  # the parse would be asked again: never graded as right
            if wrong:
                caught += 1
            else:
                missed.append((fx.name, field))
    assert not missed
    assert caught >= 110


def test_ode_parses_are_graded_by_their_meaning():
    swap = lambda d: {**d, "ode": {"rhs": ["-4*y2", "y1"], "initial": ["0", "1"], "exact": []}}  # x = y2, x' = y1
    assert _grade_reply("ode-spring", swap) == []
    named = lambda d: {**d, "coefficients": [{"name": "r", "value": 2.0}], "ode": {**d["ode"], "rhs": ["r*y1 - r*y1**2"]}}
    assert _grade_reply("ode-logistic", named) == []
    assert _grade_reply("ode-damped", lambda d: {**d, "ode": {**d["ode"], "rhs": ["y2", "-y1 + 0.2*y2"]}})  # sign
    assert any("右端项" in m for m in _grade_reply("ode-predator-prey",
                                                lambda d: {**d, "ode": {**d["ode"], "rhs": ["1.5*y1 - y1*y2", "-3*y2"]}}))
    from pde_agent.llm import problem_from_reply

    as_pde = problem_from_reply({**expected_reply(_fixture("heat-rod"))})
    assert compare_parse(_fixture("ode-logistic"), as_pde, builtin_equations())[0].startswith("方程类型")


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


def test_eval_command(tmp_path, capsys, monkeypatch):
    import pde_agent.evaluation as evaluation

    every = evaluation.all_cases
    monkeypatch.setattr(evaluation, "all_cases", lambda parts: [c for c in every(parts) if c.name in ODE_SMALL])
    out_json = tmp_path / "eval.json"
    assert main(["--home", str(tmp_path), "eval", "--only", "ode", "--seed", "1", "--gate", "--json", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "假成功率" in out and "无经验" in out and "冻结" in out and "在线" in out and "--parse" in out
    assert "经验收益" in out and "参数留出" in out and "种子 1" in out
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(data["solve"]["summary"]) == set(ARMS)
    assert all(r["part"] == "ode" for r in data["solve"]["results"])
    assert data["environment"]["pde_agent"] and data["solve"]["seed"] == 1
    assert set(data["solve"]["memory_gain"]) == {"memory_frozen", "memory_online"}


def test_eval_gate_fails_on_a_false_success(tmp_path, capsys, monkeypatch):
    import pde_agent.evaluation as evaluation

    case = next(c for c in eval_cases(("ode",)) if c.split == "test")
    liar = CaseResult(case, "memory_off", True, "ode.liar", 8, 1e-9, 1.0, 1, 1.0, 0.0)
    monkeypatch.setattr(evaluation, "run_eval", lambda *a, **k: [liar])
    assert main(["--home", str(tmp_path), "eval", "--only", "ode"]) == 0  # without --gate: report only
    assert main(["--home", str(tmp_path), "eval", "--only", "ode", "--gate"]) == 1
    assert "评测门槛没有通过" in capsys.readouterr().out


def test_eval_parse_only_uses_the_configured_model(tmp_path, capsys, monkeypatch):
    import pde_agent.llm

    monkeypatch.setattr(pde_agent.llm, "make_brain", lambda *a, **k: FixtureBrain())
    assert main(["--home", str(tmp_path), "--llm", "deepseek", "eval", "--parse-only"]) == 0
    out = capsys.readouterr().out
    n = len(parse_fixtures())
    assert f"解析正确 {n}/{n}" in out and "无经验" not in out
