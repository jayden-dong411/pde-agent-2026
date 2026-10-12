import json

from pde_agent.cli import main


def test_curriculum_stats_and_skills(tmp_path, capsys):
    home = str(tmp_path)
    assert main(["--home", home, "curriculum", "--level", "1", "--rounds", "2"]) == 0
    out = capsys.readouterr().out
    assert "poisson-1d" in out and "第 2 轮：成功 2/2" in out

    assert main(["--home", home, "stats"]) == 0
    assert "poisson/1d/dirichlet" in capsys.readouterr().out

    assert main(["--home", home, "skills"]) == 0
    assert "heat.crank_nicolson" in capsys.readouterr().out


def test_solve_problem_file(tmp_path, capsys):
    problem = {
        "kind": "heat", "domain": [[0, 1]], "t_final": 0.5, "coefficients": {"alpha": 0.2},
        "initial": "sin(pi*x)", "exact": "sin(pi*x)*exp(-0.2*pi**2*t)", "name": "rod",
    }
    path = tmp_path / "rod.json"
    path.write_text(json.dumps(problem), encoding="utf-8")
    assert main(["--home", str(tmp_path / "home"), "solve", str(path), "--tol", "1e-4"]) == 0
    assert "rod：成功" in capsys.readouterr().out


def test_deepseek_without_a_key_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-deepseek")
    assert main(["--home", str(tmp_path), "--llm", "deepseek", "ask", "求解一维热方程"]) == 2
    err = capsys.readouterr().err
    assert "调用 DeepSeek 失败" in err and "DEEPSEEK_API_KEY" in err
