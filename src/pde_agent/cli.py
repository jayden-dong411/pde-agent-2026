"""Command line: ``pde-agent curriculum | solve | ask | learn | stats | skills | app``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import __version__
from .agent import PDEAgent
from .curriculum import basic_curriculum, frontier_lessons, run_curriculum
from .llm import PROVIDERS, LLMError, default_provider
from .problem import PDEProblem


def _brain(args):
    if not getattr(args, "learn", False):
        return None
    from .llm import make_brain

    return make_brain(args.llm, args.model)


def _agent(args) -> PDEAgent:
    agent = PDEAgent(home=args.home, brain=_brain(args), explore=not args.no_explore)
    for problem in agent.skill_errors:
        print(f"警告：技能加载失败 {problem}", file=sys.stderr)
    return agent


def _fmt(value, spec=".2e") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return format(value, spec)


def cmd_curriculum(args) -> int:
    agent = _agent(args)
    lessons = basic_curriculum() + (frontier_lessons() if args.frontier else [])
    if args.level:
        lessons = [lesson for lesson in lessons if lesson.level <= args.level]
    print(f"{'轮':>2} {'课程':<24} {'结果':<4} {'方法':<26} {'n':>5} {'误差':>9} {'求解次数':>6} {'工作量':>10}")
    totals: dict[int, list[float]] = {}
    for round_, lesson, report in run_curriculum(agent, lessons, args.rounds, learn=agent.brain is not None):
        ok = "成功" if report.success else "失败"
        print(f"{round_:>2} {lesson.name:<24} {ok:<4} {report.method or '-':<26} {report.n or '-':>5} "
              f"{_fmt(report.error):>9} {report.solver_runs:>8} {report.work:>12.3g}")
        if not report.success:
            print(f"   └ {report.message}")
        for outcome in report.learned:
            status = f"学会了 {outcome.solver}" if outcome.success else f"学习失败：{outcome.message}"
            print(f"   └ {status}（{outcome.rounds} 轮）")
        t = totals.setdefault(round_, [0, 0, 0.0])
        t[0] += report.success
        t[1] += report.solver_runs
        t[2] += report.work
    for round_, (ok, runs, work) in totals.items():
        print(f"第 {round_} 轮：成功 {ok}/{len(lessons)}，共求解 {runs} 次，工作量 {work:.3g}")
    return 0


def cmd_solve(args) -> int:
    data = json.loads(Path(args.problem).read_text(encoding="utf-8"))
    agent = _agent(args)
    report = agent.solve(PDEProblem.from_dict(data), args.tol, learn=agent.brain is not None)
    print(report.summary())
    return 0 if report.success else 1


def cmd_ask(args) -> int:
    args.learn = True
    agent = _agent(args)
    report = agent.ask(args.text, args.tol)
    p = report.problem
    print(f"理解为：{p.kind}（{p.equation or p.signature()}），区域 {p.domain}"
          + (f"，t ∈ [0, {p.t_final:g}]" if p.t_final else ""))
    if p.description:
        print(f"假设：{p.description}")
    print(report.summary())
    return 0 if report.success else 1


def cmd_learn(args) -> int:
    args.learn = True
    data = json.loads(Path(args.problem).read_text(encoding="utf-8"))
    agent = _agent(args)
    problem = agent.equations.complete(PDEProblem.from_dict(data))
    outcome = agent.learn(problem, reason=args.reason or "user asked for a new method")
    if outcome.success:
        print(f"学会了 {outcome.solver}（{outcome.rounds} 轮）：{outcome.message}")
        return 0
    print(f"学习失败（{outcome.rounds} 轮）：{outcome.message}")
    if outcome.report:
        print(outcome.report.feedback())
    return 1


def cmd_stats(args) -> int:
    agent = _agent(args)
    rows = agent.memory.summary()
    if not rows:
        print("还没有经验记录。先运行：pde-agent curriculum")
        return 0
    print(f"{'问题类别':<26} {'方法':<34} {'尝试':>4} {'成功':>4} {'平均工作量':>12}")
    for r in rows:
        print(f"{r['signature']:<26} {r['method']:<34} {r['tries']:>6} {r['successes']:>6} {_fmt(r['mean_work'], '.3g'):>14}")
    print(f"共 {len(agent.memory.episodes)} 次求解记录：{agent.memory.path}")
    return 0


def cmd_skills(args) -> int:
    agent = _agent(args)
    print(f"已知方程：{', '.join(eq.kind for eq in agent.equations)}")
    for spec in agent.registry:
        print(f"  [{'内置' if spec.origin == 'builtin' else '学会'}] {spec.name:<36} {spec.order:g} 阶  {spec.description}")
    return 0


def cmd_app(args) -> int:
    from .app import AppState, serve

    agent = _agent(args)
    try:
        serve(AppState(agent, args.llm, args.model), args.port, open_browser=not args.no_browser)
    except OSError as exc:
        print(f"无法在端口 {args.port} 启动：{exc}（换一个端口：--port 8766）", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pde-agent", description="会学习进化的 PDE 求解 agent")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--home", help="经验和技能的存放目录（默认 ~/.pde_agent 或 $PDE_AGENT_HOME）")
    parser.add_argument("--llm", choices=sorted(PROVIDERS),
                        help="用哪个大模型（默认：只设置了 DEEPSEEK_API_KEY 时用 deepseek，否则用 claude）")
    parser.add_argument("--model", help="模型名（默认 Claude 用 claude-opus-5-5，DeepSeek 用 deepseek-flash）")
    parser.add_argument("--no-explore", action="store_true", help="只用已知最好的方法，不尝试新方法")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("curriculum", help="按课程表从基础方程开始练习")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--level", type=int, help="只做到这一级")
    p.add_argument("--frontier", action="store_true", help="包含需要学习新方法的课程")
    p.add_argument("--learn", "--claude", action="store_true", help="允许请大模型学习新方法")
    p.set_defaults(func=cmd_curriculum)

    p = sub.add_parser("solve", help="求解 JSON 描述的问题")
    p.add_argument("problem")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--learn", "--claude", action="store_true", help="没有合适方法时请大模型学习新方法")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("ask", help="用自然语言描述问题（需要大模型）")
    p.add_argument("text")
    p.add_argument("--tol", type=float, default=1e-3)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("learn", help="让大模型为某类问题写一个新方法")
    p.add_argument("problem")
    p.add_argument("--reason", default="")
    p.set_defaults(func=cmd_learn)

    sub.add_parser("stats", help="查看经验记忆").set_defaults(func=cmd_stats)
    sub.add_parser("skills", help="查看已知方程和方法").set_defaults(func=cmd_skills)

    p = sub.add_parser("app", help="打开网页版（只在本机可访问）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.set_defaults(func=cmd_app)

    args = parser.parse_args(argv)
    args.llm = args.llm or default_provider()
    try:
        return args.func(args)
    except ModuleNotFoundError as exc:
        if exc.name == "anthropic":
            print('需要先安装 Claude SDK：pip install -e ".[llm]"', file=sys.stderr)
        elif exc.name == "openai":
            print('需要先安装 DeepSeek 用的 OpenAI SDK：pip install -e ".[deepseek]"', file=sys.stderr)
        else:
            raise
    except LLMError as exc:
        name, key = PROVIDERS[args.llm]
        print(f"调用 {name} 失败：{exc}（检查 {key} 和网络）", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
