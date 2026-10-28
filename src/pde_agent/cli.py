"""Command line: ``pde-agent curriculum | solve | ask | learn | bench | stats | skills | app``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import __version__
from .agent import PDEAgent, default_home
from .curriculum import basic_curriculum, frontier_lessons, run_curriculum
from .llm import PROVIDERS, LLMError, default_provider
from .problem import PDEProblem
from .settings import Settings


def _brain(args):
    if not getattr(args, "learn", False):
        return None
    from .llm import make_brain

    home = Path(args.home) if args.home else default_home()
    saved_key = Settings(home).key(args.llm)  # a key saved on the web page
    return make_brain(args.llm, args.model, log_path=home / "deepseek_requests.jsonl", api_key=saved_key or None)


def _agent(args) -> PDEAgent:
    agent = PDEAgent(home=args.home, brain=_brain(args), explore=not args.no_explore)
    for problem in agent.skill_errors:
        print(f"警告：{problem}", file=sys.stderr)
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
    if report.choice():
        print(report.choice())
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


def cmd_bench(args) -> int:
    from .benchmark import reference_problems, run_benchmark

    problems = reference_problems()
    if args.problem:
        unknown = set(args.problem) - {b.name for b in problems}
        if unknown:
            print(f"没有这道基准题：{', '.join(sorted(unknown))}（可选：{', '.join(b.name for b in problems)}）", file=sys.stderr)
            return 1
        problems = [b for b in problems if b.name in args.problem]
    if args.tol:
        for bench in problems:
            bench.tol = args.tol
    agent = _agent(args)  # its registry includes learned skills
    scores = []
    print(f"{'基准题':<20} {'方法':<32} {'结果':<6} {'n':>5} {'误差':>9} {'耗时(秒)':>9} {'实测阶':>6}")
    for score in run_benchmark(problems, agent.registry, args.method, agent.home, not args.no_agent, args.budget):
        scores.append(score)
        best = score.best
        for m in score.methods:
            run = m.best or (m.runs[-1] if m.runs else None)
            mark = " ★" if m is best else ""
            outcome = "达到" if m.best else {"unstable": "发散", "error": "出错"}.get(m.outcome, "未达到")
            print(f"{score.bench.name:<20} {m.method + mark:<32} {outcome:<6} {run.n if run else '-':>5} "
                  f"{_fmt(run.error if run else None):>9} {_fmt(m.time_to_tol, '.4f'):>9} {_fmt(m.observed_order, '.2f'):>6}")
        if not score.methods:
            print(f"{score.bench.name:<20} （没有能解这类问题的方法）")
        a = score.agent
        if a:
            outcome = "达到" if a.success else "未达到"
            overhead = f"  是最快方法的 {score.agent_overhead:.1f} 倍" if score.agent_overhead else ""
            print(f"{score.bench.name:<20} {'agent → ' + (a.method or '-'):<32} {outcome:<6} {a.n or '-':>5} "
                  f"{_fmt(a.error):>9} {a.runtime:>9.4f} {'':>6}  求解 {a.solver_runs} 次{overhead}")
    if not args.no_agent:
        picked = [s for s in scores if s.best and s.agent and s.agent.success]
        best_pick = sum(s.agent.method == s.best.method for s in picked)
        print(f"agent 解出 {sum(bool(s.agent and s.agent.success) for s in scores)}/{len(scores)} 道，"
              f"其中 {best_pick}/{len(picked)} 道选中了最快的方法")
    if args.json:
        Path(args.json).write_text(json.dumps([s.to_dict() for s in scores], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"详细结果已写入 {args.json}")
    return 0


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
    if agent.memory.shared:
        print(f"另有 {len(agent.memory.shared)} 条来自经验中心的共享记录：{agent.memory.shared_path}")
    return 0


def cmd_sync(args) -> int:
    from .cloud import CloudError, HubClient, sync

    home = Path(args.home) if args.home else default_home()
    settings = Settings(home)
    cloud = settings.cloud
    url = args.url or cloud.get("url")
    token = args.token or (cloud.get("token") if url == cloud.get("url") else None)
    if not url or not token:
        print("还没有设置经验中心：pde-agent sync --url https://… --token …（或在网页设置里填）", file=sys.stderr)
        return 1
    try:
        client = HubClient(url, token)
        agent = _agent(args)
        result = sync(agent.memory, home, client)
        if args.url or args.token:  # remember them once they work
            settings.update_cloud(client.url, token, enabled=cloud.get("enabled", True))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    except CloudError as exc:
        print(f"同步失败：{exc}", file=sys.stderr)
        return 2
    print(f"已同步 {client.url}：上传 {result.pushed} 条，下载 {result.pulled} 条新经验，"
          f"本机共有 {result.shared_total} 条共享经验")
    return 0


def cmd_hub(args) -> int:
    from .hub import Hub, default_hub_dir, make_hub_server

    hub = Hub(Path(args.data) if args.data else default_hub_dir())
    if args.hub_command == "add-member":
        try:
            token = hub.add_member(args.name)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"成员 {args.name} 的访问令牌（只显示这一次，发给他本人）：\n{token}")
        return 0
    if args.hub_command == "remove-member":
        if hub.remove_member(args.name):
            print(f"已移除 {args.name}，他的令牌立即失效（已上传的经验保留）")
            return 0
        print(f"没有成员 {args.name}", file=sys.stderr)
        return 1
    if args.hub_command == "members":
        stats = hub.stats()
        for name in sorted(hub.members()):
            print(f"{name}：上传 {stats['by_member'].get(name, 0)} 条")
        print(f"共 {stats['members']} 个成员，{stats['episodes']} 条经验，数据在 {hub.dir}")
        return 0
    if not hub.members():
        print("还没有成员。先运行：pde-agent hub add-member 你的名字", file=sys.stderr)
        return 1
    try:
        server = make_hub_server(hub, args.host, args.port)
    except OSError as exc:
        print(f"无法在 {args.host}:{args.port} 启动：{exc}", file=sys.stderr)
        return 1
    print(f"经验中心已启动：http://{args.host}:{server.server_address[1]}/（数据在 {hub.dir}，按 Ctrl+C 退出）")
    if args.host not in ("127.0.0.1", "localhost"):
        print("提醒：令牌会随每个请求发送，对外开放时请放在 HTTPS 反向代理后面（见 README）。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        server.server_close()
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
                        help="用哪个大模型（默认：网页“设置”里选的；没选过时，只设置了 DEEPSEEK_API_KEY 就用 deepseek，否则用 claude）")
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

    p = sub.add_parser("bench", help="在有精确解的基准题上给各个方法和 agent 的选择打分")
    p.add_argument("--problem", action="append", help="只跑这道题（可重复）")
    p.add_argument("--method", action="append", help="只跑这个方法（可重复）")
    p.add_argument("--tol", type=float, help="统一的精度要求（默认每道题自己的，一般是 1e-3）")
    p.add_argument("--budget", type=float, default=20.0, help="每个方法在每道题上最多用多少秒（默认 20）")
    p.add_argument("--no-agent", action="store_true", help="只给方法打分，不跑 agent")
    p.add_argument("--json", help="把详细结果写到这个 JSON 文件")
    p.set_defaults(func=cmd_bench)

    sub.add_parser("stats", help="查看经验记忆").set_defaults(func=cmd_stats)
    sub.add_parser("skills", help="查看已知方程和方法").set_defaults(func=cmd_skills)

    p = sub.add_parser("sync", help="和经验中心同步：上传自己的经验，下载别人的")
    p.add_argument("--url", help="经验中心地址（第一次用时填，之后记住）")
    p.add_argument("--token", help="经验中心发给你的访问令牌")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("hub", help="运行经验中心（给多人共享经验用）")
    p.add_argument("--data", help="经验中心的数据目录（默认 ~/.pde_agent_hub 或 $PDE_AGENT_HUB）")
    hub_sub = p.add_subparsers(dest="hub_command", required=True)
    q = hub_sub.add_parser("serve", help="启动经验中心")
    q.add_argument("--host", default="127.0.0.1", help="监听地址（对外开放用 0.0.0.0）")
    q.add_argument("--port", type=int, default=8800)
    q = hub_sub.add_parser("add-member", help="添加成员，生成他的访问令牌")
    q.add_argument("name")
    q = hub_sub.add_parser("remove-member", help="移除成员，令牌立即失效")
    q.add_argument("name")
    hub_sub.add_parser("members", help="查看成员和各自上传的经验数")
    p.set_defaults(func=cmd_hub)

    p = sub.add_parser("app", help="打开网页版（只在本机可访问）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.set_defaults(func=cmd_app)

    args = parser.parse_args(argv)
    saved = Settings(Path(args.home) if args.home else default_home())  # choices saved on the web page
    args.llm = args.llm or saved.provider or default_provider()
    args.model = args.model or saved.model(args.llm)
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
        name, _ = PROVIDERS[args.llm]
        print(f"调用 {name} 失败：{exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
