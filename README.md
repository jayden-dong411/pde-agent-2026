# pde-agent：会学习进化的偏微分方程求解 agent

从最基础的方程开始，agent 自己选方法、自己验证结果、记住每次的经验；遇到不会的方程时请大模型（Claude 或 DeepSeek）写新的求解器，新求解器考试通过才收进技能库。

## 现在会解什么

| 方程 | 形式（`L(u) = f`） | 内置方法（收敛阶） |
| --- | --- | --- |
| Poisson（1D、2D，Dirichlet 边界） | `-k Δu = f` | `poisson.fd2` 中心差分 + 稀疏直接解（2） |
| 热方程（1D，Dirichlet） | `u_t = α u_xx + f` | `heat.ftcs` 显式（2）、`heat.backward_euler`（1）、`heat.crank_nicolson`（2） |
| 对流方程（1D，周期边界） | `u_t + c u_x = f` | `advection.upwind`（1）、`advection.lax_wendroff`（2） |
| 波动方程（1D，Dirichlet） | `u_tt = c² u_xx + f` | `wave.leapfrog`（2） |

所有内置方法都用制造解（method of manufactured solutions）测过，实测收敛阶和标称一致。

## agent 怎么学习进化

1. **自我验证**：有精确解就直接算误差；没有精确解时，用两套嵌套网格的差做 Richardson 估计，并在观测收敛阶低于标称阶时改用观测值（偏保守）。测试里拿更细网格的解做参照，确认真实误差不超过容差的 2 倍。
2. **经验记忆**：每次求解都写进 `~/.pde_agent/experience.jsonl`，按问题类别（如 `heat/1d/dirichlet`）统计每个方法的成功率和计算量。
   - 选方法：没试过的方法先试一次（探索），之后优先用成功率高、计算量小的（利用）。
   - 选网格：根据以前在多大网格上达到多高精度，直接预测这次需要的网格，不用每次从粗网格加密上去。
3. **技能库**：没有方法能解的问题（新方程、新维度），或者现有方法都达不到精度时，agent 请大模型写一个新的求解器模块。新模块要过两关才能入库：
   - 静态检查：只允许 numpy、scipy、sympy、math 和 `pde_agent.skill_api`，不能读写文件、开进程、用 `eval` 等；
   - 独立进程里的考试：在 16、32、64 网格上解几个已知答案的制造解问题，误差要足够小，收敛阶要达到声明的阶。

   没通过就把考试报告发回给大模型修改，最多 3 轮。入库的技能下次启动自动加载。
4. **课程表**：从一维 Poisson 开始，依次是热方程、对流、波动、二维 Poisson，再到没有精确解、要自己估计误差的问题。最后是“前沿课程”：反应扩散方程、二维热方程、非线性 Burgers 方程，这些没有内置方法，要 agent 自己学。

不接大模型，把课程表跑两轮的结果：

```
第 1 轮：成功 12/12，共求解 26 次，工作量 1.23e+07
第 2 轮：成功 12/12，共求解 14 次，工作量 4.77e+04
```

第一轮在探索，比如对流方程试了一阶迎风格式，要 4096 个网格才达到精度。第二轮根据经验改用 Lax–Wendroff 和 Crank–Nicolson，网格大小直接预测，求解次数少了将近一半，计算量少了两个多数量级。

## 快速开始

```bash
pip install -e ".[dev]"            # 只用内置方法
pip install -e ".[dev,llm]"        # 加上 Claude（需要 ANTHROPIC_API_KEY）
pip install -e ".[dev,deepseek]"   # 加上 DeepSeek（需要 DEEPSEEK_API_KEY）

pde-agent app                      # 打开网页版（见下面“网页版”）
pde-agent curriculum --rounds 2    # 从基础方程开始练习
pde-agent stats                    # 看学到的经验
pde-agent skills                   # 看已知的方程和方法
pde-agent solve examples/heat_rod.json --tol 1e-4

# 下面这些需要大模型（见“接入大模型”）
pde-agent ask "在 [0,1] 上解 u_t = 0.01 u_xx，初值 sin(pi x)，两端为 0，算到 t = 1"
pde-agent curriculum --frontier --learn   # 让 agent 自己学会反应扩散、二维热方程、Burgers
```

## 网页版

```bash
pde-agent app        # 打开 http://127.0.0.1:8765/
```

在浏览器里选一道课程题或自己填写题目，点“求解”就能看到解的图像、误差和 agent 试过的方法；也可以切到“用文字描述”，让大模型理解题目后再解（需要先设置好大模型，见“接入大模型”）。页面下方是经验记忆和已知方法。网页只在本机能打开，按 Ctrl+C 退出。

![网页版](docs/app.png)

## 在 Python 里用



```python
from pde_agent import PDEAgent, PDEProblem

agent = PDEAgent()  # 经验和技能存在 ~/.pde_agent，可以用 PDE_AGENT_HOME 改
problem = PDEProblem(
    kind="heat", domain=[(0, 1)], t_final=0.5,
    coefficients={"alpha": 0.1}, initial="sin(pi*x)",
)
report = agent.solve(problem, tol=1e-4)
print(report.summary())   # 方法、网格、误差、求解次数
u = report.result.u       # 终止时刻的解，网格是 report.result.grid
```

接上大模型：

```python
from pde_agent.llm import ClaudeBrain, DeepSeekBrain

agent = PDEAgent(brain=DeepSeekBrain())   # 或 ClaudeBrain()
report = agent.ask("求解反应扩散方程 u_t = 0.5 u_xx - 2u ……")
```

## 代码结构

```
src/pde_agent/
  problem.py      问题描述（JSON 可存），表达式安全解析
  equations.py    已知方程的符号算子 L(u)，制造解
  solvers/        内置数值方法（统一的网格和结果格式）
  verify.py       误差、收敛阶、Richardson 估计
  memory.py       经验记忆：方法排序、网格预测
  skills.py       技能库：静态检查、考试、安装和加载
  agent.py        主循环：选方法、求解、验证、加密、记忆、学习
  llm.py          大模型（Claude、DeepSeek）：理解自然语言问题、编写和修复新求解器
  curriculum.py   课程表
  app.py          网页版的本地服务
  templates/      技能模块示例（反应扩散方程）、网页
```

## 接入大模型

只有“理解自然语言”和“写新求解器”用到大模型。数值求解、验证、选方法都在本地，不联网也能用。现在支持 Claude 和 DeepSeek，用 `--llm` 选：

```bash
export DEEPSEEK_API_KEY=...   # 在 https://platform.deepseek.com/api_keys 创建
pde-agent --llm deepseek ask "在 [0,1] 上解 u_t = 0.01 u_xx，初值 sin(pi x)，两端为 0，算到 t = 1"
pde-agent --llm deepseek --model deepseek-v4-pro curriculum --frontier --learn   # 换成更强的 pro 模型
```

不写 `--llm` 时，只设置了 `DEEPSEEK_API_KEY` 就用 DeepSeek，否则用 Claude。

DeepSeek 解析题目时使用 low 思考强度和 16,384 tokens 预算；写求解器时使用 high 和 131,072 tokens。思考过程也占生成预算。遇到长度截断时，最多自动重试一次，提高预算并改用 low，从完整对话重新生成。JSON 格式修复仍最多尝试 3 次，所以一次调用最多发起 4 次生成请求。

网页和命令行把 DeepSeek 请求统计写入经验目录中的 `deepseek_requests.jsonl`（默认 `~/.pde_agent/deepseek_requests.jsonl`，也跟随 `--home` / `PDE_AGENT_HOME`）。记录包含预算、用量、结束原因、耗时和请求 ID，不含 Key、题目、代码或思考正文。旧程序需要重启后才能使用这些改动。

| | Claude | DeepSeek |
| --- | --- | --- |
| 安装 | `pip install -e ".[llm]"` | `pip install -e ".[deepseek]"` |
| API key（环境变量） | `ANTHROPIC_API_KEY` | `DEEPSEEK_API_KEY` |
| 默认模型 | `claude-opus-5-5` | `deepseek-flash`，需要更强的可以用 `--model deepseek-v4-pro` |
| 回复格式 | 结构化输出，JSON 一定符合格式 | JSON 输出模式，本地再按格式检查，不合格就让它重写，最多 3 次 |
| 其他 | 开启服务端 fallback：模型拒绝请求时，API 在同一次调用里换模型重试（不需要可以在 `llm.py` 里去掉） | 开启思考模式 |

不管哪个模型写的求解器，都要通过同样的静态检查和制造解考试才能入库；技能旁边的 `.json` 文件记录了是哪个模型写的。

## 安全提示

大模型写的求解器代码会在本机运行。静态检查和独立进程能挡住常见的意外，但不是安全沙箱。入库前请看一下 `~/.pde_agent/skills/` 里的代码，最好在容器或虚拟环境里跑学习流程。

## 版本和更新日志

每个版本都发布在 [GitHub Releases](https://github.com/jayden-dong411/pde-agent-2026/releases)，每一版的改动写在 [CHANGELOG.md](CHANGELOG.md)。

发新版本的步骤：

1. 改 `src/pde_agent/__init__.py` 里的 `__version__`；
2. 在 `CHANGELOG.md` 最上面加一节 `## v版本号 — 日期`，写清这一版改了什么（没写的话 CI 不通过）；
3. 合并到 `main` 后，GitHub Actions 会自动跑测试、打包，并用这一节内容发布 Release。

## 路线图

- [x] 基础方程、自我验证、经验记忆、技能库、课程表
- [x] 可以选用 Claude 或 DeepSeek
- [x] 网页版，带结果可视化
- [ ] 把 PINN / 神经网络求解器作为一种技能接入，用同一套考试评判
- [ ] 更多内置方程：二维热和波动、Neumann/Robin 边界、变系数
- [ ] 让时间步长、CFL 数等参数也从经验中学习
