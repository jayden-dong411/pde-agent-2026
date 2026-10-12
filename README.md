# pde-agent：会学习进化的偏微分方程求解 agent

从最基础的方程开始，agent 自己选方法、自己验证结果、记住每次的经验；遇到不会的方程时请 Claude 写新的求解器，新求解器考试通过才收进技能库。

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
3. **技能库**：没有方法能解的问题（新方程、新维度），或者现有方法都达不到精度时，agent 请 Claude 写一个新的求解器模块。新模块要过两关才能入库：
   - 静态检查：只允许 numpy、scipy、sympy、math 和 `pde_agent.skill_api`，不能读写文件、开进程、用 `eval` 等；
   - 独立进程里的考试：在 16、32、64 网格上解几个已知答案的制造解问题，误差要足够小，收敛阶要达到声明的阶。

   没通过就把考试报告发回给 Claude 修改，最多 3 轮。入库的技能下次启动自动加载。
4. **课程表**：从一维 Poisson 开始，依次是热方程、对流、波动、二维 Poisson，再到没有精确解、要自己估计误差的问题。最后是“前沿课程”：反应扩散方程、二维热方程、非线性 Burgers 方程，这些没有内置方法，要 agent 自己学。

不接 Claude，把课程表跑两轮的结果：

```
第 1 轮：成功 12/12，共求解 26 次，工作量 1.23e+07
第 2 轮：成功 12/12，共求解 14 次，工作量 4.77e+04
```

第一轮在探索，比如对流方程试了一阶迎风格式，要 4096 个网格才达到精度。第二轮根据经验改用 Lax–Wendroff 和 Crank–Nicolson，网格大小直接预测，求解次数少了将近一半，计算量少了两个多数量级。

## 快速开始

```bash
pip install -e ".[dev]"            # 只用内置方法
pip install -e ".[dev,llm]"        # 加上 Claude（需要 ANTHROPIC_API_KEY）

pde-agent curriculum --rounds 2    # 从基础方程开始练习
pde-agent stats                    # 看学到的经验
pde-agent skills                   # 看已知的方程和方法
pde-agent solve examples/heat_rod.json --tol 1e-4

# 下面这些需要 Claude
pde-agent ask "在 [0,1] 上解 u_t = 0.01 u_xx，初值 sin(pi x)，两端为 0，算到 t = 1"
pde-agent curriculum --frontier --claude   # 让 agent 自己学会反应扩散、二维热方程、Burgers
```

在 Python 里用：

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

接上 Claude：

```python
from pde_agent.llm import ClaudeBrain

agent = PDEAgent(brain=ClaudeBrain())
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
  llm.py          Claude：理解自然语言问题、编写和修复新求解器
  curriculum.py   课程表
  templates/      技能模块示例（反应扩散方程）
```

## 关于 Claude

- 默认模型 `claude-opus-5-5`，用结构化输出，返回的 JSON 一定符合格式。
- 开启了服务端 fallback（`fallbacks="default"`）：如果模型拒绝某个请求，API 会在同一次调用里换模型重试。不需要可以在 `llm.py` 里去掉。
- 只有“理解自然语言”和“写新求解器”用到 Claude。数值求解、验证、选方法都在本地，不联网也能用。

## 安全提示

Claude 写的求解器代码会在本机运行。静态检查和独立进程能挡住常见的意外，但不是安全沙箱。入库前请看一下 `~/.pde_agent/skills/` 里的代码，最好在容器或虚拟环境里跑学习流程。

## 路线图

- [x] 基础方程、自我验证、经验记忆、技能库、课程表
- [ ] 把 PINN / 神经网络求解器作为一种技能接入，用同一套考试评判
- [ ] 更多内置方程：二维热和波动、Neumann/Robin 边界、变系数
- [ ] 让时间步长、CFL 数等参数也从经验中学习
- [ ] 结果可视化
