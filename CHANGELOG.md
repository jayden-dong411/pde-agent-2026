# 更新日志

每个版本都发布在 [GitHub Releases](https://github.com/jayden-dong411/pde-agent-2026/releases)，发布说明就是下面对应版本的内容。

## v0.2.0 — 2026-10-12

可以用 DeepSeek 代替 Claude：理解自然语言描述的题目、给新方程写求解器，都能交给 DeepSeek。

### 新增

- **接入 DeepSeek**：`llm.py` 新增 `DeepSeekBrain`，通过 DeepSeek 的 OpenAI 兼容接口调用，默认模型 `deepseek-v4-pro`（用 `--model deepseek-flash` 可以换成更快更便宜的版本），开启思考模式和 JSON 输出。DeepSeek 不能强制 JSON 的结构，所以回复会在本地按格式检查；空回复或格式不对时让它重写，最多 3 次。
- **命令行选模型**：新增全局选项 `--llm claude|deepseek`。不写时，只设置了 `DEEPSEEK_API_KEY` 就用 DeepSeek，否则用 Claude。
- **安装和 key**：`pip install -e ".[deepseek]"`，API key 放在环境变量 `DEEPSEEK_API_KEY`。没有这个变量时直接报错，不会改用 `OPENAI_API_KEY`，避免把别家的 key 发给 DeepSeek。
- **技能来源**：技能库的 `.json` 文件用 `origin` 记录是哪个模型写的（`claude` 或 `deepseek`）。

### 改动

- 提示词和对话流程抽成通用的 `Brain`，Claude 和 DeepSeek 只各自实现调用接口的部分，以后接别的模型（比如通义千问、Kimi）也只需要写这一部分。
- `--claude` 改名为 `--learn`（允许请大模型学习新方法），旧的 `--claude` 写法仍然可用。
- `--model` 不写时，按所选的模型取默认值。
- 提示信息里的“Claude”改成“大模型”；调用失败时会说明是哪个模型、该检查哪个 API key。

### 已知限制

- DeepSeek 部分和 Claude 一样，只用模拟接口测过，还没用真实 API key 跑过。

## v0.1.0 — 2026-10-12

第一个版本：一个会学习进化的 PDE 求解 agent，从最基础的方程开始。

### 新增

- **基础方程和内置方法**（都用制造解测过收敛阶）：
  - Poisson 方程（一维、二维，Dirichlet 边界）：`poisson.fd2` 二阶中心差分；
  - 一维热方程：`heat.ftcs` 显式格式、`heat.backward_euler` 向后 Euler、`heat.crank_nicolson` Crank–Nicolson；
  - 一维对流方程（周期边界）：`advection.upwind` 迎风格式、`advection.lax_wendroff`；
  - 一维波动方程：`wave.leapfrog` 蛙跳格式。
- **自我验证**：有精确解时直接算误差；没有精确解时用嵌套网格做 Richardson 估计。数值发散、加密后误差不降、网格到上限都会被识别出来，agent 会换方法。
- **经验记忆**：每次求解写进 `~/.pde_agent/experience.jsonl`。agent 按问题类别统计每个方法的成功率和计算量，先把每个方法都试一次，之后挑最可靠、最省的；网格大小也按经验直接预测。课程表跑两轮，求解次数从 26 次降到 14 次，计算量少了 250 倍左右。
- **技能库**：遇到没有方法的问题，agent 请 Claude 写新的求解器。新求解器要先过静态检查（只能用 numpy、scipy、sympy、math），再在独立进程里做制造解考试，收敛阶达到它声明的阶才能入库；不及格就把考试报告发回去修改，最多 3 轮。
- **接入 Claude**：可以用自然语言描述问题（`pde-agent ask`）。默认模型 `claude-opus-5-5`，用结构化输出，开启服务端 fallback。
- **课程表**：从一维 Poisson 到二维 Poisson，再到没有精确解的问题；最后三课（反应扩散、二维热方程、Burgers）没有内置方法，留给 agent 自己学。
- **命令行**：`pde-agent curriculum | solve | ask | learn | stats | skills`，以及 `pde-agent --version`。
- **版本发布流程**：版本号写在 `src/pde_agent/__init__.py`，每个版本在本文件写更新说明；合并到 main 后，GitHub Actions 自动跑测试、打包，并发布 Release。

### 已知限制

- Claude 部分只用模拟客户端测过，还没用真实 API key 跑过。
- 热方程、对流、波动方程目前只有一维，边界只支持 Dirichlet 和周期。
- Claude 写的求解器代码在本机运行。静态检查和独立进程能防住常见的意外，但不是安全沙箱。
