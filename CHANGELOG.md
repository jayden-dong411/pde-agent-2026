# 更新日志

每个版本都发布在 [GitHub Releases](https://github.com/jayden-dong411/pde-agent-2026/releases)，发布说明就是下面对应版本的内容。

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
