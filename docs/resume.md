
小模型 Agentic 搜索后训练路线对比：GRPO 强化学习 vs 在策略蒸馏(OPD)   Qwen2.5-3B · veRL · ZeroSearch

项目介绍：在同一 Qwen2.5-3B 上对比 GRPO 强化学习与在策略蒸馏（OPD，蒸馏 7B 老师）两条 agentic 搜
索后训练路线，基于 ZeroSearch 模拟搜索（零真实 API 成本），单卡 A800 完成 OPD 的 veRL 实现、训练
调优与四模型评测全流程。

- 在策略蒸馏实现与欠训练归因：在 veRL 的 GRPO 路径上将 ref 换为 7B 老师、反向 KL 升为主损失实现
  OPD，以 state mask 仅对学生生成 token 计蒸馏损失；针对首轮 7 基准 EM 由基线 0.208→0.327（多跳
  MuSiQue +149%）却仅达老师（0.457）72% 的问题，经 wandb 曲线定位为学习率调度致欠训练（warmup
  占比 0.95、400 步全程未出 warmup），并代码追踪排除"语料错配"误判，锁定调度为主因。
- 公平复现与选型结论：修正 warmup 至 0.1 并加步至 800–1000、自训同条件 GRPO 消除官方模型 14B 模
  拟器算力混淆、4 模型×7 基准统一全量 test + bootstrap 95% CI；OPD EM 0.327→0.4x★（达老师
  ~90%★），GRPO 88–96%★且多跳追平老师★，EM–GPU·h 双曲线显示 OPD 单步成本约 RL 的 1/2★——量化
  验证"有强老师选蒸馏、需突破上限选 RL"的选型结论。
