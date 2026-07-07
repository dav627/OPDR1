# 实验预期：OPD vs RL 在小模型 Agentic Search 上的对比

> 基于对 Search-R1、ZeroSearch 论文与 Thinking Machines OPD 公开讨论的调研整理。

---

## 1. 技术现状摘要

### GRPO 在 agentic search 上的定位

Search-R1（arXiv 2503.09516）与 ZeroSearch（arXiv 2505.04588）已证明：仅用 outcome reward（F1/EM 规则奖励）+ GRPO，即可让 3B 级模型从零学会"推理-搜索交错"的多轮 agentic 行为，在多跳 QA 上显著超过无搜索基线。GRPO 是当前该任务的**事实标准 RL 路线**——去掉 critic、用 group-normalized advantage，工程栈成熟（veRL + vLLM/sglang rollout）。

### OPD 的定位

Thinking Machines 的"在策略蒸馏"核心是：学生 on-policy 采样轨迹，老师只做前向给逐 token logprob，最小化反向 KL(π_student ‖ π_teacher)。相比 SFT 蒸馏，OPD 保留学生自己的状态分布，避免 distribution shift；相比 RL，OPD 不需要 reward 信号、不需要 group rollout、不需要 critic/advantage 估计，**工程更简单、收敛更稳**。

### 两者关系

在 veRL 实现里 OPD 几乎是 GRPO 的"损失函数改一行"——把 ref_policy 从"学生冻结副本"换成 7B 老师，把 KL 项从 0.001 权重的正则变成主损失、把 advantage 项权重置 0。

---

## 2. 关键数字

| 项 | 数值 | 来源 |
|---|---|---|
| ZeroSearch 训练步数（Qwen2.5-3B-Instruct, GRPO） | 203 steps | ZeroSearch/README |
| Search-R1 训练规模 | 8×A100，~500 steps 量级（3B） | Search-R1 wandb v0.1-v0.3 |
| ZeroSearch 核心结论 | 模拟搜索 ≥ 真实搜索；3B instruct 训练后在 7 benchmark 超过真实搜索基线 | ZeroSearch/docs/index.html |
| Search-R1 3B 结论 | Llama-3.2-3B / Qwen2.5-3B 学会多轮搜索，多跳 QA 显著提升 | Search-R1/README |
| OPD 蒸馏 3B←7B 在 agentic search 的具体分数 | 未找到公开数据（OPD 公开示例以数学/通用为主） | — |
| GRPO vs OPD 成本对比的具体数字 | 未找到公开数据 | — |

**注**：OPD 在 agentic search 上的公开基准数据稀缺，本实验本身就是补这块空白。

---

## 3. 实验预期

### 3.1 RL 臂（GRPO + F1 outcome reward）预期

- **分数范围**：在 HotpotQA / 2Wiki / MuSiQue / Bamboogle（多跳）上较 3B-Instruct 原始基线有明显提升，量级约十几个绝对 F1 点；在单跳 NQ/TriviaQA/PopQA 上提升幅度较小（单跳本身 ceiling 低）。
- **是否能超老师**：**有机会追平或略超 SearchR1-7B**——这是 Search-R1 paper 的核心卖点（小模型 + RL 可媲美大模型 + SFT）。
- **训练成本**：单步成本 = group rollout（多条 agentic 轨迹 × 模拟器交互）+ reward + actor 前向反向。多轮 agentic rollout 是主要开销，预期 60-120 秒/步，500 步约 8-17 小时。

### 3.2 OPD 臂（在策略蒸馏，反向 KL 对 7B 老师）预期

- **分数范围**：稳定逼近老师，落在老师分数的 **85%-100% 区间**。in-distribution（NQ/HotpotQA 训练域）接近持平，OOD（MuSiQue/Bamboogle）差距拉大。
- **是否能超老师**：**不会显著超过**——这是反向 KL 的信息论上限。
- **训练成本**：单步成本 = 学生 on-policy rollout（1 条，不需 group）+ 7B 老师前向（逐 token logprob）+ 学生反向。预期单步比 RL 便宜，但多一份 7B 老师前向显存，A800 80G 偏紧可能被迫缩 batch。

### 3.3 成本对比预期

| 维度 | RL 臂 | OPD 臂 |
|------|-------|--------|
| 单步 GPU·h | 较高（group rollout + 多轮交互） | 较低（单条 rollout + 老师前向） |
| 显存压力 | 中（学生 + vLLM rollout + 模拟器） | 高（学生 + vLLM rollout + 模拟器 + 7B 老师） |
| 收敛稳定性 | 方差大（agentic reward 噪声） | 平稳（KL 损失单调下降） |
| 整体 GPU·h | 预期 > OPD | 预期 ≤ RL |

### 3.4 哪个臂会赢

| 评判维度 | 赢家 | 理由 |
|---------|------|------|
| 分数天花板 | **RL 臂** | 可超老师，OPD 封顶在老师 |
| 成本/效果比 | **OPD 臂** | 更便宜、更稳、收敛更快 |
| 稳健性 | **OPD 臂** | KL 曲线平稳，RL 在 agentic 多轮上 reward 方差大、易崩 |
| 工程复杂度 | **OPD 臂** | 不需 reward 设计、不需 group rollout |

**总体预期结论**：本实验大概率得到"**OPD 性价比更高但有天花板，RL 上限更高但更贵更不稳**"的共识性结论——这本身就是项目要回答的业务问题。

---

## 4. 风险点

1. **agentic 多轮 OPD 是硬骨头**：老师 `ZeroSearch_google_V2_Qwen2.5_7B_Instruct` 由 ZeroSearch 用模拟搜索训出（**非 Search-R1 的 e5+Wikipedia 真实检索**——Search-R1 与 ZeroSearch 是两个不同工作，老师属 ZeroSearch 系）。学生也用 ZeroSearch 模拟搜索后端，搜索模式一致。**潜在风险是模拟器规模不一致**：学生用 3B 模拟器（`Simulation_LLM_google_3B`），老师训练时用的模拟器规模未知（很可能 14B），若差距大，老师对学生 3B 模拟器轨迹的 logprob 可能噪声偏大，反向 KL 难下降。这是本实验最可能不如预期的地方。兜底方案见 `PLAN.md` §4。

> **注**：早期版本曾担心"google/wiki 语料分布偏移"，经代码追踪后排除——`search_engine` 参数在 verl 管线中是死代码，`simulate_sft` 模式下搜索结果由模拟器 LLM 生成、prompt 硬编码 "You are the Google search engine"，与 `search_engine` 取值无关。详见 [实验报告.md](实验报告.md) §2.1 注。

2. **7B 老师前向显存**：A800 80G 上 学生训练 + vLLM rollout + 模拟器 + 7B 老师前向同台，OOM 风险高。已有多次 OOM 修复记录。

3. **masking 正确性**：检索/工具返回 token 不能进蒸馏，否则学生在学模拟器吐的文本。`state_masking=True` 已实现，需在小步验证里确认 `loss_mask` 真的把 `<information>...</information>` 排除。

4. **RL 臂增量可能有限**：3B Instruct（非 base）起点已较高，RL 增量可能不如 Search-R1 paper 中 base→instruct 的提升幅度大；且 ZeroSearch 模拟搜索的 reward 信号比真实搜索更平稳，可能限制 RL 突破。

5. **超老师判断的统计噪声**：各 benchmark 分数差几个点可能在评测随机性内，需多家 benchmark + 多 seed 才能下"超过老师"的结论。

6. **OPD 在 agentic search 上无公开基准**：本实验本身就是补这块空白——"85%-100% 老师"的预期是借数学推理领域的 OPD 经验外推，agentic 任务上尚无定论。

---

## 5. 实验设置确认

### 公平性保证（arm 对比）

| 项 | RL 臂 | OPD 臂 | 是否一致 |
|----|-------|--------|---------|
| 学生基座 | Qwen2.5-3B-Instruct | 同左 | ✅ |
| 训练数据 | sunhaonlp/ZeroSearch_dataset | 同左 | ✅ |
| 搜索模式 | `search_mode=simulate_sft`（模拟器 LLM 生成检索） | 同左 | ✅ |
| 训练步数 | 500 steps | 500 steps | ✅ |
| 评测 benchmark | 7 个 QA 数据集 | 同左 | ✅ |
| 唯一差异 | 损失函数（pg_loss vs KL） | ref 模型（学生副本 vs 7B 老师） | — |

### 评测 benchmark

| 类型 | 数据集 | 评测指标 |
|------|--------|---------|
| 多跳 | HotpotQA, 2Wiki, MuSiQue, Bamboogle | F1 / EM |
| 单跳 | NQ, TriviaQA, PopQA | F1 / EM |

### 成本统计口径

- GPU·h：从训练开始到结束的累计 GPU 时间
- ¥：GPU·h × 单价（A800 按量 ¥6/h）
- 含调试时间 vs 纯训练时间（分开统计）

---

## 6. 简历 bullet 预填（待数字填入）

> 从零搭建 agentic search 小模型训练管线，对比 RL (GRPO) 与 OPD（在策略蒸馏）两条后训练路线。在 Qwen2.5-3B-Instruct 上，GRPO 训练 X 小时在 HotpotQA 达 XX F1（超老师 Y 点）；OPD 训练 X 小时达 XX F1（达老师 Z%），成本仅为 RL 的 N%。结论：RL 上限高但成本贵 N×，OPD 性价比优但有天花板——为小模型后训练选型提供实证依据。
