# RL vs OPD 对比实验 —— 实施计划

> 本文档从 `OPD实验实现文档` 提炼，作为执行阶段的 checklist 与改动蓝图。

---

## 1. 项目一句话

从零搭一个 agentic search 小模型（Qwen2.5-3B-Instruct），分别用 **GRPO (RL)** 和 **OPD（在策略蒸馏，老师 = ZeroSearch-7B-Instruct）** 两条后训练路线训练，系统对比 **成本 / 效果 / 能力上限**，给出企业选型建议。

---

## 2. 核心改动：OPD 损失模式

### 2.1 改动原理

veRL GRPO 的 actor 损失结构：
```
loss = policy_gradient_loss(group-normalized reward advantage)   # GRPO 主项
     + kl_coef * KL(π_actor ‖ π_ref)                              # KL 正则（小权重）
```

OPD 的改动（"基本改一行"精神）：

| 项 | GRPO 原版 | OPD 改动 |
|----|----------|---------|
| 参考模型 `π_ref` | 初始 actor 冻结副本 | **ZeroSearch-7B-Instruct（teacher）** |
| KL 的角色 | 小权重正则 | **主目标**（反向 KL 对老师） |
| reward / advantage 项 | 保留 | **权重置 0 / 去掉** |
| masking（response_mask + 检索 token mask） | 保留 | **保留**（只蒸学生生成的 token） |
| rollout 机制 | ZeroSearch 模拟搜索 | **不变**（复用 on-policy 轨迹） |

### 2.2 代码改动清单

用 `loss_mode: grpo | opd` 开关切换，**不破坏原 GRPO 路径**。

| 改动点 | 内容 |
|-------|------|
| ① 参考模型加载 | 支持从配置指向 7B 老师路径（`ref.model_path`） |
| ② 损失函数 | 新增 OPD 分支：`loss = KL(π_student ‖ π_teacher)`，复用 veRL 已有的逐 token logprob（`logπ_s − logπ_t`） |
| ③ 训练入口 | 在 yaml/config 中新增 `loss_mode` 字段；GRPO 模式下行为不变 |
| ④ masking | 不改逻辑；OPD 分支复用同一 `response_mask` + 检索 token mask |

### 2.3 关键约束（动手前必须确认）

- [ ] 老师与学生 **tokenizer 一致**（都是 Qwen2.5 系） → 反向 KL 逐 token 对齐成立
- [ ] masking **只作用于学生生成的 token**，检索/工具返回内容不参与蒸馏
- [ ] KL 估计器先用**最便宜版本**（复用现有逐 token logprob）跑通，再考虑升级
- [ ] 7B 老师前向显存压力：A800 80G 偏紧时需缩 batch

---

## 2.5 代码地图（A2 探索结果）

### 2.5.1 训练调用链（从入口到损失）

```
train_grpo.sh
  └→ python3 -m verl.trainer.main_ppo          (verl/trainer/main_ppo.py)
       └→ RayPPOTrainer.init_workers()          (verl/trainer/ppo/ray_trainer.py)
            ├→ 创建 ref_policy_wg（参考模型 worker）
            ├→ 创建 actor_rollout_wg（actor + vLLM rollout）
            └→ 创建 critic_wg（GRPO 模式不使用）
       └→ RayPPOTrainer.fit()
            ├→ ref_policy_wg.compute_ref_log_prob(batch)   # 参考模型前向
            ├→ reward_fn(batch)                             # outcome reward
            ├→ compute_advantage(grpo)                      # GRPO advantage
            └→ actor_rollout_wg.update_actor(batch)         # actor 更新
                 └→ DataParallelPPOActor.update_policy()   (verl/workers/actor/dp_actor.py)
                      └→ _forward_micro_batch() → log_prob, entropy
                      └→ core_algos.compute_policy_loss()   # pg_loss
                      └→ core_algos.kl_penalty()            # kl_loss
                      └→ policy_loss = pg_loss - entropy*coeff - kl_loss*kl_coef
```

### 2.5.2 六个关键位置

| # | 位置 | 文件:行号 | 说明 |
|---|------|----------|------|
| ① | **Actor 损失聚合** | `ZeroSearch/verl/workers/actor/dp_actor.py:249-275` | pg_loss + entropy_loss + kl_loss 的聚合点，**OPD 的主改点** |
| ② | **策略梯度损失** | `ZeroSearch/verl/trainer/ppo/core_algos.py:173-204` | `compute_policy_loss()`：PPO-clip 风格 |
| ③ | **KL 估计器** | `ZeroSearch/verl/trainer/ppo/core_algos.py:252-284` | `kl_penalty()`：支持 kl/abs/mse/low_var_kl 四种模式 |
| ④ | **参考模型加载** | `ZeroSearch/verl/workers/fsdp_workers.py:334-348` | `ref_module_fsdp` 从 `self.config.model.path` 加载，**当前与 actor 共用 path** |
| ⑤ | **参考模型调用** | `ZeroSearch/verl/trainer/ppo/ray_trainer.py:803-807` | `ref_policy_wg.compute_ref_log_prob(batch)` |
| ⑥ | **Masking（检索 token）** | `ZeroSearch/verl/trainer/ppo/ray_trainer.py:884-897` | `_create_loss_mask()`：从 `info_mask` 生成 `loss_mask`（mask 掉 `<information>...</information>`） |

### 2.5.3 配置文件

| 文件 | 关键字段 |
|------|---------|
| `ZeroSearch/verl/trainer/config/ppo_trainer.yaml` | `actor_rollout_ref.model.path`（模型路径）、`actor_rollout_ref.ref`（ref FSDP 配置）、`algorithm.kl_penalty`（KL 估计类型）、`algorithm.state_masking`（检索 mask 开关） |
| `ZeroSearch/train_grpo.sh` | `use_kl_loss=true`、`kl_loss_coef=0.001`、`kl_loss_type=low_var_kl`、`state_masking=True` |

### 2.5.4 损失计算细节（OPD 改动的核心）

**GRPO 原版（dp_actor.py:249-275）**：
```python
# 1. 前向：算当前 actor 的 log_prob 和 entropy
entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

# 2. 策略梯度损失（PPO-clip）
pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
    old_log_prob, log_prob, advantages, response_mask, clip_ratio)

# 3. 熵正则（鼓励探索）
entropy_loss = verl_F.masked_mean(entropy, response_mask)

# 4. 聚合：pg_loss - entropy * coeff
policy_loss = pg_loss - entropy_loss * entropy_coeff

# 5. KL 正则（对参考模型）
if self.config.use_kl_loss:
    ref_log_prob = data['ref_log_prob']
    kld = core_algos.kl_penalty(log_prob, ref_log_prob, self.config.kl_loss_type)
    kl_loss = masked_mean(kld, response_mask)
    policy_loss = policy_loss - kl_loss * self.config.kl_loss_coef  # ← 注意减号
```

**OPD 分支（待实现）**：
```python
# loss_mode == "opd" 时：
# 1. 前向：算学生的 log_prob
entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

# 2. 反向 KL（学生对老师）作为主目标
ref_log_prob = data['ref_log_prob']  # 此时 ref 是 7B 老师
kld = core_algos.kl_penalty(log_prob, ref_log_prob, kl_penalty="kl")
kl_loss = masked_mean(kld, response_mask)

# 3. OPD 损失 = KL（不加 pg_loss、不加 entropy）
policy_loss = kl_loss  # 最小化 KL(π_student ‖ π_teacher)
```

### 2.5.5 参考模型加载的关键约束

**当前实现**（fsdp_workers.py:334-348）：
```python
if self._is_ref:
    self.ref_module_fsdp = self._build_model_optimizer(
        model_path=self.config.model.path,  # ← 与 actor 共用 path！
        ...)
    self.ref_policy = DataParallelPPOActor(
        config=self.config.ref, actor_module=self.ref_module_fsdp)
```

**OPD 需要的改动**：
- 在 `actor_rollout_ref.ref` 下新增 `model_path` 字段（可选）
- 如果 `ref.model_path` 存在，用它；否则回退到 `actor_rollout_ref.model.path`（保持 GRPO 行为）
- 这样 GRPO 模式下 ref 仍是 actor 的冻结副本，OPD 模式下 ref 是 7B 老师

### 2.5.6 Masking 机制

**ZeroSearch 特有的 state_masking**：
- 配置：`state_masking=True`（train_grpo.sh:56）
- 标记：`<information>...</information>`（ppo_trainer.yaml:166-167）
- 实现：`ray_trainer.py:884-897` 的 `_create_loss_mask()` 把 `info_mask` 转成 `loss_mask`
- 应用：`dp_actor.py:241-242` 用 `loss_mask` 覆盖 `response_mask`

**OPD 直接复用**：不需要改 masking 逻辑，`state_masking=True` 自动把检索内容排除在蒸馏之外。

### 2.6 改动日志（A3 实施结果）

| 文件 | 改动 | 说明 |
|------|------|------|
| `verl/trainer/config/ppo_trainer.yaml:39` | 新增 `loss_mode: grpo` | actor 段内，默认 grpo 保持向后兼容 |
| `verl/trainer/config/ppo_trainer.yaml:58` | 新增 `ref.model_path: null` | null 时回退到 actor model path（GRPO 行为） |
| `verl/workers/fsdp_workers.py:337` | `ref_model_path` 读取 `config.ref.model_path` | 用 `or` 回退，不破坏 GRPO |
| `verl/workers/actor/dp_actor.py:214` | `select_keys` 在 OPD 模式下也选 `ref_log_prob` | 用 `getattr` 安全访问 |
| `verl/workers/actor/dp_actor.py:252-270` | 新增 `if loss_mode == 'opd':` 分支 | KL 对老师为主损失；pg_loss/clipfrac/ppo_kl 置零 |
| `train_opd.sh` | 新文件，从 train_grpo.sh 派生 | 参数位置调整：`$6=TEACHER_PATH`；`kl_loss_coef=1.0` |

**未改动（无需改动）**：
- `megatron_actor.py`：本实验使用 FSDP 策略，不走 Megatron 路径
- `ray_trainer.py`：`use_reference_policy` 始终为 True；`compute_ref_log_prob` 始终被调用；`use_kl_loss=True` 时正确跳过 `apply_kl_penalty`
- `core_algos.py`：`kl_penalty()` 函数直接复用，不需要新增估计器

---

## 3. 实施阶段与 TODO

### Phase A —— 本地：理解代码 + 改 OPD（主战场，不花 GPU 钱）

| # | 任务 | 状态 |
|---|------|------|
| A1 | clone 三个仓库（ZeroSearch、Search-R1、StepSearch）；配本地环境（§3.2） | ✅ |
| A2 | **代码探索**：用 grep/find 定位 §4.3 六个关键位置，记录调用关系到注释 | ✅ |
| A3 | 实现 OPD 改动（§2.2 四点）：加 `loss_mode` 开关，保留 GRPO 原路径 | ✅ |
| A4 | 纯逻辑部分（KL 计算、mask 应用）抽出来写最小单测，`pytest` 通过 | ✅ |
| A5 | lint / 类型检查通过（容忍 vllm/flash_attn 未解析告警） | ✅ |
| A6 | 写好两条训练命令草稿（RL 臂 + OPD 臂） | ✅ |

### Phase B —— 服务器：环境验证（有卡模式，挂 A800）

> **一键验证**：`bash scripts/verify_phase_b.sh [模型路径]`
> 脚本自动执行 B1-B5，输出 PASS/FAIL/WARN 报告。详见 `scripts/README.md`。

| # | 任务 | 状态 |
|---|------|------|
| B1 | `nvidia-smi` 见 A800 | ✅ |
| B2 | torch 2.4.0+cu121 / cuda / flash-attn / verl 自检全 PASS | ✅ |
| B3 | vllm 加载 Qwen2.5-3B-Instruct 并 `generate` 一句话 | ✅ |
| B4 | `sglang` 环境 `import sglang` 成功 | ✅ |
| B5 | 同步本地改动，确认改动文件无误 | ✅ |

### Phase C —— 服务器：跑 Arm RL（baseline）

> **一键启动**：`bash scripts/run_phase_c.sh {smoke|full}`
> - `smoke` — 小步数验证（20 步，~30 分钟）
> - `full` — 正式训练（500 步，8-12 小时）
> - 其他：`serve`（仅启动模拟器）、`stop`、`status`、`download`（资源下载命令）

| # | 任务 | 状态 |
|---|------|------|
| C1 | serve 模拟器（sglang 环境）/ 或由训练脚本拉起 | ⬜ |
| C2 | 小步数验证 pipeline 不崩 | ⬜ |
| C3 | 正式训练；记录 GPU·小时、reward 曲线、最终 checkpoint | ⬜ |

### Phase D —— 服务器：跑 Arm OPD（核心）

| # | 任务 | 状态 |
|---|------|------|
| D1 | serve / 加载 7B 老师供 ref-logprob 计算 | ⬜ |
| D2 | 小步数验证反向 KL 损失正常下降、masking 生效 | ⬜ |
| D3 | 正式训练；记录 GPU·小时、loss 曲线、最终 checkpoint | ⬜ |
| D4 | 若 agentic OPD 卡住 → 启用兜底（§5） | ⬜ |

### Phase E —— 评测与对比（核心产出）

| # | 任务 | 状态 |
|---|------|------|
| E1 | 在 HotpotQA / 2Wiki / MuSiQue / Bamboogle + NQ/TriviaQA/PopQA 上评所有 arm + 老师 | ⬜ |
| E2 | 出对比表：路线 × {训练成本 GPU·h/¥, 各 benchmark 分数, 是否超过老师} | ⬜ |
| E3 | 画样本效率曲线（性能 vs 步数） | ⬜ |
| E4 | 写天花板分析结论 + README + 简历 bullet | ⬜ |

### (选) Phase F —— Hybrid / 过程奖励

| # | 任务 | 状态 |
|---|------|------|
| F1 | Arm Hybrid：先 OPD 冷启动 → 再 RL 微调 | ⬜ |
| F2 | RL 臂叠加 StepSearch 过程奖励（用 MuSiQue 数据集） | ⬜ |

---

## 4. 兜底方案

若 agentic 轨迹上的 OPD 一时打不通：
→ 先在**简单非 agentic 任务**（如数学推理，OPD 经典设定）上跑通，拿到 RL-vs-OPD 对比数据，保证项目有可交付结论，再回头攻 agentic。

---

## 5. 已知坑清单

- **CUDA 12.1 是头号杀手**：torch=2.4.0+cu121 / vllm=0.6.3 / flash-attn ABI=False 三处任一错位 → `undefined symbol`
- **本地别 import 训练模块**：顶部有 `import vllm/flash_attn` 的在本地必失败，属预期
- **masking 范围**：OPD 只蒸学生生成的 token，检索/工具内容必须 mask
- **不破坏 GRPO 原路径**：RL 臂还要用，必须用开关切换
- **对比公平性**：所有 arm 同基座、同数据、同后端、同评测；成本用统一口径
- **OPD 上限错觉**：OPD 不会超过老师——这是要呈现的结论，不是 bug

---

## 6. 给 Claude Code 的硬约束

**DO**
- 先探索代码、定位 §4.3 的位置再改
- 用开关新增 OPD 模式、保留原 GRPO
- 改动小而可读、加注释
- 对纯逻辑写单测

**DON'T**
- 不要在本地跑训练 / `import vllm` / `import flash_attn` / 任何需 GPU 的代码
- 不要在本地下载模型权重
- 不要重写整个训练流程（最小改动达成 OPD）

**不确定就停下问**：涉及 KL 估计器选择、参考模型加载方式、masking 字段语义时，若代码里看不明确，先说明发现与候选方案，再确认。

---

## 7. 下一步行动

1. ~~执行 A1：clone 仓库 + 配本地环境~~ ✅
2. ~~进入 A2：代码探索~~ ✅
3. ~~A3：实施 OPD 改动~~ ✅
   - 改 `fsdp_workers.py`：ref.model_path 支持独立配置
   - 改 `dp_actor.py`：新增 `loss_mode == "opd"` 分支
   - 改 `ppo_trainer.yaml`：新增 `loss_mode` 字段
   - 新增 `train_opd.sh`，设置新参数
4. ~~A4：纯逻辑单测~~ ✅（19/19 通过）
5. ~~A5：lint / 类型检查~~ ✅
6. ~~A6：训练命令草稿~~ ✅（见 COMMANDS.md）

**Phase A 已完成。**
**Phase B 已完成（22/22 全部通过）。** 服务器环境就绪：A800 80GB + torch 2.4.0+cu121 + flash-attn 2.6.3 + verl + vLLM + sglang 0.5.13。

下一步是 Phase C（RL 基线训练，小步数验证 pipeline 不崩）和 Phase D（OPD 蒸馏训练）。
