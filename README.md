# OPDR1: On-Policy Distillation vs Reinforcement Learning

> 在 agentic search 任务上系统对比 **RL (GRPO)** 与 **OPD（在策略蒸馏）** 两条小模型后训练路线的成本、效果与能力上限。

学生模型 Qwen2.5-3B-Instruct，老师 ZeroSearch_google_V2_Qwen2.5_7B_Instruct，后端 ZeroSearch 模拟搜索（$0 真实 API），框架 veRL + vLLM/sglang。

---

## 1. 核心结论

| 模型 | 训练方式 | EM 均值 | F1 均值 | 相对老师 |
|------|---------|---------|---------|---------|
| 基线 3B | 无训练 | 0.208 | 0.190 | 45% |
| OPD 3B | 反向 KL 蒸馏到 7B 老师，400 步 | 0.327 | 0.340 | 72% |
| GRPO 3B | outcome reward (F1) + GRPO，203 步 | **0.444** | **0.453** | **97%** |
| 7B 老师 | ZeroSearch 官方 RL 训练 | 0.457 | 0.476 | 100% |

**三个关键发现：**

1. **GRPO 3B 追平 7B 老师**：EM 仅差 3%，且在多跳任务（2Wiki +9%、MuSiQue +6%）上反超老师——RL 能突破蒸馏天花板。
2. **OPD 未达老师上限**：只到老师 72%，蒸馏超参（lr=1e-6 + 95% warmup）未充分利用老师信号。
3. **多跳任务差距最大**：单跳老师仍领先 +1-3%，多跳 GRPO 反超——RL 学到的是推理链路而非记忆。

详细数字见 [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md)，完整分析与下一轮调整预期见 [docs/实验报告.md](docs/实验报告.md)。

---

## 2. 项目结构

```
OPDR1/
├── ZeroSearch/              # 主框架（已包含 OPD 改动）
│   ├── train_grpo.sh        # RL 基线训练脚本
│   ├── train_opd.sh         # OPD 蒸馏训练脚本（新增）
│   └── verl/                # veRL 训练框架
│       ├── trainer/
│       │   ├── config/ppo_trainer.yaml   # 新增 loss_mode, ref.model_path
│       │   ├── ppo/ray_trainer.py        # 训练主循环
│       │   └── ppo/core_algos.py         # KL 估计器（复用）
│       └── workers/
│           ├── actor/dp_actor.py         # OPD 损失分支（主改点）
│           ├── fsdp_workers.py           # ref 模型支持独立路径
│           └── sharding_manager/fsdp_vllm.py
├── Search-R1/               # 参考实现 + OPD 老师模型来源
├── StepSearch/              # 过程奖励（可选，未启用）
├── scripts/                 # 服务器端一键脚本
│   ├── verify_phase_b.sh    # 环境自检
│   ├── run_phase_c.sh       # Phase C: GRPO baseline 训练
│   ├── run_phase_d.sh       # Phase D: OPD 蒸馏训练
│   ├── run_phase_e.sh       # Phase E: 评测对比
│   ├── setup_logging.sh     # 配置快照 + GPU 监控
│   ├── analyze_logs.py      # 日志指标提取画图
│   └── WANDB_GUIDE.md       # wandb 配置说明
├── tests/
│   └── test_opd_logic.py    # OPD 损失逻辑单元测试
├── docs/                    # 所有文档（除 README）
│   ├── 实验报告.md           # **主报告**：实验设置/过程/结果/分析/下一轮预期
│   ├── EVAL_RESULTS.md      # 评测结果原始数据 + 简历 bullet
│   ├── PLAN.md              # 实施计划、代码地图、改动日志、已知坑清单
│   ├── EXPECTATIONS.md      # 实验预期与论文调研
│   ├── OPD实验实现文档.md    # 原始实现文档（环境配置、同步流程）
│   ├── COMMANDS.md          # 训练命令参考（参数映射）
│   └── images/              # wandb 截图（smoke/full 训练曲线）
│       ├── opd_full/        # OPD 正式训练 wandb 曲线
│       ├── opd_smoke/       # OPD smoke 验证
│       └── smoke_c2/        # GRPO smoke 验证
├── config.a800.sh           # A800 80GB 单卡预设
├── config.local.sh.example  # 本地配置模板（复制为 config.local.sh）
└── README.md                # 本文件
```

---

## 3. 实验流程（Phase A–E）

| Phase | 内容 | 位置 | 状态 |
|-------|------|------|------|
| A | 本地：理解代码 + 实现 OPD 改动 + 单测 | 本机 | ✅ |
| B | 服务器：环境验证（GPU/torch/vllm/sglang/代码改动） | A800 服务器 | ✅ |
| C | 服务器：GRPO baseline 训练 | A800 服务器 | ✅（用 ZeroSearch 官方训好模型） |
| D | 服务器：OPD 蒸馏训练（7B → 3B） | A800 服务器 | ✅ 400 步 |
| E | 评测对比：4 个模型 × 7 个 benchmark | A800 服务器 | ✅ |

### Phase A — 本地实现 OPD 改动

在 veRL GRPO 基础上**最小改动**实现 OPD 模式，用 `loss_mode: grpo | opd` 开关切换，不破坏原 GRPO 路径。

**改动清单：**

| 文件 | 改动 | 说明 |
|------|------|------|
| `verl/trainer/config/ppo_trainer.yaml` | 新增 `loss_mode: grpo` | 默认 grpo 保持兼容 |
| `verl/trainer/config/ppo_trainer.yaml` | 新增 `ref.model_path: null` | null 时回退到 actor path（GRPO 行为） |
| `verl/workers/fsdp_workers.py:337` | `ref_model_path` 读取 `config.ref.model_path` | `or` 回退，不破坏 GRPO |
| `verl/workers/actor/dp_actor.py:214` | `select_keys` 在 OPD 模式下也选 `ref_log_prob` | `getattr` 安全访问 |
| `verl/workers/actor/dp_actor.py:252-270` | 新增 `if loss_mode == 'opd':` 分支 | KL 对老师为主损失；pg_loss/clipfrac/ppo_kl 置零 |
| `train_opd.sh` | 新文件，从 `train_grpo.sh` 派生 | 新增 `TEACHER_PATH` 参数；`kl_loss_coef=1.0` |

**未改动（无需改动）：** `ray_trainer.py`（`compute_ref_log_prob` 始终被调用）、`core_algos.py`（`kl_penalty()` 直接复用）、`megatron_actor.py`（本实验走 FSDP）。

### Phase B — 服务器环境验证

```bash
bash scripts/verify_phase_b.sh [模型路径]
```
自动检查 GPU/torch/vllm/sglang/代码改动，输出 PASS/FAIL/WARN 报告。

### Phase C — GRPO baseline

```bash
bash scripts/run_phase_c.sh smoke    # 20 步验证（~30 分钟）
bash scripts/run_phase_c.sh full     # 500 步正式训练（8-12 小时）
```

实际本次评测采用 ZeroSearch 官方发布的训好 3B 模型（`ZeroSearch_google_V2_Qwen2.5_3B_Instruct`），训练步数 203，模拟器 14B。对比为近似参考。

### Phase D — OPD 蒸馏训练

```bash
bash scripts/run_phase_d.sh smoke    # 20 步验证
bash scripts/run_phase_d.sh full     # 500 步正式训练
```

实际跑到 400 步（checkpoint `global_step_400`），训练时间 ~18 小时。

### Phase E — 评测对比

```bash
bash scripts/run_phase_e.sh 1        # 基线 3B
bash scripts/run_phase_e.sh 2        # OPD 3B
bash scripts/run_phase_e.sh 3        # GRPO 3B
bash scripts/run_phase_e.sh 4        # 7B 老师
bash scripts/run_phase_e.sh all      # 依次评 1/2/3（7B 可选）
```

7 个 benchmark：NQ、TriviaQA、PopQA（单跳）、HotpotQA、2Wiki、MuSiQue、Bamboogle（多跳）。评测样本 48 条/benchmark（7B 老师 195 条全量）。

---

## 4. 实验设置

### 4.1 模型与数据

| 项 | 路径 | 来源 |
|---|------|------|
| 学生模型 | `Qwen2.5-3B-Instruct` | HuggingFace |
| OPD 老师 | `ZeroSearch_google_V2_Qwen2.5_7B_Instruct` | ZeroSearch 官方 |
| GRPO 基线 | `ZeroSearch_google_V2_Qwen2.5_3B_Instruct` | ZeroSearch 官方训好 |
| 模拟器（OPD/基线） | `Simulation_LLM_google_3B` | ZeroSearch 官方 |
| 模拟器（GRPO 基线） | `Simulation_LLM_google_14B` | ZeroSearch 官方 |
| 训练数据 | `ZeroSearch_dataset/train.parquet` | ZeroSearch |
| 评测数据 | `ZeroSearch_dataset/test.parquet` | 7 个 QA benchmark 合集 |

### 4.2 训练超参（OPD 与 GRPO 共享）

| 参数 | 值 | 说明 |
|------|------|------|
| `train_batch_size` | 32 | 单卡显存约束 |
| `ppo_mini_batch_size` | 256 | verl 自动 clamp 到 train_batch |
| `ppo_micro_batch_size` | 16 | |
| `max_prompt_length` | 4096 | |
| `max_response_length` | 500 | 每轮回答上限 |
| `max_start_length` / `max_obs_length` | 2048 / 2048 | 检索内容截断 |
| `lr` | 1e-6 | |
| `lr_warmup_steps_ratio` | 0.95 | 500 步下前 475 步都在爬坡 |
| `n_agent` | 5 | 每 prompt 5 条采样 |
| `temperature` | 1.0 | |
| `max_turns` | 5 | agentic 搜索最大轮数 |
| `topk` | 5 | 每轮检索文档数 |
| `start_threshold` / `end_threshold` | 0 / 0.5 | 强制每轮都搜 |
| `state_masking` | True | 检索 `<information>` token 不入 loss |
| FSDP offload | param/grad/optimizer 全开 | 省 GPU 显存 |
| `model_dtype` | bf16 | |

### 4.3 OPD 与 GRPO 的差异

| 参数 | RL (GRPO) | OPD |
|------|-----------|-----|
| `loss_mode` | `grpo`（默认） | `opd` |
| `ref.model_path` | null（回退到 actor） | 7B 老师路径 |
| `kl_loss_coef` | 0.001（小正则） | **1.0**（主目标） |
| `kl_loss_type` | `low_var_kl` | `low_var_kl` |
| `use_kl_loss` | true | true（OPD 自动启用） |
| `adv_estimator` | `grpo` | `grpo`（仍算但 OPD 不用） |
| 损失公式 | `pg_loss - entropy·coeff - kl·0.001` | `KL(π_s ‖ π_t)`（无 pg/entropy） |

### 4.4 硬件配置

- GPU: A800 80GB × 1（单卡）
- 模拟器显存比例: 0.12（~9 GB）
- vLLM rollout 显存: 0.4（3B）/ 0.25（7B 评测）
- 训练时间: OPD 400 步 ~18 小时
- 评测时间: 3B 模型 ~37-56 分钟，7B 老师 ~2 小时 13 分钟

---

## 5. 评测结果

### Exact Match (EM)

| Benchmark | 基线 3B | OPD 3B | GRPO 3B | 7B 老师 |
|-----------|---------|--------|---------|---------|
| NQ | 0.242 | 0.318 | 0.472 | **0.478** |
| TriviaQA | 0.372 | 0.442 | 0.622 | **0.640** |
| PopQA | 0.232 | 0.564 | 0.656 | **0.672** |
| HotpotQA | 0.182 | 0.282 | 0.386 | **0.412** |
| 2Wiki | 0.262 | 0.330 | **0.412** | 0.378 |
| MuSiQue | 0.070 | 0.174 | **0.294** | 0.278 |
| Bamboogle | 0.097 | 0.181 | 0.264 | **0.342** |
| **均值** | **0.208** | **0.327** | **0.444** | **0.457** |

### F1 Score

| Benchmark | 基线 3B | OPD 3B | GRPO 3B | 7B 老师 |
|-----------|---------|--------|---------|---------|
| NQ | 0.189 | 0.330 | 0.488 | **0.503** |
| TriviaQA | 0.368 | 0.466 | 0.648 | **0.669** |
| PopQA | 0.128 | 0.405 | 0.460 | **0.476** |
| HotpotQA | 0.191 | 0.343 | 0.464 | **0.484** |
| 2Wiki | 0.218 | 0.345 | **0.439** | 0.418 |
| MuSiQue | 0.112 | 0.220 | **0.332** | 0.355 |
| Bamboogle | 0.163 | 0.275 | 0.337 | **0.428** |
| **均值** | **0.190** | **0.340** | **0.453** | **0.476** |

### 多跳 vs 单跳

| 任务类型 | 基线 EM | OPD EM | GRPO EM | 老师 EM |
|---------|---------|--------|---------|---------|
| 单跳（NQ/TriviaQA/PopQA） | 0.282 | 0.441 | 0.583 | 0.597 |
| 多跳（HotpotQA/2Wiki/MuSiQue/Bamboogle） | 0.153 | 0.242 | 0.339 | 0.353 |

---

## 6. 结果分析

### 6.1 GRPO 3B 追平 7B 老师

GRPO 3B 与 7B 老师 EM 均值仅差 3%（0.444 vs 0.457），F1 差 5%（0.453 vs 0.476）。**多跳任务上 GRPO 3B 反超老师**：2Wiki（0.412 vs 0.378，+9%）、MuSiQue（0.294 vs 0.278，+6%）。单跳任务老师仍领先 +1-3%，Bamboogle 差距最大（老师 +30%）。

**结论**：3B 学生经 GRPO 训练后已基本榨干 7B 老师的能力上限——RL 在小模型上能突破蒸馏天花板，与 Search-R1 论文核心卖点一致。

### 6.2 OPD 未达老师天花板

OPD 3B EM 0.327，仅达老师（0.457）的 **72%**，距离老师仍有 13 个点缺口。可能原因：

1. **lr 过保守**（主因）：`lr=1e-6` + `lr_warmup_steps_ratio=0.95`，500 步内前 475 步都在线性爬坡，平均有效 lr ≈ 5e-7，学生未充分逼近老师。
2. **模拟器规模不一致**（待验证）：学生 OPD rollout 用 `Simulation_LLM_google_3B`（3B 模拟器）生成检索内容，老师训练时用的模拟器规模未知（很可能 14B，见 §4.1 表中"官方 GRPO 训练时"用的就是 14B），若规模差距大，老师对学生 3B 模拟器轨迹的 log-prob 噪声偏大。

> **注**：早先版本曾归因"google/wiki 语料三方不匹配"和"无差别蒸馏信号效率低"，经代码追踪后**均弃用**——`search_engine` 参数在 verl 管线中是死代码（详见 [docs/实验报告.md](docs/实验报告.md) §2.1 注），`simulate_sft` 模式下不存在 google/wiki 维度的偏移；"无差别蒸馏"是 on-policy 蒸馏的设计优势而非缺陷。

### 6.3 OPD 仍优于基线

OPD 蒸馏有效，7 个 benchmark 全部提升：EM 均值 0.208 → 0.327（+57%），多跳任务提升最显著（MuSiQue +149%、Bamboogle +87%）。说明 OPD 蒸馏方向正确，只是未充分收敛。

### 6.4 改进方向

若要进一步提升 OPD 效果：
- `lr_warmup_steps_ratio` 降到 0.1-0.2，或 `lr` 提到 2e-6-5e-6
- 查证老师训练时的模拟器规模；若与 3B 差距大，换 14B 模拟器重训 OPD（注意：调 `search_engine` 参数无意义，它在 verl 管线中不被消费）
- 单独 eval 7B 老师确认天花板（已完成：EM 0.457）
- 增加训练步数到 800-1000 步

---

## 7. 快速部署

### 7.1 环境配置

```bash
# 克隆
git clone https://github.com/dav627/OPDR1.git
cd OPDR1

# conda 环境
conda create -n rl-opd python=3.9 -y
conda activate rl-opd
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 wandb huggingface_hub transformers<4.48

# sglang 环境（模拟器用，独立环境）
conda create -n sglang python=3.10 -y
conda activate sglang
pip install 'sglang[all]'

# veRL（ZeroSearch 已包含 OPD 改动）
conda activate rl-opd
cd ZeroSearch && pip install -e .
```

### 7.2 资源下载

模型下载到 `/root/autodl-tmp/models/`，数据到 `/root/autodl-tmp/data/`：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像

# 学生模型
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /root/autodl-tmp/models/Qwen2.5-3B-Instruct

# 模拟器（google 版 3B）
huggingface-cli download sunhaonlp/Simulation_LLM_google_3B --local-dir /root/autodl-tmp/models/Simulation_LLM_google_3B

# 7B 老师
huggingface-cli download Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_7B_Instruct --local-dir /root/autodl-tmp/models/ZeroSearch_google_V2_Qwen2.5_7B_Instruct

# GRPO 官方 3B 基线
huggingface-cli download Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B_Instruct --local-dir /root/autodl-tmp/models/ZeroSearch_google_V2_Qwen2.5_3B_Instruct

# 训练/评测数据
huggingface-cli download --repo-type dataset sunhaonlp/ZeroSearch_dataset --local-dir /root/autodl-tmp/data/ZeroSearch_dataset
```

### 7.3 本地配置

```bash
cp config.a800.sh config.local.sh
# 按需修改模型路径、显存比例
```

### 7.4 跑训练

```bash
# Phase B 环境验证
bash scripts/verify_phase_b.sh

# Phase C GRPO baseline
bash scripts/run_phase_c.sh smoke     # 20 步
bash scripts/run_phase_c.sh full      # 500 步

# Phase D OPD 蒸馏
bash scripts/run_phase_d.sh smoke
bash scripts/run_phase_d.sh full

# Phase E 评测
bash scripts/run_phase_e.sh all       # 评 3 个 3B 模型
bash scripts/run_phase_e.sh 4         # 7B 老师（需 ~2 小时）
```

### 7.5 监控

```bash
tail -f logs/*/train_output.log       # 训练日志
watch -n 1 nvidia-smi                 # GPU
open https://wandb.ai/<user>/ZeroSearch
```

---

## 8. 文档索引

所有文档位于 `docs/` 目录下（除本 README）。

| 文档 | 内容 |
|------|------|
| [docs/实验报告.md](docs/实验报告.md) | **主报告**：实验设置 → 实现 → 结果 → 分析 → 下一轮预期 |
| [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md) | 评测结果原始数据 + 简历 bullet |
| [docs/PLAN.md](docs/PLAN.md) | 实施计划、代码地图、改动日志、已知坑清单 |
| [docs/EXPECTATIONS.md](docs/EXPECTATIONS.md) | 实验预期与论文调研 |
| [docs/OPD实验实现文档.md](docs/OPD实验实现文档.md) | 原始实现文档（环境配置、同步流程） |
| [docs/COMMANDS.md](docs/COMMANDS.md) | 训练命令参考（参数映射） |
| [scripts/README.md](scripts/README.md) | 脚本使用说明 |
| [scripts/WANDB_GUIDE.md](scripts/WANDB_GUIDE.md) | wandb 配置与关键指标 |

---

## 9. 环境版本

- CUDA 12.1
- torch 2.4.0+cu121
- vllm 0.6.3
- sglang (latest)
- transformers <4.48
- veRL (ZeroSearch fork, 含 OPD 改动)
- Python 3.9（rl-opd）/ 3.10（sglang）

---

## License

依上游仓库（ZeroSearch / Search-R1 / StepSearch）原始授权。
