# OPDR1: On-Policy Distillation vs Reinforcement Learning

对比 RL (GRPO) 与 OPD（在策略蒸馏）在 agentic search 小模型后训练上的成本/效果/能力上限。

## 项目目标

回答一个业务问题：企业要低成本拿到一个可部署的小模型，该用 RL 训，还是从更强的老师蒸？

- **学生模型**: Qwen2.5-3B-Instruct
- **OPD 老师**: Search-R1-Qwen2.5-7B-GRPO
- **后端**: ZeroSearch 模拟搜索（$0 真实 API 调用）
- **框架**: veRL (based on ZeroSearch/Search-R1)

## 两条训练路线

| | RL (GRPO) | OPD (在策略蒸馏) |
|---|-----------|-----------------|
| 损失函数 | policy gradient + KL 正则 | reverse KL(π_student ‖ π_teacher) |
| 参考模型 | 初始 actor 冻结副本 | 7B 老师模型 |
| 上限 | 可能超过老师 | 不超过老师 |
| 成本 | rollout + reward 计算 | rollout + teacher forward |

## 代码改动

在 ZeroSearch/veRL 基础上最小改动实现 OPD 模式：

1. `verl/trainer/config/ppo_trainer.yaml` — 新增 `loss_mode` 和 `ref.model_path`
2. `verl/workers/fsdp_workers.py` — ref 模型支持独立路径
3. `verl/workers/actor/dp_actor.py` — OPD 损失分支
4. `train_opd.sh` — OPD 训练脚本

详见 [PLAN.md](PLAN.md) 的 §2.5 代码地图和 §2.6 改动日志。

## 快速部署

本项目已包含完整的第三方仓库代码，服务器端可直接 clone 使用：

```bash
# 1. 克隆项目
git clone https://github.com/dav627/OPDR1.git
cd OPDR1

# 2. 配置环境（参考 OPD实验实现文档.md §3.1）
conda create -n rl-opd python=3.9 -y
conda activate rl-opd
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 wandb huggingface_hub transformers<4.48

# 3. 安装 veRL（ZeroSearch 已包含 OPD 改动）
cd ZeroSearch
pip install -e .

# 4. 准备模型和数据（参考 COMMANDS.md）
# 模型下载到 /root/autodl-tmp/models/
# 数据下载到 /root/autodl-tmp/data/

# 5. 运行训练（参考 COMMANDS.md）
bash train_grpo.sh  # RL 基线
bash train_opd.sh   # OPD 蒸馏
```

**项目结构：**
- `ZeroSearch/` — 主框架（已包含 OPD 改动）
- `Search-R1/` — 参考实现 + OPD 老师模型来源
- `StepSearch/` — 过程奖励（可选）
- `tests/` — 单元测试
- `PLAN.md` — 实施计划与代码地图
- `COMMANDS.md` — 训练命令参考

## 训练命令

见 [COMMANDS.md](COMMANDS.md)。

## 环境

- CUDA 12.1
- torch 2.4.0+cu121, vllm 0.6.3, transformers <4.48
- veRL (ZeroSearch fork)

## 文档

- [PLAN.md](PLAN.md) — 实施计划与代码地图
- [COMMANDS.md](COMMANDS.md) — 训练命令参考
- [OPD实验实现文档.md](OPD实验实现文档.md) — 原始实现文档
