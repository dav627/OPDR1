# wandb 配置指南

## 1. 注册账号并登录

1. 注册：https://wandb.ai/signup
2. 获取 API Key：https://wandb.ai/authorize
3. 在服务器上登录：
   ```bash
   conda activate rl-opd
   wandb login <你的KEY>
   ```

## 2. 项目配置（已在训练脚本里写好）

| 参数 | 值 | 位置 |
|------|-----|------|
| `project_name` | `ZeroSearch` | `train_grpo.sh:65` / `train_opd.sh` |
| `experiment_name` | `{模型名}_{GRPO/OPD}_{search_mode}_...` | 脚本自动生成 |
| `logger` | `['wandb']` | `train_grpo.sh:57` |

## 3. 记录的指标

### 通用指标（每步记录）
- `actor/entropy_loss` — 策略熵
- `actor/pg_loss` — 策略梯度损失（GRPO 模式有效，OPD 模式为 0）
- `actor/pg_clipfrac` — PPO clip 比例
- `actor/ppo_kl` — 当前策略与旧策略的 KL
- `critic/...` — critic 相关（GRPO 不用 critic）
- `timing/...` — 各阶段耗时
- `state_tokens/coverage` — state_masking 覆盖率（检索 token 占比）

### GRPO 专属
- `actor/kl_loss` — KL 正则损失
- `actor/kl_coef` — KL 系数（0.001）
- `reward/...` — outcome reward 相关

### OPD 专属（本次新增）
- `actor/opd_loss` — OPD 主损失（= kl_loss）
- `actor/kl_loss` — 同上
- `actor/student_logp_mean` — 学生平均 log 概率
- `actor/teacher_logp_mean` — 老师平均 log 概率
- `actor/logp_gap` — teacher_logp - student_logp（应逐步缩小）
- `actor/response_tokens` — 响应 token 总数

## 4. 关键曲线（重点关注）

### RL 臂（GRPO）
- ✅ `reward/mean` 应上升
- ✅ `actor/pg_loss` 应下降后稳定
- ⚠️ `actor/kl_loss` 应保持小值（0.001 系数下）

### OPD 臂
- ✅ `actor/opd_loss`（= `actor/kl_loss`）应稳步下降
- ✅ `actor/logp_gap` 应逐步缩小（学生逼近老师）
- ⚠️ `actor/student_logp_mean` 不应爆炸（数值稳定）
- ⚠️ `actor/response_tokens` 应保持合理范围

## 5. wandb 面板访问

训练启动后，console 会打印 wandb URL，形如：
```
View wandb run: https://wandb.ai/<用户名>/ZeroSearch/runs/<run_id>
```

也可以直接访问：https://wandb.ai/<你的用户名>/ZeroSearch

## 6. 离线模式（服务器无法联网时）

如果服务器无法访问 wandb.ai，启用离线模式：
```bash
export WANDB_MODE=offline
```
训练会在本地保存日志到 `wandb/offline-run-*.log`，之后用 `wandb sync` 同步：
```bash
wandb sync /root/autodl-tmp/code/ZeroSearch/wandb/offline-run-*
```

## 7. 多实验对比

wandb 项目页面会自动把同 `project_name` 的所有 run 分组，可以：
- 对比 RL vs OPD 的 reward/loss 曲线
- 对比不同超参（如不同 `kl_loss_coef`）
- 导出数据用于论文/简历
