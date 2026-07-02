#!/bin/bash
# 训练日志辅助脚本
# 用途：保存配置快照、启动 GPU 监控、重定向训练输出到本地文件
# 用法：在 run_phase_c.sh / run_phase_d.sh 中被自动调用
#      也可手动: bash scripts/setup_logging.sh <experiment_name> <checkpoint_dir>

set -u

EXPERIMENT_NAME="${1:-unknown_exp}"
CHECKPOINT_DIR="${2:-/root/autodl-tmp/code/ZeroSearch/verl_checkpoints/$EXPERIMENT_NAME}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/code/logs/$EXPERIMENT_NAME}"

# 创建日志目录
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# ── 1. 保存配置快照 ─────────────────────────────────
CONFIG_SNAPSHOT="$LOG_DIR/config_snapshot.yaml"
if [ -f "/root/autodl-tmp/code/ZeroSearch/verl/trainer/config/ppo_trainer.yaml" ]; then
    cp /root/autodl-tmp/code/ZeroSearch/verl/trainer/config/ppo_trainer.yaml "$CONFIG_SNAPSHOT"
    echo "[LOG] 配置快照已保存: $CONFIG_SNAPSHOT"
fi

# ── 2. 保存环境信息 ─────────────────────────────────
ENV_INFO="$LOG_DIR/env_info.txt"
{
    echo "=== 训练环境快照 ==="
    echo "时间: $(date)"
    echo "主机: $(hostname)"
    echo "conda 环境: ${CONDA_DEFAULT_ENV:-base}"
    echo ""
    echo "=== Git 状态 ==="
    cd /root/autodl-tmp/code 2>/dev/null && git log --oneline -1 2>/dev/null
    git status --short 2>/dev/null | head -5
    echo ""
    echo "=== GPU 信息 ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
    echo ""
    echo "=== Python 包版本 ==="
    python -c "
import torch, transformers
print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'transformers: {transformers.__version__}')
try:
    import vllm; print(f'vllm: {vllm.__version__}')
except: print('vllm: N/A')
try:
    import flash_attn; print(f'flash_attn: {flash_attn.__version__}')
except: print('flash_attn: N/A')
try:
    import verl; print(f'verl: OK')
except: print('verl: N/A')
" 2>&1
} > "$ENV_INFO" 2>&1
echo "[LOG] 环境信息已保存: $ENV_INFO"

# ── 3. 启动 GPU 监控（每 30 秒记录一次）────────────
GPU_LOG="$LOG_DIR/gpu_monitor.csv"
echo "timestamp,memory.used_MiB,memory.total_MiB,utilization_pct,temperature_C" > "$GPU_LOG"

(
    while true; do
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | \
            awk -v ts="$(date +%s)" '{print ts","$1","$2","$3","$4}' >> "$GPU_LOG"
        sleep 30
    done
) &
GPU_MONITOR_PID=$!
echo "[LOG] GPU 监控已启动 (PID: $GPU_MONITOR_PID)"
echo "[LOG] GPU 日志: $GPU_LOG"
echo "$GPU_MONITOR_PID" > "$LOG_DIR/gpu_monitor.pid"

# ── 4. 训练输出重定向说明 ───────────────────────────
TRAIN_LOG="$LOG_DIR/train_output.log"
echo "[LOG] 训练输出将保存到: $TRAIN_LOG"
echo "[LOG] 实时查看: tail -f $TRAIN_LOG"
echo "[LOG] wandb 面板: https://wandb.ai/<你的用户名>/ZeroSearch"

# 输出 LOG_DIR 供调用脚本使用
echo "$LOG_DIR"
