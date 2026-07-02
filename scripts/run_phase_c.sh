#!/bin/bash
# Phase C: RL 基线训练启动脚本（GRPO）
# 用途：在服务器上启动模拟器 + 小步数验证 + 正式训练
# 用法：bash scripts/run_phase_c.sh [模式]
#   模式: smoke  — 小步数验证（~20 步，~30 分钟）
#         full   — 正式训练（~500 步，8-12 小时）
#         serve  — 仅启动模拟器（不训练）
#         stop   — 停止模拟器

set -u

# ── 颜色 ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_pass()  { echo -e "${GREEN}[PASS]${NC} $1"; }
print_fail()  { echo -e "${RED}[FAIL]${NC} $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_section() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

# ── 路径配置（按服务器实际情况调整）─────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZS_DIR="$PROJECT_ROOT/ZeroSearch"

# ── 加载本地配置（如存在）────────────────────────────
# 在项目根目录创建 config.local.sh 可覆盖下面所有默认值
# 该文件已加入 .gitignore，不会提交到仓库
LOCAL_CONFIG="$PROJECT_ROOT/config.local.sh"
if [ -f "$LOCAL_CONFIG" ]; then
    print_info "加载本地配置: $LOCAL_CONFIG"
    source "$LOCAL_CONFIG"
fi

# 模型与数据路径（按 AutoDL 约定）
STUDENT_MODEL="${STUDENT_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
SIMULATOR_MODEL="${SIMULATOR_MODEL:-/root/autodl-tmp/models/Simulation_LLM_wiki_3B_V2}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/data/ZeroSearch_dataset}"

# 训练参数
NUM_GPUS="${NUM_GPUS:-1}"
SEARCH_MODE="${SEARCH_MODE:-simulate_sft}"   # simulate_sft = 用微调版模拟器
SEARCH_ENGINE="${SEARCH_ENGINE:-wiki}"       # wiki 不需要 API key
START_THRESHOLD="${START_THRESHOLD:-0}"
END_THRESHOLD="${END_THRESHOLD:-0.5}"
MAX_TURNS="${MAX_TURNS:-5}"
TOPK="${TOPK:-5}"
SIMULATOR_PORT="${SIMULATOR_PORT:-6001}"
SIMULATOR_IP="${SIMULATOR_IP:-localhost}"

# 显存分配（A800 80GB：模拟器 30% + vLLM rollout 40% + FSDP actor/ref 剩余）
# 如仍 OOM，调低 SIMULATOR_MEM_FRACTION 或 ROLLOUT_GPU_MEM_UTIL
export SIMULATOR_MEM_FRACTION="${SIMULATOR_MEM_FRACTION:-0.12}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}"
# batch size（单卡需要小 micro batch 避免 logits 张量 OOM）
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-16}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
export REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-16}"
# 减少显存碎片
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# vLLM 推理优化（A800 需要 True 省显存，RTX6000 可 False 提速）
export ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
export FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
# Ray 临时目录（/tmp 空间有限，用数据盘）
export RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/ray_tmp}"
mkdir -p "${RAY_TMPDIR}" 2>/dev/null

# 训练步数（smoke vs full）
MODE="${1:-smoke}"
if [ "$MODE" = "smoke" ]; then
    TOTAL_STEPS="${TOTAL_STEPS:-20}"
    print_warn "小步数验证模式：仅跑 $TOTAL_STEPS 步"
elif [ "$MODE" = "full" ]; then
    TOTAL_STEPS="${TOTAL_STEPS:-500}"
    print_info "正式训练模式：跑 $TOTAL_STEPS 步"
fi

# ── 函数定义 ──────────────────────────────────────────

check_paths() {
    print_section "路径检查"
    local missing=0

    for path_name in "学生模型:$STUDENT_MODEL" "模拟器模型:$SIMULATOR_MODEL" "数据集:$DATA_PATH"; do
        name="${path_name%%:*}"
        path="${path_name#*:}"
        if [ -d "$path" ]; then
            print_pass "$name: $path"
        else
            print_fail "$name缺失: $path"
            missing=1
        fi
    done

    if [ "$missing" -eq 1 ]; then
        print_fail "有路径缺失，请先下载资源"
        print_info "下载命令见 scripts/run_phase_c.sh 顶部注释"
        exit 1
    fi
}

start_simulator() {
    print_section "启动模拟器（sglang 环境）"

    # 检查是否已经在运行
    if pgrep -f "sglang.launch_server" > /dev/null 2>&1; then
        print_warn "模拟器已在运行，跳过启动"
        print_info "如需重启: bash scripts/run_phase_c.sh stop && bash scripts/run_phase_c.sh serve"
        return 0
    fi

    # 检查 sglang 环境
    if ! command -v conda &> /dev/null; then
        print_fail "conda 未安装"
        exit 1
    fi

    if ! conda env list 2>/dev/null | grep -q "sglang"; then
        print_fail "sglang conda 环境不存在"
        print_info "创建: conda create -n sglang python=3.10 -y && conda activate sglang && pip install 'sglang[all]'"
        exit 1
    fi

    print_info "在 sglang 环境中启动模拟器..."
    print_info "模型: $SIMULATOR_MODEL"
    print_info "端口: $SIMULATOR_PORT"
    print_info "显存比例: $SIMULATOR_MEM_FRACTION（留空间给训练）"
    print_info "日志: /tmp/sglang_simulator.log"

    # 后台启动 sglang server
    # --mem-fraction-static 限制模拟器显存，否则 sglang 默认占 88% 导致训练 OOM
    nohup conda run -n sglang python -m sglang.launch_server \
        --model-path "$SIMULATOR_MODEL" \
        --host 0.0.0.0 \
        --port "$SIMULATOR_PORT" \
        --tp 1 \
        --dp 1 \
        --mem-fraction-static "$SIMULATOR_MEM_FRACTION" \
        > /tmp/sglang_simulator.log 2>&1 &

    SIMULATOR_PID=$!
    print_info "模拟器 PID: $SIMULATOR_PID"

    # 等待模拟器就绪（最多 5 分钟）
    print_info "等待模拟器就绪（最多 5 分钟）..."
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$SIMULATOR_PORT/health" > /dev/null 2>&1; then
            print_pass "模拟器已就绪（等待 ${i}0 秒）"
            return 0
        fi
        sleep 10
    done

    print_fail "模拟器启动超时，查看日志: tail -50 /tmp/sglang_simulator.log"
    exit 1
}

stop_simulator() {
    print_section "停止模拟器"
    pkill -f "sglang.launch_server" 2>/dev/null && print_pass "已停止" || print_warn "无运行中的模拟器"
}

run_training() {
    print_section "启动 RL 训练（GRPO，$MODE 模式）"

    # 切换到训练环境
    if [ "${CONDA_DEFAULT_ENV:-}" != "rl-opd" ]; then
        print_warn "当前不在 rl-opd 环境，尝试切换..."
        source activate rl-opd 2>/dev/null || conda activate rl-opd 2>/dev/null || {
            print_fail "无法激活 rl-opd 环境"
            exit 1
        }
    fi

    cd "$ZS_DIR" || { print_fail "ZeroSearch 目录不存在: $ZS_DIR"; exit 1; }

    print_info "训练脚本: train_grpo.sh"
    print_info "学生模型: $STUDENT_MODEL"
    print_info "数据集: $DATA_PATH"
    print_info "总步数: $TOTAL_STEPS"
    print_info "搜索模式: $SEARCH_MODE (模拟器)"
    print_info "搜索后端: $SEARCH_ENGINE"
    print_info "日志: 训练输出直接打印，同时写 wandb"

    # 确认模拟器可达
    if ! curl -s "http://$SIMULATOR_IP:$SIMULATOR_PORT/health" > /dev/null 2>&1; then
        print_fail "模拟器不可达: http://$SIMULATOR_IP:$SIMULATOR_PORT"
        print_info "请先启动: bash scripts/run_phase_c.sh serve"
        exit 1
    fi

    # wandb 登录检查
    if ! python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
        print_warn "wandb 未登录，训练会失败"
        print_info "登录命令: wandb login <你的KEY>"
        print_info "获取KEY: https://wandb.ai/authorize"
        read -p "是否已登录 wandb？(y/N) " confirm
        [ "$confirm" != "y" ] && exit 1
    fi

    # 设置本地日志（配置快照 + GPU 监控 + 输出重定向）
    # 日志目录加 mode 后缀，避免 smoke 和 full 互相覆盖
    EXPERIMENT_NAME="${STUDENT_MODEL##*/}_GRPO_${SEARCH_MODE}_${START_THRESHOLD}_${END_THRESHOLD}_${SEARCH_ENGINE}_turns_${MAX_TURNS}"
    LOG_SUBDIR="${EXPERIMENT_NAME}_${MODE}"
    CHECKPOINT_DIR="$ZS_DIR/verl_checkpoints/$EXPERIMENT_NAME"
    LOG_DIR="$PROJECT_ROOT/logs/$LOG_SUBDIR"
    mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
    TRAIN_LOG="$LOG_DIR/train_output.log"

    print_info "实验名: $EXPERIMENT_NAME (wandb)"
    print_info "本地日志目录: $LOG_DIR"
    print_info "训练输出: $TRAIN_LOG"
    print_info "GPU 监控: $LOG_DIR/gpu_monitor.csv"
    print_info "配置快照: $LOG_DIR/config_snapshot.yaml"
    print_info "环境快照: $LOG_DIR/env_info.txt"
    print_info "实时查看: tail -f $TRAIN_LOG"

    # 调用日志辅助脚本（保存配置 + 启动 GPU 监控）
    bash "$PROJECT_ROOT/scripts/setup_logging.sh" "$LOG_SUBDIR" "$CHECKPOINT_DIR" > /dev/null

    print_info "开始训练（输出同时写入 $TRAIN_LOG）..."
    set -o pipefail  # 让管道返回训练命令的退出码，而非 tee 的
    # 注意: train_grpo.sh 用 $2/$4/$6... 取偶数位参数，必须用 KEY VALUE 成对传递
    bash train_grpo.sh \
        NUM_GPUS_PER_NODE "$NUM_GPUS" \
        MODEL_PATH "$STUDENT_MODEL" \
        DATA_PATH "$DATA_PATH" \
        TOTAL_STEPS "$TOTAL_STEPS" \
        IP "$SIMULATOR_IP:$SIMULATOR_PORT" \
        SEARCH_MODE "$SEARCH_MODE" \
        SIMULATION_LLM "$SIMULATOR_MODEL" \
        START_THRESHOLD "$START_THRESHOLD" \
        END_THRESHOLD "$END_THRESHOLD" \
        SEARCH_ENGINE "$SEARCH_ENGINE" \
        MAX_TURNS "$MAX_TURNS" \
        TOPK "$TOPK" 2>&1 | tee "$TRAIN_LOG"
    TRAIN_EXIT_CODE=$?
    set +o pipefail

    # 训练结束后停止 GPU 监控
    if [ -f "$LOG_DIR/gpu_monitor.pid" ]; then
        kill "$(cat $LOG_DIR/gpu_monitor.pid)" 2>/dev/null && print_info "GPU 监控已停止"
    fi
}

show_status() {
    print_section "当前状态"
    echo "模拟器进程:"
    pgrep -af "sglang.launch_server" || echo "  未运行"
    echo ""
    echo "GPU 占用:"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 不可用"
    echo ""
    echo "训练日志: tail -f /tmp/sglang_simulator.log"
    echo "wandb: https://wandb.ai/<你的用户名>/ZeroSearch"
}

# ── 资源下载说明（放在脚本里方便查阅）─────────────────
show_download_help() {
    print_section "资源下载指南"
    cat <<'EOF'
如需下载模型/数据，在 rl-opd 环境中执行：

# 1. 学生模型（Qwen2.5-3B-Instruct）
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /root/autodl-tmp/models/Qwen2.5-3B-Instruct

# 2. 模拟器模型（Simulation_LLM_wiki_3B_V2，省显存）
huggingface-cli download sunhaonlp/Simulation_LLM_wiki_3B_V2 --local-dir /root/autodl-tmp/models/Simulation_LLM_wiki_3B_V2

# 3. 训练数据
huggingface-cli download --repo-type dataset sunhaonlp/ZeroSearch_dataset --local-dir /root/autodl-tmp/data/ZeroSearch_dataset

# 4. （OPD 用）7B 老师模型（Phase D 需要）
huggingface-cli download Alibaba-NLP/Search-R1-Qwen2.5-7B-GRPO --local-dir /root/autodl-tmp/models/Search-R1-Qwen2.5-7B-GRPO

如 HF 下载慢，用镜像:
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ...
EOF
}

# ── 主流程 ────────────────────────────────────────────
print_section "Phase C: RL 基线训练（GRPO）"
print_info "模式: $MODE"
print_info "项目根目录: $PROJECT_ROOT"

case "$MODE" in
    smoke|full)
        check_paths
        start_simulator
        run_training
        show_status
        ;;
    serve)
        check_paths
        start_simulator
        show_status
        ;;
    stop)
        stop_simulator
        ;;
    status)
        show_status
        ;;
    download)
        show_download_help
        ;;
    *)
        echo "用法: bash $0 {smoke|full|serve|stop|status|download}"
        echo ""
        echo "  smoke     — 小步数验证（20 步，~30 分钟）"
        echo "  full      — 正式训练（500 步，8-12 小时）"
        echo "  serve     — 仅启动模拟器"
        echo "  stop      — 停止模拟器"
        echo "  status    — 查看状态"
        echo "  download  — 显示资源下载命令"
        exit 1
        ;;
esac

print_section "完成"
if [ "${TRAIN_EXIT_CODE:-0}" -ne 0 ]; then
    print_fail "训练失败（退出码 $TRAIN_EXIT_CODE）"
    echo ""
    print_info "日志: $LOG_DIR/train_output.log"
    print_info "查看报错: tail -50 $LOG_DIR/train_output.log"
    echo ""
    print_warn "模拟器仍在后台运行（如需重试可复用）"
    print_info "  重试: bash scripts/run_phase_c.sh $MODE"
    print_info "  释放显存: bash scripts/run_phase_c.sh stop"
    exit $TRAIN_EXIT_CODE
elif [ "$MODE" = "smoke" ]; then
    print_pass "C1+C2 完成：模拟器已启动 + 小步数验证跑完"
    echo ""
    print_info "日志: $LOG_DIR/train_output.log"
    print_info "指标分析: python scripts/analyze_logs.py $LOG_DIR"
    echo ""
    print_warn "模拟器仍在后台运行（占用 GPU 显存）"
    print_info "  保留到跑 C3: 不用操作，直接 bash scripts/run_phase_c.sh full"
    print_info "  立即释放显存: bash scripts/run_phase_c.sh stop"
    print_info "  查看状态: bash scripts/run_phase_c.sh status"
    echo ""
    print_info "确认 reward 有上升趋势后，跑 C3 正式训练:"
    print_info "  bash scripts/run_phase_c.sh full"
elif [ "$MODE" = "full" ]; then
    print_pass "C3 完成：正式训练结束"
    echo ""
    print_info "checkpoint: $CHECKPOINT_DIR"
    print_info "日志: $LOG_DIR/train_output.log"
    print_info "指标分析: python scripts/analyze_logs.py $LOG_DIR"
    echo ""
    print_warn "模拟器仍在后台运行"
    print_info "  释放显存: bash scripts/run_phase_c.sh stop"
    print_info "下一步: Phase D (OPD 蒸馏训练)"
fi
