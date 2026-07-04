#!/bin/bash
# Phase D: OPD 蒸馏训练启动脚本
# 用途：在服务器上启动模拟器 + OPD 训练（7B 老师蒸馏到 3B 学生）
# 用法：bash scripts/run_phase_d.sh [模式]
#   模式: smoke  — 小步数验证（~20 步）
#         full   — 正式训练（~500 步）
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

# ── 路径配置 ──────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZS_DIR="$PROJECT_ROOT/ZeroSearch"

# ── 加载本地配置（如存在）────────────────────────────
LOCAL_CONFIG="$PROJECT_ROOT/config.local.sh"
if [ -f "$LOCAL_CONFIG" ]; then
    print_info "加载本地配置: $LOCAL_CONFIG"
    source "$LOCAL_CONFIG"
fi

# 模型与数据路径
STUDENT_MODEL="${STUDENT_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
SIMULATOR_MODEL="${SIMULATOR_MODEL:-/root/autodl-tmp/models/Simulation_LLM_wiki_3B_V2}"
TEACHER_MODEL="${TEACHER_MODEL:-/root/autodl-tmp/models/Search-R1-Qwen2.5-7B-GRPO}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/data/ZeroSearch_dataset}"

# 训练参数
NUM_GPUS="${NUM_GPUS:-1}"
SEARCH_MODE="${SEARCH_MODE:-simulate_sft}"
SEARCH_ENGINE="${SEARCH_ENGINE:-wiki}"
START_THRESHOLD="${START_THRESHOLD:-0}"
END_THRESHOLD="${END_THRESHOLD:-0.5}"
MAX_TURNS="${MAX_TURNS:-5}"
TOPK="${TOPK:-5}"
SIMULATOR_PORT="${SIMULATOR_PORT:-6001}"
SIMULATOR_IP="${SIMULATOR_IP:-localhost}"

# 显存分配（A800 80GB 单卡优化）
export SIMULATOR_MEM_FRACTION="${SIMULATOR_MEM_FRACTION:-0.12}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-16}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
# OPD 7B ref 前向需要更小的 micro batch
export REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
export FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
export RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/ray_tmp}"
mkdir -p "${RAY_TMPDIR}" 2>/dev/null

# 训练步数
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

    for path_name in "学生模型:$STUDENT_MODEL" "模拟器模型:$SIMULATOR_MODEL" "老师模型:$TEACHER_MODEL" "数据集:$DATA_PATH"; do
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
        print_info "老师模型下载: huggingface-cli download Alibaba-NLP/Search-R1-Qwen2.5-7B-GRPO --local-dir $TEACHER_MODEL"
        exit 1
    fi
}

start_simulator() {
    print_section "启动模拟器（sglang 环境）"

    if pgrep -f "sglang.launch_server" > /dev/null 2>&1; then
        print_warn "模拟器已在运行，跳过启动"
        return 0
    fi

    if ! conda env list 2>/dev/null | grep -q "sglang"; then
        print_fail "sglang conda 环境不存在"
        exit 1
    fi

    print_info "模型: $SIMULATOR_MODEL"
    print_info "端口: $SIMULATOR_PORT"
    print_info "显存比例: $SIMULATOR_MEM_FRACTION"

    nohup conda run -n sglang python -m sglang.launch_server \
        --model-path "$SIMULATOR_MODEL" \
        --host 0.0.0.0 \
        --port "$SIMULATOR_PORT" \
        --tp 1 --dp 1 \
        --mem-fraction-static "$SIMULATOR_MEM_FRACTION" \
        > /tmp/sglang_simulator.log 2>&1 &

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
    print_section "启动 OPD 蒸馏训练（$MODE 模式）"

    if [ "${CONDA_DEFAULT_ENV:-}" != "rl-opd" ]; then
        source activate rl-opd 2>/dev/null || conda activate rl-opd 2>/dev/null || {
            print_fail "无法激活 rl-opd 环境"; exit 1
        }
    fi

    cd "$ZS_DIR" || { print_fail "ZeroSearch 目录不存在: $ZS_DIR"; exit 1; }

    print_info "训练脚本: train_opd.sh"
    print_info "学生模型: $STUDENT_MODEL"
    print_info "老师模型: $TEACHER_MODEL"
    print_info "数据集: $DATA_PATH"
    print_info "总步数: $TOTAL_STEPS"

    if ! curl -s "http://$SIMULATOR_IP:$SIMULATOR_PORT/health" > /dev/null 2>&1; then
        print_fail "模拟器不可达: http://$SIMULATOR_IP:$SIMULATOR_PORT"
        print_info "请先启动: bash scripts/run_phase_d.sh serve"
        exit 1
    fi

    if ! python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
        print_warn "wandb 未登录"
        print_info "登录: wandb login <KEY>"
        read -p "是否已登录 wandb？(y/N) " confirm
        [ "$confirm" != "y" ] && exit 1
    fi

    EXPERIMENT_NAME="${STUDENT_MODEL##*/}_OPD_${SEARCH_MODE}_${START_THRESHOLD}_${END_THRESHOLD}_${SEARCH_ENGINE}_turns_${MAX_TURNS}"
    LOG_SUBDIR="${EXPERIMENT_NAME}_${MODE}"
    CHECKPOINT_DIR="$ZS_DIR/verl_checkpoints/$EXPERIMENT_NAME"
    LOG_DIR="$PROJECT_ROOT/logs/$LOG_SUBDIR"
    mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
    TRAIN_LOG="$LOG_DIR/train_output.log"

    print_info "实验名: $EXPERIMENT_NAME (wandb)"
    print_info "本地日志: $LOG_DIR"
    print_info "实时查看: tail -f $TRAIN_LOG"

    bash "$PROJECT_ROOT/scripts/setup_logging.sh" "$LOG_SUBDIR" "$CHECKPOINT_DIR" > /dev/null

    print_info "开始训练..."
    set -o pipefail
    bash train_opd.sh \
        NUM_GPUS_PER_NODE "$NUM_GPUS" \
        MODEL_PATH "$STUDENT_MODEL" \
        TEACHER_PATH "$TEACHER_MODEL" \
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
}

# ── 主流程 ────────────────────────────────────────────
print_section "Phase D: OPD 蒸馏训练（7B → 3B）"
print_info "模式: $MODE"

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
    *)
        echo "用法: bash $0 {smoke|full|serve|stop|status}"
        echo ""
        echo "  smoke  — 小步数验证（20 步）"
        echo "  full   — 正式训练（500 步）"
        echo "  serve  — 仅启动模拟器"
        echo "  stop   — 停止模拟器"
        echo "  status — 查看状态"
        exit 1
        ;;
esac

print_section "完成"
if [ "${TRAIN_EXIT_CODE:-0}" -ne 0 ]; then
    print_fail "训练失败（退出码 $TRAIN_EXIT_CODE）"
    print_info "查看报错: tail -50 $TRAIN_LOG"
    print_warn "模拟器仍在后台运行"
    print_info "  重试: bash scripts/run_phase_d.sh $MODE"
    print_info "  释放显存: bash scripts/run_phase_d.sh stop"
    exit $TRAIN_EXIT_CODE
elif [ "$MODE" = "smoke" ]; then
    print_pass "D1 smoke 完成"
    print_info "确认 kl_loss 在下降后，跑正式训练:"
    print_info "  bash scripts/run_phase_d.sh full"
elif [ "$MODE" = "full" ]; then
    print_pass "D2 完成：OPD 正式训练结束"
    print_info "checkpoint: $CHECKPOINT_DIR"
    print_info "下一步: Phase C (GRPO RL 基线) 或 Phase E (对比分析)"
fi
