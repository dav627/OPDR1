#!/bin/bash
# Phase E: 评测对比脚本
# 在 ZeroSearch test set 上评测所有模型，输出 EM/F1 分数
# 用法：bash scripts/run_phase_e.sh [模型编号]
#   无参数  — 依次评测所有模型
#   1       — 未训练基线 (Qwen2.5-3B-Instruct)
#   2       — OPD checkpoint (你的蒸馏结果)
#   3       — 官方 GRPO 3B (RL 基线)
#   4       — 7B 老师 (天花板，可选)

set -u

# ── 颜色 ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_pass()    { echo -e "${GREEN}[PASS]${NC} $1"; }
print_fail()    { echo -e "${RED}[FAIL]${NC} $1"; }
print_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_section() { echo ""; echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"; }

# ── 路径 ──
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZS_DIR="$PROJECT_ROOT/ZeroSearch"

# ── 加载本地配置 ──
LOCAL_CONFIG="$PROJECT_ROOT/config.local.sh"
if [ -f "$LOCAL_CONFIG" ]; then
    source "$LOCAL_CONFIG"
fi

# ── 模型路径 ──
BASELINE_MODEL="${BASELINE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
OPD_CHECKPOINT="${OPD_CHECKPOINT:-$(find /root/autodl-tmp -path "*/global_step_400*" -type d 2>/dev/null | head -1)}"
GRPO_MODEL="${GRPO_MODEL:-/root/autodl-tmp/models/ZeroSearch_google_V2_Qwen2.5_3B_Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-/root/autodl-tmp/models/ZeroSearch_google_V2_Qwen2.5_7B_Instruct}"

DATA_PATH="${DATA_PATH:-/root/autodl-tmp/data/ZeroSearch_dataset}"
SIMULATOR_MODEL="${SIMULATOR_MODEL:-/root/autodl-tmp/models/Simulation_LLM_google_3B}"
SIMULATOR_PORT="${SIMULATOR_PORT:-6001}"
SIMULATOR_IP="${SIMULATOR_IP:-localhost}"

# 显存配置
export SIMULATOR_MEM_FRACTION="${SIMULATOR_MEM_FRACTION:-0.12}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
export FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
export RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/ray_tmp}"
mkdir -p "${RAY_TMPDIR}" 2>/dev/null

EVAL_DIR="$PROJECT_ROOT/eval_results"
mkdir -p "$EVAL_DIR"

# ── 准备 OPD checkpoint ──
prepare_opd_checkpoint() {
    if [ -z "$OPD_CHECKPOINT" ] || [ ! -d "$OPD_CHECKPOINT" ]; then
        print_fail "OPD checkpoint 未找到: $OPD_CHECKPOINT"
        return 1
    fi
    if [ ! -f "$OPD_CHECKPOINT/config.json" ]; then
        print_info "补齐 OPD checkpoint 的 tokenizer/config..."
        cp "$BASELINE_MODEL"/*.json "$OPD_CHECKPOINT/" 2>/dev/null
        cp "$BASELINE_MODEL"/tokenizer.json "$OPD_CHECKPOINT/" 2>/dev/null
        cp "$BASELINE_MODEL"/merges.txt "$OPD_CHECKPOINT/" 2>/dev/null
        cp "$BASELINE_MODEL"/vocab.json "$OPD_CHECKPOINT/" 2>/dev/null
    fi
    print_pass "OPD checkpoint 就绪: $OPD_CHECKPOINT"
}

# ── 启动模拟器 ──
# 参数: $1 = 期望的 mem_fraction（可选，默认 SIMULATOR_MEM_FRACTION）
# 用 /tmp/sglang_simulator.state 记录上次启动时的 mem，避免解析进程命令行出错
SGLANG_STATE_FILE="/tmp/sglang_simulator.state"
ensure_simulator() {
    local want_mem="${1:-$SIMULATOR_MEM_FRACTION}"

    if curl -s "http://$SIMULATOR_IP:$SIMULATOR_PORT/health" > /dev/null 2>&1; then
        local last_mem=""
        [ -f "$SGLANG_STATE_FILE" ] && last_mem=$(cat "$SGLANG_STATE_FILE" 2>/dev/null)
        if [ "$last_mem" = "$want_mem" ]; then
            print_pass "模拟器已在运行（mem=$last_mem）"
            return 0
        else
            print_warn "模拟器已在运行但显存比例 ${last_mem:-未知} ≠ 需要 $want_mem，重启中..."
            pkill -f "sglang.launch_server" 2>/dev/null
            # 等 GPU 显存释放
            for i in $(seq 1 30); do
                if ! pgrep -f "sglang.launch_server" > /dev/null 2>&1; then break; fi
                sleep 2
            done
            sleep 3
        fi
    fi

    print_info "启动模拟器（mem_fraction=$want_mem）..."
    nohup conda run -n sglang python -m sglang.launch_server \
        --model-path "$SIMULATOR_MODEL" \
        --host 0.0.0.0 --port "$SIMULATOR_PORT" \
        --tp 1 --dp 1 \
        --mem-fraction-static "$want_mem" \
        > /tmp/sglang_simulator.log 2>&1 &
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$SIMULATOR_PORT/health" > /dev/null 2>&1; then
            print_pass "模拟器就绪（${i}0 秒，mem=$want_mem）"
            echo "$want_mem" > "$SGLANG_STATE_FILE"
            return 0
        fi
        sleep 10
    done
    print_fail "模拟器启动超时，查看日志: tail -50 /tmp/sglang_simulator.log"
    exit 1
}

# ── 单模型评测 ──
eval_model() {
    local model_path="$1"
    local model_name="$2"
    local model_size="${3:-3b}"
    local log_file="$EVAL_DIR/${model_name}.log"
    local result_file="$EVAL_DIR/${model_name}_record.json"

    print_section "评测: $model_name"
    print_info "模型: $model_path"
    print_info "日志: $log_file"

    if [ ! -d "$model_path" ]; then
        print_fail "模型不存在: $model_path"
        return 1
    fi

    # 切换环境
    if [ "${CONDA_DEFAULT_ENV:-}" != "rl-opd" ]; then
        source activate rl-opd 2>/dev/null || conda activate rl-opd 2>/dev/null || true
    fi

    cd "$ZS_DIR" || { print_fail "ZeroSearch 目录不存在"; return 1; }

    # 7B 模型：单卡 80GB 同时塞 模拟器+vLLM+FSDP 同步，需大幅压低显存
    local rollout_mem="$ROLLOUT_GPU_MEM_UTIL"
    local val_batch=64
    local n_agent=5
    local sim_mem="$SIMULATOR_MEM_FRACTION"
    local ref_micro=16
    local ref_model_path="$model_path"
    if [ "$model_size" = "7b" ]; then
        rollout_mem=0.15
        val_batch=4
        n_agent=1
        sim_mem=0.06
        ref_micro=2
        # 评测时 ref 不参与计算（use_kl_loss=false），但 verl 仍会实例化 ref。
        # 把 ref 指向 3B baseline，避免再加载一份 7B。
        ref_model_path="$BASELINE_MODEL"
        print_warn "7B 模型：rollout_mem=0.15, val_batch=4, n_agent=1, sim_mem=0.06, ref→3B"
    fi
    # 确保模拟器以 sim_mem 比例运行（不匹配会自动重启）
    ensure_simulator "$sim_mem"

    export VLLM_ATTENTION_BACKEND=XFORMERS

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files=$DATA_PATH/train.parquet \
        data.val_files=$DATA_PATH/test.parquet \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size=32 \
        data.val_batch_size=$val_batch \
        data.max_prompt_length=4096 \
        data.max_response_length=500 \
        data.max_start_length=2048 \
        data.max_obs_length=2048 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path=$model_path \
        actor_rollout_ref.model.enable_gradient_checkpointing=false \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
        actor_rollout_ref.actor.use_kl_loss=false \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size=16 \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.grad_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
        +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=$ref_micro \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_mem \
        actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
        actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE} \
        actor_rollout_ref.ref.model.path=$ref_model_path \
        actor_rollout_ref.ref.log_prob_micro_batch_size=$ref_micro \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.state_masking=True \
        algorithm.no_think_rl=false \
        actor_rollout_ref.rollout.n_agent=$n_agent \
        actor_rollout_ref.rollout.temperature=1 \
        trainer.logger=[] \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=0 \
        trainer.test_freq=9999 \
        trainer.total_epochs=10 \
        trainer.total_training_steps=1 \
        trainer.default_local_dir=$EVAL_DIR/$model_name \
        trainer.max_turns=5 \
        trainer.reward_function=f1 \
        trainer.do_search=True \
        retriever.start_threshold=0 \
        retriever.end_threshold=0.5 \
        retriever.llm_ip=${SIMULATOR_IP}:${SIMULATOR_PORT} \
        retriever.search_mode=simulate_sft \
        retriever.search_engine=google \
        retriever.topk=5 \
        retriever.simulate_llm=${SIMULATOR_MODEL} 2>&1 | tee "$log_file"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        print_pass "$model_name 评测完成"
        # 提取分数
        grep -oP "val/test_score_(EM|F1)/\S+.*?[\d.]+" "$log_file" | tail -20
    else
        print_fail "$model_name 评测失败（退出码 $exit_code）"
    fi

    return $exit_code
}

# ── 汇总结果 ──
summarize() {
    print_section "评测结果汇总（7 个 benchmark）"

    # 提取所有 benchmark 名称
    local benchmarks="nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle"

    # 表头
    printf "%-22s" "模型"
    for b in $benchmarks; do
        printf " | %-8s" "$b"
    done
    echo ""
    printf "%-22s" "----------------------"
    for b in $benchmarks; do
        printf "-+----------"
    done
    echo ""

    # 每个模型一行
    for log_file in "$EVAL_DIR"/*.log; do
        [ -f "$log_file" ] || continue
        local name=$(basename "$log_file" .log)
        printf "%-22s" "$name"
        for b in $benchmarks; do
            # 提取 EM 和 F1，格式: val/test_score_EM/<bench>': 0.242
            local em=$(grep "test_score_EM/${b}':" "$log_file" | grep -oP "[\d.]+$" | tail -1 || echo "-")
            local f1=$(grep "test_score_F1/${b}':" "$log_file" | grep -oP "[\d.]+$" | tail -1 || echo "-")
            printf " | %-4s/%-4s" "$em" "$f1"
        done
        echo ""
    done
    echo ""
    echo "（每个格子格式: EM/F1）"
    echo ""
}

# ── 主流程 ──
print_section "Phase E: 评测对比"

TARGET="${1:-all}"

# 检查路径
print_section "模型路径确认"
echo "  1. 未训练基线: $BASELINE_MODEL"
echo "  2. OPD 蒸馏:   $OPD_CHECKPOINT"
echo "  3. 官方 GRPO:  $GRPO_MODEL"
echo "  4. 7B 老师:    $TEACHER_MODEL"
echo ""

# 根据目标模型决定模拟器显存比例（7B 需要 0.06，其余 0.12）
case "$TARGET" in
    4) TARGET_SIM_MEM=0.06 ;;
    *) TARGET_SIM_MEM="$SIMULATOR_MEM_FRACTION" ;;
esac

ensure_simulator "$TARGET_SIM_MEM"

# 仅在需要 OPD 评测时准备其 checkpoint
case "$TARGET" in
    2|all) prepare_opd_checkpoint ;;
esac

case "$TARGET" in
    1) eval_model "$BASELINE_MODEL" "1_baseline_3B" "3b" ;;
    2) eval_model "$OPD_CHECKPOINT" "2_opd_3B" "3b" ;;
    3) eval_model "$GRPO_MODEL" "3_grpo_3B" "3b" ;;
    4) eval_model "$TEACHER_MODEL" "4_teacher_7B" "7b" ;;
    all)
        eval_model "$BASELINE_MODEL" "1_baseline_3B" "3b"
        eval_model "$OPD_CHECKPOINT" "2_opd_3B" "3b"
        eval_model "$GRPO_MODEL" "3_grpo_3B" "3b"
        # 7B 可选，取消注释启用
        # eval_model "$TEACHER_MODEL" "4_teacher_7B" "7b"
        ;;
    *)
        echo "用法: bash $0 {1|2|3|4|all}"
        exit 1
        ;;
esac

summarize

print_section "完成"
print_info "详细日志: $EVAL_DIR/"
print_info "推理记录: $EVAL_DIR/*/record_init.json"
