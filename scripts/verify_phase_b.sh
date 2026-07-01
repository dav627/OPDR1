#!/bin/bash
# Phase B: 服务器环境验证脚本
# 用途：在 GPU 服务器上验证训练环境是否就绪
# 用法：bash scripts/verify_phase_b.sh [STUDENT_MODEL_PATH]
# 示例：bash scripts/verify_phase_b.sh /root/autodl-tmp/models/Qwen2.5-3B-Instruct

set -u

# ── 颜色定义 ─────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    WARN_COUNT=$((WARN_COUNT + 1))
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_section() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

# ── 参数处理 ─────────────────────────────────────────
STUDENT_MODEL_PATH="${1:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

print_section "Phase B: 服务器环境验证"
print_info "项目根目录: $PROJECT_ROOT"
print_info "学生模型路径: $STUDENT_MODEL_PATH"
print_info "当前 conda 环境: ${CONDA_DEFAULT_ENV:-未激活}"
print_info "Python: $(which python3 2>/dev/null || echo '未找到')"

# ════════════════════════════════════════════════════════════════
# B1: GPU 可用性检查
# ════════════════════════════════════════════════════════════════
print_section "B1: GPU 可用性检查"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)

    if [ "$GPU_COUNT" -ge 1 ]; then
        print_pass "检测到 $GPU_COUNT 张 GPU: $GPU_NAME (总显存: $GPU_MEM)"
        if echo "$GPU_NAME" | grep -qi "A800\|A100\|H100\|4090"; then
            print_pass "GPU 型号符合训练要求: $GPU_NAME"
        else
            print_warn "GPU 型号为 $GPU_NAME，建议 A800/A100/H100"
        fi
    else
        print_fail "未检测到 GPU（可能处于无卡模式）"
        print_info "修复建议: AutoDL 控制台切换到「有卡模式」"
    fi
else
    print_fail "nvidia-smi 命令未找到"
    print_info "修复建议: 检查 NVIDIA 驱动是否安装"
fi

# ════════════════════════════════════════════════════════════════
# B2: Python 依赖栈自检
# ════════════════════════════════════════════════════════════════
print_section "B2: Python 依赖栈自检"

# 检查 Python 版本
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
if [ "$PY_VERSION" = "3.9" ]; then
    print_pass "Python 版本: 3.9 (符合要求)"
else
    print_warn "Python 版本: ${PY_VERSION:-未知}，建议 3.9"
fi

# 检查 torch
print_info "检查 torch..."
python3 -c "
import torch
print(f'  torch 版本: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA 版本: {torch.version.cuda}')
    print(f'  GPU 数量: {torch.cuda.device_count()}')
    print(f'  GPU 名称: {torch.cuda.get_device_name(0)}')

    # 检查版本匹配
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda or ''
    if '2.4.0' in torch_ver and '12.1' in cuda_ver:
        print('  版本匹配: torch 2.4.0 + cu121 ✅')
    else:
        print(f'  ⚠️ 版本不匹配: torch={torch_ver}, cuda={cuda_ver}，建议 torch==2.4.0+cu121')

    if not torch.cuda.is_available():
        print('  ❌ torch.cuda.is_available() = False')
        exit(1)
else:
    print('  ❌ torch.cuda.is_available() = False')
    print('  可能原因: 无卡模式 / CUDA 版本不匹配 / torch 装的是 CPU 版')
    exit(1)
" 2>&1 | sed 's/^/  /' && print_pass "torch 检查通过" || print_fail "torch 检查失败"

# 检查 flash-attn
print_info "检查 flash-attn..."
if python3 -c "import flash_attn" 2>/dev/null; then
    FA_VERSION=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
    print_pass "flash-attn 可用 (版本: $FA_VERSION)"
else
    print_warn "flash-attn 不可用（可能需要预编译 wheel）"
    print_info "  修复建议: pip install flash-attn --no-build-isolation"
    print_info "  或使用预编译 wheel: cu12 + torch2.4 + cp39 + abiFALSE"
fi

# 检查 verl
print_info "检查 verl..."
if python3 -c "import verl" 2>/dev/null; then
    print_pass "verl 可用"
else
    print_fail "verl 不可用"
    print_info "  修复建议: cd ZeroSearch && pip install -e ."
fi

# 检查 transformers 版本
print_info "检查 transformers 版本..."
TF_VERSION=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
if [ -n "$TF_VERSION" ]; then
    if python3 -c "
import transformers
from packaging import version
assert version.parse(transformers.__version__) < version.parse('4.48'), '版本过高'
" 2>/dev/null; then
        print_pass "transformers 版本: $TF_VERSION (<4.48，符合 vllm 0.6.3 要求)"
    else
        print_warn "transformers 版本: $TF_VERSION，建议 <4.48（vllm 0.6.3 兼容性）"
    fi
else
    print_fail "transformers 未安装"
fi

# ════════════════════════════════════════════════════════════════
# B3: vLLM 加载测试
# ════════════════════════════════════════════════════════════════
print_section "B3: vLLM 加载测试"

if [ ! -d "$STUDENT_MODEL_PATH" ]; then
    print_fail "学生模型路径不存在: $STUDENT_MODEL_PATH"
    print_info "  修复建议: 检查模型是否已下载到该路径"
    print_info "  或指定路径: bash scripts/verify_phase_b.sh <模型路径>"
else
    print_info "模型路径存在: $STUDENT_MODEL_PATH"

    # 检查 vllm 是否安装
    if python3 -c "import vllm" 2>/dev/null; then
        VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
        print_pass "vllm 可用 (版本: $VLLM_VERSION)"

        print_info "尝试用 vllm 加载模型并生成一句话（可能需要 1-2 分钟）..."
        if python3 -c "
from vllm import LLM, SamplingParams
import sys

model_path = '$STUDENT_MODEL_PATH'
print(f'  加载模型: {model_path}', flush=True)
llm = LLM(model=model_path, dtype='bfloat16', gpu_memory_utilization=0.5)
print('  模型加载成功', flush=True)

prompts = ['你好，请用一句话介绍你自己。']
sampling = SamplingParams(temperature=0.7, max_tokens=50)
outputs = llm.generate(prompts, sampling)
print(f'  生成结果: {outputs[0].outputs[0].text}', flush=True)
print('  vLLM 生成测试通过 ✅', flush=True)
" 2>&1 | sed 's/^/  /' | tail -20; then
            print_pass "vLLM 加载并生成成功"
        else
            print_fail "vLLM 加载或生成失败"
            print_info "  常见原因: GPU 显存不足 / 模型路径错误 / CUDA 版本不匹配"
        fi
    else
        print_fail "vllm 未安装"
        print_info "  修复建议: pip install vllm==0.6.3"
    fi
fi

# ════════════════════════════════════════════════════════════════
# B4: sglang 环境检查
# ════════════════════════════════════════════════════════════════
print_section "B4: sglang 环境检查"

# sglang 通常在独立 conda 环境
print_info "当前环境: ${CONDA_DEFAULT_ENV:-base}"
print_info "sglang 通常在独立 conda 环境（如 sglang）"

# 检查是否有 sglang 环境
if command -v conda &> /dev/null; then
    if conda env list 2>/dev/null | grep -q "sglang"; then
        print_pass "检测到 sglang conda 环境"
        print_info "切换到 sglang 环境测试: conda activate sglang"

        # 尝试在 sglang 环境中检查
        if conda run -n sglang python -c "import sglang; print(f'sglang 版本: {sglang.__version__}')" 2>/dev/null; then
            print_pass "sglang 在独立环境中可用"
        else
            print_warn "sglang 环境存在但 import 失败"
            print_info "  修复建议: conda activate sglang && pip install 'sglang[all]'"
        fi
    else
        print_warn "未检测到 sglang conda 环境"
        print_info "  修复建议: conda create -n sglang python=3.10 -y"
        print_info "            conda activate sglang"
        print_info "            pip install 'sglang[all]'"
    fi
else
    # 当前环境检查
    if python3 -c "import sglang" 2>/dev/null; then
        SGL_VERSION=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null)
        print_pass "sglang 可用 (版本: $SGL_VERSION)"
    else
        print_warn "sglang 未安装（当前环境）"
        print_info "  建议: 创建独立 sglang 环境避免与训练环境冲突"
    fi
fi

# ════════════════════════════════════════════════════════════════
# B5: 代码改动同步确认
# ════════════════════════════════════════════════════════════════
print_section "B5: 代码改动同步确认"

# 检查关键文件是否存在
print_info "检查 OPD 改动文件..."

FILE_CHECKS=(
    "ZeroSearch/verl/trainer/config/ppo_trainer.yaml"
    "ZeroSearch/verl/workers/fsdp_workers.py"
    "ZeroSearch/verl/workers/actor/dp_actor.py"
    "ZeroSearch/train_opd.sh"
    "tests/test_opd_logic.py"
)

for f in "${FILE_CHECKS[@]}"; do
    full_path="$PROJECT_ROOT/$f"
    if [ -f "$full_path" ]; then
        print_pass "存在: $f"
    else
        print_fail "缺失: $f"
    fi
done

# 检查关键改动内容
print_info "检查 OPD 改动内容..."

# 检查 ppo_trainer.yaml 是否包含 loss_mode
if grep -q "loss_mode" "$PROJECT_ROOT/ZeroSearch/verl/trainer/config/ppo_trainer.yaml" 2>/dev/null; then
    print_pass "ppo_trainer.yaml 包含 loss_mode 字段"
else
    print_fail "ppo_trainer.yaml 缺少 loss_mode 字段"
fi

# 检查 ppo_trainer.yaml 是否包含 ref.model_path
if grep -q "model_path" "$PROJECT_ROOT/ZeroSearch/verl/trainer/config/ppo_trainer.yaml" 2>/dev/null; then
    print_pass "ppo_trainer.yaml 包含 ref.model_path 字段"
else
    print_fail "ppo_trainer.yaml 缺少 ref.model_path 字段"
fi

# 检查 dp_actor.py 是否包含 OPD 分支
if grep -q "loss_mode == 'opd'" "$PROJECT_ROOT/ZeroSearch/verl/workers/actor/dp_actor.py" 2>/dev/null; then
    print_pass "dp_actor.py 包含 OPD 损失分支"
else
    print_fail "dp_actor.py 缺少 OPD 损失分支"
fi

# 检查 fsdp_workers.py 是否支持独立 ref 路径
if grep -q "ref_model_path" "$PROJECT_ROOT/ZeroSearch/verl/workers/fsdp_workers.py" 2>/dev/null; then
    print_pass "fsdp_workers.py 支持 ref 模型独立路径"
else
    print_fail "fsdp_workers.py 缺少 ref 模型独立路径支持"
fi

# 检查 train_opd.sh 是否存在且包含关键参数
if [ -f "$PROJECT_ROOT/ZeroSearch/train_opd.sh" ]; then
    if grep -q "loss_mode=opd" "$PROJECT_ROOT/ZeroSearch/train_opd.sh" 2>/dev/null; then
        print_pass "train_opd.sh 配置了 loss_mode=opd"
    else
        print_fail "train_opd.sh 未配置 loss_mode=opd"
    fi
    if grep -q "ref.model_path" "$PROJECT_ROOT/ZeroSearch/train_opd.sh" 2>/dev/null; then
        print_pass "train_opd.sh 配置了 ref.model_path"
    else
        print_fail "train_opd.sh 未配置 ref.model_path"
    fi
fi

# ════════════════════════════════════════════════════════════════
# 总结报告
# ════════════════════════════════════════════════════════════════
print_section "验证总结"

TOTAL=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT))
echo -e "  通过: ${GREEN}$PASS_COUNT${NC} / $TOTAL"
echo -e "  失败: ${RED}$FAIL_COUNT${NC} / $TOTAL"
echo -e "  警告: ${YELLOW}$WARN_COUNT${NC} / $TOTAL"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    if [ "$WARN_COUNT" -eq 0 ]; then
        echo -e "${GREEN}🎉 Phase B 验证全部通过！可以进入 Phase C/D 训练。${NC}"
    else
        echo -e "${YELLOW}✅ Phase B 验证通过（有警告项，建议处理但不阻塞训练）。${NC}"
    fi
    echo -e "${GREEN}下一步: bash ZeroSearch/train_grpo.sh ... (RL 臂) 或 bash ZeroSearch/train_opd.sh ... (OPD 臂)${NC}"
else
    echo -e "${RED}❌ Phase B 验证未通过，请修复上述 FAIL 项后再继续。${NC}"
    exit 1
fi
