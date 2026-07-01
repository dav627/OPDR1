# scripts/ — 服务器验证脚本

本目录存放服务器端使用的自动化验证脚本。

## 文件说明

| 脚本 | 用途 | 对应 Phase |
|------|------|-----------|
| `verify_phase_b.sh` | 服务器环境验证（GPU/torch/vllm/sglang/代码改动） | Phase B |

## 使用方法

### Phase B 环境验证

在服务器上 clone 项目后，进入项目目录运行：

```bash
cd /root/autodl-tmp/code/OPDR1

# 基本用法（默认模型路径 /root/autodl-tmp/models/Qwen2.5-3B-Instruct）
bash scripts/verify_phase_b.sh

# 指定模型路径
bash scripts/verify_phase_b.sh /path/to/Qwen2.5-3B-Instruct
```

### 验证内容

脚本会按顺序检查以下 5 项（对应 Phase B 的 B1-B5）：

| 项 | 检查内容 | 通过标准 |
|----|---------|---------|
| B1 | GPU 可用性 | `nvidia-smi` 能看到 A800/A100/H100 |
| B2 | Python 依赖栈 | torch 2.4.0+cu121 / flash-attn / verl / transformers<4.48 |
| B3 | vLLM 加载测试 | 能加载 Qwen2.5-3B-Instruct 并 generate 一句话 |
| B4 | sglang 环境 | sglang 在独立 conda 环境中可用 |
| B5 | 代码改动确认 | OPD 改动的 4 个文件 + 关键字段都在 |

### 输出说明

- `[PASS]` 绿色 — 该项通过
- `[FAIL]` 红色 — 该项失败，需修复后才能训练
- `[WARN]` 黄色 — 该项有警告，建议处理但不阻塞

脚本最后会输出总结报告：
- 全部 PASS → 可进入 Phase C/D 训练
- 有 FAIL → 需修复后重新运行

### 常见问题修复

**B1 GPU 不可见**：
- AutoDL 控制台切换到「有卡模式」

**B2 torch.cuda.is_available() = False**：
- 检查是否在无卡模式
- 检查 torch 是否为 CPU 版：`pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121`

**B2 flash-attn 安装失败**：
- 使用预编译 wheel：`pip install flash-attn --no-build-isolation`
- 或从 https://github.com/Dao-AILab/flash-attention/releases 下载匹配版本

**B2 verl 不可用**：
- `cd ZeroSearch && pip install -e .`

**B3 vLLM 加载失败**：
- 检查 GPU 显存是否足够（3B 模型至少需要 10GB）
- 检查模型路径是否正确
- 检查 CUDA 版本：`nvcc --version` 应为 12.1

**B4 sglang 未安装**：
- 创建独立环境：`conda create -n sglang python=3.10 -y`
- `conda activate sglang && pip install 'sglang[all]'`
