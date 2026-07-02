#!/bin/bash
# RTX PRO 6000 96GB 单卡预设 — 充裕显存，速度优先
# 用法: cp config.rtx6000.sh config.local.sh

# ── 模型路径（按需修改）──
SIMULATOR_MODEL=/root/autodl-tmp/models/Simulation_LLM_google_3B
SEARCH_ENGINE=google

# ── 显存分配（模拟器 19GB + vLLM 48GB + FSDP 剩余 29GB）──
SIMULATOR_MEM_FRACTION=0.2
ROLLOUT_GPU_MEM_UTIL=0.5

# ── Batch size（96GB 可以用更大 batch）──
TRAIN_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE=64
LOG_PROB_MICRO_BATCH_SIZE=64
REF_LOG_PROB_MICRO_BATCH_SIZE=64

# ── 速度优先策略 ──
ENFORCE_EAGER=False         # 启用 CUDA 图，推理加速 20-30%
FREE_CACHE_ENGINE=False     # 保留 KV cache，省去每步重建开销
