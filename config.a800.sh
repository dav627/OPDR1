#!/bin/bash
# A800 80GB 单卡预设 — 极限显存优化
# 用法: cp config.a800.sh config.local.sh

# ── 模型路径（按需修改）──
SIMULATOR_MODEL=/root/autodl-tmp/models/Simulation_LLM_google_3B
SEARCH_ENGINE=google

# ── 显存分配（模拟器 10GB + vLLM 32GB + FSDP 剩余 38GB）──
SIMULATOR_MEM_FRACTION=0.12
ROLLOUT_GPU_MEM_UTIL=0.4

# ── Batch size（小 batch 避免 OOM）──
TRAIN_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE=16
LOG_PROB_MICRO_BATCH_SIZE=16
REF_LOG_PROB_MICRO_BATCH_SIZE=16

# ── 显存节省策略 ──
ENFORCE_EAGER=True          # 禁用 CUDA 图，省显存
FREE_CACHE_ENGINE=True      # 训练时释放 KV cache，省 ~26GB
