#!/bin/bash
# A800 80GB 单卡预设
# 用法: cp config.a800.sh config.local.sh

# ── 模型路径（按需修改）──
SIMULATOR_MODEL=/root/autodl-tmp/models/Simulation_LLM_google_3B
TEACHER_MODEL=/root/autodl-tmp/models/ZeroSearch_google_V2_Qwen2.5_7B_Instruct
SEARCH_ENGINE=google

# ── 显存分配 ──
SIMULATOR_MEM_FRACTION=0.12
ROLLOUT_GPU_MEM_UTIL=0.4

# ── Batch size ──
TRAIN_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE=16
LOG_PROB_MICRO_BATCH_SIZE=16
REF_LOG_PROB_MICRO_BATCH_SIZE=8

# ── 显存节省策略 ──
ENFORCE_EAGER=True
FREE_CACHE_ENGINE=True
