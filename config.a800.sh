#!/bin/bash
# A800 80GB 单卡预设 — 极限显存优化
# 用法: cp config.a800.sh config.local.sh

# ── 训练模式：grpo 或 opd ──
TRAIN_MODE=opd          # 改为 grpo 跑 RL 基线

# ── 模型路径（按需修改）──
SIMULATOR_MODEL=/root/autodl-tmp/models/Simulation_LLM_google_3B
TEACHER_MODEL=/root/autodl-tmp/models/Search-R1-Qwen2.5-7B-GRPO
SEARCH_ENGINE=google

# ── 显存分配（模拟器 10GB + vLLM 32GB + FSDP 剩余 38GB）──
SIMULATOR_MEM_FRACTION=0.12
ROLLOUT_GPU_MEM_UTIL=0.4

# ── Batch size（小 batch 避免 OOM）──
TRAIN_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE=16
LOG_PROB_MICRO_BATCH_SIZE=16
REF_LOG_PROB_MICRO_BATCH_SIZE=8     # OPD 7B ref 需要更小的 micro batch

# ── 显存节省策略 ──
ENFORCE_EAGER=True          # 禁用 CUDA 图，省显存
FREE_CACHE_ENGINE=True      # 训练时释放 KV cache，省 ~26GB
