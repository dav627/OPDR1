# Training Commands Reference

Server path conventions (from implementation doc):
- Models: `/root/autodl-tmp/models/`
- Data: `/root/autodl-tmp/data/`
- Code: `/root/autodl-tmp/code/`

---

## Arm RL (GRPO baseline)

```bash
cd /root/autodl-tmp/code/ZeroSearch

bash train_grpo.sh \
  -n 1 \
  -g 8 \
  -m /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  -d /root/autodl-tmp/data/ZeroSearch_dataset \
  -t 500 \
  -i localhost:8000 \
  -s simulate_prompt \
  -l /root/autodl-tmp/models/SearchSimulation_3B \
  -a 0.0 \
  -e 0.5 \
  -g wiki \
  -r 5 \
  -k 5
```

**Parameter mapping** (train_grpo.sh positional args):
- `$2` NUM_GPUS_PER_NODE = 8
- `$4` MODEL_PATH = Qwen2.5-3B-Instruct
- `$6` DATA_PATH = ZeroSearch_dataset
- `$8` TOTAL_STEPS = 500
- `$10` IP = localhost:8000 (simulator endpoint)
- `$12` SEARCH_MODE = simulate_prompt
- `$14` SIMULATION_LLM = SearchSimulation_3B
- `$16` START_THRESHOLD = 0.0
- `$18` END_THRESHOLD = 0.5
- `$20` SEARCH_ENGINE = wiki
- `$22` MAX_TURNS = 5
- `$24` TOPK = 5

---

## Arm OPD (on-policy distillation)

```bash
cd /root/autodl-tmp/code/ZeroSearch

bash train_opd.sh \
  -n 1 \
  -g 8 \
  -m /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  -T /root/autodl-tmp/models/Search-R1-Qwen2.5-7B-GRPO \
  -d /root/autodl-tmp/data/ZeroSearch_dataset \
  -t 500 \
  -i localhost:8000 \
  -s simulate_prompt \
  -l /root/autodl-tmp/models/SearchSimulation_3B \
  -a 0.0 \
  -e 0.5 \
  -g wiki \
  -r 5 \
  -k 5
```

**Parameter mapping** (train_opd.sh positional args):
- `$2` NUM_GPUS_PER_NODE = 8
- `$4` MODEL_PATH = Qwen2.5-3B-Instruct (student)
- `$6` TEACHER_PATH = Search-R1-Qwen2.5-7B-GRPO (7B teacher)
- `$8` DATA_PATH = ZeroSearch_dataset
- `$10` TOTAL_STEPS = 500
- `$12` IP = localhost:8000
- `$14` SEARCH_MODE = simulate_prompt
- `$16` SIMULATION_LLM = SearchSimulation_3B
- `$18` START_THRESHOLD = 0.0
- `$20` END_THRESHOLD = 0.5
- `$22` SEARCH_ENGINE = wiki
- `$24` MAX_TURNS = 5
- `$26` TOPK = 5

---

## Key differences between arms

| Parameter | RL (GRPO) | OPD |
|-----------|-----------|-----|
| `loss_mode` | `grpo` (default) | `opd` |
| `ref.model_path` | null → uses actor model | teacher model path |
| `kl_loss_coef` | 0.001 (small regularizer) | 1.0 (primary objective) |
| Teacher model | N/A | Search-R1-7B |

---

## Pre-flight checklist (before running either arm)

1. **Simulator running**: sglang environment serving SearchSimulation_3B on localhost:8000
2. **GPU memory**: A800 80G sufficient for 3B student + 7B teacher + simulator?
   - If OOM: reduce `train_batch_size` from 64 to 32
3. **WandB login**: `wandb login <key>` for experiment tracking
4. **Checkpoint dir**: `/root/autodl-tmp/verl_checkpoints/` has enough space (~50GB per run)

---

## Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Tail training logs
tail -f verl_checkpoints/<EXPERIMENT_NAME>/training.log

# Check wandb
open https://wandb.ai/<username>/ZeroSearch
```

---

## Expected metrics

**RL arm** (GRPO):
- `actor/reward` should increase over training
- `actor/pg_loss` should decrease then stabilize
- `actor/kl_loss` should stay small (coef=0.001)

**OPD arm**:
- `actor/opd_loss` (= `actor/kl_loss`) should decrease steadily
- `actor/pg_loss` should be 0 (disabled in OPD mode)
- `actor/ppo_kl` should be 0 (not used in OPD mode)
