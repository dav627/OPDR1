# OPD vs RL 对比实验 —— 实现文档 / TODO（供本地 Claude Code 使用）

## 0. 怎么用这份文档

- 这是一份**自包含的实现简报 + 任务清单**，目标是在本地把代码改对、再上 GPU 服务器验证，**省 GPU 租用成本**。
- **最重要的约束（Claude Code 必读）**：你现在工作在**本地"只改代码"环境**。
  - ✅ 可以做：读代码、跨文件定位、改代码、lint、对**纯 Python 逻辑**写小单测。
  - ❌ 不能做：跑训练、`import vllm` / `import flash_attn`（本地无 GPU，会直接失败）、验证 CUDA 内核。
  - 因此：**所有"能不能跑通"的验证都留给 GPU 服务器**。本地的成功标准是"逻辑改对、能静态分析通过"，而不是"能运行"。
- 用 `- [ ]` 勾选跟踪进度。标注 `[本地]` 的在本地做，`[服务器]` 的在服务器（有卡模式）做。

---

## 1. 实验整体目标

**一句话**：从零搭一个 agentic search 小模型，分别用 **RL（GRPO）** 和 **在策略蒸馏（OPD）** 两条后训练路线训练它，**系统对比成本 / 效果 / 能力上限**。

**要回答的业务问题**：企业要低成本拿到一个可部署的小模型，是该用 RL 训，还是从更强的老师蒸？

**最终交付物（项目的灵魂）**：一张 "RL vs OPD" 的成本/效果对比表 + 天花板分析。这是面试最值钱的部分——证明懂两条后训练路线的选型判断。

**arm 设计**（同基座、同 benchmark、同后端）：
- **Arm RL**：GRPO + outcome 奖励 —— **必做**
- **Arm OPD**：从 SearchR1-7B 老师做在策略蒸馏 —— **必做核心**
- *(选)* **Arm Hybrid**：先 OPD 冷启动、再 RL 微调
- *(选)* RL 臂叠加 StepSearch 过程奖励

**关键技术决策**：
- 后端用 **ZeroSearch 模拟搜索**：$0 真实 API、无需托管 Wikipedia 语料、不装 faiss。
- 学生 = **Qwen2.5-3B-Instruct**；OPD 老师 = **SearchR1-7B**。
- 老师与学生**同为 Qwen2.5 系、tokenizer 一致** → 反向 KL 可逐 token 直接计算（这是选 SearchR1-7B 当老师的关键原因）。

---

## 2. 仓库与关键资源

**仓库（本地和服务器都 clone 同样的）：**
- `https://github.com/Alibaba-NLP/ZeroSearch`（主力：RL 训练 + 模拟搜索后端，内置 veRL）
- `https://github.com/PeterGriffinJin/Search-R1`（参考框架 + OPD 老师来源）
- *(选)* `https://github.com/Zillwang/StepSearch`（过程奖励代码 + MuSiQue 数据集）

**模型 / 数据（服务器上已下到 `/root/autodl-tmp`，本地不需要下权重）：**
- 学生：`Qwen/Qwen2.5-3B-Instruct`
- 模拟器：`sunhaonlp/SearchSimulation_3B`（微调版，省显存；准确仓库名以 ZeroSearch README 为准）
- OPD 老师：Search-R1 的某个 **7B GRPO 变体**（到其 HF 模型列表挑选）
- 训练数据：`sunhaonlp/ZeroSearch_dataset`

**服务器路径约定：** 代码在 `/root/autodl-tmp/code/`，模型在 `/root/autodl-tmp/models/`，数据在 `/root/autodl-tmp/data/`。

---

## 3. 环境配置

### 3.1 服务器环境（A800 80G，已装好，供对照）

- 镜像 **CUDA 12.1**（关键，版本不对全栈会炸）。
- 训练环境 `rl-opd`（conda，Python 3.9）：
  - `torch==2.4.0`（cu121）、`vllm==0.6.3`、`wandb`、`huggingface_hub`
  - veRL：在 `ZeroSearch/` 目录下 `pip install -e .`
  - flash-attn：**预编译 wheel**，匹配 `cu12 + torch2.4 + cp39 + abiFALSE`
- 服务模拟器的环境 `sglang`（**单独 conda 环境**，py3.10，`pip install "sglang[all]"`，避免与训练环境冲突）。
- **不装** faiss、**不需要**真实搜索 API key（走模拟搜索）。

### 3.2 本地"只改代码"环境（Claude Code 在这里工作）

```bash
conda create -n rl-opd-local python=3.9 -y
conda activate rl-opd-local

# CPU 版 torch，钉 2.4.0，保证 API 与服务器一致
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu   # macOS 改用: pip install torch==2.4.0
# 仅为代码智能/静态分析的轻量依赖：
pip install transformers datasets numpy pandas pyyaml omegaconf hydra-core tensordict

# 让 verl.* 能被解析（不拉重依赖）：
cd ~/code/ZeroSearch && pip install -e . --no-deps
```

- **本地不要装 vllm / flash-attn**（无 CUDA 装不上，也不需要）。编辑器把这两个 import 标"未解析"是正常的。
- 若 `pip install -e . --no-deps` 报错，则不装，改在编辑器里把仓库目录加进解析路径（如 VSCode `python.analysis.extraPaths: ["./ZeroSearch", "./Search-R1"]`）。

---

## 4. 实验修改思路（核心 —— OPD 实现）

### 4.1 背景：veRL GRPO 的损失结构

ZeroSearch 基于 veRL。其 actor 训练损失（概念上）≈：
```
loss = policy_gradient_loss(advantage from group-normalized reward)   # GRPO 主项
     + kl_coef * KL(π_actor || π_ref)                                  # 对参考策略的 KL 正则（小权重）
聚合时用 response_mask 只在「响应 token」上算损失；
且 ZeroSearch/SearchR1 额外把「检索/工具返回的 token」mask 掉（不计入损失）。
```
- 参考策略 `π_ref` 通常是**初始 actor 的冻结副本**，由一个 ref worker 做前向、返回逐 token logprob。

### 4.2 OPD 怎么改（"基本改一行"的精神）

OPD = 让学生在自己生成的 on-policy 轨迹上，**逐 token 反向 KL 对齐老师**。改动是把上面的结构重定向：

1. **把参考模型换成老师**：`π_ref` 从「初始 actor 冻结副本」改为 **SearchR1-7B**（teacher）。
2. **把损失目标改成对老师的反向 KL**：让 `KL(π_student || π_teacher)` 成为**主目标**，而不是小权重正则；**去掉 reward / advantage 那一项**（或将其权重置 0）。
3. **保留 masking**：`response_mask` + 检索/工具 token 的 mask **不变**——**只蒸学生自己生成的 reasoning/query token**。
4. **rollout 机制不变**：学生照常通过 ZeroSearch 模拟搜索生成多轮 agentic 轨迹，OPD 复用这些 on-policy rollout。

> 这正是 Thinking Machines 说的"在带 KL 正则的 RL 上基本改一行"——把 KL 的参考指向老师、把这个 KL 当目标。

### 4.3 需要在代码里定位的位置（Claude Code 用 grep 找）

**第一件事：先探索代码库，定位下面这些，再动手改。** 建议的搜索关键词：
- actor 损失 / 策略损失：`pg_loss`、`policy_loss`、`compute_policy_loss`、`ppo`、`actor`
- KL 计算 / KL 系数：`kl_penalty`、`kl_coef`、`compute_kl`、`kl_ctrl`、`kld`
- 参考策略 worker / 加载：`ref`、`reference`、`ref_policy`、`RefWorker`、`ref_log_prob`、`compute_ref_log_prob`
- masking（尤其检索 token）：`response_mask`、`loss_mask`、`eos_mask`、`info_mask`（ZeroSearch/SearchR1 很可能用类似 `info_mask` 把检索内容 mask 掉）
- 参考模型路径配置：训练 yaml/config 里 `ref` / `model.path` 相关字段
- 训练入口：`train_grpo.sh` → 它调用的 python main → veRL 的 PPO trainer

把这些位置和它们的调用关系**先在文档/注释里记下来**，再实施 4.2 的改动。

### 4.4 关键注意点

- **老师 7B、学生 3B，但同 vocab**：反向 KL 逐 token 对齐成立。务必确认两者 tokenizer 一致（都是 Qwen2.5）。
- **KL 估计器是个旋钮**：最省事的起点是**复用 veRL 已有的逐 token logprob**（`logπ_student(a_t) − logπ_teacher(a_t)` 这类采样 token 估计）当损失；想降方差，再扩展 ref worker 返回 teacher 的 top-k logits、算更完整的反向 KL。**先用便宜版跑通，再考虑升级。**
- **masking 必须只作用在「学生生成的 token」**：检索/工具返回的 token 不参与蒸馏，否则学的是模拟器吐的文本。
- **显存（服务器侧，本地不涉及）**：OPD 阶段同时在跑 学生训练 + 学生 rollout(vLLM) + 模拟器 + 7B 老师前向；A800 80G 偏紧时缩 batch / 换更小老师。

### 4.5 兜底方案（写进风险预案）

把 OPD 用在**多轮 agentic 轨迹**上是本项目最硬的部分。**若 agentic 上的 OPD 一时打不通**：先在一个简单非 agentic 任务（如数学推理，也是 OPD 的经典演示设定）上把 OPD 跑通、拿到 RL-vs-OPD 对比数据，保证项目有可交付结论，再回头攻 agentic。

---

## 5. TODO（按阶段，标注 本地 / 服务器）

### Phase A —— 本地：理解代码 + 改 OPD（不花 GPU 钱，主战场）
- [ ] `[本地]` clone 三个仓库；按 §3.2 配好本地环境与编辑器解析路径。
- [ ] `[本地]` 探索代码库，按 §4.3 定位 actor 损失 / KL 计算 / 参考策略加载 / masking / 配置 / 训练入口，记录调用关系。
- [ ] `[本地]` 实现 OPD 改动（§4.2）：参考模型可配置为老师路径；新增"OPD 损失模式"（反向 KL 对老师为主目标、去掉 reward 项）；保留 masking。建议用开关（如 `loss_mode: opd|grpo`）切换，**不破坏原 GRPO 路径**（RL 臂还要用）。
- [ ] `[本地]` 把可纯逻辑测试的部分（如 KL/损失计算函数、mask 应用）抽出来写最小单测，本地 `pytest` 跑过。
- [ ] `[本地]` lint / 类型检查通过（容忍 vllm/flash_attn 的"未解析"告警）。
- [ ] `[本地]` 写好两条训练命令的草稿：RL 臂（原版 GRPO）和 OPD 臂（新模式），参数见 §5 末尾备注。

### Phase B —— 服务器：环境验证（有卡模式，先开机挂 A800）
- [ ] `[服务器]` `nvidia-smi` 能看到 A800。
- [ ] `[服务器]` 基础栈自检：`torch.__version__`=2.4.0+cu121、`torch.cuda.is_available()`=True、flash-attn 内核能跑、`import verl` 成功。
- [ ] `[服务器]` 决定性测试：用 vllm 加载 `Qwen2.5-3B-Instruct` 并成功 `generate` 一句话。
- [ ] `[服务器]` `sglang` 环境里 `import sglang` 成功。
- [ ] `[服务器]` 同步本地改动上来（§7），确认改动文件无误。

### Phase C —— 服务器：跑 Arm RL（baseline，必做）
- [ ] `[服务器]` 先 serve 模拟器（sglang 环境）/ 或按脚本由训练流程拉起模拟器。
- [ ] `[服务器]` 用 `train_grpo.sh` 走 `SEARCH_MODE simulate*`、`MODEL_PATH` 指向 3B、配 `START/END_THRESHOLD`，小步数先验证 pipeline 不崩。
- [ ] `[服务器]` 正式训练 RL 臂；**记录 GPU·小时、reward 曲线、最终 checkpoint**。

### Phase D —— 服务器：跑 Arm OPD（必做核心）
- [ ] `[服务器]` serve / 加载 7B 老师以供 ref-logprob 计算。
- [ ] `[服务器]` 用 OPD 模式训练同一个 3B（同数据、同后端、同步数）；小步数先验证反向 KL 损失正常下降、masking 生效。
- [ ] `[服务器]` 正式训练；记录 GPU·小时、loss 曲线、最终 checkpoint。
- [ ] `[服务器]` 若 agentic OPD 卡住，启用 §4.5 兜底。

### Phase E —— 评测与对比（核心产出）
- [ ] `[服务器]` 在 benchmark（HotpotQA / 2Wiki / MuSiQue / Bamboogle + 单跳 NQ/TriviaQA/PopQA）上评所有 arm + 老师本身。
- [ ] `[本地/服务器]` 出对比表：路线 × {训练成本(GPU·h/¥), 各 benchmark 分数, 是否超过老师}。
- [ ] `[本地/服务器]` 画样本效率曲线（性能 vs 步数）。
- [ ] `[本地]` 写天花板分析结论 + README + 简历 bullet（填入真实数字）。

### (选) Phase F —— Hybrid / 过程奖励
- [ ] `[服务器]` Arm Hybrid：先 OPD 后 RL。
- [ ] `[服务器]` RL 臂叠加 StepSearch 过程奖励（用其 MuSiQue 数据集）。

> **训练命令备注**（来自 ZeroSearch README，按需调整）：
> `bash train_grpo.sh NUM_GPUS_PER_NODE 1 MODEL_PATH <3B路径> DATA_PATH <数据路径> SEARCH_MODE simulate_prompt SIMULATION_LLM <模拟器> START_THRESHOLD 0 END_THRESHOLD 0.5 MAX_TURNS 5 TOPK 5 ...`
> OPD 臂用你新增的开关切到 OPD 损失模式、并把参考模型指向老师。

---

## 6. 验收 / 验证清单

**本地（Phase A 完成的标准）：**
- 代码改动逻辑自洽、单测通过、lint 通过；原 GRPO 路径未被破坏（有开关可切回）。

**服务器三步自检（Phase B）：**
1. `nvidia-smi` 见 A800。
2. torch/cuda/flash-attn/verl 自检全 PASS。
3. vllm 能加载 3B 并生成 → 说明 torch+CUDA+vllm 整栈通。

**训练正确性（Phase C/D 小步验证）：**
- RL：reward 曲线总体上行、不崩。
- OPD：反向 KL 损失下降；确认检索/工具 token 未参与损失（masking 生效）。

---

## 7. 同步流程（本地 → 服务器）

**git（推荐，可追溯）：**
```bash
cd ~/code/ZeroSearch
git checkout -b opd-mod
# ...改代码...
git add -A && git commit -m "OPD: swap KL ref to teacher, add opd loss mode"
# 推到自己的 fork，服务器 git pull
```

**rsync（快速直传，Mac/Linux/WSL）：**
```bash
rsync -avz -e "ssh -p <端口>" ~/code/ZeroSearch/ \
  root@<AutoDL主机>:/root/autodl-tmp/code/ZeroSearch/ \
  --exclude '.git' --exclude '__pycache__'
```

**换行符坑**：Windows 改过的 `.sh` 传到 Linux 可能因 CRLF 报 `bad interpreter`。先 `git config --global core.autocrlf input`，或确保 `train_*.sh` 存为 LF。

---

## 8. 已知坑清单（汇总）

- **CUDA 版本不匹配是头号杀手**：服务器镜像必须 12.1；torch=2.4.0+cu121、vllm=0.6.3、flash-attn ABI=False，三处任一错位 → 运行时 `undefined symbol`。
- **本地别 import 训练模块**：顶部带 `import vllm/flash_attn` 的模块在本地必失败，属预期；要测就抽纯逻辑单测。
- **OPD 的 tokenizer 一致性**：老师必须与学生同 tokenizer 系（Qwen2.5），否则要做跨 tokenizer 蒸馏（复杂得多，避免）。
- **masking 范围**：OPD 只蒸学生生成的 token，检索/工具返回内容必须 mask。
- **不破坏原 GRPO 路径**：用开关切换 OPD/GRPO，RL 臂还要用。
- **磁盘**：服务器模型/env 放 `/root/autodl-tmp`，别塞满系统盘。
- **对比要公平**：所有 arm 同基座、同数据、同后端、同评测；成本用统一口径（GPU·h 或 ¥）。
- **OPD 上限错觉**：OPD 不会超过老师——这是要呈现的结论，不是 bug。
- **GPU 测试要在有卡模式**：无卡模式 `torch.cuda` 必为 False，不代表环境错。

---

## 9. 给 Claude Code 的硬约束（DO / DON'T）

- **DO**：先探索代码、定位 §4.3 的位置再改；用开关新增 OPD 模式、保留原 GRPO；改动小而可读、加注释；对纯逻辑写单测。
- **DON'T**：不要在本地尝试运行训练、`import vllm`、`import flash_attn` 或任何需要 GPU 的代码；不要在本地下模型权重；不要重写整个训练流程（最小改动达成 OPD 即可）。
- **不确定就停下问**：涉及 KL 估计器选择、参考模型加载方式、masking 字段的确切语义时，若代码里看不明确，先说明你的发现与候选方案，再让我确认，别擅自大改。