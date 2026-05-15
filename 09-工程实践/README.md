# 工程实践：真实项目怎么做

前面几章我们学了 SFT 的原理、方法和技巧。但在真实项目中，你面对的第一个问题往往不是"怎么训练"，而是"要不要训练"。

这一章我们聊聊工程实践中的关键决策和常见问题。

---

## 一、SFT vs Prompt Engineering vs RAG

### 三种方法的定位

这三种方法经常被拿来比较，但它们其实解决的是不同层面的问题：

```
Prompt Engineering（提示词工程）: 不改模型，改提示词 —— 给员工写更好的工作说明
RAG（检索增强生成）:           不改模型，给模型查资料 —— 给员工配一个资料库
SFT（有监督微调）:            改模型本身           —— 把员工送去培训班
```

### 什么时候用哪个？

做项目时，**按成本从低到高依次尝试**：

```
第一步：试 Prompt Engineering（零成本，分钟级见效）
  │
  ├── 效果够了？→ 用它！不需要继续了
  │
  └── 效果不够？→ 分析原因
        │
        ├── 模型缺少某些知识？→ 试 RAG（给它查资料）
        │     │
        │     ├── 效果够了？→ 用它！
        │     └── 效果不够？→ 继续往下
        │
        └── 模型能力本身不够？→ SFT（教它新技能）
              │
              ├── 效果够了？→ 用它！
              └── 效果不够？→ 考虑换更大的模型 / 更多数据
```

### 详细对比

| 对比维度 | Prompt Engineering | RAG | SFT |
|---------|-------------------|-----|-----|
| 成本 | 几乎为零 | 低（搭建检索系统） | 高（GPU + 数据 + 时间） |
| 见效时间 | 分钟 | 小时到天 | 天到周 |
| 效果上限 | 有限（受模型能力限制） | 中等（受检索质量限制） | 高（改变模型能力） |
| 适合场景 | 简单任务、格式控制 | 知识密集型任务 | 需要特定能力的任务 |
| 风险 | 极低 | 低 | 中等（可能过拟合、忘记原有能力） |
| 是否需要训练 | 否 | 否 | 是 |
| 知识更新 | 改提示词即可 | 更新知识库即可 | 需要重新训练 |
| 技术门槛 | 低 | 中等 | 高 |

### 典型场景分析

**场景一：客服机器人**

```
第一步：Prompt Engineering
  → 写好系统提示词，定义角色、语气、回答范围
  → 效果：能回答简单问题，但对产品细节不熟

第二步：+ RAG
  → 接入产品文档知识库
  → 效果：大部分问题都能回答了

第三步（可选）：+ SFT
  → 如果需要特定的回答风格或处理复杂业务逻辑
  → 效果：回答更专业、更符合公司调性
```

**场景二：代码助手**

```
推荐方案：SFT
  → 模型需要学会特定的编码规范和框架用法
  → Prompt Engineering 和 RAG 难以教会"写代码的感觉"
```

**场景三：知识问答系统**

```
推荐方案：RAG
  → 知识会经常更新，重新训练成本太高
  → 检索最新文档 + 大模型理解 = 最佳组合
```

**场景四：风格化写作（如古风诗词生成）**

```
推荐方案：SFT
  → 需要模型学会特定的语言风格
  → Prompt Engineering 只能做到"像"，SFT 才能做到"是"
```

**场景五：简单分类任务**

```
推荐方案：Prompt Engineering
  → "请将以下文本分类为：正面/负面/中性"
  → 够用了，不需要更复杂的方案
```

### 组合使用：最强方案

实际项目中，最好的做法往往是**三者组合**：

```
最强组合: SFT（基础能力）+ RAG（实时知识）+ PE（具体调控）

示例：企业级AI助手
├── SFT: 微调模型，学会企业的专业领域和回答风格
├── RAG: 接入企业知识库，获取最新的产品信息、政策文档
└── PE:  针对不同场景写不同的系统提示词
    ├── 客服场景: "你是XX公司客服，语气友好专业..."
    ├── 技术支持: "你是技术工程师，回答要精确..."
    └── 销售咨询: "你是销售顾问，突出产品优势..."
```

---

## 二、训练框架生态

### 手写 Trainer（本项目方式）

我们这个教程项目就是手写的训练代码。

```python
# 我们的 train.py 就是手写 Trainer 的例子
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

| 优点 | 缺点 |
|------|------|
| 理解每一行代码 | 样板代码多 |
| 完全可控 | 缺少高级功能（混合精度、分布式等需要自己加） |
| 适合学习 | 不适合生产 |

### HuggingFace TRL

TRL（Transformer Reinforcement Learning）是 HuggingFace 官方的训练库，专门用于 SFT、DPO、RLHF 等训练。

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

training_args = SFTConfig(
    output_dir="sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    bf16=True,
    gradient_accumulation_steps=4,
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

| 优点 | 缺点 |
|------|------|
| 官方维护，更新快 | 抽象程度适中，有时需要读源码 |
| 支持 SFT/DPO/RLHF/ORPO | 配置项多，初学者可能困惑 |
| 与 PEFT（LoRA）无缝集成 | - |
| 社区活跃，文档丰富 | - |

### LLaMA-Factory

LLaMA-Factory 是目前中文社区最受欢迎的微调框架，提供了一站式的训练解决方案。

```bash
# 安装
pip install llamafactory

# 用 Web UI 启动（零代码微调！）
llamafactory-cli webui
```

特点：
- **Web UI 配置**：不用写代码，点点鼠标就能微调
- **支持 100+ 模型**：Qwen、LLaMA、ChatGLM、Baichuan 等开箱即用
- **全功能**：SFT、DPO、RLHF、预训练都支持
- **中文社区活跃**：遇到问题容易找到解答

| 优点 | 缺点 |
|------|------|
| 零代码上手 | 过于封装，不利于深度理解 |
| 支持模型最全 | 自定义灵活度较低 |
| 中文文档完善 | 升级时可能有兼容性问题 |
| 适合快速出结果 | - |

### Axolotl

Axolotl 是基于 YAML 配置的微调框架，在英文社区很受欢迎。

```yaml
# axolotl 配置文件示例
base_model: Qwen/Qwen2.5-7B
model_type: AutoModelForCausalLM

load_in_8bit: false
load_in_4bit: true

datasets:
  - path: my_dataset.json
    type: alpaca

adapter: lora
lora_r: 16
lora_alpha: 32

learning_rate: 2e-4
num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 4
```

| 优点 | 缺点 |
|------|------|
| YAML 配置，灵活直观 | 中文社区较小 |
| 适合做对比实验 | 文档以英文为主 |
| 功能全面 | 更新较快，偶尔不稳定 |

### 框架选择建议

| 你的需求 | 推荐框架 | 原因 |
|---------|---------|------|
| 学习原理，理解每一步 | 手写（像本项目） | 知其然也知其所以然 |
| 快速微调，尽快出结果 | LLaMA-Factory | 零代码，开箱即用 |
| 研究实验，对比方法 | Axolotl 或 TRL | 灵活配置，方便调参 |
| 生产部署，长期维护 | TRL + 自定义流程 | 官方支持，稳定可靠 |

---

## 三、分布式训练

### 为什么需要分布式

当你的模型或数据规模增长时，单张 GPU 会遇到瓶颈：

```
问题一：显存不够
  7B 模型全量训练 ≈ 需要 60GB+ 显存
  单张 A100 = 80GB → 勉强够
  单张 4090 = 24GB → 放不下

问题二：训练太慢
  7B 模型 + 100K 数据 + 单卡 ≈ 几天
  70B 模型 + 100K 数据 + 单卡 ≈ 根本跑不起来

解决方案：用多张 GPU！
```

### 数据并行（Data Parallelism / DDP）

最简单的分布式方法：每张 GPU 存一份完整的模型，但处理不同的数据。

```
GPU 0: 完整模型副本 + batch 1 → 计算梯度 1 ─┐
GPU 1: 完整模型副本 + batch 2 → 计算梯度 2 ─┼→ 平均梯度 → 同步更新所有副本
GPU 2: 完整模型副本 + batch 3 → 计算梯度 3 ─┘
```

打个比方：三个完全相同的老师，各自批改不同的作业，最后把评分标准统一一下。

```python
# PyTorch DDP（分布式数据并行）
# 启动命令: torchrun --nproc_per_node=4 train.py

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model.to(local_rank), device_ids=[local_rank])

# 训练代码和单卡几乎一样！
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()  # DDP 自动同步梯度
    optimizer.step()
```

**优点**：简单，代码改动小
**缺点**：每张 GPU 都要存完整模型，显存没省

### 模型并行（Model / Tensor Parallelism）

把模型切成几块，分到不同 GPU 上：

```
一个 24 层的模型:

GPU 0: 层 0-7    ───→ 中间结果传给 GPU 1
GPU 1: 层 8-15   ───→ 中间结果传给 GPU 2
GPU 2: 层 16-23  ───→ 最终输出
```

打个比方：一个工厂的流水线，每个工人负责不同的工序。

**优点**：可以放下大模型
**缺点**：GPU 之间需要频繁通信，效率较低（一个 GPU 在算的时候，其他 GPU 在等）

### DeepSpeed ZeRO

DeepSpeed 是微软开发的分布式训练框架，其核心技术是 ZeRO（零冗余优化器），分为三个阶段：

```
训练时 GPU 上存的东西 = 模型参数 + 梯度 + 优化器状态

以 7B 模型为例（FP16 训练）:
  模型参数:    14 GB
  梯度:        14 GB
  优化器状态:  28 GB（AdamW 需要存 momentum 和 variance）
  总计:        56 GB  ← 单卡放不下！

ZeRO 的思路：把这些东西分摊到多张卡上
```

| 阶段 | 分摊内容 | 显存节省 | 通信开销 |
|------|---------|---------|---------|
| ZeRO-1 | 优化器状态 | 约 4x | 低 |
| ZeRO-2 | + 梯度 | 约 8x | 中等 |
| ZeRO-3 | + 模型参数 | 几乎线性 | 较高 |

```python
# DeepSpeed ZeRO-3 配置示例 (ds_config.json)
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"        # 参数卸载到 CPU，进一步省显存
        },
        "offload_optimizer": {
            "device": "cpu"        # 优化器卸载到 CPU
        },
        "overlap_comm": true       # 通信和计算重叠
    },
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2
}
```

```bash
# 启动 DeepSpeed 训练
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

### FSDP（Fully Sharded Data Parallel）

FSDP 是 PyTorch 官方的解决方案，原理和 ZeRO-3 类似：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 类似 ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    ),
)
```

| 对比 | DeepSpeed ZeRO | FSDP |
|------|---------------|------|
| 来源 | 微软 | PyTorch 官方 |
| 成熟度 | 更成熟 | 在快速追赶 |
| 生态 | 独立配置 | 原生集成 |
| 适合 | 超大模型训练 | 中大模型训练 |

### 分布式选择建议

```
你有几张 GPU?
├── 1 张 → 不需要分布式
│         └── 显存不够? → LoRA / QLoRA / 梯度累积
│
├── 2-8 张，模型放得下单卡
│         └── DDP（数据并行，最简单）
│
├── 2-8 张，模型放不下单卡
│         └── DeepSpeed ZeRO-3 或 FSDP
│
└── 超大模型（100B+）
          └── DeepSpeed ZeRO-3 + 张量并行 + 流水线并行
```

---

## 四、完整项目流程

一个真实的 SFT 项目大概是这样的流程：

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│ 需求分析 │───→│ 数据准备 │───→│ 模型选择 │───→│ 训练配置 │───→│ 训练执行 │
└─────────┘    └─────────┘    └─────────┘    └──────────┘    └──────────┘
                                                                   │
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐         │
│ 持续优化 │←───│  监控   │←───│  部署   │←───│ 模型评估 │←────────┘
└─────────┘    └─────────┘    └─────────┘    └──────────┘
```

### 第一步：需求分析

问自己几个关键问题：

```
1. 我到底需要什么能力？
   → 具体描述，越具体越好
   → "让模型更好" 不是好需求
   → "让模型用200字以内回答客户退货问题" 是好需求

2. 现有模型（+Prompt Engineering+RAG）能做到吗？
   → 先试最便宜的方案！
   → 80% 的情况不需要 SFT

3. 需要多高的质量？
   → 对话场景：90分就够（SFT足够）
   → 数学推理：需要99分（可能要 GRPO）

4. 有什么约束？
   → 预算、时间、GPU资源、数据量
```

### 第二步：数据准备

```
数据收集 → 数据清洗 → 数据格式化 → 质量检查 → 数据划分
```

关键原则：**数据质量 > 数据数量**

```python
# 数据质量检查清单
checklist = {
    "格式是否统一": "所有样本都是相同的对话格式",
    "有无空值或乱码": "检查 prompt 和 response 是否完整",
    "回答质量是否达标": "随机抽样 100 条人工检查",
    "有无重复数据": "去重",
    "长度分布是否合理": "太短的可能没信息量，太长的可能有噪声",
    "有无有害内容": "检查并过滤",
}
```

### 第三步：选择基座模型

```
模型大小选择:
├── 场景简单 + 资源有限  →  0.5B - 1.5B（如 Qwen2.5-0.5B）
├── 一般场景            →  7B（如 Qwen2.5-7B, LLaMA-3-8B）
├── 复杂场景            →  14B - 72B
└── 不差钱              →  72B+

中文场景推荐:
├── Qwen2.5 系列（阿里）   → 中文能力强，社区活跃
├── ChatGLM 系列（清华）    → 中文特化
├── DeepSeek 系列          → 推理能力强
└── LLaMA 3 系列（Meta）   → 英文强，中文还行
```

### 第四步：训练配置

```python
# 关键超参数决策
config = {
    # 微调方法
    "method": "LoRA",           # 首选 LoRA，除非有明确理由用全量微调

    # 学习率
    "learning_rate": 2e-4,      # LoRA 用 1e-4 ~ 3e-4
                                # 全量微调用 1e-5 ~ 5e-5

    # 训练轮次
    "num_epochs": 3,            # 通常 1-5 轮
                                # 数据少（<1K）→ 少轮次，防过拟合
                                # 数据多（>10K）→ 可以多轮

    # 批次大小
    "batch_size": 4,            # 受显存限制
    "gradient_accumulation": 8,  # 等效 batch_size = 4 * 8 = 32

    # 精度
    "bf16": True,               # A100/4090 用 BF16
                                # 老显卡用 FP16

    # LoRA 配置
    "lora_r": 16,               # 8-64，越大越强但越慢
    "lora_alpha": 32,           # 通常 = 2 * r
    "lora_target": "all",       # 所有线性层
}
```

### 第五步到第八步：训练、评估、部署、监控

这些内容在前面的章节中已经详细讲解，这里不再重复。

---

## 五、成本估算

做项目预算时，这张表可以参考：

| 模型大小 | 微调方法 | GPU 需求 | 训练时间 | 大致费用 |
|---------|---------|---------|---------|---------|
| 0.5B | 全量微调 | 1x 消费级显卡（4060等） | 几分钟 | 免费（自己的卡） |
| 7B | LoRA | 1x A100-80G 或 4090-24G | 几小时 | 约几十元（云GPU） |
| 7B | 全量微调 | 4x A100-80G | 几小时 | 约几百元 |
| 14B | LoRA | 1x A100-80G | 数小时 | 约百元 |
| 14B | 全量微调 | 8x A100-80G | 数小时 | 约千元 |
| 70B | QLoRA | 2x A100-80G | 1-2天 | 约千元 |
| 70B | 全量微调 | 8x A100-80G | 数天 | 约万元 |

> 注意：以上费用按云 GPU 租用估算（如 AutoDL、智星云等平台），仅供参考。实际费用取决于数据量、训练轮次和平台定价。

### 省钱建议

```
1. 先用小模型验证方案可行性（0.5B/1.5B），再换大模型
2. 优先使用 LoRA/QLoRA，少用全量微调
3. 用梯度累积代替大 batch size，省显存
4. 关注云 GPU 平台的折扣和竞价实例
5. 训练前做好充分的实验设计，减少无效尝试
```

---

## 六、常见踩坑与解决方案

### 坑1：OOM（显存不足）

```
RuntimeError: CUDA out of memory.
```

**这是最常见的错误**。解决方案按优先级排列：

| 方法 | 效果 | 副作用 |
|------|------|--------|
| 减小 batch_size | 立竿见影 | 可能需要更多梯度累积步 |
| 开启梯度累积 | 等效大batch，不多用显存 | 训练稍慢 |
| 使用 LoRA/QLoRA | 显存大幅下降 | 效果可能略低于全量微调 |
| 开启 BF16/FP16 | 显存减半 | 极少数情况有精度问题 |
| 减小 max_seq_length | 减少显存 | 长文本会被截断 |
| 开启 gradient checkpointing | 用时间换空间 | 训练速度降低约20-30% |

```python
# gradient checkpointing 示例
model.gradient_checkpointing_enable()

# QLoRA 示例：4bit 量化 + LoRA
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config
)
```

### 坑2：Loss 不下降

```
Epoch 1: loss = 2.35
Epoch 2: loss = 2.34
Epoch 3: loss = 2.35    ← 几乎没动！
```

排查步骤：

```
1. 检查数据质量
   → 数据格式对不对？tokenizer 有没有正确处理？
   → 随机看几条训练样本，确认输入输出是否正确

2. 调整学习率
   → 太小：loss 下降极慢或不动
   → 太大：loss 上下跳动或爆炸
   → 建议先用 1e-4 试试，再微调

3. 检查标签（labels）
   → 非常常见的错误：labels 全是 -100（被mask了），模型无法学习
   → 确认只有 prompt 部分是 -100，response 部分是正常 token id

4. 检查数据量
   → 数据太少（<100条）可能不够模型学到模式
```

### 坑3：模型输出乱码

```
输入: "你好"
输出: "嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯" 或 "asdkjh2k3j"
```

原因和解决方案：

```
1. Tokenizer 不匹配
   → 确保训练和推理用的是同一个 tokenizer
   → 确保 tokenizer 的 pad_token 设置正确

2. 数据格式错误
   → 训练数据的对话模板是否和模型要求的一致？
   → 不同模型的 chat template 不一样（ChatML、Alpaca等）

3. 训练过度（过拟合）
   → 减少训练轮次
   → 增加数据多样性
```

### 坑4：灾难性遗忘

```
微调前: 能正常对话、做数学题、写代码
微调后: 学会了新任务，但基础对话能力变差了
```

这是 SFT 的经典问题。解决方案：

```
1. 使用 LoRA
   → 只改少量参数，保留大部分原始能力
   → 这是最有效的防遗忘方法

2. 降低学习率
   → 学习率越大，遗忘越严重
   → 全量微调建议用 1e-5 ~ 2e-5

3. 减少训练轮次
   → 通常 1-3 个 epoch 就够了
   → epoch 越多，遗忘越多

4. 混入通用数据
   → 训练数据中加入一些通用对话数据（占 10-20%）
   → 让模型在学新任务的同时复习旧知识
```

### 坑5：训练速度很慢

排查清单：

```
□ 是否开启了 BF16/FP16？
  → 混合精度训练速度提升 2-3 倍

□ GPU 利用率是不是很低？
  → nvidia-smi 看 GPU 利用率
  → 如果低于 80%，可能是数据加载是瓶颈
  → 增加 dataloader 的 num_workers

□ 是不是用了 gradient checkpointing？
  → 这会降速约 20-30%，但省显存
  → 如果显存够用，可以关掉

□ 数据预处理是不是在线做的？
  → 改成离线预处理，避免训练时重复计算

□ 序列长度是否合理？
  → 过长的序列会大幅降速
  → 考虑截断或过滤超长样本
```

### 坑6：评估指标好但实际效果差

```
验证集 loss 很低，但实际使用时回答质量不行
```

原因：

```
1. 验证集和真实场景分布不一致
   → 验证集应该尽量接近真实使用场景
   → 加入一些边界情况和困难样本

2. 只看了 loss，没做人工评估
   → loss 低不代表回答好
   → 务必做人工抽样评估

3. 过拟合到训练集
   → 模型背住了训练数据，遇到新问题就不行
   → 增加数据多样性，减少训练轮次
```

---

## 七、总结：实战检查清单

开始一个 SFT 项目前，过一遍这个清单：

```
□ 1. 确认是否真的需要 SFT（先试 PE 和 RAG）
□ 2. 准备高质量数据（质量 > 数量）
□ 3. 选择合适的基座模型（先小后大）
□ 4. 选择微调方法（优先 LoRA）
□ 5. 设置合理的超参数（学习率、轮次、批次大小）
□ 6. 小规模验证（先用少量数据跑通流程）
□ 7. 正式训练 + 监控 loss 曲线
□ 8. 多维度评估（自动指标 + 人工评估）
□ 9. 迭代优化（数据 → 训练 → 评估 → 改进数据）
□ 10. 部署前的最终检查（安全性、推理速度、资源消耗）
```

记住一句话：**做 SFT 项目，80% 的时间应该花在数据上，而不是训练上。** 数据好，简单的方法就能出好效果；数据差，再高级的方法也救不了。
