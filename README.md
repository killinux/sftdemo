# 大模型有监督微调 (SFT) Demo

基于 Qwen2.5-0.5B 的有监督微调教程，演示如何通过 label masking 让模型只学习生成回答，而不学习复述用户指令。

> 参考：[JieShenAI/csdn - SFT](https://github.com/JieShenAI/csdn/tree/main/25/02/SFT)

## 核心原理

SFT 与预训练的代码流程基本一致，**唯一的区别是不对用户输入部分计算 loss**。

训练时，每条数据被拼接为一个完整序列：

```
[instruct] + [input] + [label] + <eos>
```

其中 `instruct + input` 部分的 label 被设置为 `-100`（PyTorch CrossEntropyLoss 会自动忽略该值），只有 `label + <eos>` 部分参与 loss 计算：

```
tokens:  请你给哪吒写一首诗：哪吒降世，意气飞扬。逆天改命，破障冲霄。红绫缠腕，风火踏浪。不屈不悔，笑傲苍茫。<eos>
labels:  -100 -100 -100 ...（指令+输入部分全部 -100）...  红绫缠腕，风火踏浪。不屈不悔，笑傲苍茫。<eos>
```

这样模型只学习「给定指令和上文后，如何生成正确的回答」，而不会学习复述指令本身。

## 项目结构

```
sftdemo/
├── data.json    # 训练数据（5 条哪吒角色诗歌样本）
├── train.py     # 训练脚本
├── infer.py     # 推理脚本
└── README.md
```

## 数据格式

每条样本包含三个字段：

| 字段 | 含义 | 示例 |
|------|------|------|
| `instruct` | 任务指令 | `请你给敖丙写一首诗：` |
| `input` | 上文/提示 | `碧海生龙子，云中舞雪霜。` |
| `label` | 期望输出 | `恩仇难两忘，何处是家乡？` |

## 关键实现

### 1. PreTrainDataset

自定义 `Dataset`，在 `__getitem__` 中完成 tokenization 和 label masking：

```python
# 拼接完整文本并 tokenize
text = instruct + input + label + eos_token
text_token = tokenizer(text, ...)

# 单独 tokenize 指令部分，获取长度
instruct_token = tokenizer(instruct + input, ...)
instruct_len = instruct_token["input_ids"].size(-1)

# 将指令部分的 label 设为 -100，不参与 loss 计算
label[:, :instruct_len] = -100
```

### 2. DataCollatorForSFTDataset

处理 batch 内不同长度序列的 padding：

- `input_ids` — 用 `pad_token_id` 填充
- `attention_mask` — 用 `0` 填充（padding 位置不参与注意力计算）
- `labels` — 用 `-100` 填充（padding 位置也不计算 loss）

### 3. 推理验证

推理脚本分两组测试：

- **instruct + input → label**：给定完整上文，验证模型能否生成正确的续写
- **instruct only → label**：仅给指令不给上文，观察模型泛化能力

## 环境要求

```
torch
transformers
```

模型 `Qwen/Qwen2.5-0.5B` 会从 HuggingFace Hub 自动下载（约 1GB）。有 GPU 会快很多，CPU 也可以运行。

## 运行

```bash
# 第一步：训练（输出保存到 ./sft_output）
python train.py

# 第二步：推理验证
python infer.py
```

## 训练参数

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen/Qwen2.5-0.5B |
| 训练轮数 | 10 |
| batch size | 2 |
| 模型保存格式 | safetensors |
| 日志策略 | 每个 epoch 打印一次 |

参考原仓库在 GPU 上约 20 秒即可完成训练（5 条样本，10 个 epoch）。
