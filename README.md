# SFT 学习指南：大模型有监督微调

从零理解 SFT（Supervised Fine-Tuning），从基础概念到工程实践，按章节组织，方便复习。

基于 Qwen2.5-0.5B 的完整 demo，含可运行的训练和推理代码。

## 目录

| 章节 | 内容 | 关键词 |
|------|------|--------|
| [01-基础概念](01-基础概念/) | 大模型是什么、SFT 原理、Attention、Tokenizer | Transformer, Loss, Label Masking |
| [02-数据工程](02-数据工程/) | 数据格式、清洗、合成数据、数据增强 | Alpaca, ShareGPT, Chat Template |
| [03-全量微调](03-全量微调/) | 完整训练 demo、推理验证、参数对比 | train.py, infer.py, Epoch |
| [04-LoRA微调](04-LoRA微调/) | LoRA 原理、QLoRA、超参数选择、训练代码 | rank, alpha, PEFT |
| [05-训练技巧](05-训练技巧/) | 混合精度、梯度累积、学习率调度、过拟合诊断 | BF16, Warmup, Early Stop |
| [06-推理与部署](06-推理与部署/) | 解码策略、量化、KV Cache、推理框架 | temperature, top_p, vLLM, GGUF |
| [07-模型评估](07-模型评估/) | 离线评估、场景化评估、在线评估 | BLEU, ROUGE, Benchmark |
| [08-对齐训练](08-对齐训练/) | RLHF、DPO、GRPO 原理与对比 | Reward Model, PPO, DeepSeek-R1 |
| [09-工程实践](09-工程实践/) | SFT vs RAG vs PE、训练框架、分布式训练 | LLaMA-Factory, DeepSpeed |

## 推荐学习路径

```
入门:  01 → 02 → 03（跑通 demo）
进阶:  04 → 05 → 07（学会调优）
实战:  06 → 08 → 09（部署上线）
```

## 快速开始

```bash
# 安装依赖
pip install torch transformers accelerate

# 跑通全量微调 demo
cd 03-全量微调
python train.py
python infer.py

# 跑通 LoRA 微调
cd ../04-LoRA微调
pip install peft
python train_lora.py
```

## 项目结构

```
sftdemo/
├── 01-基础概念/         # 大模型、SFT、Attention、Tokenizer
├── 02-数据工程/         # 数据格式、清洗、合成、增强
│   └── data.json       # 示例训练数据（5条）
├── 03-全量微调/         # 全量微调 demo（可运行）
│   ├── train.py        # 训练脚本
│   ├── infer.py        # 推理脚本
│   ├── compare.py      # 参数对比
│   └── show_layers.py  # 模型结构
├── 04-LoRA微调/         # LoRA 微调（可运行）
│   └── train_lora.py   # LoRA 训练脚本
├── 05-训练技巧/         # 混合精度、梯度累积、调参
├── 06-推理与部署/       # 解码策略、量化、部署
├── 07-模型评估/         # 评估方法与指标
├── 08-对齐训练/         # RLHF / DPO / GRPO
└── 09-工程实践/         # 框架选型、分布式、完整流程
```

## 大模型训练全链路

```
预训练          →  SFT           →  对齐训练         →  部署
读几万亿字的书      教它听指令        教它回答得好        给用户用
（01-基础概念）   （02~05）        （08-对齐训练）    （06-推理与部署）
```
