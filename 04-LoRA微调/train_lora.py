"""
LoRA 微调示例脚本
================
使用 PEFT 库对 Qwen2.5-0.5B 进行 LoRA 微调。
数据格式与全量微调完全一致（data.json: instruct/input/label）。

依赖安装:
    pip install torch transformers peft datasets accelerate
"""

from typing import List, Dict, Sequence
import json
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, TaskType

IGNORE_INDEX = -100
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════
# 第一步：加载基座模型与分词器
# ═══════════════════════════════════════════════════════════════
model_dir = "Qwen/Qwen2.5-0.5B"

# 加载原始模型（后面会用 PEFT 给它"贴便利贴"）
model = AutoModelForCausalLM.from_pretrained(model_dir)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# 打印原始模型的参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"基座模型参数量: {total_params:,}")

# ═══════════════════════════════════════════════════════════════
# 第二步：配置 LoRA
# ═══════════════════════════════════════════════════════════════
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
    r=8,                            # rank: 便利贴的大小，8 是推荐默认值
    lora_alpha=16,                  # 缩放系数，一般设为 2×r
    lora_dropout=0.05,              # dropout 防过拟合
    target_modules=[                # 在哪些层加 LoRA（Qwen2.5 的注意力层）
        "q_proj",                   # Query 投影
        "v_proj",                   # Value 投影
    ],
    # 如果想要更好的效果，可以加上 k_proj 和 o_proj：
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# 用 PEFT 给模型"贴上便利贴"
model = get_peft_model(model, lora_config)

# 打印可训练参数的比例（应该远小于 1%）
model.print_trainable_parameters()

# ═══════════════════════════════════════════════════════════════
# 第三步：加载数据集
# ═══════════════════════════════════════════════════════════════
data_path = "../02-数据工程/data.json"

with open(data_path, "r") as f:
    data = json.load(f)

print(f"\n训练样本数: {len(data)}")
for item in data:
    print(item)


# ═══════════════════════════════════════════════════════════════
# 第四步：自定义 Dataset（与全量微调完全一致）
# ═══════════════════════════════════════════════════════════════
# 核心逻辑：
#   - 把 instruct + input + label 拼成完整文本
#   - 对 instruct + input 部分的 label 设为 IGNORE_INDEX（不计算 loss）
#   - 只对 label 部分计算 loss（让模型学会"回答"）
class PreTrainDataset(Dataset):

    def __init__(self, data: List):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        # 拼接完整文本：指令 + 输入 + 标签 + 结束符
        text = item["instruct"] + item["input"] + item["label"] + tokenizer.eos_token
        text_token = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        label = text_token["input_ids"].clone()

        # 计算指令部分的长度，将其 label 设为 IGNORE_INDEX
        instruct = item["instruct"] + item["input"]
        instruct_token = tokenizer(
            instruct,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        instruct_len = instruct_token["input_ids"].size(-1)

        # 指令部分不计算 loss（label masking）
        label[:, :instruct_len] = IGNORE_INDEX
        text_token["labels"] = label

        return text_token


# ═══════════════════════════════════════════════════════════════
# 第五步：DataCollator（批量 padding，与全量微调一致）
# ═══════════════════════════════════════════════════════════════
@dataclass
class DataCollatorForSFTDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, items: Sequence) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"].squeeze(0) for item in items]
        attention_mask = [item["attention_mask"].squeeze(0) for item in items]
        labels = [item["labels"].squeeze(0) for item in items]

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,

        )
        attention_mask = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,

        )
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,

        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ═══════════════════════════════════════════════════════════════
# 第六步：验证数据集
# ═══════════════════════════════════════════════════════════════
dataset = PreTrainDataset(data)
sample = dataset[0]
print("\n=== 第一条样本 ===")
print("input_ids shape:", sample["input_ids"].shape)
print("labels shape:", sample["labels"].shape)

test_label = sample["labels"].squeeze()
loss_tokens = test_label[test_label != IGNORE_INDEX]
print("需要计算 loss 的文本:", tokenizer.decode(loss_tokens))

ignored_ids = sample["input_ids"].squeeze()[test_label == IGNORE_INDEX]
print("不计算 loss 的文本:", tokenizer.decode(ignored_ids))

# ═══════════════════════════════════════════════════════════════
# 第七步：配置训练参数并开始训练
# ═══════════════════════════════════════════════════════════════
output_dir = "./lora_output"

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,               # 训练轮数
    per_device_train_batch_size=2,      # 批大小
    learning_rate=2e-4,                 # LoRA 通常用稍大的学习率
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,                 # 只保留最近 2 个 checkpoint
    fp16=torch.cuda.is_available(),     # 有 GPU 就用半精度加速
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=dataset,
    eval_dataset=None,
    data_collator=DataCollatorForSFTDataset(tokenizer=tokenizer),
)

print("\n开始 LoRA 训练...")
train_result = trainer.train()
print("\n训练指标:", train_result.metrics)

# ═══════════════════════════════════════════════════════════════
# 第八步：保存 LoRA adapter（只有几十 MB）
# ═══════════════════════════════════════════════════════════════
adapter_dir = "./lora_output/adapter"

trainer.save_state()
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"\nLoRA adapter 已保存至: {adapter_dir}")
print("（注意：这里只保存了 LoRA 的参数，不包含基座模型）")

# ═══════════════════════════════════════════════════════════════
# 第九步：合并 LoRA 到基座模型（可选，用于部署）
# ═══════════════════════════════════════════════════════════════
# 合并后就是一个普通的完整模型，不再需要 PEFT 库
merged_dir = "./lora_output/merged_model"

print("\n正在合并 LoRA 权重到基座模型...")
merged_model = model.merge_and_unload()  # 把便利贴内容写进教科书
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print(f"合并后的完整模型已保存至: {merged_dir}")
print("（这个模型可以像普通模型一样加载和部署，不需要 PEFT 库）")

# ═══════════════════════════════════════════════════════════════
# 第十步：快速验证 —— 用合并后的模型生成一首诗
# ═══════════════════════════════════════════════════════════════
print("\n=== 验证：用 LoRA 微调后的模型生成 ===")
merged_model = merged_model.to(device)
merged_model.eval()

test_prompt = "请你给哪吒写一首诗：哪吒降世，意气飞扬。\n逆天改命，破障冲霄。"
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = merged_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"输入: {test_prompt}")
print(f"生成: {generated_text[len(test_prompt):]}")
