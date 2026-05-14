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

IGNORE_INDEX = -100
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── 模型与分词器 ──────────────────────────────────────────────
model_dir = "Qwen/Qwen2.5-0.5B"

model = AutoModelForCausalLM.from_pretrained(model_dir)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# ── 加载数据集 ────────────────────────────────────────────────
with open("data.json", "r") as f:
    data = json.load(f)

print(f"训练样本数: {len(data)}")
for item in data:
    print(item)


# ── 自定义 Dataset ────────────────────────────────────────────
class PreTrainDataset(Dataset):

    def __init__(self, data: List):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        text = item["instruct"] + item["input"] + item["label"] + tokenizer.eos_token
        text_token = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        label = text_token["input_ids"].clone()

        instruct = item["instruct"] + item["input"]
        instruct_token = tokenizer(
            instruct,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        instruct_len = instruct_token["input_ids"].size(-1)

        label[:, :instruct_len] = IGNORE_INDEX
        text_token["labels"] = label

        return text_token


# ── DataCollator: 批量 padding ────────────────────────────────
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
            padding_side="right",
        )
        attention_mask = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="right",
        )
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
            padding_side="right",
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ── 验证 dataset & collator ──────────────────────────────────
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

# ── 训练 ──────────────────────────────────────────────────────
output_dir = "./sft_output"

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    logging_strategy="epoch",
)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=dataset,
    eval_dataset=None,
    data_collator=DataCollatorForSFTDataset(tokenizer=tokenizer),
)

train_result = trainer.train()
print("\n训练指标:", train_result.metrics)

trainer.save_state()
trainer.save_model(output_dir=output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n模型已保存至: {output_dir}")
