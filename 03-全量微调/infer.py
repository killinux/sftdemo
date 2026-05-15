import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_model = "./sft_output"

model = AutoModelForCausalLM.from_pretrained(train_model)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(train_model, padding_side="right")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

with open("../02-数据工程/data.json", "r") as f:
    data = json.load(f)


def infer(text):
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**input_ids, max_new_tokens=128)
    generated_ids = [
        output_ids[len(inp) :]
        for inp, output_ids in zip(input_ids.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ── 测试 1: instruct + input -> label ─────────────────────────
print("=" * 50 + " instruct + input " + "=" * 50)
for item in data:
    instruct, inp, label = item["instruct"], item["input"], item["label"]
    print(f"text_input: {instruct + inp}")
    print(f"predict:    {infer(instruct + inp)}")
    print(f"label:      {label}")
    print("-" * 118)

# ── 测试 2: 仅 instruct -> label ──────────────────────────────
print("\n" + "=" * 50 + " instruct only " + "=" * 50)
for item in data:
    instruct, inp, label = item["instruct"], item["input"], item["label"]
    print(f"text_input: {instruct}")
    print(f"predict:    {infer(instruct)}")
    print(f"label:      {label}")
    print("-" * 118)
