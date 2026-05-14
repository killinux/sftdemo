import torch
from transformers import AutoModelForCausalLM

original = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
finetuned = AutoModelForCausalLM.from_pretrained("./sft_output")

print("参数名 | 变化量(均值绝对值) | 变化比例")
print("-" * 60)

total_params = 0
total_changed = 0

for name, p_orig in original.named_parameters():
    p_ft = finetuned.state_dict()[name]
    diff = (p_ft - p_orig).abs()
    mean_diff = diff.mean().item()
    mean_orig = p_orig.abs().mean().item()
    ratio = mean_diff / mean_orig if mean_orig > 0 else 0

    total_params += p_orig.numel()
    total_changed += diff.sum().item()

    if "layers.0." in name and "weight" in name:
        print(f"{name[:50]:50s} | {mean_diff:.6f} | {ratio:.4%}")

print("-" * 60)
print(f"总参数量: {total_params:,}")
print(f"平均每个参数变化量: {total_changed / total_params:.6f}")
