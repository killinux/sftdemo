from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

print("=== 模型结构 ===\n")

layer_params = {}
for name, p in model.named_parameters():
    parts = name.split(".")
    if "layers" in parts:
        layer_idx = parts[parts.index("layers") + 1]
        key = f"Transformer 层 {int(layer_idx):2d}"
    elif "embed" in name:
        key = "Embedding 层"
    elif "norm" in name:
        key = "LayerNorm"
    elif "lm_head" in name:
        key = "输出层 (lm_head)"
    else:
        key = "其他"
    layer_params[key] = layer_params.get(key, 0) + p.numel()

for key, count in layer_params.items():
    print(f"  {key:20s}  {count:>12,} 参数")

print(f"\n  {'总计':20s}  {sum(layer_params.values()):>12,} 参数")
print(f"\n每层 Transformer 约 {layer_params.get('Transformer 层  0', 0):,} 参数")
