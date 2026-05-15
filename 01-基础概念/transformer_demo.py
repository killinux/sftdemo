"""
从零手写 Transformer
===================
不用任何预训练模型，纯 PyTorch 搭建 Transformer 的每个组件。
训练一个字符级语言模型，学会根据上文预测下一个字。

运行: python transformer_demo.py

组件清单:
  1. Token Embedding     —— 把字变成向量
  2. Positional Encoding —— 告诉模型每个字的位置
  3. Self-Attention       —— 核心：每个字去"看"其他字
  4. Multi-Head Attention —— 多组注意力，关注不同维度
  5. Feed-Forward Network —— 对每个位置做非线性变换
  6. Transformer Block    —— Attention + FFN + 残差 + LayerNorm
  7. Transformer LM       —— 完整模型：Embedding + N个Block + 输出层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ═══════════════════════════════════════════════════════════════
# 训练数据：几句简单的中文，让模型学会这些模式
# ═══════════════════════════════════════════════════════════════
corpus = """天问一号成功着陆火星。嫦娥五号带回月球样本。
北斗导航覆盖全球。神舟飞船载人航天。天宫空间站建设完成。
长征火箭发射成功。祝融号探测火星表面。玉兔号探测月球表面。
天问一号成功着陆火星。嫦娥五号带回月球样本。
北斗导航覆盖全球。神舟飞船载人航天。天宫空间站建设完成。"""


# ═══════════════════════════════════════════════════════════════
# 第零步：构建字符级词表
# ═══════════════════════════════════════════════════════════════
chars = sorted(set(corpus))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}

print(f"词表大小: {vocab_size} 个字符")
print(f"词表内容: {''.join(chars)}")

def encode(text):
    return [char_to_id[c] for c in text]

def decode(ids):
    return ''.join(id_to_char[i] for i in ids)


# ═══════════════════════════════════════════════════════════════
# 第一步：Token Embedding —— 把整数ID变成向量
# ═══════════════════════════════════════════════════════════════
# 就是一个查找表：每个字对应一个 d_model 维的向量
# 这个向量是可学习的参数，训练过程中会不断更新
#
#   "天" → id=15 → [0.12, -0.34, 0.56, ...]  (长度=d_model)
#
# PyTorch 的 nn.Embedding 就是干这个的


# ═══════════════════════════════════════════════════════════════
# 第二步：Positional Encoding —— 告诉模型位置信息
# ═══════════════════════════════════════════════════════════════
# Transformer 没有循环结构（不像 RNN），它看到的是一堆向量的集合
# 如果不加位置信息，"我爱你" 和 "你爱我" 对模型来说完全一样
# 所以要给每个位置加一个独特的"位置编码"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维用 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════
# 第三步：Self-Attention —— Transformer 的灵魂
# ═══════════════════════════════════════════════════════════════
# 核心问题：对于序列中的每个字，它应该"关注"哪些其他字？
#
# 做法：每个字生成三个向量
#   Q (Query)  —— "我在找什么？"
#   K (Key)    —— "我有什么特征？"
#   V (Value)  —— "我的实际内容是什么？"
#
# 然后用 Q 和所有 K 做点积，得到注意力权重，再加权求和 V
#
# 例子：处理 "天问一号" 中的 "号" 时
#   "号" 的 Q 跟 "天" 的 K 点积 = 0.1  （不太相关）
#   "号" 的 Q 跟 "问" 的 K 点积 = 0.2  （有点相关）
#   "号" 的 Q 跟 "一" 的 K 点积 = 0.3  （比较相关）
#   softmax 归一化后得到注意力权重，用来加权各个字的 V

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 点积注意力: softmax(Q·K^T / √d) · V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


# ═══════════════════════════════════════════════════════════════
# 第四步：Multi-Head Attention —— 多角度关注
# ═══════════════════════════════════════════════════════════════
# 一个注意力头只能关注一种模式。比如一个头学会了关注"前一个字"，
# 那它就没法同时关注"语义相关的字"。
#
# 解决方法：用多个头（head），每个头关注不同的模式
#   Head 1: 关注相邻的字（语法）
#   Head 2: 关注语义相关的字（语义）
#   Head 3: 关注标点和结构（结构）
#   ...
#
# 最后把所有头的结果拼起来

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape

        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # 现在形状: (batch, n_heads, seq_len, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # 把多个头拼回去
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.W_o(attn_output)

        return output, attn_weights


# ═══════════════════════════════════════════════════════════════
# 第五步：Feed-Forward Network —— 逐位置的非线性变换
# ═══════════════════════════════════════════════════════════════
# Attention 负责"看其他位置的信息"，FFN 负责"消化这些信息"
# 结构很简单：两层全连接 + ReLU
# 先升维（d_model → 4*d_model），再降维（4*d_model → d_model）

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ═══════════════════════════════════════════════════════════════
# 第六步：Transformer Block —— 把以上组件拼起来
# ═══════════════════════════════════════════════════════════════
# 一个 Block = Multi-Head Attention + FFN + 残差连接 + LayerNorm
#
#  输入 ──┐
#         ├─→ LayerNorm → Multi-Head Attention ──┐
#         │                                       ├─→ 加起来（残差）
#         └───────────────────────────────────────┘
#                                                 ──┐
#         ┌───────────────────────────────────────┐  ├─→ 加起来（残差）
#         ├─→ LayerNorm → Feed-Forward ───────────┘
#         │
#  输出 ──┘
#
# 残差连接的作用：让梯度能直接流过去，防止深层网络训不动

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 子层 1: 多头注意力 + 残差
        normed = self.norm1(x)
        attn_output, attn_weights = self.attention(normed, mask)
        x = x + attn_output

        # 子层 2: FFN + 残差
        normed = self.norm2(x)
        ff_output = self.ffn(normed)
        x = x + ff_output

        return x, attn_weights


# ═══════════════════════════════════════════════════════════════
# 第七步：完整的 Transformer 语言模型
# ═══════════════════════════════════════════════════════════════
# 完整流程：
#   字符 → Embedding → 加位置编码 → N个Block → 输出层 → 预测下一个字

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, max_len=128):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape

        # causal mask: 每个位置只能看到自己和前面的位置（不能偷看未来）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            all_attn_weights.append(attn_weights)

        x = self.norm(x)
        logits = self.output_head(x)  # (batch, seq_len, vocab_size)

        return logits, all_attn_weights

    def generate(self, start_text, max_new_tokens=30):
        self.eval()
        input_ids = torch.tensor([encode(start_text)])

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self(input_ids)
                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits / 0.8, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_id], dim=1)

        return decode(input_ids[0].tolist())


# ═══════════════════════════════════════════════════════════════
# 第八步：准备训练数据
# ═══════════════════════════════════════════════════════════════
seq_len = 32

all_ids = encode(corpus)
print(f"\n语料总长: {len(all_ids)} 个字符")
print(f"编码示例: '{corpus[:6]}' → {all_ids[:6]}")

# 滑动窗口切分训练样本
inputs, targets = [], []
for i in range(0, len(all_ids) - seq_len):
    inputs.append(all_ids[i : i + seq_len])
    targets.append(all_ids[i + 1 : i + seq_len + 1])

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)
print(f"训练样本数: {len(inputs)}")


# ═══════════════════════════════════════════════════════════════
# 第九步：训练
# ═══════════════════════════════════════════════════════════════
model = TransformerLM(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=4)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print("（对比：GPT-3 = 175,000M，Qwen2.5-0.5B = 500M）\n")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("开始训练...")
print("-" * 50)

for epoch in range(100):
    model.train()
    logits, _ = model(inputs)
    # logits: (batch, seq_len, vocab_size)
    # 展平后计算交叉熵
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/100  Loss: {loss.item():.4f}")


# ═══════════════════════════════════════════════════════════════
# 第十步：生成文本
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("训练完成，测试生成：")
print("=" * 50)

test_prompts = ["天问", "北斗", "神舟", "嫦娥"]
for prompt in test_prompts:
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"  '{prompt}' → {generated}")


# ═══════════════════════════════════════════════════════════════
# 第十一步：可视化注意力权重（看模型在"看"什么）
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("注意力可视化（第1层，第1个头）：")
print("=" * 50)

viz_text = "天问一号成功着陆火星"
viz_ids = torch.tensor([encode(viz_text)])

model.eval()
with torch.no_grad():
    _, attn_weights = model(viz_ids)

# 取第一层、第一个头的注意力权重
attn = attn_weights[0][0, 0].numpy()  # (seq_len, seq_len)

print(f"\n输入: {viz_text}")
print(f"每个字在关注哪些字（注意力权重，越大表示越关注）:\n")

chars_list = list(viz_text)
header = "        " + "  ".join(f"{c:>4s}" for c in chars_list)
print(header)
print("        " + "─" * (len(chars_list) * 6))

for i, c in enumerate(chars_list):
    weights = attn[i, :len(chars_list)]
    row = f"  {c}  │ " + "  ".join(f"{w:.2f}" for w in weights)
    max_j = weights.argmax()
    row += f"  ← 最关注'{chars_list[max_j]}'"
    print(row)

print("""
\n读法：每一行表示该字在"看"哪些字
      比如 "号" 那一行，哪个数值最大，就说明 "号" 最关注那个字
      因为有 causal mask，每个字只能看到自己和前面的字（右上角都是 0）

这就是 Attention 的本质：让每个字自己学会该关注谁
""")
