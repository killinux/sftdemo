# 推理与部署：从训练完到能用

> 模型训练好了只是万里长征第一步，怎么让它**快速、省钱、稳定**地跑起来，才是真正的挑战。

---

## 一、推理解码策略

模型生成文本的过程，本质上是**一个 token 一个 token 往外蹦**。每一步，模型会给词表中每个 token 算一个概率，然后从中选一个。

关键问题：怎么"选"？不同的选法（解码策略）会产生截然不同的结果。

---

### 1.1 Greedy Decoding（贪心解码）

最简单粗暴的方法：**每一步都选概率最高的那个 token**。

```python
# HuggingFace 默认就是贪心解码
output = model.generate(input_ids, do_sample=False)
```

**特点：**
- 确定性输出：同样的输入永远得到同样的输出
- 速度快，不需要额外计算
- 但是输出很"无聊"，容易**反复重复**同一句话

> 类比：考试的时候只写最有把握的答案，绝对不冒险。分数稳定，但没有任何亮点。

---

### 1.2 Temperature（温度）

温度是控制"随机程度"的旋钮。它会改变概率分布的形状：

```
temperature=0.1: 概率分布更尖锐 → 几乎等于贪心，输出非常确定
temperature=1.0: 原始分布 → 正常随机
temperature=2.0: 概率分布更平坦 → 更随机、更有创意，但可能胡说八道
```

**原理：** 把每个 token 的 logit 除以 temperature，再做 softmax。温度越低，高概率 token 越突出；温度越高，各 token 概率越接近。

> 类比：温度就是考试答题时的心理状态。
> - 低温（0.1）= 严谨学霸，只写最有把握的答案
> - 中温（0.7）= 正常发挥，偶尔有灵感
> - 高温（2.0）= 天马行空随便写，可能写出天才答案，也可能写出离谱的东西

```python
output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.7,  # 推荐的通用温度
)
```

---

### 1.3 Top-K 采样

问题：即使用了 temperature，词表里有几万个 token，很多完全不靠谱的词也有微小概率被选中。

**Top-K 的做法：只从概率最高的 K 个 token 里随机选，其余的直接忽略。**

```
假设词表有 5 万个词，当前概率分布：
  的(30%), 了(20%), 是(15%), 在(10%), 我(8%), 有(7%), ... 几万个词

top_k=3: 只从 [的, 了, 是] 里随机选，其余全部排除
top_k=50: 只从前50个高概率词里选（常用设置）
```

> 类比：点外卖时，top_k=3 就是"只看销量前3的店"，不管后面那几百家。

```python
output = model.generate(
    input_ids,
    do_sample=True,
    top_k=50,
    temperature=0.7,
)
```

---

### 1.4 Top-P（Nucleus 采样）

Top-K 有个问题：有时候前3个词就已经占了 95% 的概率（选50个太多），有时候前100个词加起来才 80%（选50个太少）。**K 是固定的，但概率分布不是固定的。**

**Top-P 的做法更聪明：选概率加起来刚好达到 P 的最少的那些词。**

```
top_p=0.9 的例子：

情况1（概率集中）:
  的(60%) + 了(25%) + 是(8%) = 93% > 90%
  → 只从这 3 个词里选

情况2（概率分散）:
  的(15%) + 了(12%) + 是(10%) + 在(9%) + 我(8%) + 有(7%) + ... 
  → 需要更多词才能凑到 90%，可能从 10+ 个词里选
```

> 类比：Top-K 是"只看前 K 名"，Top-P 是"看够分数线的人"。人多就多看几个，人少就少看几个，更灵活。

```python
output = model.generate(
    input_ids,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
)
```

**实践中 Top-K 和 Top-P 经常一起用**，取两者的交集。

---

### 1.5 Repetition Penalty（重复惩罚）

模型有个坏毛病：特别喜欢重复自己说过的话（"我觉得这个很好。我觉得这个很好。我觉得..."）。

**重复惩罚会降低已经生成过的 token 的概率：**

```
repetition_penalty=1.0: 不惩罚，原始概率
repetition_penalty=1.2: 适度惩罚，减少重复（推荐）
repetition_penalty=1.5: 惩罚较重，可能导致用词过于分散
```

> 类比：写作文时老师说"不要用重复的词"。惩罚=1.2 是"尽量换个说法"，惩罚=1.5 是"绝对不许重复"。

---

### 1.6 Beam Search（束搜索）

前面的方法都是"走一步看一步"，Beam Search 不一样：**同时探索多条路径，最后选最好的那条。**

```
普通生成（每步只保留1个）:
  今天 → 天气 → 很好 → 。

Beam Search（num_beams=4，每步保留4个候选）:
  路径1: 今天 → 天气 → 很好 → 。         (得分: 0.85)
  路径2: 今天 → 天气 → 不错 → 。         (得分: 0.82)
  路径3: 今天 → 是 → 个 → 好日子          (得分: 0.78)
  路径4: 今天 → 的 → 天气 → 真好          (得分: 0.75)
  → 选得分最高的路径1
```

**特点：**
- 质量更好（因为考虑了全局最优）
- 但速度更慢（同时维护多条路径）
- **不适合对话/聊天场景**（生成内容太"标准"，缺乏多样性）
- 适合翻译、摘要等追求准确性的任务

```python
output = model.generate(
    input_ids,
    num_beams=4,         # 保留4条候选路径
    early_stopping=True, # 所有 beam 都生成结束符就停
)
```

---

### 1.7 推荐配置速查

不同场景用不同参数，这是经过大量实践总结出来的经验值：

| 场景 | temperature | top_p | top_k | repetition_penalty | 说明 |
|------|------------|-------|-------|-------------------|------|
| 代码生成 | 0.1~0.3 | 0.9 | 50 | 1.0 | 要求精确，不能瞎编 |
| 创意写作 | 0.7~1.0 | 0.95 | 50 | 1.2 | 鼓励多样性和创造力 |
| 日常对话 | 0.5~0.7 | 0.9 | 50 | 1.1 | 平衡准确性和自然度 |
| 事实问答 | 0.1~0.3 | 0.8 | 40 | 1.0 | 尽量准确，减少幻觉 |

> 经验法则：**不确定就用 temperature=0.7, top_p=0.9**，这是一个不会太差的万金油配置。

---

## 二、量化（Quantization）

### 2.1 什么是量化？

模型的每个参数默认用 FP16（半精度浮点数）存储，占 2 个字节。**量化就是用更少的位数来表示这些参数**，牺牲一点精度换取更小的体积和更快的速度。

```
7B 模型的显存占用：

FP16（16位）: 7B × 2字节 = 14GB → 需要 16GB+ 显存的显卡
INT8（8位） : 7B × 1字节 =  7GB → 普通游戏显卡就能跑
INT4（4位） : 7B × 0.5字节= 3.5GB → 笔记本电脑都能跑！
```

> 类比：就像把高清照片压缩成缩略图。缩略图比原图小很多，但主要内容还是看得清。
> INT8 相当于压缩到一半大小（几乎看不出区别），INT4 相当于压缩到四分之一（仔细看能发现一些模糊）。

---

### 2.2 常见量化方法

| 方法 | 特点 | 适用场景 | 速度 |
|------|------|---------|------|
| **GPTQ** | 训练后量化，质量好 | GPU 推理 | 快 |
| **AWQ** | 感知激活值的量化，比 GPTQ 略好 | GPU 推理 | 快 |
| **GGUF** | 专为 CPU 推理设计（llama.cpp 格式） | 本地/边缘部署 | CPU 上较快 |
| **BitsAndBytes** | 使用简单，和 HuggingFace 无缝集成 | 快速实验 | 中等 |

**怎么选？**
- 想在 GPU 上部署 → **AWQ** 或 **GPTQ**
- 想在 MacBook / CPU 上跑 → **GGUF**（配合 Ollama 或 llama.cpp）
- 想快速试一下能不能跑 → **BitsAndBytes**（代码最简单）

---

### 2.3 量化对质量的影响

```
FP16 → INT8: 质量损失很小（约 1%），几乎感觉不到差别
FP16 → INT4: 有一定损失（约 3~5%），但对于大模型影响更小
```

**重要规律：模型越大，量化带来的质量损失越小。**

```
INT4 量化的质量损失：
  1B 模型：约 8~10%，损失明显
  7B 模型：约 3~5%，可以接受
  70B 模型：约 1~2%，几乎无感
```

> 直觉理解：大模型的参数有很多"冗余"，量化掉的那些精度，其他参数可以"补偿"回来。
> 小模型每个参数都很关键，精度损失更难弥补。

---

### 2.4 代码示例：BitsAndBytes 4-bit 量化加载

这是最简单的量化方式，几行代码就能把显存砍到四分之一：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 启用 4-bit 量化
    bnb_4bit_quant_type="nf4",            # NF4 量化格式（效果最好）
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时用 bfloat16
    bnb_4bit_use_double_quant=True,       # 双重量化，进一步省内存
)

# 加载量化后的模型
model_id = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配到可用的 GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 正常使用，和非量化模型完全一样
inputs = tokenizer("你好，请介绍一下自己", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 三、KV Cache

### 3.1 问题：为什么不缓存会很慢？

Transformer 模型生成文本时，是一个 token 一个 token 往外蹦的。生成每个新 token 时，需要对**所有之前的 token** 做 Attention 计算。

```
不用缓存的情况：

生成第1个token: 计算 "今"          的 attention
生成第2个token: 重新计算 "今天"      的 attention（"今" 又算了一遍！）
生成第3个token: 重新计算 "今天天"    的 attention（"今天" 又算了一遍！！）
生成第4个token: 重新计算 "今天天气"  的 attention（"今天天" 又算了一遍！！！）
...

每一步都要从头算，总计算量是 O(n²)，越到后面越慢
```

> 类比：就像读一本书，每读一页新内容，都要从第一页重新读一遍。100 页的书要反复读 100 次！

---

### 3.2 解决方案：缓存 Key 和 Value

Attention 计算需要三个东西：Query（Q）、Key（K）、Value（V）。其中 K 和 V 只跟输入有关，之前的 token 算过的 K 和 V **不会变**，完全可以缓存起来复用。

```
用 KV Cache 的情况：

生成第1个token: 计算 K1, V1 → 缓存起来
生成第2个token: 从缓存取 K1,V1 + 计算新的 K2,V2 → 缓存 K1,K2,V1,V2
生成第3个token: 从缓存取 K1,K2,V1,V2 + 计算新的 K3,V3 → 缓存更新
生成第4个token: 从缓存取之前所有的 KV + 只算新的 K4,V4
...

每一步只需要计算 1 个新 token 的 KV，然后和缓存拼在一起做 attention
```

**效果：每步只做 O(n) 的计算，总共 O(n²) 降到了线性级别。**

> 类比：读书时用书签标记已读内容。每次只需要读新的一页，之前的内容用书签标记好，随时可以翻到。不需要每次都从第一页重新读。

---

### 3.3 KV Cache 的代价

天下没有免费的午餐。KV Cache 用空间换时间，**会占用大量显存**：

```
KV Cache 显存 ≈ 2 × 层数 × 隐藏维度 × 序列长度 × batch_size × 数据精度

举例（7B 模型，序列长度 4096）:
  = 2 × 32层 × 4096维 × 4096长度 × 2字节(FP16)
  ≈ 2GB（单条请求）

如果 batch_size=32:
  ≈ 64GB（光 KV Cache 就要 64GB！）
```

**这就是为什么：**
- 长上下文模型（128K、1M）推理时需要巨大显存
- 同时服务多个用户时，显存很容易不够
- KV Cache 管理是推理框架的核心优化点

---

## 四、推理框架

训练好的模型不能直接拿 `model.generate()` 上线服务——太慢了。需要专业的推理框架来榨干硬件性能。

---

### 4.1 vLLM

目前最流行的大模型推理框架，是生产部署的事实标准。

**核心技术：**

- **PagedAttention**：像操作系统管理虚拟内存一样管理 KV Cache
  - 传统方式：给每个请求预分配最大长度的 KV Cache → 大量浪费
  - PagedAttention：按需分配，用多少分多少 → 显存利用率大幅提升
- **Continuous Batching**：连续批处理
  - 传统方式：等一批请求都结束了，才处理下一批 → 先完成的请求在空等
  - Continuous Batching：某个请求完成了立刻填入新请求 → GPU 永远不闲着

**性能：比原生 HuggingFace 推理快 10~24 倍。**

```bash
# 启动 vLLM 服务（兼容 OpenAI API 格式）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# 调用方式和 OpenAI 完全一样
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "你好"}]
    }'
```

---

### 4.2 Text Generation Inference（TGI）

HuggingFace 官方出品的生产级推理服务器。

**特点：**
- 和 HuggingFace 生态无缝集成
- 支持量化、LoRA adapter 热加载
- Docker 一键部署

```bash
# Docker 启动 TGI
docker run --gpus all -p 8080:80 \
    -v /data/models:/models \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen2.5-7B-Instruct
```

**vLLM vs TGI：** vLLM 吞吐量通常更高，TGI 和 HuggingFace 生态集成更好。大多数场景推荐 vLLM。

---

### 4.3 Ollama / llama.cpp

专为本地和边缘部署设计，让普通人也能在自己电脑上跑大模型。

**特点：**
- 支持 CPU 推理（不需要 GPU！）
- 使用 GGUF 格式的量化模型
- 安装和使用极其简单

```bash
# 安装 Ollama（macOS / Linux）
curl -fsSL https://ollama.ai/install.sh | sh

# 下载并运行模型，就这么简单
ollama run qwen2.5:7b

# 直接开始对话
>>> 你好，介绍一下你自己
```

> 适用场景：个人使用、本地开发调试、隐私要求高不能联网的场景。
> 不适合：高并发的生产服务。

---

### 4.4 推理框架对比

| 框架 | 适用场景 | 硬件要求 | 性能 | 易用性 |
|------|---------|---------|------|--------|
| **vLLM** | 生产部署 | GPU | 最高 | 中等 |
| **TGI** | 生产部署 | GPU | 高 | 中等 |
| **Ollama** | 本地使用 | CPU/GPU | 中等 | 极简 |
| **llama.cpp** | 嵌入式/边缘 | CPU | 中等 | 需要编译 |
| **HuggingFace** | 开发调试 | CPU/GPU | 低 | 最简单 |

---

## 五、部署方式

### 5.1 简单部署（适合开发测试）

用 FastAPI 包一层 HTTP 接口，几分钟就能跑起来：

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# 启动时加载模型（只加载一次）
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{"role": "user", "content": req.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True,
    )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"response": response}
```

```bash
# 启动服务
uvicorn app:app --host 0.0.0.0 --port 8000

# 测试
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "什么是机器学习？"}'
```

> 注意：这种方式**只适合开发测试**。没有批处理、没有并发支持、没有 KV Cache 优化，性能很差。

---

### 5.2 生产部署架构

真正上线服务时，架构通常是这样的：

```
                     ┌──────────────┐
用户请求 ──→ 网关/限流 ──→│   负载均衡    │
                     └──────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ vLLM 实例1│  │ vLLM 实例2│  │ vLLM 实例3│
        │ (GPU×2)  │  │ (GPU×2)  │  │ (GPU×2)  │
        └──────────┘  └──────────┘  └──────────┘
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                     ┌──────────────┐
                     │   监控告警    │
                     │ (延迟/吞吐/  │
                     │  错误率/显存) │
                     └──────────────┘
```

**生产部署的关键考虑：**

| 方面 | 做法 | 为什么 |
|------|------|-------|
| **批处理** | 使用 vLLM 的 continuous batching | 提高 GPU 利用率，提升吞吐量 |
| **流式输出** | 用 SSE（Server-Sent Events）逐 token 返回 | 用户体验好，不用等全部生成完 |
| **限流** | 设置每用户/每 IP 的 QPS 限制 | 防止被刷爆，保护服务稳定 |
| **监控** | 跟踪延迟、吞吐量、错误率、显存使用 | 及时发现问题，容量规划 |
| **弹性伸缩** | 根据负载自动增减 GPU 实例 | 高峰期不崩，低谷期省钱 |
| **模型热更新** | 支持不停服更换模型或 LoRA adapter | 快速迭代，不影响用户 |

---

### 5.3 流式输出示例

用户不想等 10 秒才看到完整回复。流式输出让文字像打字机一样一个一个蹦出来：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        # 使用 vLLM 的流式接口
        async for chunk in engine.generate_stream(req.prompt):
            data = json.dumps({"text": chunk.text}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 六、概念速查表

一张表总结本章所有核心概念：

| 概念 | 一句话解释 | 关键点 |
|------|-----------|--------|
| **Greedy Decoding** | 每步选概率最高的 token | 确定性输出，容易重复 |
| **Temperature** | 控制随机程度的旋钮 | 低温精确，高温创意 |
| **Top-K** | 只从前 K 个高概率词里选 | K 是固定值，简单粗暴 |
| **Top-P** | 选概率总和达到 P 的最少的词 | 动态调整候选词数量 |
| **Repetition Penalty** | 降低已生成 token 的概率 | 1.2 左右比较合适 |
| **Beam Search** | 同时探索多条路径选最优 | 适合翻译/摘要，不适合对话 |
| **量化** | 用更少的位数存储参数 | FP16→INT4 体积缩小4倍 |
| **GPTQ/AWQ** | GPU 上的量化方法 | 生产部署用 |
| **GGUF** | CPU 推理的量化格式 | 本地部署用 |
| **BitsAndBytes** | HuggingFace 集成的量化工具 | 快速实验用 |
| **KV Cache** | 缓存之前 token 的 Key 和 Value | 用空间换时间，大幅加速推理 |
| **PagedAttention** | 像虚拟内存一样管理 KV Cache | vLLM 的核心技术 |
| **Continuous Batching** | 动态填充和移除请求 | 提高 GPU 利用率 |
| **vLLM** | 最流行的生产推理框架 | 吞吐量最高 |
| **TGI** | HuggingFace 的推理服务 | 生态集成好 |
| **Ollama** | 本地运行大模型的工具 | 最简单易用 |
| **流式输出** | 逐 token 返回，像打字机一样 | 用户体验的关键 |

---

## 七、下一步

模型能跑了，但**效果好不好**还需要评估。下一章我们会学习如何科学地评估你微调后的模型：

```
训练完成 → 推理部署（本章）→ 模型评估（下一章）→ 发现问题 → 回去改数据/调参 → 重新训练
             ↑                                                          │
             └──────────────────────────────────────────────────────────┘
```

> 这是一个不断迭代的循环。好的模型不是一次训练出来的，而是在"训练 → 部署 → 评估 → 改进"的循环中不断打磨出来的。
