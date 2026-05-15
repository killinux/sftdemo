# 数据工程：SFT 成败的关键

做 SFT，大家最容易忽略的就是数据。很多人一上来就调模型、调参数，结果发现效果不好，回头一看——数据有问题。

**一句话：数据决定了 SFT 效果的上限，模型和训练技巧只是在逼近这个上限。**

好比开餐厅，模型是厨师，训练是厨艺，数据是食材。食材不新鲜，米其林大厨也做不出好菜。

---

## 一、数据格式

SFT 数据的本质就是"问题 → 回答"的配对。不同社区和框架定义了不同的格式标准，搞清楚它们之间的区别很重要。

### Alpaca 格式（单轮对话）

斯坦福 Alpaca 项目定义的格式，是最经典的 SFT 数据格式，适合单轮问答：

```json
{
    "instruction": "将以下句子翻译成英文",
    "input": "今天天气真好",
    "output": "The weather is really nice today."
}
```

三个字段的含义：

| 字段 | 含义 | 是否必填 |
|------|------|---------|
| `instruction` | 任务指令，告诉模型要做什么 | 必填 |
| `input` | 补充输入，比如要处理的文本 | 选填（没有就留空） |
| `output` | 期望的输出 | 必填 |

`input` 为空的例子：

```json
{
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支，它让计算机通过数据自动学习规律..."
}
```

**适用场景**：分类、翻译、摘要、问答等单轮任务。简单直接，大多数 SFT 框架都支持。

### ShareGPT 格式（多轮对话）

实际应用中，用户和模型不只说一轮话，而是连续对话。ShareGPT 格式专门解决多轮对话的表示问题：

```json
{
    "conversations": [
        {"from": "system", "value": "你是一个专业的Python编程助手。"},
        {"from": "human", "value": "帮我写一个快速排序"},
        {"from": "gpt", "value": "好的，这是一个Python实现的快速排序：\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)"},
        {"from": "human", "value": "能加上注释吗？"},
        {"from": "gpt", "value": "当然，这是加了注释的版本：\n\ndef quicksort(arr):\n    # 基线条件：空列表或只有一个元素，直接返回\n    if len(arr) <= 1:\n        return arr\n    # 选第一个元素作为基准\n    pivot = arr[0]\n    # 比基准小的放左边\n    left = [x for x in arr[1:] if x <= pivot]\n    # 比基准大的放右边\n    right = [x for x in arr[1:] if x > pivot]\n    # 递归排序左右两边，再拼起来\n    return quicksort(left) + [pivot] + quicksort(right)"}
    ]
}
```

关键点：
- `from` 字段标识说话角色：`system`（系统提示）、`human`（用户）、`gpt`（助手）
- `conversations` 数组里按时间顺序排列，`human` 和 `gpt` 交替出现
- 训练时，所有 `human` 和 `system` 部分不算 loss，只有 `gpt` 部分算 loss

**这是目前多轮对话 SFT 的事实标准**，LLaMA-Factory、FastChat 等主流框架都支持。

### Chat Template（对话模板）

不管用什么数据格式，最终喂给模型的都是一串 token。模型怎么知道"谁在说话"？靠**特殊 token**：

```
<|im_start|>system
你是一个专业的Python编程助手<|im_end|>
<|im_start|>user
帮我写一个快速排序<|im_end|>
<|im_start|>assistant
好的，这是一个Python实现的快速排序...
<|im_end|>
```

这些 `<|im_start|>` `<|im_end|>` 就是特殊 token，模型在预训练时就见过它们，知道遇到 `assistant` 标记后应该开始生成回答。

**不同模型有不同的模板**：

| 模型 | 角色标记方式 | 示例 |
|------|------------|------|
| Qwen | `<\|im_start\|>role` ... `<\|im_end\|>` | 上面的例子 |
| LLaMA 3 | `<\|start_header_id\|>role<\|end_header_id\|>` | `<\|start_header_id\|>user<\|end_header_id\|>你好` |
| ChatGLM | `[gMASK]<sop>` + 角色标记 | 自定义格式 |

**重要**：用哪个模型做 SFT，就必须用那个模型的 Chat Template。模板用错了，模型会很困惑——就像一个只懂中文语法的人突然看到法语语法，完全不知道从哪里开始回答。

好消息是，HuggingFace 的 tokenizer 自带 `apply_chat_template()` 方法，自动帮你处理：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"},
]

# 自动套用该模型的 Chat Template
text = tokenizer.apply_chat_template(messages, tokenize=False)
print(text)
# <|im_start|>system
# 你是一个助手<|im_end|>
# <|im_start|>user
# 你好<|im_end|>
# <|im_start|>assistant
```

### 本项目的数据格式

本项目 demo 用了一个简化格式，三个字段：`instruct`、`input`、`label`：

```json
{
    "instruct": "请你给哪吒写一首诗：",
    "input": "哪吒降世，意气飞扬。\n逆天改命，破障冲霄。",
    "label": "红绫缠腕，风火踏浪。\n不屈不悔，笑傲苍茫。"
}
```

对应关系：

| 本项目字段 | 对应 Alpaca 字段 | 作用 |
|-----------|-----------------|------|
| `instruct` | `instruction` | 任务指令 |
| `input` | `input` | 补充输入/上文 |
| `label` | `output` | 期望输出 |

训练时拼接成 `instruct + input + label + <eos>`，其中 `instruct + input` 部分设为 -100 不算 loss。具体实现见 `train.py`。

完整数据在 `data.json` 中，共 5 条哪吒角色诗歌样本。

---

## 二、数据质量（最重要！）

如果你只记住这篇文章的一件事，请记住：**数据质量是 SFT 效果的决定性因素**。

### 质量 > 数量

这不是空话，是无数实验验证过的结论：

```
1000 条高质量数据 > 10000 条低质量数据
```

什么叫"高质量"？

| 维度 | 要求 | 反例 |
|------|------|------|
| 准确性 | 回答内容正确无误 | "太阳从西边升起"（事实错误） |
| 多样性 | 涵盖不同问法、不同场景 | 1000 条数据全是"你好→你好" |
| 格式规范 | 统一的输出风格 | 有的用 Markdown，有的用纯文本 |
| 风格一致 | 语气、措辞保持一致 | 一会儿"您好"一会儿"你好呀~" |
| 完整性 | 回答完整，不截断 | 话说一半就没了 |
| 相关性 | 回答和问题对应 | 问天气，答美食 |

类比：你教新员工，给他看的范例文档如果错误百出、格式混乱、风格不统一，他学出来的工作质量能好吗？

### 数据清洗

拿到原始数据后，不能直接扔进去训练。必须先做清洗，就像做菜前要洗菜、择菜：

**1. 去除重复数据**

```python
import json
from collections import Counter

with open("data.json", "r") as f:
    data = json.load(f)

# 用 instruction 作为去重 key
seen = set()
unique_data = []
for item in data:
    key = item["instruct"] + item["input"]
    if key not in seen:
        seen.add(key)
        unique_data.append(item)

print(f"去重前: {len(data)} 条, 去重后: {len(unique_data)} 条")
```

**2. 过滤低质量数据**

```python
def is_low_quality(item):
    """判断是否为低质量数据"""
    # 回答太短（少于10个字可能信息量不足）
    if len(item["label"]) < 10:
        return True
    # 包含乱码
    if any(c in item["label"] for c in ["�", "\x00"]):
        return True
    # 指令为空
    if not item["instruct"].strip():
        return True
    return False

clean_data = [item for item in data if not is_low_quality(item)]
```

**3. 格式标准化**

```python
def normalize(item):
    """标准化格式"""
    return {
        "instruct": item["instruct"].strip(),
        "input": item["input"].strip(),
        # 统一去掉结尾的句号/换行
        "label": item["label"].strip().rstrip("。").rstrip("\n")
    }

normalized_data = [normalize(item) for item in clean_data]
```

**4. 去除敏感信息（PII）**

训练数据中混入手机号、身份证号、邮箱等个人信息是很危险的——模型可能在推理时把这些信息吐出来。

```python
import re

def remove_pii(text):
    """去除常见个人敏感信息"""
    # 手机号
    text = re.sub(r'1[3-9]\d{9}', '[手机号]', text)
    # 邮箱
    text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[邮箱]', text)
    # 身份证号
    text = re.sub(r'\d{17}[\dXx]', '[身份证号]', text)
    return text
```

### 常见质量问题及解决方案

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 回答太短 | "好的"、"是的" 这种敷衍回答 | 设置最小长度阈值，太短的丢弃或重新标注 |
| 格式不一致 | 有的用列表、有的用段落、有的用 Markdown | 制定格式规范，统一转换 |
| 标注错误 | 答非所问，事实错误 | 抽样人工检查，交叉验证 |
| 重复数据 | 大量相似或完全相同的样本 | 基于文本相似度去重（exact match + 语义去重） |
| 领域偏差 | 90% 都是闲聊，只有 10% 是专业问答 | 调整数据配比，欠采样多数类或过采样少数类 |
| 输入输出不匹配 | 指令说翻译，输出却是摘要 | 自动检查 + 人工抽查 |
| 幻觉数据 | 回答包含虚构的事实 | 引入事实核查环节 |

---

## 三、数据配比

实际项目中，训练数据通常来自多个领域。怎么混合这些数据，直接影响模型的能力分布。

### 问题：一个领域独大

假设你在做一个客服助手，收集了这些数据：

```
客服对话:  50000 条
通用问答:  2000 条
安全拒答:  500 条
```

如果直接全混在一起训练，模型会变成一个"只会客服话术"的模型——通用能力下降，安全意识也不够强。

### 推荐做法：按比例混合

```
客服数据:   60%    ← 主要能力
通用对话:   20%    ← 保持通用性
安全数据:   20%    ← 确保安全底线
```

具体实现很简单：

```python
import random

def mix_datasets(datasets, ratios, total_size):
    """
    按比例混合多个数据集
    datasets: [客服数据, 通用数据, 安全数据]
    ratios:   [0.6, 0.2, 0.2]
    total_size: 目标总量
    """
    mixed = []
    for dataset, ratio in zip(datasets, ratios):
        n = int(total_size * ratio)
        if len(dataset) >= n:
            mixed.extend(random.sample(dataset, n))
        else:
            # 数据不够就重复采样
            mixed.extend(random.choices(dataset, k=n))
    
    random.shuffle(mixed)
    return mixed

# 示例
mixed_data = mix_datasets(
    datasets=[customer_service, general_qa, safety],
    ratios=[0.6, 0.2, 0.2],
    total_size=10000
)
```

### 配比的经验法则

| 数据类型 | 建议占比 | 作用 |
|---------|---------|------|
| 核心任务数据 | 50-70% | 模型的主要能力来源 |
| 通用对话数据 | 15-25% | 防止灾难性遗忘，保持通用性 |
| 安全/拒答数据 | 10-20% | 确保模型不输出有害内容 |
| 格式数据 | 5-10% | 教模型正确使用 Markdown、JSON 等格式 |

注意：配比没有万能公式，需要根据实际效果调整。一个好的做法是先跑一版，看哪方面弱就加哪方面的数据。

---

## 四、合成数据

真实标注数据贵、慢、难以规模化。越来越多的团队开始用强模型（GPT-4、Claude 等）来生成训练数据。

这就像"以大带小"——用能力强的模型当老师，生成数据来教能力弱的模型。

### Self-Instruct：自己出题自己答

核心思路：用少量种子数据，让模型自动生成更多指令和回答。

```
种子数据（人写的，10-20 条）
      ↓
让模型参考种子数据，生成新的指令
      ↓
让模型回答这些新指令
      ↓
过滤低质量数据
      ↓
得到大量训练数据
```

### Evol-Instruct：让问题越来越难

WizardLM 提出的方法。不是简单地生成新问题，而是把简单问题**进化**成复杂问题：

```
原始指令: "写一个冒泡排序"
    ↓ 深度进化
进化后:  "写一个冒泡排序，要求支持自定义比较函数、
         能处理包含None的列表、时间复杂度要在注释中标明"
```

进化方向有很多：增加约束、增加步骤、更换领域、提高推理深度......

### 蒸馏：直接让强模型当老师

最简单粗暴的方法——直接拿强模型的输出当训练数据：

```python
from openai import OpenAI

client = OpenAI()

def generate_training_data(instruction):
    """用 GPT-4 生成训练数据"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是一个专业的助手，请详细、准确地回答问题。"},
            {"role": "user", "content": instruction}
        ],
        temperature=0.7
    )
    return {
        "instruction": instruction,
        "input": "",
        "output": response.choices[0].message.content
    }

# 批量生成
instructions = [
    "解释什么是梯度下降",
    "Python的GIL是什么？为什么它会影响多线程？",
    "用通俗的语言解释TCP三次握手",
    # ... 更多指令
]

training_data = []
for inst in instructions:
    item = generate_training_data(inst)
    training_data.append(item)
    print(f"已生成: {inst[:20]}...")

# 保存
with open("synthetic_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)
```

### 合成数据的陷阱

合成数据不是银弹，有几个要注意的坑：

| 陷阱 | 说明 | 应对方法 |
|------|------|---------|
| 模型幻觉 | 强模型也会编造事实，错误数据会"教坏"小模型 | 对关键事实做自动或人工核查 |
| 多样性不足 | 模型生成的数据风格趋同，缺乏变化 | 调高 temperature，用不同 prompt 引导 |
| 法律风险 | 部分模型的使用条款禁止用输出训练其他模型 | 确认模型的许可协议 |
| 分布偏移 | 合成数据的分布和真实场景不一致 | 混入一定比例的真实数据 |

**经验法则**：合成数据占比不要超过 70%，至少混入 30% 的真实标注数据，保证数据分布的真实性。

---

## 五、数据增强

数据增强的核心思想：在不改变语义的前提下，通过变换来扩充训练数据。比原始标注便宜得多，但效果显著。

### 1. 指令改写

同一个任务，换不同的说法：

```json
[
    {"instruction": "请帮我翻译这句话", "input": "今天天气不错", "output": "..."},
    {"instruction": "把下面的中文翻成英文", "input": "今天天气不错", "output": "..."},
    {"instruction": "翻译以下内容为英语", "input": "今天天气不错", "output": "..."},
    {"instruction": "将这段文字转成英文表达", "input": "今天天气不错", "output": "..."}
]
```

这样模型就不会只认一种指令说法，泛化能力大大增强。

### 2. 添加/变换系统提示

给同一条数据加上不同的 system prompt：

```json
{"system": "你是一个严谨的学术助手", "instruction": "解释梯度下降", ...}
{"system": "你是一个善于用类比的老师", "instruction": "解释梯度下降", ...}
{"system": "请用通俗易懂的语言回答", "instruction": "解释梯度下降", ...}
```

这能让模型学会根据 system prompt 调整回答风格。

### 3. 回译增强

先翻成另一种语言，再翻回来，得到同义但不同措辞的表达：

```
原文:     "机器学习是让计算机从数据中学习规律的技术"
→ 英文:   "Machine learning is a technology that enables computers to learn patterns from data"
→ 回译:   "机器学习是一种使计算机能够从数据中学习模式的技术"
```

措辞变了，但含义一样，相当于免费多了一条训练数据。

### 4. 组合增强

将以上方法组合使用，一条数据可以扩展出多条：

```python
def augment_data(item, num_variations=3):
    """对一条数据做增强"""
    augmented = [item]  # 保留原始数据
    
    # 改写指令
    instruction_variants = rewrite_instruction(item["instruction"], num_variations)
    for variant in instruction_variants:
        new_item = item.copy()
        new_item["instruction"] = variant
        augmented.append(new_item)
    
    return augmented
```

**注意**：增强不能过度。如果所有变体的含义都一样，模型还是只学到了一种知识，但训练时间增加了。增强的目标是增加**形式的多样性**，而不是无意义地重复。

---

## 六、数据量参考

"到底需要多少数据？"这是最常被问到的问题。答案取决于任务复杂度：

| 场景 | 推荐数据量 | 说明 |
|------|-----------|------|
| 简单分类（情感分析、意图识别） | 100 - 500 条 | 类别明确，模型容易学会 |
| 领域问答（客服、医疗咨询） | 1,000 - 5,000 条 | 需要覆盖常见问题和多种问法 |
| 通用助手（多领域综合能力） | 10,000 - 50,000 条 | 需要涵盖多个领域和能力维度 |
| 复杂推理（数学、代码、逻辑） | 50,000+ 条 | 推理链长、变化多，需要大量示例 |

几个关键参考：

- **斯坦福 Alpaca**：52K 条合成数据，就让 LLaMA-7B 有了不错的指令跟随能力
- **LIMA 论文**：仅 1000 条精选高质量数据，效果超过大量低质量数据，证明了"质量 > 数量"
- **本项目 demo**：5 条数据，只能死记硬背（但足以演示 SFT 的原理）

**实际建议**：先用少量数据（几百条）跑通流程，验证效果，然后再逐步扩充。不要一上来就去标注 10 万条数据，万一方向错了全白费。

---

## 七、数据准备 Checklist

训练前对照这个清单检查一遍，能避免很多低级错误：

**格式检查：**
- [ ] 数据格式正确（JSON 合法，字段名无误）
- [ ] 编码统一（建议全部使用 UTF-8）
- [ ] 特殊字符处理妥当（换行符、引号、转义符）
- [ ] 文本长度在模型的最大上下文长度范围内

**质量检查：**
- [ ] 抽样 50-100 条人工检查，确认回答质量
- [ ] 去除重复数据（exact match + 近似去重）
- [ ] 过滤掉过短/过长/乱码/无意义的样本
- [ ] 回答内容准确无事实错误
- [ ] 无敏感个人信息（手机号、身份证号、地址等）

**分布检查：**
- [ ] 统计各类别/领域的数据量分布，是否均衡
- [ ] 检查指令的多样性，避免过多重复模式
- [ ] 如有多领域数据，确认配比合理

**一致性检查：**
- [ ] 输出格式统一（比如全用 Markdown 或全用纯文本）
- [ ] 语气风格一致（"您好" vs "你好"，选一种）
- [ ] 对同类问题的回答逻辑一致（不能自相矛盾）

**工程检查：**
- [ ] 训练集/验证集/测试集已拆分（推荐 8:1:1 或 9:0.5:0.5）
- [ ] 拆分时确保同一用户/同一会话的数据不跨集（防止数据泄漏）
- [ ] 数据加载脚本能正确读取并解析
- [ ] tokenize 后的长度分布已统计，超长样本已处理（截断或丢弃）

```python
# 快速统计 token 长度分布
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

lengths = []
for item in data:
    text = item["instruct"] + item["input"] + item["label"]
    tokens = tokenizer(text)["input_ids"]
    lengths.append(len(tokens))

print(f"最短: {min(lengths)}, 最长: {max(lengths)}, 平均: {sum(lengths)/len(lengths):.0f}")
print(f"超过 512 token 的: {sum(1 for l in lengths if l > 512)} 条")
```

---

## 小结

```
数据工程要点：

1. 格式要对    → Alpaca（单轮）、ShareGPT（多轮）、Chat Template（模型专属）
2. 质量为王    → 1000 条好数据胜过 10000 条烂数据
3. 配比合理    → 别让一个领域霸占所有比例
4. 善用合成    → 强模型生成数据，但要注意幻觉和多样性
5. 适度增强    → 改写指令、变换 prompt、回译
6. 数据量适中  → 先少量验证再逐步扩充
7. 上线前检查  → 对照 Checklist 逐项过
```

**下一步**：数据准备好了，就可以开始训练了。前往 `03-全量微调` 目录学习如何用这些数据微调模型。
