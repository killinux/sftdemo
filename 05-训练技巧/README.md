# 训练技巧：从能跑到跑好

前面我们跑通了 SFT 训练，模型能学会背诗了。但在真实项目中，"能跑"只是第一步，"跑好"才是关键。

这一章讲的就是那些让训练效果从 60 分提升到 90 分的实战技巧。

---

## 一、混合精度训练（FP16 / BF16）

### 什么是精度

计算机存数字要占空间。精度越高，数字越精确，但占的空间也越大：

```
FP32: 占 4 字节，精度高，速度慢        ← 默认精度，什么都好就是太占地方
FP16: 占 2 字节，精度够用，但数值范围小（容易溢出）  ← 省空间但有风险
BF16: 占 2 字节，数值范围和 FP32 一样，精度略低    ← 推荐用这个
```

类比：

- FP32 就像用**尺子量到毫米**——非常精确，但量起来很慢
- FP16 就像用**手指比划**——快是快，但太大或太小的东西量不了（溢出）
- BF16 就像用**卷尺量到厘米**——速度快，什么尺寸都能量，就是没那么精确

### 三者对比

```
类型     字节数    数值范围              精度         训练速度
─────────────────────────────────────────────────────────────
FP32     4 字节    ±3.4×10^38          最高         慢（基准）
FP16     2 字节    ±6.5×10^4           较高         快（~2x）
BF16     2 字节    ±3.4×10^38（同FP32） 略低于FP16   快（~2x）
```

注意 FP16 的数值范围只到 6.5 万，训练中梯度如果超过这个值就会变成 `inf`（溢出），导致训练崩溃。BF16 保持了 FP32 的数值范围，所以不会溢出。

### 为什么 SFT 几乎都用 BF16

三个理由，个个都很实在：

**1. 显存直接砍半**

```
7B 模型参数占用:
  FP32: 7B × 4 字节 = 28 GB
  BF16: 7B × 2 字节 = 14 GB     ← 省了 14 GB！
```

本来一张 GPU 放不下的模型，换成 BF16 就能放下了。

**2. 训练速度快 ~2 倍**

现代 GPU（A100、4090 等）有专门的半精度计算单元，BF16 运算速度几乎是 FP32 的两倍。

**3. 效果几乎不受影响**

大量实验证明，BF16 训练出来的模型质量和 FP32 几乎一样。毕竟我们调的是几十亿个参数，每个参数差那么一丁点精度，对最终结果影响微乎其微。

### 怎么开启

```python
# 如果你的 GPU 支持 BF16（A100, H100, 4090, 3090 等较新的卡）
TrainingArguments(bf16=True)

# 如果你的 GPU 不支持 BF16（V100, 2080 等较老的卡）
TrainingArguments(fp16=True)
```

怎么知道自己的 GPU 支不支持 BF16？跑一行代码：

```python
import torch
print(torch.cuda.is_bf16_supported())  # True 就支持
```

> **一句话总结：BF16 是 SFT 训练的标配，省显存、速度快、效果不打折，没有理由不开。**

---

## 二、梯度累积（Gradient Accumulation）

### 问题：显存不够放大 batch

理想情况下，我们希望用 batch size=32 来训练（梯度更稳定，收敛更快）。但 GPU 显存只够放 batch size=2，怎么办？

### 解决方案：分多次攒梯度

核心思路：跑 16 次小 batch，把梯度攒起来，最后一起更新参数。效果等价于跑 1 次大 batch。

```
梯度累积步数 = 16
实际 batch size = 2（物理） × 16（累积） = 32（等效）

Step 1:  batch_size=2, 算梯度, 先存着不更新
Step 2:  batch_size=2, 算梯度, 累加到之前的梯度上
Step 3:  batch_size=2, 算梯度, 继续累加
...
Step 16: batch_size=2, 算梯度, 累加完毕, 一起更新参数！
```

画个图更直观：

```
不用梯度累积（显存不够，跑不了）:
  ┌──────────── batch=32 ────────────┐
  │ 样本1 样本2 样本3 ... 样本32      │ → 算梯度 → 更新参数
  └──────────────────────────────────┘

用梯度累积（显存够了）:
  ┌─ batch=2 ─┐
  │ 样本1 样本2│ → 算梯度, 攒着
  ├─ batch=2 ─┤
  │ 样本3 样本4│ → 算梯度, 累加
  ├─ batch=2 ─┤
  │ ...       │ → ...
  ├─ batch=2 ─┤
  │ 样本31 32 │ → 算梯度, 累加, 然后更新参数！
  └───────────┘
```

### 类比：搬砖

你要把 32 块砖从 A 搬到 B：

- **batch=32**（不用累积）：一次搬 32 块 —— 搬不动，砖太重了（显存不够）
- **batch=2 + 累积 16 步**：每次搬 2 块，跑 16 趟 —— 虽然慢一点，但搬完了效果一样

### 代码

```python
TrainingArguments(
    per_device_train_batch_size=2,       # 每次实际放进 GPU 的样本数
    gradient_accumulation_steps=16,      # 累积 16 步再更新
    # 等效 batch_size = 2 × 16 = 32
)
```

### 注意事项

```
等效 batch size = per_device_batch_size × gradient_accumulation_steps × GPU数量

单卡: 2 × 16 × 1 = 32
4卡: 2 × 16 × 4 = 128   ← 多卡训练时要注意调整累积步数
```

> **一句话总结：显存不够？梯度累积来凑。小 batch 跑多次 = 大 batch 跑一次。**

---

## 三、学习率调度器（LR Scheduler）

### 为什么不用固定学习率

固定学习率有个两难问题：

```
学习率太大:  训练不稳定，loss 上蹿下跳          → 像喝醉酒找路，走两步退三步
学习率太小:  收敛太慢，训练半天 loss 不动         → 像蜗牛爬，爬到天亮也到不了
学习率刚好:  前期嫌慢，后期嫌大（快到终点了还大步走，容易走过头）
```

所以我们需要**动态调整学习率**——开始慢慢加速，然后逐渐减速。就像开车：

```
起步 → 慢慢加油（warmup）→ 上高速巡航 → 快到了减速 → 停车
```

### Warmup（预热）

```
学习率
  ^
  │         ╱─────────...
  │       ╱
  │     ╱
  │   ╱
  │ ╱
  │╱
  └──────────────────────→ 训练步数
  0    warmup结束     ...

训练开始时，学习率从 0 慢慢升到目标值
```

为什么需要 warmup？

训练刚开始的时候，模型参数还是预训练的状态，梯度方向不太靠谱。如果一上来就用大学习率，容易把预训练学到的好参数直接改坏。先用小学习率"热热身"，等梯度方向稳定了再加大步伐。

类比：你刚起床就跑百米冲刺，容易拉伤（梯度爆炸）。先热身几分钟（warmup），再跑才安全。

### 常见调度策略

**1. Linear（线性衰减）**

warmup 之后，学习率匀速降到 0：

```
学习率
  ^
  │     ╱╲
  │   ╱    ╲
  │  ╱       ╲
  │╱           ╲
  │              ╲
  └────────────────╲──→ 训练步数
  0  warmup        结束
```

特点：简单直接，适合训练步数确定的场景。

**2. Cosine（余弦衰减）--- SFT 最常用**

warmup 之后，学习率按余弦曲线平滑下降：

```
学习率
  ^
  │     ╱╲
  │   ╱    ╲
  │  ╱       ╲
  │╱           ──╲
  │                ──╲
  └────────────────────╲→ 训练步数
  0  warmup             结束
```

特点：前期下降慢（充分学习），后期下降快（精细收敛），效果通常最好。

**3. Constant with Warmup（常数+预热）**

warmup 之后，学习率保持不变：

```
学习率
  ^
  │     ╱──────────────────
  │   ╱
  │  ╱
  │╱
  │
  └──────────────────────→ 训练步数
  0  warmup              结束
```

特点：最简单，适合数据量小、训练步数少的情况（比如本 demo）。

### 代码

```python
TrainingArguments(
    learning_rate=2e-5,              # 目标学习率
    warmup_steps=100,                # warmup 步数
    # 或者用 warmup_ratio=0.1,      # warmup 占总步数的 10%
    lr_scheduler_type="cosine",      # 余弦衰减，SFT 最常用
)
```

常用的 `lr_scheduler_type` 选项：

```
"linear"              线性衰减
"cosine"              余弦衰减（推荐）
"constant"            恒定不变
"constant_with_warmup" 预热后恒定
"cosine_with_restarts" 余弦衰减+重启（周期性回升）
```

> **一句话总结：用 cosine 调度 + warmup，让学习率先升后降，训练更稳定、效果更好。**

---

## 四、过拟合 vs 欠拟合

这是训练中最核心的判断——你的模型是"学得太死"还是"没学到"。

### 怎么判断

```
               train_loss    eval_loss    诊断          该怎么办
───────────────────────────────────────────────────────────────────
正常:           ↓             ↓           都在下降       继续训练
过拟合:         ↓↓            ↑           train降eval升  该停了！
欠拟合:         →             →           都不降         模型没学到
学得刚好:       ↓(稳定)       ↓(稳定)     两者都收敛     可以结束了
```

用图来看更直观：

```
Loss
  ^
  │╲  ╱ eval_loss（开始上升 = 过拟合！）
  │  ╲╱
  │  ╱╲
  │╱    ╲───── train_loss（一直在降）
  │
  └──────────────────→ 训练步数
        ↑
    最佳停止点
```

### 过拟合的信号

过拟合 = 模型把训练数据背住了，但遇到新数据就不会了。

```
训练数据: "1+1=?" → "2"    ← 记住了
训练数据: "2+3=?" → "5"    ← 记住了
新数据:   "3+4=?" → "？"   ← 没见过，不会！
```

就像本 demo 的情况：5 条诗歌数据全部精确背住了（loss=0.0003），但换个说法就答不上来。

过拟合的典型表现：

- train_loss 很低，但 eval_loss 反而在上升
- 训练集上回答完美，测试集上乱说一通
- loss 降到极低值（比如 0.001 以下），模型很可能在死记硬背

### 解决过拟合

按优先级排列：

```
方法                  原理                           难度    效果
──────────────────────────────────────────────────────────────────
加数据                见过更多样本，不容易死记硬背       高     最好
减少 epoch            少训几轮，别背太死               低     立竿见影
Early stopping        eval_loss 不降了就停             低     非常实用
LoRA                  冻住大部分参数，不容易过拟合       中     推荐
Dropout               随机关闭一些神经元，防止死记硬背   低     有效
Weight decay          惩罚过大的参数值                 低     标配
减小学习率            调参数幅度更小，改动更谨慎         低     有效
```

### 解决欠拟合

欠拟合 = 模型没学到东西，train_loss 一直降不下去。

```
方法                  原理
─────────────────────────────────────────────────
换更大的模型           0.5B 学不会的，7B 可能学得会
加大学习率            步子迈大一点，学快一点
增加 epoch            多训几轮，给模型更多学习时间
检查数据质量           数据本身有问题，模型学不了
```

### Early Stopping（早停）

最实用的防过拟合技巧：监控 eval_loss，如果连续几轮不降了，就停止训练。

```
Epoch 1: eval_loss = 2.5
Epoch 2: eval_loss = 1.8    ← 在降，继续
Epoch 3: eval_loss = 1.2    ← 在降，继续  ★ 记录为最佳模型
Epoch 4: eval_loss = 1.3    ← 升了！patience 计数 = 1
Epoch 5: eval_loss = 1.5    ← 又升了！patience 计数 = 2，达到上限
                                → 停止训练，回滚到 Epoch 3 的参数
```

`patience`（耐心值）= 允许 eval_loss 连续不改善的轮数。通常设 2~3。

```python
from transformers import EarlyStoppingCallback

TrainingArguments(
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,       # 训练结束后加载最佳模型
    metric_for_best_model="eval_loss", # 用 eval_loss 判断最佳
)

Trainer(
    ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
```

> **一句话总结：看 train_loss 和 eval_loss 的走势——同降是正常，train 降 eval 升是过拟合，都不降是欠拟合。**

---

## 五、验证集的重要性

### 本 demo 的问题

回看我们的 `train.py`：

```python
trainer = Trainer(
    ...
    eval_dataset=None,    # ← 没有验证集！
)
```

没有验证集意味着：

- 训练过程中**看不到 eval_loss**，不知道模型有没有过拟合
- **Early stopping 无法使用**，不知道什么时候该停
- 只能凭 train_loss 判断，而 train_loss 永远在降（即使模型已经过拟合了）

这就像考试只看平时作业成绩，不做模拟考 —— 你觉得学得很好，一上考场就傻了。

### 正确做法：划分数据集

```
全部数据
  │
  ├── 训练集（80%）   ← 用来训练模型
  │
  ├── 验证集（10%）   ← 训练过程中监控，决定什么时候停
  │
  └── 测试集（10%）   ← 训练结束后最终评估，之前不能碰
```

类比：

```
训练集 = 课后练习题        ← 平时做的，用来学习
验证集 = 模拟考            ← 定期测试，看学得怎么样，决定要不要继续学
测试集 = 高考              ← 最终考核，只考一次，不能提前看
```

### 代码示例

```python
from sklearn.model_selection import train_test_split

# 划分数据集
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"训练集: {len(train_data)} 条")
print(f"验证集: {len(eval_data)} 条")
print(f"测试集: {len(test_data)} 条")

# 创建 Dataset
train_dataset = PreTrainDataset(train_data)
eval_dataset = PreTrainDataset(eval_data)

# 训练时传入验证集
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./sft_output",
        evaluation_strategy="epoch",    # 每个 epoch 评估一次
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,          # ← 传入验证集
    data_collator=DataCollatorForSFTDataset(tokenizer=tokenizer),
)
```

> **一句话总结：验证集是训练的"后视镜"，没有它你根本不知道模型有没有过拟合。**

---

## 六、Weight Decay（权重衰减）

### 是什么

训练过程中，有些参数会变得特别大。参数太大意味着模型对某些特征"过度依赖"，容易过拟合。

Weight decay 的做法很简单：每次更新参数时，**额外把所有参数往 0 的方向缩一点**。

```
没有 weight decay:
  新参数 = 旧参数 - 学习率 × 梯度

有 weight decay:
  新参数 = 旧参数 × (1 - weight_decay) - 学习率 × 梯度
                    ↑
              每次都往 0 缩一点，防止参数长太大
```

类比：你种了一棵树，不修剪的话枝叶会疯长（参数变大），定期修剪（weight decay）让它长得更健康（不过拟合）。

### 代码

```python
TrainingArguments(
    weight_decay=0.01,   # 典型值，一般 0.01 ~ 0.1
)
```

这个值不用特别调，0.01 是大多数 SFT 任务的默认推荐值。

> **一句话总结：Weight decay 就是给参数"减肥"，防止某些参数长太胖导致过拟合。**

---

## 七、Batch Size 选择

Batch size 对训练效果影响很大，但很多人只关注"显存能放多少"而忽略了它对训练质量的影响。

### Batch Size 的影响

```
太小（batch=1~2）:
  ┌─┐ ┌─┐ ┌─┐ ┌─┐     每次只看1~2个样本
  │↗│ │↙│ │↖│ │↘│     梯度方向跳来跳去（噪声大）
  └─┘ └─┘ └─┘ └─┘     收敛慢，训练不稳定

适中（batch=16~64）:
  ┌──────────┐          每次看一批样本
  │   →→→    │          梯度方向比较稳定
  └──────────┘          收敛快，训练稳定  ← 推荐

太大（batch=256+）:
  ┌────────────────────┐  每次看很多样本
  │      →             │  梯度方向太稳定了
  └────────────────────┘  可能收敛到不好的局部最优解
                          而且显存占用巨大
```

类比：

- **batch=1**：只问一个人的意见就做决定 —— 太片面，决策不稳定
- **batch=32**：问 32 个人的意见再做决定 —— 比较靠谱
- **batch=1024**：问 1024 个人的意见 —— 太保守，容易走向"平庸"的方案

### SFT 推荐 Batch Size

```
场景              推荐等效 batch size
────────────────────────────────────
小数据(<1000条)    8 ~ 16
中等数据           16 ~ 32
大数据(>10000条)   32 ~ 64
```

记住：这里说的是**等效 batch size**（物理 batch size × 梯度累积步数 × GPU 数）。

GPU 放不下就用梯度累积：

```python
# 显存只够 batch=4，但想要等效 batch=32
TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,    # 4 × 8 = 32
)
```

> **一句话总结：SFT 的 batch size 推荐 16~64，显存不够就用梯度累积凑。**

---

## 八、训练参数推荐配置

不同数据规模下的推荐参数：

### 参数速查表

| 参数 | 小数据（<1000条） | 中等数据（1000~10000条） | 大数据（>10000条） |
|------|-------------------|--------------------------|---------------------|
| **learning_rate** | 1e-5 ~ 2e-5 | 2e-5 ~ 5e-5 | 2e-5 ~ 1e-4 |
| **num_train_epochs** | 3 ~ 5 | 2 ~ 3 | 1 ~ 2 |
| **等效 batch_size** | 8 ~ 16 | 16 ~ 32 | 32 ~ 64 |
| **warmup_ratio** | 0.1（10%） | 0.05 ~ 0.1 | 0.03 ~ 0.05 |
| **weight_decay** | 0.01 ~ 0.1 | 0.01 | 0.01 |
| **lr_scheduler** | cosine | cosine | cosine |
| **bf16** | True | True | True |
| **过拟合风险** | 高（数据少） | 中等 | 低 |
| **重点关注** | 防止过拟合 | 平衡学习与泛化 | 训练效率 |

### 小数据配置示例（<1000条）

数据少，最怕过拟合。策略：小学习率 + 少 epoch + 强正则化。

```python
TrainingArguments(
    output_dir="./sft_output",
    num_train_epochs=3,                    # 少训几轮，别背太死
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,         # 等效 batch=16
    learning_rate=1e-5,                    # 小学习率，温柔地调
    warmup_ratio=0.1,                      # 10% 预热
    lr_scheduler_type="cosine",
    weight_decay=0.05,                     # 稍大的正则化
    bf16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
)
```

### 中等数据配置示例（1000~10000条）

数据量适中，标准配置即可。

```python
TrainingArguments(
    output_dir="./sft_output",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,         # 等效 batch=32
    learning_rate=2e-5,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=True,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    logging_steps=10,
)
```

### 大数据配置示例（>10000条）

数据多，过拟合风险低，可以用更大的学习率加速训练。

```python
TrainingArguments(
    output_dir="./sft_output",
    num_train_epochs=1,                    # 数据多，1~2轮就够
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,         # 等效 batch=32
    learning_rate=5e-5,                    # 可以稍大
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=True,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,                    # 只保留最新3个checkpoint，省硬盘
    load_best_model_at_end=True,
    logging_steps=10,
)
```

### 与本 Demo 对比

我们的 `train.py` 用的是最简配置：

```python
# 本 demo 的配置（能跑，但不够好）
TrainingArguments(
    output_dir="./sft_output",
    num_train_epochs=10,              # 10轮太多，过拟合了
    per_device_train_batch_size=2,    # 没用梯度累积
    logging_strategy="epoch",         # 没有 eval
    # 缺少: bf16, warmup, weight_decay, lr_scheduler, eval_dataset...
)

# 改进后的配置
TrainingArguments(
    output_dir="./sft_output",
    num_train_epochs=3,               # 减少轮数
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,    # 等效 batch=8
    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=True,                        # 开启混合精度
    evaluation_strategy="epoch",      # 每轮评估
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=1,
)
```

---

## 九、调参的一般原则

最后总结一些实战中的调参经验：

### 优先级

不是所有参数都一样重要，按影响大小排序：

```
影响从大到小:
1. 数据质量和数量          ← 最重要！垃圾数据怎么调参都没用
2. 学习率                  ← 最敏感的超参数
3. 训练轮数（epoch）       ← 直接决定有没有过拟合
4. Batch size              ← 影响训练稳定性
5. Warmup / Scheduler      ← 锦上添花
6. Weight decay            ← 微调，通常用默认值就行
```

### 调参口诀

```
数据为王:        数据质量 > 数据数量 > 模型大小 > 训练技巧
先跑通再调优:     先用默认参数跑通，再一个个调
一次只改一个:     同时改多个参数，分不清哪个有用
看 loss 曲线:    train_loss 和 eval_loss 是最重要的信号
```

### 常见问题排查

```
问题                        可能原因                 解决方案
────────────────────────────────────────────────────────────────
loss 不降                   学习率太小 / 数据有问题    加大学习率 / 检查数据
loss 跳来跳去               学习率太大 / batch太小     减小学习率 / 加大batch
train_loss降 eval_loss升    过拟合                    减epoch / 加数据 / 早停
loss 变 NaN                 学习率太大 / 数值溢出      减小学习率 / 用BF16
训练很慢                    没开混合精度 / batch太小   开BF16 / 加大batch
显存不够(OOM)               batch太大 / 模型太大      减小batch / 用梯度累积 / 用LoRA
```

---

## 总结

```
训练技巧一览:

┌─────────────────────────────────────────────────────────┐
│  混合精度 (BF16)     → 省显存、加速度、必须开            │
│  梯度累积            → 显存不够时凑大 batch               │
│  学习率调度 (Cosine)  → 先升后降，训练更稳定              │
│  过拟合/欠拟合判断    → 看 train_loss vs eval_loss        │
│  验证集              → 训练的后视镜，必须有                │
│  Weight Decay        → 给参数减肥，防过拟合               │
│  Batch Size          → 16~64 适中，太大太小都不好          │
│  Early Stopping      → eval_loss 不降了就停               │
└─────────────────────────────────────────────────────────┘

记住：好的训练 = 好的数据 + 合理的参数 + 及时停止
```
