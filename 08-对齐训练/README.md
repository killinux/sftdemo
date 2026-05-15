# 对齐训练：让模型不仅听话，还要回答得好

## 一、大模型训练全链路

在前面的章节中，我们学会了用 SFT 让模型"听话"。但你有没有想过一个问题：**模型听话了，回答的质量就一定好吗？**

打个比方：SFT 就像教一个新员工"领导问什么你就答什么"，但没教他**怎么答才算好**。你问他"如何提高团队效率？"，他可能回答"提高团队效率很重要，你应该好好提高"——格式对了，但内容是废话。

这就是对齐训练要解决的问题：**教模型区分什么是好回答，什么是差回答**。

```
预训练 → SFT → 对齐训练（RLHF / DPO / GRPO）
学语言    学听话   学什么回答更好
```

### 为什么 SFT 不够？

| 能力 | SFT 能做到 | SFT 做不到 |
|------|-----------|-----------|
| 按指令回答 | 能 | - |
| 输出格式正确 | 能 | - |
| 区分好答案和差答案 | - | 不能 |
| 拒绝有害问题 | 部分能 | 很难全面覆盖 |
| 回答更有深度 | - | 不能主动优化 |

SFT 的训练信号是"模仿标准答案"，它没有"比较"的概念。就像一个学生只会背范文，但不知道为什么这篇范文比另一篇好。对齐训练就是要补上这个"比较和判断"的能力。

---

## 二、强化学习基础：DQN 与 PPO

RLHF 里的 RL = Reinforcement Learning（强化学习）。在看 RLHF 之前，先搞懂两个最重要的 RL 算法。

### 强化学习的基本框架

```
智能体（Agent）在环境中行动，获得奖励，目标是让总奖励最大化

        ┌───────────┐
        │   环境     │
        │ (Environment)│
        └─────┬─────┘
    观察状态 ↓   ↑ 执行动作
        ┌─────┴─────┐
        │   智能体    │  ← 收到奖励，调整策略
        │  (Agent)   │
        └───────────┘
```

类比：训狗。狗（智能体）做了一个动作 → 你给它一块饼干（奖励）→ 狗学会了多做这个动作。

### DQN（Deep Q-Network）

**核心思想：学一个"打分表"——给每个动作打分，选分最高的。**

```
状态: 游戏画面
动作: 上/下/左/右（只有4个选择）

Q表:
  Q(当前画面, 上) = 3.2 分
  Q(当前画面, 下) = 1.1 分  → 选"上"（分最高）
  Q(当前画面, 左) = 0.5 分
  Q(当前画面, 右) = 2.8 分
```

DQN 用神经网络来近似这个 Q 表（所以叫 Deep Q-Network）。

经典应用：DeepMind 用 DQN 打 Atari 游戏，超越人类水平。

**为什么 DQN 不适合 LLM？**

```
Atari 游戏: 动作空间 = 上/下/左/右/开火 → 几个到几十个动作
语言模型:   动作空间 = 从词表中选下一个词 → 15 万个动作！

每一步都要给 15 万个词打分，每个词的最优分数还取决于前面所有的词
→ Q 表爆炸，根本存不下、算不完
```

### PPO（Proximal Policy Optimization）

**核心思想：不给动作打分了，直接学一个"策略"——输出每个动作的概率。**

```
DQN:  状态 → Q(动作1)=3.2, Q(动作2)=1.1, ... → 选分最高的
PPO:  状态 → P(动作1)=60%, P(动作2)=5%, ...   → 按概率采样
```

关键发现：语言模型本身就是一个策略！给它一段文字，它输出每个词的概率——这不就是"给定状态，输出动作概率分布"吗？

```
LLM 就是一个策略（Policy）:
  状态 = "请你给哪吒写一首诗：哪吒降世，意气飞扬。"
  动作 = 从词表中选下一个词
  策略 = 模型输出的概率分布：P(红)=12%, P(风)=8%, P(的)=3%, ...
```

PPO 的核心约束——**Proximal（近端）**：每次更新不能跨太大步。

```
为什么要限制步幅？

想象你在山上找最高点：
  大步走: 可能一步跨过山顶，掉到另一边去了
  小步走: 慢但稳，能准确到达顶峰

PPO 就是"小步走"——每次更新限制策略变化幅度，防止模型突然跑偏
```

### DQN vs PPO 对比

| | DQN | PPO |
|------|-----|-----|
| 学什么 | Q 值表（每个动作的分数） | 策略（动作的概率分布） |
| 怎么选动作 | 选分最高的 | 按概率采样 |
| 动作空间 | 小（几个到几十个） | 大（几万到几十万） |
| 用于 LLM | 不适合（词表太大） | **RLHF 的标准算法** |
| 代表应用 | Atari 游戏 | ChatGPT、Claude |

**一句话：DQN 给每个动作打分选最高的，PPO 直接调整概率让好动作更可能发生。LLM 用 PPO，因为词表太大没法给每个词都打分。**

---

## 三、RLHF（基于人类反馈的强化学习）

RLHF（Reinforcement Learning from Human Feedback）是最早被广泛使用的对齐方法，ChatGPT 和早期 Claude 都用了这个技术。

### 核心思路

RLHF 的逻辑非常直觉：

1. 让模型对同一个问题生成多个回答
2. 让人类标注员给这些回答排序（哪个好，哪个差）
3. 用排序数据训练一个"奖励模型"（Reward Model），让它学会自动打分
4. 用强化学习（PPO）让模型朝着高分方向优化

### 完整流程图

```
                    ┌──────────────────────────────────────────────────┐
                    │              RLHF 完整流程                       │
                    └──────────────────────────────────────────────────┘

SFT模型 → 生成多个回答 → 人工排序 → 训练奖励模型(RM) → PPO强化学习 → 对齐模型
                │                         │                    │
                │                         │                    │
           同一个问题              "回答A > 回答B"         最大化奖励分数
           生成3-5个回答           学习人的偏好            同时别跑太远
```

### 奖励模型（Reward Model）

奖励模型的本质就是一个**打分器**。

想象你请了一位经验丰富的老师，给学生的作文打分。这位老师看了成千上万篇范文的排名后，学会了判断什么样的文章是好文章。以后新学生交作文，老师不用排序，直接打个分就行。

```python
# 奖励模型的伪代码
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model        # 用预训练模型做骨干
        self.reward_head = nn.Linear(hidden_size, 1)  # 输出一个分数

    def forward(self, prompt, response):
        hidden = self.backbone(prompt + response)
        score = self.reward_head(hidden[:, -1, :])  # 取最后一个token的表示
        return score  # 返回一个标量分数，越高越好
```

训练奖励模型时，用的是**排序损失**：

```python
# 排序损失：让好回答的分数 > 差回答的分数
def reward_loss(score_chosen, score_rejected):
    return -torch.log(torch.sigmoid(score_chosen - score_rejected))
```

### PPO（近端策略优化）

PPO 是 RLHF 中使用的强化学习算法。它要解决一个微妙的平衡问题：

- **目标**：让模型获得更高的奖励分数
- **约束**：不能跟 SFT 模型差太远（用 KL 散度惩罚）

为什么要有这个约束？因为如果只追求高分，模型可能学会"作弊"——比如它发现奖励模型对某些模式给高分，就拼命生成那种模式，结果输出变得很奇怪。

```python
# PPO 的核心目标函数（简化版）
def ppo_objective(model, ref_model, reward_model, prompt):
    response = model.generate(prompt)
    reward = reward_model(prompt, response)

    # KL 散度惩罚：不要偏离 SFT 模型太远
    kl_penalty = kl_divergence(model(prompt), ref_model(prompt))

    # 最终目标 = 奖励 - KL惩罚
    objective = reward - beta * kl_penalty
    return objective
```

这里的 `beta` 是一个超参数，控制"追求高分"和"保持稳定"之间的平衡。beta 太小，模型可能跑偏；beta 太大，模型学不到什么新东西。

### RLHF 的缺点

说实话，RLHF 虽然效果好，但**工程上是个噩梦**：

1. **成本高**：需要同时维护 4 个模型（SFT模型、奖励模型、PPO策略模型、参考模型），显存需求巨大
2. **训练不稳定**：PPO 有大量超参数（学习率、KL系数、clip范围等），调参非常痛苦
3. **需要大量人工标注**：人工排序数据既贵又慢
4. **奖励模型可能有偏差**：奖励模型学到的偏好可能不完全准确，导致"奖励黑客"现象

> 代表作品：ChatGPT、早期 Claude

---

## 四、DPO（直接偏好优化）

### 关键洞察

2023 年，斯坦福大学的研究者发现了一件了不起的事：**RLHF 的数学目标可以被简化成一个简单的分类问题**。

这意味着什么？不需要训练奖励模型了！不需要 PPO 了！直接从偏好数据训练就行！

这就好比：以前你要先请一个评委（奖励模型），再让选手根据评委的反馈练习（PPO）。现在发现，直接把"好作品"和"差作品"拿给选手看，让他自己学就行了，效果一样好。

### 数据格式

DPO 的训练数据非常简单，就是"偏好对"：

```json
{
  "prompt": "如何学编程？",
  "chosen": "推荐从Python开始。首先学习基础语法（变量、循环、函数），可以用《Python编程：从入门到实践》这本书。然后做小项目练手，比如写一个计算器、爬虫。遇到问题多查文档和Stack Overflow。坚持每天写代码，3个月就能入门。",
  "rejected": "编程很重要，你应该好好学习，加油！相信你一定可以的！"
}
```

```json
{
  "prompt": "解释什么是机器学习",
  "chosen": "机器学习就像教小孩认东西。你给他看很多猫的照片，告诉他'这是猫'。看多了之后，他看到新的猫也能认出来。机器学习的原理类似：给计算机大量数据和标签，让它自己找出规律，之后就能对新数据做预测。",
  "rejected": "机器学习是人工智能的一个子领域，它使用统计学方法使计算机系统能够从数据中学习并改进其性能，而无需进行显式编程。机器学习的主要类型包括监督学习、无监督学习和强化学习。"
}
```

注意第二个例子：rejected 的回答并不是错的，但它太"教科书"了，不够通俗易懂。DPO 就是在训练模型学习这种细微的偏好差异。

### DPO 的损失函数

```python
import torch
import torch.nn.functional as F

def dpo_loss(model, ref_model, chosen, rejected, beta=0.1):
    """
    DPO 损失函数

    核心思想：
    - 增大好回答的生成概率
    - 减小差回答的生成概率
    - 同时参考SFT模型，不要偏离太远
    """
    # 计算当前模型对好/差回答的对数概率
    log_prob_chosen = model.log_prob(chosen)
    log_prob_rejected = model.log_prob(rejected)

    # 计算参考模型（SFT模型）对好/差回答的对数概率
    with torch.no_grad():
        ref_log_prob_chosen = ref_model.log_prob(chosen)
        ref_log_prob_rejected = ref_model.log_prob(rejected)

    # DPO 的核心公式
    # 直觉：好回答的"相对概率提升"应该大于差回答的
    chosen_reward = beta * (log_prob_chosen - ref_log_prob_chosen)
    rejected_reward = beta * (log_prob_rejected - ref_log_prob_rejected)

    # 二分类交叉熵风格的损失
    loss = -F.logsigmoid(chosen_reward - rejected_reward)

    return loss.mean()
```

### DPO vs RLHF 对比

| 对比项 | RLHF | DPO |
|--------|------|-----|
| 需要奖励模型 | 需要 | 不需要 |
| 训练步骤 | 多步（RM训练 + PPO） | 一步到位 |
| 内存占用 | 很大（4个模型） | 较小（2个模型） |
| 训练稳定性 | 不稳定，调参困难 | 稳定，容易收敛 |
| 超参数 | 很多 | 很少（主要就一个 beta） |
| 效果 | 好 | 相当甚至更好 |
| 数据需求 | 偏好排序数据 | 偏好对数据 |

### DPO 的实际使用

```python
# 使用 HuggingFace TRL 库进行 DPO 训练（简化示例）
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("your-sft-model")
ref_model = AutoModelForCausalLM.from_pretrained("your-sft-model")  # 参考模型
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")

# DPO 配置
training_args = DPOConfig(
    output_dir="dpo_output",
    beta=0.1,                  # KL 惩罚系数
    learning_rate=5e-7,        # DPO 通常用很小的学习率
    num_train_epochs=1,        # 通常只需要 1-3 个 epoch
    per_device_train_batch_size=4,
    bf16=True,
)

# 创建 DPO 训练器
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dpo_dataset,  # 包含 prompt, chosen, rejected 的数据集
)

trainer.train()
```

> 代表作品：LLaMA 2、Zephyr、Mistral

---

## 五、GRPO（群体相对策略优化）

GRPO（Group Relative Policy Optimization）是 DeepSeek 在训练 DeepSeek-R1 时提出的方法。它的核心突破在于：**完全不需要人工标注！**

### 核心思路

GRPO 的想法简单到令人拍案叫绝：

1. 对同一个问题，让模型生成 K 个回答（比如 K=8）
2. 用**规则**（不是人、不是奖励模型）给每个回答打分
3. 在这一组回答内部做比较：高于平均分的鼓励，低于平均分的抑制

这就像一个班级的考试：不需要外部评委，也不需要标准答案模板，只需要能判卷（规则打分），然后让好学生教差学生就行。

### 具体流程

```
问题: "计算 15 x 7 = ?"

模型生成 8 个回答:
  回答1: "105" → 正确 → 得分 1.0 → 鼓励 ↑
  回答2: "115" → 错误 → 得分 0.0 → 抑制 ↓
  回答3: "105" → 正确 → 得分 1.0 → 鼓励 ↑
  回答4: "107" → 错误 → 得分 0.0 → 抑制 ↓
  回答5: "105" → 正确 → 得分 1.0 → 鼓励 ↑
  回答6: "150" → 错误 → 得分 0.0 → 抑制 ↓
  回答7: "105" → 正确 → 得分 1.0 → 鼓励 ↑
  回答8: "95"  → 错误 → 得分 0.0 → 抑制 ↓

组内平均分: 0.5
高于平均(1.0 > 0.5) → 增大这些回答的生成概率
低于平均(0.0 < 0.5) → 减小这些回答的生成概率
```

### 伪代码实现

```python
def grpo_step(model, ref_model, question, K=8, beta=0.04):
    """
    GRPO 一步训练

    参数:
        model: 当前策略模型
        ref_model: 参考模型（SFT模型）
        question: 输入问题
        K: 每个问题生成的回答数量
        beta: KL 惩罚系数
    """
    # 第一步：生成 K 个回答
    responses = [model.generate(question) for _ in range(K)]

    # 第二步：用规则打分（不是人工！不是奖励模型！）
    scores = [rule_based_score(question, response) for response in responses]

    # 第三步：组内归一化（关键！）
    mean_score = sum(scores) / K
    std_score = std(scores)
    advantages = [(s - mean_score) / (std_score + 1e-8) for s in scores]

    # 第四步：计算策略梯度
    loss = 0
    for response, advantage in zip(responses, advantages):
        log_prob = model.log_prob(question, response)
        ref_log_prob = ref_model.log_prob(question, response)

        # 类似 PPO 的 clip 机制
        ratio = torch.exp(log_prob - ref_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

        # 策略梯度 + KL 惩罚
        pg_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        kl_penalty = beta * (log_prob - ref_log_prob)

        loss += pg_loss + kl_penalty

    return loss / K


def rule_based_score(question, response):
    """
    基于规则的打分函数

    这是 GRPO 的关键：用可验证的规则代替人工标注
    """
    # 数学题：答案对不对
    if is_math_question(question):
        correct_answer = solve(question)
        return 1.0 if extract_answer(response) == correct_answer else 0.0

    # 代码题：测试能不能通过
    if is_code_question(question):
        test_results = run_tests(response)
        return test_results.pass_rate

    # 格式检查：有没有按要求的格式输出
    format_score = check_format(question, response)

    return format_score
```

### 适用场景

GRPO 的关键限制在于：**必须有客观的打分规则**。

| 场景 | 能否用 GRPO | 原因 |
|------|-----------|------|
| 数学推理 | 非常适合 | 答案对错可以验证 |
| 代码生成 | 非常适合 | 可以跑测试用例 |
| 逻辑推理 | 适合 | 可以检查逻辑规则 |
| 知识问答 | 部分适合 | 有些答案可以验证 |
| 创意写作 | 不适合 | 没有客观标准 |
| 开放对话 | 不适合 | 好坏很主观 |

### 重大发现：涌现行为

DeepSeek 在用 GRPO 训练 DeepSeek-R1 时，发现了一个令人震惊的现象：

**模型自发学会了思维链推理！**

训练过程中，模型的回答逐渐从：

```
早期: "15 x 7 = 105"  （直接给答案）

中期: "15 x 7，我来算一下。15 x 7 = 105"  （开始有思考过程）

后期: "让我想想...15 x 7，可以拆成 15 x 5 + 15 x 2 = 75 + 30 = 105。
       等等，让我验算一下：105 / 7 = 15，没错。
       所以 15 x 7 = 105"  （完整的思考 + 自我验证）
```

注意：**没有任何人教它"要思考"或"要验算"**。这完全是模型在 GRPO 训练中自发涌现的行为。为什么？因为那些经过思考和验算的回答正确率更高，得到了更多的"鼓励"，模型就自然学会了这种策略。

这个发现意义重大：它说明**推理能力可能不需要人工设计，只需要合适的训练信号就能涌现**。

> 代表作品：DeepSeek-R1

---

## 六、技术演进对比

| 对比维度 | RLHF | DPO | GRPO |
|---------|------|-----|------|
| 需要奖励模型 | 需要 | 不需要 | 不需要 |
| 需要人工标注 | 大量 | 中等（偏好对） | 不需要 |
| 打分方式 | 训练的奖励模型 | 隐式（偏好对） | 规则/程序化 |
| 适用场景 | 通用对话 | 通用对话 | 有客观标准的任务 |
| 可扩展性 | 低（受人工瓶颈） | 中等 | 高（自动化） |
| 代表工作 | ChatGPT, 早期Claude | LLaMA 2, Zephyr | DeepSeek-R1 |
| 训练复杂度 | 非常高 | 低 | 中等 |
| 训练稳定性 | 低 | 高 | 中等 |
| 出现时间 | 2022 | 2023 | 2024-2025 |

### 演进脉络

```
RLHF (2022)                    DPO (2023)                  GRPO (2024-2025)
  │                               │                            │
  │  "效果好但太复杂了"             │  "简单多了，效果一样好"       │  "连标注都不用了！"
  │                               │                            │
  │  4个模型，训练不稳定            │  2个模型，简单稳定            │  自动打分，可扩展
  │  需要大量人工标注               │  需要偏好对数据               │  只需要规则
  │                               │                            │
  └───── 简化 ─────────────────────┘                            │
                                  └───── 去标注化 ────────────────┘
```

---

## 七、SFT 之后怎么接强化学习？

SFT 训完的模型可以直接作为对齐训练的起点，不需要额外处理。

### 衔接关系

```
SFT 模型（学会了听指令）
    │
    │  直接作为初始模型
    ↓
对齐训练（学会什么回答更好）
    │
    │  输出最终模型
    ↓
部署上线
```

SFT 教会了**格式**（问什么答什么），对齐训练教会了**质量**（怎么答才好）。两者是递进关系，不是替代关系。

### 实操：SFT → DPO（最常用的接法）

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══ 关键：直接加载 SFT 训完的模型 ═══
sft_model_path = "./sft_output"  # 或 LoRA 合并后的模型路径

model = AutoModelForCausalLM.from_pretrained(sft_model_path)
ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path)  # 冻结副本
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

# DPO 配置 —— 注意学习率比 SFT 小很多
training_args = DPOConfig(
    output_dir="dpo_output",
    beta=0.1,                      # KL 惩罚系数
    learning_rate=5e-7,            # 比 SFT 的 2e-4 小 400 倍！
    num_train_epochs=1,            # 通常 1-3 个 epoch 就够
    per_device_train_batch_size=4,
    bf16=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,           # 防止模型跑偏太远
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=preference_data, # 偏好数据
)

trainer.train()
```

### 偏好数据怎么来？

```json
[
  {
    "prompt": "解释量子计算",
    "chosen": "量子计算利用量子比特的叠加态，可以同时表示0和1...",
    "rejected": "量子计算就是很快的计算机..."
  }
]
```

获取方式：

| 方法 | 成本 | 质量 | 适用场景 |
|------|------|------|---------|
| 人工标注 | 高 | 最好 | 关键业务场景 |
| AI 辅助生成 | 低 | 中等 | 快速迭代、冷启动 |
| 用户反馈收集 | 低 | 高（真实偏好） | 已上线产品 |
| 自动对比（强模型 vs 弱模型） | 低 | 中等 | 缺少标注资源时 |

### 关键注意事项

1. **学习率要小很多**：SFT 常用 2e-4，对齐训练用 5e-7 ~ 1e-6。因为只是微调偏好，不想破坏 SFT 学到的能力
2. **ref_model 不能省**：它通过 KL 散度约束防止模型"跑偏"——比如为了讨好打分器而说废话
3. **epoch 不要多**：1-3 个 epoch 通常就够，多了容易过拟合到偏好数据
4. **先确认 SFT 效果**：如果 SFT 模型本身质量不行，对齐训练也救不回来

### 三种接法对比

| | SFT → RLHF | SFT → DPO | SFT → GRPO |
|------|------------|-----------|------------|
| 额外需要 | 奖励模型 + PPO | 偏好数据 | 打分规则 |
| 显存占用 | 4 个模型 | 2 个模型 | 2 个模型 |
| 工程难度 | 很高 | 低 | 中等 |
| 推荐度 | 除非团队有经验 | **首选** | 有客观标准时首选 |

---

## 八、怎么选？

做对齐训练时，按以下决策树选择（对齐方法 + 数据来源）：

```
你的任务有客观评价标准吗？（数学/代码/逻辑）
├── 有 → GRPO（首选）
│       不需要人工标注，可自动扩展
│       特别适合训练推理能力
│
└── 没有（开放对话/创意写作等）
    │
    你有偏好标注数据吗？（chosen/rejected 对）
    ├── 有 → DPO（强烈推荐）
    │       简单稳定，效果好
    │       大多数场景的最优选择
    │
    └── 没有
        │
        预算充足吗？
        ├── 充足 → 先标注数据，再用 DPO
        └── 有限 → 用 AI 辅助生成偏好数据，再用 DPO
```

### 实用建议

1. **先做 SFT**：对齐训练是建立在 SFT 基础上的。先把 SFT 做好，再考虑对齐
2. **首选 DPO**：除非你的任务特别适合 GRPO，否则 DPO 是性价比最高的选择
3. **数据质量 > 数据数量**：1000 条高质量偏好对 > 10000 条低质量偏好对
4. **不一定需要对齐**：很多实际应用中，做好 SFT 就足够了。对齐训练是锦上添花
5. **可以组合使用**：先 SFT，再 DPO/GRPO，效果通常最好

---

## 总结

```
对齐训练的三种方法：

RLHF：效果好，但工程复杂度高，像开着一辆手动挡赛车
DPO： 效果好，工程简单，像开自动挡 —— 大多数人的最佳选择
GRPO：不需要人工标注，适合有客观标准的任务 —— 推理能力训练的未来

演进趋势：越来越简单、越来越少依赖人工、越来越可扩展
```
