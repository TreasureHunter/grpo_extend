# 注意力驱动的强化学习推理优化 —— 有代码的两个方法

> 调研日期：2026-08-31
> 主题：利用模型内部信号（注意力/感知）改进 RLVR（RL with Verifiable Rewards）训练，聚焦**有开源代码**的两个方法。
> 范围：近一年（2025 下半年 ~ 2026）发表在顶会的 RL 推理优化方法。
> 相关论文速览（无代码的注意力驱动工作）：FlowTracer（ICML 2026）、Preplan-and-Anchor（ICML 2026）、CCT/CompPO（arXiv:2608.21501）、Not All Tokens Learn Alike（arXiv:2605.07660）——目前均未开源，本文只整理有代码的两篇。

---

## 总览

| 方法 | 会议 | 场景 | "内部信号"来源 | 作用粒度 | 核心收益 | 代码 |
|---|---|---|---|---|---|---|
| **VPPO**（Visually-Perceptive Policy Optimization） | **ICLR 2026 Spotlight** | 多模态视觉推理 | 扰动法（遮蔽图 KL），非注意力图 | token 级 + 轨迹级 | 7B 模型 8 基准平均 +19.2pp | [huaixuheqing/VPPO-RL](https://github.com/huaixuheqing/VPPO-RL) ✅ |
| **AttnPO**（Attention-Guided Process Supervision） | ACL 2026 Main | 文本数学推理、治"过度思考" | **真·注意力图**（Key-Focus Heads 头分数） | step 级 | +7.3pts 且推理长度 -60% | [NieSYsc20/AttnPO](https://github.com/NieSYsc20/AttnPO) ✅ |

两者的共同套路：**把序列级奖励（GRPO 里所有 token 平分）变成 token/step 级信用分配**，信号来源是模型自己前向传播时的内部状态；下游统一接到 GRPO/RLOO/PPO 的更新式，即插即用。

---

# 方法一：VPPO（Visually-Perceptive Policy Optimization）

- **会议**：ICLR 2026 Spotlight
- **论文**：[arXiv:2510.09285](https://arxiv.org/abs/2510.09285)
- **代码**：[huaixuheqing/VPPO-RL](https://github.com/huaixuheqing/VPPO-RL)（约 85★，2026-08 仍在更新）

## 一句话总结

通过量化每个 token 的视觉依赖度，在轨迹级和 token 级两个层次精细化调控 RL 学习信号，显著提升大视觉语言模型（LVLM）的多模态推理能力。

## 1. 解决的问题

多模态 RLVR（GRPO/DAPO）把统一学习信号广播给所有 token，忽视了视觉感知的关键作用：

- **Insight 1（稀疏性）**：一条轨迹里真正"依赖图像"的 token 是稀疏的——只有数字、几何概念、逻辑算符等少数 token 获得高视觉依赖分数，大部分是连接词/格式符。
- **Insight 2（异质性）**：不同轨迹的视觉参与度差异很大——有些"答对"的轨迹其实是靠语言先验蒙对的幸运猜测，并非真正的视觉推理。

GRPO 一视同仁地广播信号，等于在**稀释甚至污染梯度**，还容易鼓励模型走捷径。

## 2. 方法核心

### 第一步：token 视觉依赖度打分（零标注）

用"遮蔽前向"衡量：把图像替换成非信息性内容 $I'$，看输出分布变化多少：

$$
\mathcal{S}(s_t, I) := D_{KL}\Big(\pi_\theta(\cdot \mid s_t, I) \;\Big\|\; \pi_\theta(\cdot \mid s_t, I')\Big)
$$

直觉：遮掉图像后预测分布几乎不变 → 该 token 靠语言先验生成、与视觉无关；$\mathcal{S}$ 越大 → 该 token 的预测高度锚定在视觉证据上。只需多跑一次遮蔽前向即可，**无需任何额外标注或辅助模型**。经验上分数高度右偏，印证"视觉依赖稀疏"。

### 第二步（微观）：Token Gradient Filtering（TGF）

每条轨迹 $\tau_i$ 按 $\mathcal{S}$ 选出视觉依赖度最高的 top-$k\%$ token 集合 $\mathcal{K}_i$，构造二值掩码：

$$
m_{i,t} = \mathbb{I}(t \in \mathcal{K}_i)
$$

**策略梯度只在这些 token 上计算**，其余一律屏蔽。实验最优过滤比例 $k=0.4$（只让 40% 最"看图"的 token 出声），把梯度集中到决定推理对错的感知 token 上，避免低信息 token 把更新方向拉偏。

### 第三步（宏观）：Trajectory Advantage Shaping（TAS）

先算每条轨迹的平均视觉依赖度 $\bar{\mathcal{S}}(\tau_i)$，再组内归一化映射到塑形因子：

$$
\alpha(\tau_i) = \beta_{\min} + (\beta_{\max} - \beta_{\min})\, \frac{\bar{\mathcal{S}}(\tau_i) - \min_{j} \bar{\mathcal{S}}(\tau_j)}{\max_{j} \bar{\mathcal{S}}(\tau_j) - \min_{j} \bar{\mathcal{S}}(\tau_j)}
$$

塑形后的优势：

$$
\hat{A}'(\tau_i) = \alpha(\tau_i) \cdot \hat{A}_{\text{GRPO}}(\tau_i)
$$

让视觉参与度高的轨迹获得更大更新、低依赖路径被压低，把学习偏好引向真正基于图像的推理。

### 最终目标函数

$$
\mathcal{L}^{\text{VPPO}}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} m_{i,t} \cdot \min\Big(r_{i,t}(\theta)\hat{A}'_i,\ \text{clip}\big(r_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big)\,\hat{A}'_i\Big)\Big]
$$

结构解读：**`m` 控制"谁出声"（TGF），`Â'` 控制"声音多大"（TAS）**，外层是标准 GRPO clipped surrogate。

### 理论支撑（为什么更稳）

方差缩减定理：

$$
\text{Var}(\mathbf{g}_{\text{VPPO}}) \approx k \cdot \mathbb{E}[\alpha(\tau)^2] \cdot \text{Var}(\mathbf{g}_{\text{GRPO}})
$$

稀疏率 $k \in (0,1)$ 来自只保留 top-$k\%$ token，$\alpha(\tau)$ 被压缩在 1 附近的窄带，两者相乘使梯度方差显著低于 GRPO。

## 3. 实验结果

**主实验：8 个多模态推理基准（avg@8 准确率，Qwen2.5-VL-7B）**

| 模型 | MathVerse | DynaMath | MMK12 | Geo3k | MathVision | We-Math | LogicVista | MMMU-Pro | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| Base | 39.0 | 55.7 | 42.5 | 37.1 | 18.4 | 46.4 | 42.4 | 25.1 | 38.3 |
| + GRPO | 66.5 | 65.8 | 72.3 | 40.2 | 30.7 | 68.1 | 45.6 | 35.2 | 53.1 |
| + DAPO | 68.3 | 66.6 | 82.1 | 41.5 | 30.5 | 68.0 | 46.8 | 35.9 | 55.0 |
| **+ VPPO** | **71.6** | **68.1** | **82.8** | **46.5** | **33.3** | **71.5** | **47.9** | **37.9** | **57.5** |

- 32B 规模：VPPO 64.6% vs DAPO 63.5%。
- 消融：仅 TAS 55.8 / 仅 TGF 56.2 / 组合 57.5（各自独立有效，组合最优）。

## 4. 代码结构与上手

基于 **verl** 框架（PyTorch 2.8）：

```
examples/configs/train_vppo_7b.sh      # 一键训练脚本（Qwen2.5-VL-7B）
examples/configs/train_vppo_8b.sh      # 一键训练脚本（Qwen3-VL-8B）
examples/format_prompt/math_format_perception.jinja  # 感知奖励 prompt 模板
examples/reward_function/{dapo,math,r1v}.py          # 三种奖励函数
verl/models/                           # VPPO 算法核心（打分 + 掩码 + 优势塑形）
```

**局限**：视觉依赖度需额外一次遮蔽前向（有计算开销）；仅 Qwen2.5-VL/Qwen3-VL 系列验证；"感知"用的是扰动 KL 而非注意力图。

---

# 方法二：AttnPO（Attention-Guided Process Supervision）

- **会议**：ACL 2026 Main
- **论文**：[arXiv:2602.09953](https://arxiv.org/abs/2602.09953)
- **代码**：[NieSYsc20/AttnPO](https://github.com/NieSYsc20/AttnPO)（约 4★，已开源模型权重）

## 一句话总结

利用模型**内在注意力信号**进行步级信用分配的低开销过程监督 RL 框架：识别 Key-Focus Heads（KFH），用其注意力分数区分冗余与关键推理步骤，大幅缩短推理长度的同时显著提升准确率。

## 1. 解决的问题：过度思考

RLVR 训练的大推理模型（如 DeepSeek-R1）存在严重"过度思考"——对简单操作也生成冗长推理。现有方案：

- **轨迹级长度惩罚**：均匀对待所有步骤，无法区分冗余与必要，常导致掉点；
- **MC 采样式过程监督**：计算开销大；
- **reward model 式过程监督**：信用分配不精确。

核心矛盾：需要细粒度 step 级监督区分冗余/关键，但现有方法要么贵、要么不准。

## 2. 核心发现：Key-Focus Heads（KFH）

LRM 生成**最终答案**时，会有一小撮注意力头**天然把高注意力给关键推理步骤、低注意力给冗余步骤**——这是零成本的 step 级监督信号。

定义最终答案对某推理步骤 $s_k$ 的注意力得分：

$$
\mathcal{S}_{s_k}^{l,h} = \frac{1}{|s_k|} \sum_{m \in \mathcal{F}} \sum_{n \in s_k} a_{m \to n}^{l,h}
$$

其中 $\mathcal{F}$ 是最终答案的 token 集合。用 **SRA（Step Ranking Accuracy）**——注意力排序与真实关键性排序的一致度——筛头：

- 少数头的 SRA 高达 **95–96%**；
- KFH 主要位于**中后层**，少量（SRA > 0.9）即够用，增加更多头收益饱和；
- KFH 行为在 RL 训练中高度稳定（Pearson 相关 > 0.85），且对困难题（AIME24）有泛化能力。

## 3. 两个互补子策略（一正一负）

先定义奖励（带长度正则）与 RLOO 优势：

$$
r_i = \mathbb{I}[o_i \text{ correct}]\big(1 - \alpha \cdot \sigma(f(o_i))\big), \qquad f(o_i) = \sigma\Big(\tfrac{\text{len}(o_i) - \text{mean}(q)}{\text{std}(q)}\Big)
$$

$$
A^i = r_i - \frac{1}{G-1}\sum_{j \neq i} r_j
$$

### 策略 A：正优势 → 冗余步骤衰减（治"正确但啰嗦"）

当 $A^i > 0$ 且某步 KFH 注意力**低于基线**（判定为冗余）时，用缩放因子削弱该步正优势：

$$
\gamma_{s_k}^{i} = (1-\delta)\cdot p_i^{\lambda}\cdot\big(\mathcal{S}_{s_k}^{i} / \mathcal{S}_{\text{base}}^{i}\big) + \delta
$$

其中**难度感知基线**：

$$
\mathcal{S}_{\text{base}}^i = p_i^{\beta} \cdot \frac{|\mathcal{F}_i|}{|o_i|}
$$

$p_i$ 越难的题阈值越宽松，避免误删真正的探索步骤。

### 策略 B：负优势 → 关键步骤保护（治"错误但有价值"）

当 $A^i < 0$ 且某步注意力**高于基线**（判定为关键）时，直接令 $\gamma_{s_k}^i = 0$ **完全免除惩罚**，把负向梯度集中到冗余步骤上。

> 两者互补：A 压住正确回复里的废话步骤，B 护住错误回复里的关键推理步骤——合起来同时实现缩长度 + 提精度。

## 4. 实验结果

DeepSeek-R1-Distill-Qwen-1.5B，6 个数学基准：平均 **+7.3 点准确率**（GSM8K、MATH500、AIME24、AIME25 均显著提升）+ **推理长度缩减 60%**（对比 AutoThink：+2.5pts、-37%）。

消融：
- 仅 Pos-Adv 衰减：有效缩短长度但精度提升有限；
- 仅 Neg-Adv 保护：有效保护精度但长度缩减有限；
- 两者结合（AttnPO）：同时实现大幅缩短 + 精度提升；
- 移除高 SRA 步骤显著降低 pass@32，低 SRA 步骤影响小（因果验证）。

## 5. 代码结构与上手

基于 **verl** 框架 + Qwen2 系模型：

```
modeling_qwen2.py        # ★关键 hack：覆盖到 transformers 包，hook 出 KFH 注意力头分数
scripts/attnpo_ds1.5b.sh # 一键训练脚本（R1-Distill-1.5B）
data/train/preprocessed_data/deepscaler.parquet   # 训练数据（已预处理）
data/test/preprocessed_data/*.parquet             # 测试数据（aime/amc/gsm8k/math 等）
verl/                    # 基于 verl 的 RL 训练管线
```

安装要点：`pip install -e .` + `flash-attn`，再执行 `cp modeling_qwen2.py <transformers 包路径>/models/qwen2/modeling_qwen2.py` 覆盖。

**局限**：推理步骤切分依赖预定义的特殊短语（可能不够通用）；KFH 在 >7B 模型上未充分验证；仅数学推理评估；推理时读注意力分数有少量开销。

---

# 对比与选择建议

| 维度 | VPPO | AttnPO |
|---|---|---|
| 会议 | **ICLR 2026 Spotlight** ✅ | ACL 2026 Main |
| 场景 | 多模态（LVLM）视觉推理 | 文本数学推理、治过度思考 |
| "内部信号"来源 | **扰动法**（遮蔽图 KL），非注意力图 | **真·注意力图**（KFH 头分数） |
| 作用粒度 | token 级（TGF）+ 轨迹级（TAS） | step 级（推理步骤优势缩放） |
| 额外开销 | 一次遮蔽前向 | 几乎为零（读已有注意力） |
| 核心收益 | +19.2pp（7B，8 基准） | +7.3pts 且长度 -60% |
| 代码成熟度 | 更完整（~85★，持续更新） | 较新（~4★，含预处理数据） |
| 与"利用注意力优化 RL"的契合度 | 中（"感知"思路近似，但不是注意力图） | **高**（正是"用模型内部注意力做 RL"） |

**实操建议**：
- 想要**最接近"利用注意力优化 RL"** → 以 **AttnPO** 为骨架（读 KFH 注意力分数 → step 级优势缩放），`modeling_qwen2.py` hack + `attnpo_ds1.5b.sh` 现成可跑；
- 想在 **ICLR/ICML 范畴内**复现且有代码保障 → 选 **VPPO**；
- 结合创新点：用 VPPO 的"稀疏 top-k 掩码 + 轨迹重加权"框架，把其"遮蔽 KL 打分"替换成 AttnPO 的"注意力头分数"——这是一个较新的工作点。

---

# 相关阅读

- **Attention as a Compass**（ICLR 2026）：注意力引导过程监督 RL 中的探索，与上述思路接近。
- **FlowTracer**（ICML 2026）：注意力权重建 DAG + 流量守恒算 token 重要度 → 尚无代码。
- **Preplan-and-Anchor**（ICML 2026）：注意力头回看距离 → WAAD/FAI 优势放大 → 代码待确认。
- **CCT / CompPO**（arXiv:2608.21501）：注意力集中度门控 + 路径相关 GAE → 无代码。
- **Not All Tokens Learn Alike**（arXiv:2605.07660）：注意力熵 → token 级软重加权 → 无代码。
- 持续更新的论文列表：[opendilab/awesome-RLVR](https://github.com/opendilab/awesome-RLVR)、[smiles724/Awesome-LLM-RLVR](https://github.com/smiles724/Awesome-LLM-RLVR)。

---

# 六篇 RL 推理优化论文速读（算法 + 创新）

> 追加日期：2026-08-31。均已在线阅读论文/详细解读核实；代码仓库全部经 GitHub API 验证存在。
> 主题脉络：全部围绕 **RLVR/GRPO 家族的细粒度信用分配与推理压缩**，恰好是上面 VPPO/AttnPO 所属方向的扩展。

| # | 论文 | 会议 | 代码 | 一句话 |
|---|---|---|---|---|
| 1 | SPAE（Step Potential Advantage Estimation） | Findings of ACL 2026 | [cii030/SPAE-RL](https://github.com/cii030/SPAE-RL) ✅ | 用"步骤势能"判断推理状态，抑制过度检查与改错答案 |
| 2 | SAS（Step-Level Advantage Selection） | Findings of ACL 2026 | [HanNight/SAS](https://github.com/HanNight/SAS) ✅ | 按步骤置信度选择性置零 advantage，稳定短上下文压缩 |
| 3 | HICRA（Hierarchy-Aware Credit Assignment） | ICLR 2026 | [TIGER-AI-Lab/Hierarchical-Reasoner](https://github.com/TIGER-AI-Lab/Hierarchical-Reasoner) ✅ | 发现"先程序后策略"两阶段，重点强化规划 token |
| 4 | StepORLM | ICLR 2026 | [0xzhouchenyu/StepORLM](https://github.com/0xzhouchenyu/StepORLM) ✅ | 策略与生成式过程奖励模型共同进化，解决运筹建模 |
| 5 | TIPS | ICLR 2026 | [ucsd-wang-lab-lm/tips](https://github.com/ucsd-wang-lab-lm/tips) ✅ | 用"自老师模型"的信息势能给每轮工具调用打稠密奖励 |
| 6 | Tree-GRPO | ICLR 2026 | [AMAP-ML/Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO) ✅ | 智能体级树搜索采样 + 树内/树间双层相对优势 |

---

## 1. SPAE：Step Potential Advantage Estimation

- **论文**：[arXiv:2601.03823](https://arxiv.org/abs/2601.03823) ｜ **代码**：[cii030/SPAE-RL](https://github.com/cii030/SPAE-RL)
- **要解决的问题**：结果级奖励（GRPO 只给整条轨迹一个优势）导致两类病理行为——**过度检查**（已经算对还继续验证，浪费 token）和**改错答案 R2W**（中途正确却"想太多"把最终答案改错）。已有熵/长度启发式缺乏"语义化的推理进度度量"。

### 算法
1. **免训练的探测机制（probing）**：在每步边界暂停生成，追加 `Final Answer \n\boxed{` 提示，采样 N=5 个短补全，提取两个信号：
   - **正确性 Acc**：强制喂入标准答案 token 的平均概率（当前逻辑导向正确结果的可能性）；
   - **置信度 Conf**：探测补全的 token 熵（熵低=收敛）。
2. **步骤势能**：$$\Phi(\tau_k^i) = 1.5\cdot \mathrm{Acc}\cdot\mathrm{Conf} + 0.5\cdot\mathrm{Acc} - \mathrm{Conf} \in [-1,1]$$
   - 三种状态：**探索**（Φ≈0）、**正确确信**（Φ→+1）、**错误自信**（Φ→−1）。
   - 饱和阈值 ε_sat=0.9：Φ 超过它之后的 token 是"检查 token"，之前的才是"解题 token"。
3. **SPAE 优势**：$$\hat{A}_{i,j}^{\mathrm{SPAE}} = \hat{A}_i^{\mathrm{Group}}\cdot f(\Phi_{i,\mathcal{M}(j)}) + \xi\cdot g(\Delta\Phi_{i,\mathcal{M}(j)})$$
   - **饱和惩罚** f(Φ)：解题完成后按 (1−α) 逐 token 衰减奖励，鼓励及时终止；
   - **势能差塑形** g(ΔΦ)：放大"aha 时刻"（ΔΦ 大的步）、惩罚势能回落。

### 创新点
- 把推理当成**可测量的多步过程**而非黑箱：训练免费的 probe 提取语义化进度信号；
- 与 Potential-Based Reward Shaping（Ng et al., 1999）同构，有理论根基；
- 探测信号经统计方差分析验证稳定（非噪声），既是诊断指标也是稠密塑形信号。

### 结果
DeepSeek-R1-Distill-Qwen-7B 上 AIME24 精度 +6.7% 且长度 −25.1%；GPQA +1.1% 且 −24.4%；R2W 率最高降低 67%；Qwen3-4B 上是唯一能超越 base 的 RL 方法。

---

## 2. SAS：Stabilizing Efficient Reasoning with Step-Level Advantage Selection

- **论文**：[arXiv:2604.24003](https://arxiv.org/abs/2604.24003) ｜ **代码**：[HanNight/SAS](https://github.com/HanNight/SAS)
- **关键发现（贡献之一）**：此前"高效推理"方法把变短归功于自己的长度奖励，却偷偷把模型从 16K–24K 长上下文换到 4K 短上下文后训练。作者用对照实验证明：**短上下文 GRPO 本身就是强压缩信号**——不加任何长度惩罚，输出就逼近甚至短于 LAPO/ThinkPrune。
- **但压缩带来不稳**：8K 正确 rollout 截断到 4K 后约 **29% 被 verifier 误判为失败**（常只缺末尾 boxed 答案），导致正确的中间步骤被误罚、正确样本里的冗余/自我怀疑步骤被误强化。

### 算法（Step-level Advantage Selection）
1. 输出按**双换行切步**；每步置信度 = **平均 token log 概率**（与外部 PRM Qwen2.5-Math-PRM-7B 排序 nDCG@k=0.9022，无需额外模型）；
2. **正确 rollout（正优势）**：选出比例 r（默认 0.3）的**最低置信步骤**，其优势置 0——"去噪"而非惩罚；
3. **verifier 判失败的 rollout（负优势）**：把**最高置信步骤**的负优势置 0——"保护"被截断误杀的正确推理。

巧妙之处：**同一个置零操作，因 GRPO 优势的符号结构而在正负样本上产生不同语义**（正=去噪，负=保护）。无需改结构、无需额外 PRM、无需额外前向，仅 ~17% 步耗时开销。

### 创新点
- 解耦出"短上下文"这个隐藏变量，改变了解读高效推理论文收益的方式；
- 用模型自身 log-prob 做步骤置信度，规避 PRM 训练成本与 reward hacking；
- 置零（而不是负值）是核心洞见：只停止强化低质量步骤，不反向惩罚。

### 结果
数学：SAS Pass@1 **54.54** @3407 token（AES 0.46），vs DeepScaleR 52.37@5118、GRPO-4K 53.61@3775、ThinkPrune 53.66@3878；通用（GPQA/LSAT/MMLU）38.30@2729。消融：随机选步、token 级选步、只做单边都弱于完整方法。

---

## 3. HICRA：Emergent Hierarchical Reasoning in LLMs through RL

- **论文**：[arXiv:2509.03646](https://arxiv.org/abs/2509.03646) ｜ **代码**：[TIGER-AI-Lab/Hierarchical-Reasoner](https://github.com/TIGER-AI-Lab/Hierarchical-Reasoner)
- **核心发现（机制分析）**：RL 让 LLM 推理按**两阶段涌现层次**推进（对应人类"高层策略规划 / 低层程序执行"分离）：
  - **阶段①程序巩固**：执行 token（算术、格式）的困惑度/熵快速下降；
  - **阶段②策略探索**：规划 token 的语义熵持续上升，且与精度、链长正相关——这是后期推理的真正瓶颈。
  - 这统一解释了"aha moment""长度扩展""熵动力学"三个看似孤立的现象。

### 算法（Hierarchy-Aware Credit Assignment）
1. **Strategic Grams（SGs）**：3–5 词 n-gram 作规划语义单元（如 "let's consider the case"）。用句子向量语义聚类 + Cluster Document Frequency 频率分析，取 top 20% 簇，数据驱动免人工标注；
2. **各向异性优势**：对规划 token 放大优势、其余不变：
   $$\hat{A}_{\mathrm{HICRA}} = \hat{A} + \alpha\cdot|\hat{A}|\ \ (\text{规划 token}),\qquad \hat{A}\ \ (\text{其他})$$
   α≈0.2，把优化压力集中到高影响规划 token，程序 token 保持稳定。

### 创新点
- 给出 RL 提升推理的**机制性解释**，而非纯经验；
- **语义熵**被验证为比 token 熵更好的探索诊断指标（token 熵会被低层 token 支配而失真）；
- 误差分析（GPT-4o 分类）证明 RL 主要修复的是"规划与策略"错误，支撑其假设。

### 结果
AIME24：Qwen3-4B-Instruct +5.4%、Qwen3-4B-Base +6.1%；多模态（MathVista/MathVision）同样提升；跨 Qwen/Llama/VLM 验证。局限：依赖 base 模型已有一定程序能力。

---

## 4. StepORLM：Generative Process Supervision for Operations Research

- **论文**：[arXiv:2509.22558](https://arxiv.org/abs/2509.22558) ｜ **代码**：[0xzhouchenyu/StepORLM](https://github.com/0xzhouchenyu/StepORLM)
- **要解决的问题**（运筹建模的独特性）：
  1. 结果奖励的**信用分配问题**：答案对了但推理有缺陷，会被一起强化；
  2. **判别式过程监督太短视**：OR 建模步骤高度耦合（变量定义、约束、目标函数必须协同一致），孤立地给每步打分会误判。

### 算法（两阶段自我进化）
- **阶段一（warm-up SFT）**：老师 LLM（GPT-4o）生成运筹题 + 完整推理轨迹 → **外部求解器确定性验证**（执行代码核对目标值），失败自动自我修正 → SFT 得到初始策略。
- **阶段二（策略 π 与 GenPRM 协同进化循环）**：
  1. **轨迹收集 + 双源评估**：求解器给确定性结果标签 + **GenPRM 整体过程评估**（对整个解做链式思考的事后逐步评价）；
  2. **策略对齐**：用 **Weighted-DPO** 优化，求解器验证优先（权重 1.0），求解器结果相同时按 GenPRM 过程质量排序；
  3. **GenPRM 精炼**：用求解器验证轨迹的逐步标签持续训练 GenPRM，形成正反馈。

### 创新点
- **生成式过程监督（GenPRM）**取代判别式 PRM：能整体推理、捕捉步骤间依赖，直接回应 OR 建模的"互依短视"问题；
- **策略-奖励共同进化**：reward model 不再是静态的；
- 双源反馈（确定性求解器 + 整体过程评估）兼顾"算得对"与"建得对"，抑制 reward hacking。

### 结果
8B 模型在 6 个 OR 基准（NL4Opt/MAMO/NLP4LP/ComplexOR/IndustryOR/ReSocratic）SOTA，超越 DeepSeek-V3(671B)、Qwen2.5-72B、Chain-of-Experts 等；co-evolved GenPRM 作通用验证器可提升其他 OR 模型最高 10%。

---

## 5. TIPS：Turn-Level Information-Potential Reward Shaping

- **论文**：[arXiv:2603.22293](https://arxiv.org/abs/2603.22293) ｜ **代码**：[ucsd-wang-lab-lm/tips](https://github.com/ucsd-wang-lab-lm/tips)
- **要解决的问题**：搜索增强 LLM 的 RL 训练极不稳定——奖励只在轨迹末端，跨多轮推理与工具调用的信用分配困难（哪次搜索真正有用？），GRPO 约 320 步即坍塌、PPO 停滞。

### 算法
1. **轮次定义**：一轮 = 一段"推理 + 工具调用 + 观察"；
2. **答案势能**：$$\Phi(S) = \log\!\Big(\sum_{m=1}^{M} p_{\mathrm{teach}}(A^{(m)}\mid S)\Big)$$ 老师模型对任一正确答案的对数似然；
3. **轮次奖励**：$$\Delta_k = \alpha\big(\Phi(S_k) - \Phi(S_{k-1})\big)$$ 该轮引入证据让正确答案更可信则奖励为正，否则小或负；
4. **老师 = 延迟自拷贝**：每 ~200 步刷新一次的策略冻结快照（非外部模型），配 **KV-cache 复用**，额外开销仅 ~12% FLOPs；
5. **PBRS 理论保证**：Δ_k 恰为 Potential-Based Reward Shaping 形式（Ng et al., 1999），γ=1 时**策略不变性**成立——不改变原任务最优策略，只降低方差、给出细粒度指引。

### 创新点
- **自老师 + 势能塑形**：无需人工逐步标注、无需昂贵外部 verifier/reward model；
- 证明了密集轮次奖励的**策略不变性**（对最优解无害）；
- 优势分布从 PPO 的"一团"变成**清晰双峰**，直观说明信用分配质量。

### 结果
Qwen2.5-7B：EM +11.8%、F1 +13.6%（相对 PPO）；跨 3B–14B 多个模型族、7 个 QA 基准一致提升，多跳 OOD 任务增益最大。消融：老师必须是自拷贝（换异构强模型掉 19.5–22.2%）；α 甜点区 [0.05, 0.3]。

---

## 6. Tree-GRPO：Tree Search for LLM Agent Reinforcement Learning

- **论文**：[arXiv:2509.21240](https://arxiv.org/abs/2509.21240) ｜ **代码**：[AMAP-ML/Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO)（约 400★）
- **要解决的问题**（多轮智能体 RL 两大瓶颈）：① **rollout 成本高**——多轮轨迹动辄上千 token + 多次工具调用，独立链式采样高度冗余；② **监督稀疏**——只有最终结果奖励，无法定位哪一步决策出了问题。

### 算法（Tree-based GRPO）
1. **智能体步级树搜索**：每个节点 = 一个完整 Thought-Action-Observation 步（ReAct 单元，而非 token/句级）。**初始化-扩展**采样：先生成 M 条完整轨迹，再随机选 N 个非叶节点，从该前缀扩展 L 次。前缀共享使预算
   $$\mathbb{E}[B_{\mathrm{tree}}] = M\cdot B + L\cdot N\cdot B/2$$
   固定 token/工具预算下最多获得约 **4×** 的 rollout 数量。
2. **双层相对优势**：
   - **树内优势**（同父分支比较）：$$A_{\mathrm{Intra}} = \frac{R(H_i) - \mathrm{mean}\{R(H_j)\}}{\mathrm{std}\{R(H_j)\}}$$ 把单条轨迹的稀疏奖励变成**步级偏好信号**；
   - **树间优势**（跨树/批内）：提供稳定的全局基线；
   - 最终 $$A_{\mathrm{tree}} = A_{\mathrm{Intra}} + A_{\mathrm{Inter}}$$。
3. **理论结果**：树内 GRPO 目标在数学上等价于**步级 DPO**——仅用结果奖励就隐式完成了过程级偏好学习。
4. 更新目标：标准 clipped surrogate + 对参考模型 KL 约束。

### 创新点
- **在线 RL 内做智能体步级树搜索**（区别于只在测试时用树搜索或离线生成数据）；
- 前缀共享显著降低 rollout 成本；
- 树的分支结构天然生成偏好对，把结果奖励"回溯"成分支点的过程监督——解决稀疏奖励下的信用分配；
- 双层优势（局部精确 + 全局稳定）提升训练鲁棒性。

### 结果
11 个数据集、3 类 QA 任务（含 WebWalkerQA）：多跳 QA 上在 ~2 条轨迹/题的极低预算下相对 GRPO 提升 **112%**；通常用 25% 预算即可取得更好效果；对 1.5B–3B 小模型增益最大（学会此前 RL/SFT 都学不会的多轮工具使用）。

---

## 六篇横向脉络小结

| 维度 | SPAE | SAS | HICRA | StepORLM | TIPS | Tree-GRPO |
|---|---|---|---|---|---|---|
| 信号来源 | 探测（Conf+Acc） | 步骤 log-prob 置信度 | 语义规划 token（SG） | 求解器+GenPRM | 自老师信息势能 | 树结构/结果奖励 |
| 改造点 | 优势塑形+饱和惩罚 | 优势选择性置零 | 优势放大（规划 token） | 奖励模型+偏好对齐 | 稠密轮次奖励(PBRS) | 采样结构+双层优势 |
| 场景 | 数学推理 | 数学/通用推理压缩 | 数学/多模态 | 运筹建模 | 搜索增强 QA | 多轮智能体 |
| 是否需额外模型/前向 | 探测采样（多 5 次短补全） | 否（仅后处理） | 否（SG 离线统计） | 是（GenPRM） | 自老师（KV 复用 +12%） | 否 |
| 有代码 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**对你（注意力驱动 RL）的启发**：SAS/SPAE/HICRA 都证明"**token 级/步级信用分配**"是 GRPO 家族的主战场；你关注的"注意力"只是众多内部信号的一种来源（AttnPO 用注意力头、SAS 用 log-prob、HICRA 用语义角色、SPAE 用探测）。其中 SAS 与 AttnPO 的**置零/保护**结构几乎同构——差异只在"用什么信号判定关键步骤"。若做新工作，可考虑把 AttnPO 的注意力信号替换/融合进 SAS 的置零框架，或用 HICRA 的语义熵诊断注意力驱动的策略。
