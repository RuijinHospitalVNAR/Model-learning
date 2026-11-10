---
model: Proteina-Complexa
aliases: [Proteina-Complexa, Proteina Complexa]
family: 复合物生成（binder 设计）
granularity: 全原子坐标（binder 走 Cα + 8 维 latent 解码；靶标以 Atom37 显式坐标 + 残基类型作为条件）
backbone: Transformer（继承 La-Proteina 主干；14 层，无三角乘法更新）
generative: 流匹配（部分潜变量）+ 测试时计算（beam search / SMC）
conditioning: 靶标结构（Atom37）+ 链断点特征 + hotspot token + 催化残基片段（酶任务）+ binder 长度；无 contig、无 MSA、无模板
data_scope: Teddymer 合成复合物（TED 域-域互作，10,089,503 dimer → 3,556,223 簇）+ PDB multimer 45,856 复合物（→110,976 链用于 VAE 微调）+ AFDB 单体预训练
status: 已完成
verified_at: 2026-11-10
---

# Proteina-Complexa

> **状态**：已完成（谱系、条件张量、数据、结果、代码/权重均已回到一手材料核实；湿实验不存在，属"论文无"而非"未核实"）
> **记录人**：团队共同学习
> **最后更新**：2026-11-10

> 三句话速判：**它是 La-Proteina 的 binder 分支，不是 Proteina 的续作**——binder 侧沿用 La-Proteina 训练好的 VAE（Cα + 8 维 latent），靶标则以 Atom37 坐标 + 残基类型 + hotspot token 直接进 denoiser 做条件；**它把"测试时计算"做成了一等公民**（beam search / SMC + 外挂 AF2-Multimer reward），这是它与 RFD4-Proteina 最像、也是与 APM/RFdiffusion 差距最大的地方；**全部结果都是 in-silico，论文没有湿实验**。

---

## 1. 学习背景与目标

### 1.1 为什么学它

- 它是本仓库「部分隐变量谱系」的**复合物分支**：La-Proteina 解决"单体全原子生成"，本篇解决"给定靶标生成 binder"。
- 它是当前 binder 设计里**唯一把 test-time compute 写进主结果**的开源模型（代码 + 权重 + 合成数据三件套齐全），也是 RFD4-Proteina 技术报告里的**主对照基线**。
- 仓库 CSV 里这一行的填法与论文严重不符（见 §1.3），必须用一手材料订正。

### 1.2 出处与身份

| 项 | 内容 | 标记 |
| --- | --- | --- |
| 标题 | Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute | 【核实】 |
| 作者 | Kieran Didi, Zuobai Zhang, Guoqing Zhou, Danny Reidenbach, Zhonglin Cao, Sooyoung Cha, Tomas Geffner, Christian Dallago, Jian Tang, Michael M. Bronstein, Martin Steinegger, Emine Kucukbenli, Arash Vahdat, Karsten Kreis | 【核实：arXiv abs 页】 |
| 预印本 | arXiv:2603.27950，v1 2026-03-30 | 【核实】 |
| 会议 | ICLR 2026 Oral | 【核实：arXiv 页与会议记录】 |
| 项目页 | `research.nvidia.com/labs/genair/proteina-complexa` | 【核实】 |
| 代码 | `https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa`，Apache-2.0，2026-03-15 发布 | 【核实】 |
| 权重 | NGC，NVIDIA Open Model License，2026-03-16 发布 | 【核实：NGC 模型卡】 |

### 1.3 与 Proteina / La-Proteina / RFD4-Proteina 的关系

这是本篇最容易写错的一段，逐条钉死：

| 关系 | 结论 | 证据 |
| --- | --- | --- |
| 与 **Proteina** | **不是同一篇的续作**，但同属 NVIDIA GenAIR；Proteina 是 Geffner 等的 Cα 骨架流（ICLR 2025），Complexa 是 Didi 等（ICLR 2026 Oral） | 【核实：两篇作者与年份】 |
| 与 **La-Proteina** | **直接复用其 VAE**：binder 侧用 La-Proteina 训练好的编码器/解码器（Cα + 8 维 latent）；靶标不进 VAE，只在 denoiser 内做条件。NGC 模型卡原文："partially latent architecture derived from La-Proteina" | 【核实：论文 §3.1 + NGC 卡】 |
| 与 **RFD4-Proteina** | 互为基线：Complexa = 设计（无置信度头，评估外挂 AF2-Multimer / RF3）；RFD4 = 设计 + 内部 latent reasoning（32 维）+ 外部 beam search。RFD4 报告把 Complexa 列为主要对照 | 【核实：RFD4 报告；两篇 §4】 |

一句话谱系：**Proteina（Cα 流）→ La-Proteina（8 维部分潜变量，全原子单体）→ Proteina-Complexa（同 VAE，加靶标条件 + 测试时搜索）**；RFD4-Proteina 是这条线的**旁支合流**（32 维 + 内部推理）。

### 1.4 本次目标

1. 判粒度：它是原子级还是残基级，binder 与靶标两侧分别是什么表示。
2. 拆条件接口：链组成、contig、表位三件事它到底支持哪几件。
3. 查清"设计 vs 预测"的定位差（与 RFD4-Proteina 对照）。
4. 核实数据与结果口径，特别是湿实验到底有没有。

---

## 2. 核心概念

| 术语 | 本篇含义 |
| --- | --- |
| binder | 被生成的短链（通常 40–250 aa），条件于给定靶标结构。 |
| target（靶标） | 已知结构，以 **Atom37 坐标 + 残基类型** 输入；在 denoiser 中作为条件，**不参与加噪/去噪**。【核实：§3.1】 |
| chain break 特征 | 用链断点标记替代残基索引，使模型能处理多链靶标（如 TNF-α 三聚体），不依赖单一连续编号。【核实：§3.1】 |
| hotspot token | 逐残基的二值指示，标出希望 binder 接触的靶标位点——这是本篇的**表位接口**。【核实：§3.1】 |
| 催化残基片段 | 酶任务变体：把催化残基作为额外的 Atom37 片段（坐标 + 类型）拼进条件序列，总长 = binder + target + catalytic。【核实：§3.2】 |
| Teddymer | 自建合成复合物数据集：从 TED 的域-域互作中取**单体 AFDB 预测结构**配对成二聚体，用于让模型见"界面"而无需实验复合物。【核实：§3.3】 |
| test-time compute | 推理期搜索：在同一轨迹的若干积分步分叉保留 top-k，用 reward 打分续跑（beam search / SMC）。【核实：§3.4】 |
| reward | 外挂评估器的加权组合：AF2-Multimer 界面指标（ipAE / pLDDT）、氢键能量、fold-class 符合度；配体变体换用 RF3。【核实：§3.4】 |
| unique successes | 主指标：200 个候选中，通过"pLDDT>90 且 ipAE<7 且 binder RMSD<1.5 Å"且按序列/结构去重后的条数。【核实：§4】 |
| self-seq | 直接用模型解码出的序列评估（不重设计），与 MPNN 重设计两档并列汇报。【核实：§4 表】 |

---

## 3. 原理与推导

### 3.1 生成侧：binder 走部分潜变量，靶标只做条件

- binder 侧完全沿用 La-Proteina 的分解：`x_Cα ∈ R^{L_b×3}` + `z ∈ R^{L_b×8}` 去噪，VAE 解码出侧链与序列。【核实：§3.1】
- 靶标侧**不编码、不解码、不加噪**：坐标以 Atom37 形式（`R^{L_t×37×3}`）与残基类型一起作为 denoiser 的条件 token 输入。这一点决定了它的"原子级"是**条件端的原子级**，不是生成端的原子级。【核实：§3.1】
- 关键工程细节：**用 chain-break 特征代替残基索引**。原因是多链靶标（同源三聚体、异源二聚体）没有单一连续编号，如果按索引编码，链间相对位置就会错。换成 chain break 后，模型靠"断点 + 相对位置"理解链结构。【核实：§3.1】
- hotspot 以逐残基二值 token 注入，等价于"指定表位"的软约束——模型仍可能不接触，只是被引导。【核实：§3.1】

**为什么不用 contig 字符串**：contig 是 RFdiffusion 系的接口语言（用 `A1-100/0 B1-50` 这类 DSL 描述链段与插入）。Complexa 的接口是**张量级**的：链组成靠 chain break，长度靠 `L_b` 采样区间，表位靠 hotspot token。表达能力上二者接近，但 Complexa 没有"精确指定某段 motif 序列"的 contig 语法。【核实：§3 全文未见 contig；对照为推断】

### 3.2 酶变体：把催化残基当"第三个片段"

- 条件序列的构成从 `[binder; target]` 扩到 `[binder; target; catalytic]`，三段都用 Atom37 坐标 + 残基类型。【核实：§3.2】
- 这样做的直接好处：催化残基的**侧链朝向**成为硬条件，模型必须围绕它组织活性口袋，而不是只在序列上"写上"催化三联体。【推断：基于条件构造】
- 评测用 AME（enzyme benchmark）41 个任务，与 RFdiffusion2 对比。【核实：§4】

### 3.3 数据：Teddymer 是本篇最值得学的一块

构造链条（务必按顺序理解）：

```text
TED 数据库的域-域互作对
  → 只取"两个域都来自单体 AFDB 预测结构"的配对
  → 拼成二聚体                  : 10,089,503 条
  → 聚类去冗余                  :  3,556,223 簇
```

【核实：§3.3】

- **为什么可以这样造**：AFDB 单体是 AF2 预测的，界面没被预测过；把两个单体按 TED 的真实互作关系摆在一起，就得到一个"界面几何来自真实生物学、单体几何来自 AF2"的复合物。它绕开了实验复合物数量不足的问题。【推断：基于构造描述】
- **代价**：界面处的侧链构象与真实复合物不一致（两个单体各自是最小能量构象，没有为结合做调整）。这会系统性削弱"界面处原子精度"的可信度。【推断】
- PDB multimer 作为真实数据补充：45,856 个复合物，过滤条件为"链长 50–256 aa、分辨率 ≤5.0 Å、loop 比例 <50%、Rg <3.0 Å"，最终 110,976 条链用于 **VAE 微调 40k 步**。【核实：§3.3 / App】

### 3.4 测试时计算：这篇的真正差异化

- 搜索对象是**去噪轨迹**：在指定积分步分叉，保留 top-k 候选，用 reward 打分后继续积分。【核实：§3.4】
- 两种策略：beam search（确定性保留）与 SMC（序贯蒙特卡洛，带重采样）。【核实：§3.4】
- reward 组成（按任务切换）：

| 任务 | reward 主项 | 备注 |
| --- | --- | --- |
| 蛋白 binder | AF2-Multimer 的 ipAE / pLDDT | 打分要跑一次 AF2-Multimer，成本高 |
| 配体 binder | RF3 的界面指标 | 配体变体 |
| 通用 | 氢键能量（Rosetta 定义，Tmol 实现） | 可选项，默认配置常被注释掉 |
| 通用 | fold-class 符合度 | 与设计目标折叠类对齐 |

【核实：§3.4 + 代码 `search/reward/`】

- **成本口径**：每靶标 200 个候选、binder 长度 40–250 aa，单样本约 **15.6 s**；搜索会把成本乘上分叉数与候选数。【核实：§4 / App】
- **必须写清的一点**：reward 里的 AF2-Multimer 同时是**评估器**。于是"用 AF2 打分筛出来的样本再用 AF2 评估"存在**循环性**——这不是作弊，但意味着 in-silico 成功率会系统性高于真实湿实验成功率。论文的做法是同时报告 self-seq 与 MPNN 重设计两档，但两档都仍在 AF2 体系内。【核实：协议；循环性影响为推断】

### 3.5 成功判据

```text
pLDDT > 90
ipAE < 7（界面 PAE，单位与 AF2 输出一致）
binder RMSD < 1.5 Å（相对设计模型）
```

AF2-Multimer 通过 ColabDesign 调用，序列来源两档：模型自解码（self-seq）/ ProteinMPNN 重设计（MPNN、MPNN-FI）。【核实：§4】

### 3.6 与 RFD4-Proteina 的"设计 vs 预测"定位差

| 维度 | Proteina-Complexa | RFD4-Proteina |
| --- | --- | --- |
| 任务定位 | **设计**（给定靶标造 binder / 酶） | **设计**（多分子上下文，含小分子/DNA/RNA） |
| 是否预测器 | 不是。无 pLDDT/PAE 头，评估外挂 AF2-Multimer / RF3 | 不是。无置信度头，评估外挂 oracle |
| 隐变量 | 8 维（继承 La-Proteina） | 32 维（自训 AE） |
| 内部推理 | 无 recycling 调度 | 有（120/180/300 次 trunk pass 三档） |
| 外部搜索 | 有（beam / SMC） | 有（checkpoint beam search） |
| 湿实验 | **无** | 有（158/188） |
| 开源程度 | 代码 + 权重 + Teddymer | 代码 + 权重 + 技术报告 |

【核实：两篇 §3–§5；RFD4 报告】

**一句话**：两者都是"设计器"，都不做预测；差别在**"把算力花在哪"**——Complexa 花在外部搜索，RFD4 花在内部 latent reasoning + 更大的隐变量。

---

## 4. 训练数据构成与切分

- **数据源与规模**：
  - AFDB 单体（预训练）：VAE 500k 步 / 16×A100；FM 单体预训练 540k 步 / 32×A100。【核实：App】
  - Teddymer（复合物预训练）：10,089,503 dimer → 3,556,223 簇。【核实：§3.3】
  - PDB multimer（真实复合物微调）：45,856 复合物 → 110,976 链；VAE 微调 40k 步 / 16 GPU；FM 微调 290k 步 / 96×A100；可训练参数合计 **256M**（denoiser 160M，14 层，latent 维 8）。【核实：App】
  - **口径冲突**：NGC 模型卡称发布的合成数据约 **500,000** 对，与论文的 3,556,223 簇不一致；差异成因模型卡未解释。【核实：冲突存在；成因为未核实】
- **清洗与去重**：PDB 侧按链长 50–256 aa、分辨率 ≤5.0 Å、loop <50%、Rg <3.0 Å 过滤；Teddymer 侧按 TED 互作对 + 单体来源约束 + 聚类去冗余。【核实：§3.3】
- **切分与泄漏控制**：**作者未给出显式的训练/测试序列相似度切分**。已查 §3.3、§4 与数据附录，未见"靶标与训练集序列同一性阈值"的陈述；19 个评测靶标是否与 Teddymer/PDB 训练集有同源关系，论文未报。【未核实：已查上述三处，属"作者没写"】
  - 这是本篇最实质的评测风险：binder 任务的"新颖性"取决于靶标是否见过，而这一点没有隔离声明。【推断】
- **覆盖与偏置**：蛋白 binder 为主；配体变体单独训练；酶任务用 AME 41 任务。多链靶标靠 chain break 支持，但**同源寡聚体占比、膜蛋白占比、物种分布均未报**。【未核实】
- **可获取性与许可**：代码 Apache-2.0（可商用）；权重 NVIDIA Open Model License；Teddymer 随 NGC 发布。【核实】

---

## 5. 宣称与可验证性

| 宣称 | 证据类型 | 核查状态 | 备注 |
| --- | --- | --- | --- |
| binder 侧为部分潜变量（Cα + 8 维） | 正文 + 继承关系 | 已核实 | 与 La-Proteina 同一 VAE |
| 靶标以 Atom37 + 残基类型作条件 | 正文 + 图 | 已核实 | 不进 VAE、不加噪 |
| 支持多链靶标 | 正文（chain break）+ 算例 | 已核实 | 例：TNF-α 三聚体 |
| 支持催化残基片段条件 | 正文 + AME 基准 | 已核实 | 三段拼接 |
| 19 个 in-silico 靶标 | 表格 | 已核实（数值） | 23 个候选筛到 19 个 |
| 平均 unique successes：self-seq 9.10 / MPNN-FI 13.6 / MPNN 14.4 | 表格 | 已核实（数值） | 每靶 200 候选 |
| 对照：APM 0.31 / 1.52 / 3.15；RFdiffusion 4.68（MPNN） | 表格 | 已核实（数值） | 同协议 |
| test-time beam 可达约 77 unique successes | 表格/图 | 已核实（数值） | 上限值，非平均 |
| AME 酶基准优于 RFdiffusion2 | 表格 | 已核实（数值） | 41 任务 |
| 有湿实验验证 | — | **论文无** | 全文无湿实验章节；CSV 中"133 靶 / >1M 候选 / 63.5% hit / 碳水化合物 binder"均无出处 |
| 代码 + 权重 + 合成数据公开 | 仓库 + NGC | 已核实 | Apache-2.0 / NVIDIA Open Model License |

- **可复现所需材料**：GitHub 代码、NGC 权重、Teddymer、AF2-Multimer（ColabDesign）与 RF3（配体）作为评估器、ProteinMPNN。训练需 96×A100 量级；推理单卡可跑但 200 候选 × 搜索会显著放大成本。【核实 + 推断】
- **与官方文档 / 论文的一致性**：README / NGC 与论文在"继承 La-Proteina""部分潜变量""test-time compute"上一致；**唯一数值冲突是 Teddymer 规模**（3.56M 簇 vs 发布约 500k 对）。【核实】

---

## 6. 实践与应用

- **适用场景**：给定结构的蛋白靶标做 de novo binder；多链/寡聚靶标；酶活性口袋重设计（催化残基条件）；配体结合位点设计（配体变体）。
- **不适用**：结构预测（无置信度头，别拿它当 AF2 用）；需要 contig 精确控制链段的任务；需要 motif scaffolding（binder 侧无 motif 接口，motif 能力留在 La-Proteina）；无结构靶标（必须先有靶标结构）。
- **工程要点**：
  - 先跑无搜索（单轨迹）拿基线，再开 beam；reward 里 AF2-Multimer 是最贵的一项，先用轻量 reward（fold-class、氢键）粗筛可省大量算力（**该策略为工程推断，论文未验证**）。
  - 多链靶标务必检查 chain break 标记是否正确，否则链间相对位置会错。
  - hotspot 只是软引导，不能保证接触。
- **成本提示**：单样本 15.6 s（40–250 aa），200 候选 ≈ 52 min；搜索按分叉数近似线性放大。

---

## 7. 疑问与讨论

- **Q**：它和 Proteina 是同一个模型的两个版本吗？
  - **A**：不是。Proteina 只生成 Cα，没有隐变量、没有序列、没有靶标条件；Complexa 用的是 La-Proteina 的 VAE。二者只共享团队与"Proteina"这个命名前缀。【核实】
- **Q**：它算"原子级"模型吗？
  - **A**：分两侧看。**条件端**是原子级（靶标 Atom37 全原子坐标 + 类型）；**生成端**是残基级潜变量（3+8 维/残基）+ VAE 解码。所以准确说法是"原子级条件、部分潜变量生成"。写对比表时应与 SeedProteo 的"逐原子 atom14 生成"区分开。【核实 + 推断】
- **Q**：为什么用 chain break 而不是链 ID embedding？
  - **A**：论文的表述是用 chain break 替代残基索引。效果上等价于显式告诉模型"这里断链"，但对任意链数都成立，不需要预定义最大链数。作者的取舍理由未展开。【核实：机制；理由为推断】
- **Q**：Teddymer 的界面可信吗？
  - **A**：界面**关系**来自真实 TED 互作，界面**几何**来自两个 AF2 单体硬拼，侧链没有为结合做诱导契合。所以它适合教"哪里可能接触"，不适合教"界面原子精度"。【推断】
- **Q**：为什么 CSV 里会有"133 靶、湿实验、63.5% hit"？
  - **A**：一手论文里完全没有这些数字（19 靶、无湿实验、无碳水化合物 binder）。已登记到 `work/待订正清单.md`，需人工裁决是否与其他模型串行。【核实：论文全文】

---

## 8. 学习心得

- **"测试时计算"是本篇真正的方法论贡献**，而不是模型结构。结构上它只是"La-Proteina + 条件"，但把搜索接进生成轨迹后，unique successes 从个位数推到几十。这提示：在 binder 这类"可廉价外挂评估器"的任务上，搜索的边际收益可能大于换更大主干。
- **reward 与 evaluator 同源是这类工作的结构性弱点**。AF2-Multimer 既筛又评，in-silico 数字再高也不能直接推湿实验命中率。读这类论文时，先看"评估器是否与 reward 同源"，再看数字。
- **数据合成思路值得迁移**：用"真实互作关系 + 预测单体结构"造复合物，解决了实验复合物稀缺。这个套路（关系真实、几何近似）在其他界面任务上同样可用。
- **接口设计的代际差异**：RFDiffusion 系用 contig 字符串，Complexa 用张量条件（chain break + hotspot + 长度区间）。前者表达力强但难学，后者易用但表达受限。做工具选型时要按"是否需要精确 motif 指定"来分。

---

## 9. 参考资料

- 论文：Didi, Zhang, Zhou, Reidenbach, Cao, Cha, Geffner, Dallago, Tang, Bronstein, Steinegger, Kucukbenli, Vahdat, Kreis. *Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute.* arXiv:2603.27950（2026-03-30）；ICLR 2026 Oral.
- 项目页：`https://research.nvidia.com/labs/genair/proteina-complexa`
- 代码：`https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa`（Apache-2.0）
- 权重：NGC，NVIDIA Open Model License
- 本地材料：`d:\postdoc_work\文献学习\2026-11-10\laproteina-complexa-seedproteo-notes\work\complexa.pdf`（及 `complexa.txt`）
- 同谱系对照：[La-Proteina](../La-Proteina/README.md)、[Proteina](../Proteina/README.md)、[RFD4-Proteina](../RFD4-Proteina/README.md)
