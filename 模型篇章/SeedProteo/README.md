---
model: SeedProteo
aliases: [SeedProteo, SeedProteo-R, SeedProteo-D]
family: 复合物生成（binder 设计）
granularity: 全原子坐标（atom14，逐原子显式；非潜变量）
backbone: AF3 式 Pairformer（12 层，较 AF3 的 48 层缩减）+ diffusion module，保留 MSA 模块
generative: 扩散（AF3 式；采样步数与 σ 调度作者未写）
conditioning: 靶标链 ID + 残基范围 + hotspot 残基 + binder 长度区间 + 二级结构序列条件；无 contig；Design View 不使用 MSA
data_scope: 约 0.7M 样本（单体约 0.5M：AFDB + ESMAtlas，50–768 aa、pLDDT>80、coil<50%；Pinder 约 50k 双链；HumanPPI 约 0.1M DDI）；切分与泄漏控制作者未说明
status: 已完成（薄笔记：步数、参数量、σ 调度、代码/权重为缺口）
verified_at: 2026-11-10
---

# SeedProteo

> **状态**：已完成（薄笔记）。一手材料仅"技术报告 + 项目页 + webserver"三件，已全部读完；缺口属**作者没写**，不是我们没查
> **记录人**：团队共同学习
> **最后更新**：2026-11-10

> 三句话速判：**它是字节跳动 Seed 的技术报告（非期刊论文），走 AF3 式 atom14 全原子扩散，不是潜变量路线**；**它的关键设计是"Design View / Folding View"双视图 + MRF 序列解码**，用同一套网络既做条件设计又做结构校验；**代码与权重均未开源，只有 webserver，扩散步数/参数量/σ 调度三项核心数字作者未披露**。

---

## 1. 学习背景与目标

### 1.1 为什么学它

- 它是本仓库 binder 设计组里**唯一的"中国团队 + AF3 血统"**样本：与 La-Proteina/Complexa（潜变量流匹配）、RFD4-Proteina（部分隐变量 + 内部推理）构成三条互不相同的技术路线，横向对比价值高。
- 它与同团队的 **SeedFold**（结构预测）共享项目页，是"预测 + 设计"一体化的样本，可用来检验"设计模型是否需要自己的 folding 分支"。
- 网上关于它的二手介绍不少，且常把"atom14""全原子""开源"混着说。本篇只采信技术报告与项目页。

### 1.2 出处判定（先解决"它是什么"）

| 项 | 内容 | 标记 |
| --- | --- | --- |
| 标题 | SeedProteo: Accurate De Novo All-Atom Design of Protein Binders | 【核实】 |
| 机构 | ByteDance Seed | 【核实：报告首页】 |
| 形态 | **技术报告（technical report）**，非期刊/会议论文 | 【核实：arXiv 元数据与正文体例】 |
| 预印本 | arXiv:2512.24192，v1 2025-12，v2 2026-02-24 | 【核实】 |
| 作者 | Wei Qu（一作）, Yiming Ma, Fei Ye, Chan Lu, Yi Zhou, Kexin Zhang, Lan Wang, Minrui Gui；技术负责 Quanquan Gu | 【核实：报告作者列表】 |
| 项目页 | `seedfold.github.io` | 【核实】 |
| 在线服务 | `seedfold.io`（含 proteinDesign 入口） | 【核实】 |
| 代码 | **无公开仓库**（项目页无 GitHub / HuggingFace 链接） | 【核实：截至 2026-11-10 项目页与报告均无代码链接】 |
| 权重 | **未发布** | 【核实：同上】；内部权重是否存在【未核实】 |

**"薄笔记"声明**：本篇不追求与 La-Proteina / Complexa 同等颗粒度。原因是可核材料只有三件（PDF、项目页、webserver），且报告本身在扩散步数、参数量、噪声调度、loss 权重上留白。凡这类条目一律标【未核实】并注明"已核 PDF 全文 + 项目页 + webserver"，不用二手博客补。

### 1.3 本次目标

1. 判粒度：atom14 到底覆盖哪些原子，"全原子"是否名副其实。
2. 拆条件接口：链组成、contig、表位三件事各自支持到什么程度。
3. 查训练数据：来源、规模、过滤、课程安排。
4. 标出所有缺口，为后续"若开源再补"留接口。

---

## 2. 核心概念

| 术语 | 本篇含义 |
| --- | --- |
| atom14 | 每残基 14 个重原子的固定槽位表示（N, CA, C, O, CB + 侧链原子按残基类型映射到 14 槽）。它比 Atom37 少氢、少 OXT 等，是 AF 系里常见的"全重原子"紧凑表示。【核实：报告 §方法；"14"为核实，"槽位映射"为通用定义】 |
| Design View | 设计分支：输入自条件序列 + 自条件模板（二级结构）+ 含噪坐标，**不使用 MSA**。【核实：§方法】 |
| Folding View | 折叠/校验分支：输入序列 + MSA + 模板，输出结构——相当于内置一个轻量 folding 模型。【核实：§方法】 |
| self-conditioning | 把上一时间步去噪出的结构作为当前步的额外输入，让模型在 pair 表示上做"几何推理"。【核实：§方法】 |
| MRF 序列解码 | 用马尔可夫随机场对序列联合建模，而不是逐位取 argmax。用于抑制"等排（isosteric）氨基酸"被错判——如 Val/Ile/Thr 体积相近但身份不同。【核实：§方法】 |
| SS 序列条件 | 以二级结构序列（H/E/C 三类）作为条件，控制生成骨架的二级结构组成。【核实：§方法】 |
| SeedProteo-R / -D | 两个条件变体：R（robust）用更长连续的 SS 段、成功率更高；D（diverse）用更短段、拓扑更多样。【核实：§结果】 |
| SeedFold | 同团队的蛋白结构预测模型，被用作本篇的**评估器**（非训练组件）。【核实：项目页】 |
| inter-chain PAE / binder pTM / complex RMSD | 三个成功判据，见 §3.5。【核实：§评测协议】 |

---

## 3. 原理与推导

### 3.1 架构：AF3 的缩减版，不是潜变量路线

| 组件 | 配置 | 标记 |
| --- | --- | --- |
| 主干 | AF3 式 Pairformer，**12 层**（AF3 为 48 层） | 【核实：§方法】 |
| MSA 模块 | 保留（Folding View 使用） | 【核实】 |
| 扩散模块 | AF3 式，作用于 atom14 坐标 | 【核实】 |
| 表示 | atom14 逐原子显式；**无 VAE、无隐变量** | 【核实：全文无 latent/VAE/decoder 描述】 |
| 参数量 | **作者未给** | 【未核实：已核 PDF 全文与项目页】 |

**第一层判断**：它与 La-Proteina 系的分野不在"是否全原子"，而在**自由度怎么摆**。La-Proteina 把侧链压进 8 维潜变量再解码；SeedProteo 直接对 14 个原子坐标做扩散。前者省算力、后者表达更直接——但代价是采样成本随原子数线性上升。【推断：基于两条路线的结构差异，非论文陈述】

**第二层判断**：12 层 Pairformer 是"够用就好"的取舍。binder 设计的条件端（靶标）通常是几十到几百残基，不需要 AF3 那套为全基因组级 MSA 深度优化的 48 层容量。【推断】

### 3.2 双视图：Design 与 Folding 共用一套网络

```text
Design View :  自条件序列 s_t + 自条件模板（SS）+ 含噪坐标 x_t   → 预测 x_0 与序列
Folding View:  给定序列 s + MSA + 模板                          → 预测结构
```

【核实：§方法】

- Design View **不用 MSA**：因为设计时还没有序列，MSA 无从查起。这与 AF3 的预测范式必然不同。【核实】
- Folding View 的存在意味着模型内部具备"给定序列验结构"的能力，可用来做 self-consistency 式筛选，而不必完全外挂 ESMFold。【推断：基于双视图设计意图；论文是否这样用见 §3.5】
- 但要小心：**评估实际用的是 SeedFold，不是 Folding View**。两者是同团队的不同模型，Folding View 是否被用于筛选，报告未明确。【未核实】

### 3.3 MRF 序列解码：为什么不能逐位 argmax

- 若对每个位置独立取 argmax，等排残基（Val/Ile/Thr/Leu 等）会因为几何几乎相同而被随机分配，导致序列"看起来对、实际错"。【核实：§方法动机】
- MRF 形式：

```text
P(X = x | a, z) ∝ exp( Σ_i h_i(x_i | a_i) + Σ_{i<j} e_ij(x_i, x_j | z_ij) )
```

其中 `h_i` 为逐位置一元势、`e_ij` 为 pair 表示给出的二元势，`z_ij` 为 pair 特征。【核实：§方法】

- 工程含义：最终序列是**联合解码**的产物，不是几何头的副产品。这解释了为什么它能报 self-seq 成绩（不像某些模型必须外挂 MPNN 才能出可用序列）。【推断：基于解码方式；self-seq 结果为核实】

### 3.4 条件接口：支持表位，不支持 contig

| 条件 | 支持情况 | 标记 |
| --- | --- | --- |
| 靶标链 ID + 残基范围 | 支持（指定哪些链/区段参与结合） | 【核实：§方法 + 靶标表】 |
| hotspot 残基 | 支持（指定希望接触的靶标位点） | 【核实】 |
| binder 长度区间 | 支持（按靶标给出区间，如 60–120 aa） | 【核实：§评测靶标表】 |
| 二级结构序列条件 | 支持（H/E/C，R/D 两版策略不同） | 【核实】 |
| contig 语法 | **不支持**（无 `A1-100/0` 式链段 DSL） | 【核实：全文无 contig 相关描述】 |
| motif scaffolding | 报告未提（训练含 motif 样本，但推理接口未见 motif 指定） | 【未核实】 |
| MSA / 模板（设计侧） | 不使用 | 【核实】 |
| 能量项 / reward | 无 | 【核实】 |

**与 Complexa 的接口对照**：两者都放弃 contig，都靠"链 + 残基 + hotspot + 长度"这四个量描述任务。差别在于 SeedProteo 多一个 SS 条件，Complexa 多一个催化残基片段条件。【核实：两篇方法节】

### 3.5 评测协议与成功判据

```text
成功判据：inter-chain PAE ≤ 1.5
          binder pTM ≥ 0.8
          complex RMSD < 2.5 Å
```

【核实：§评测协议】

- 靶标：按 AlphaProteo 协议取 **10 个靶标**，每靶约 **1000 个候选**，候选长度在靶标专属区间内均匀覆盖。【核实】
- 公平性处理：**所有扩散类基线都再用 ProteinMPNN 重设计序列**（含 SeedProteo 的对照口径），使比较不被"序列生成能力"混淆。【核实】
- **评估器是 SeedFold**——同团队的 folding 模型。这构成潜在循环性：模型与评估器同源，误差相关性未知。论文未做跨评估器（如 AF2/AF3/Boltz）复核。【核实：评估器为 SeedFold；循环性影响为推断】

---

## 4. 训练数据构成与切分

- **数据源与规模**：合计约 **0.7M** 样本。【核实：§数据】

| 子集 | 规模 | 来源 | 过滤 |
| --- | --- | --- | --- |
| 单体 | 约 0.5M | AFDB + ESMAtlas（ESMAtlas 为 MGnify 序列经 AF2 重预测） | 长度 50–768 aa、pLDDT >80、coil 比例 <50%、Foldseek + MMseqs2 聚类 |
| 双链复合物 | 约 50k | Pinder | 每链 coil <50%、最小界面 Cβ 距离 <8 Å、Foldseek-Multimer 聚类 |
| DDI 对 | 约 0.1M | HumanPPI | — |

【核实：§数据与附录】

- **清洗与去重**：pLDDT、长度、coil 比例三道硬过滤 + Foldseek / MMseqs2 / Foldseek-Multimer 三级聚类。【核实】
- **训练课程（三阶段）**：

| 阶段 | 步数 | crop | batch | 样本配比 |
| --- | --- | --- | --- | --- |
| 1 | 50K | 384 | 128 | 100% 严格单体；motif 0% |
| 2 | 20K | 768 | 64 | 20% motif；20% 严格单体 / 80% 扩展单体 |
| 3 | 30K | 768 | 64 | 20% motif；10% 严格单体 / 40% 扩展单体 / 50% 多聚 |

【核实：§训练细节】

  这套课程的含义：先在纯单体上学"蛋白长什么样"，再逐步引入 motif 与复合物，避免早期就被界面数据带偏。【推断：基于配比单调变化】

- **切分与泄漏控制**：**作者未说明**。已查数据节、训练细节附录与评测节，未见"训练集与 10 个评测靶标的序列同一性阈值"陈述，也未给时间切分。考虑到 Pinder / HumanPPI 与 AlphaProteo 靶标可能同源，这一项是实质风险。【未核实：已查三处，属"作者没写"】
- **覆盖与偏置**：单体远多于复合物（约 0.5M : 0.15M）；pLDDT>80 过滤意味着只用"AF2 有把握"的区域，柔性区/无序区被系统性排除；物种与构象偏置未报。【核实 + 未核实：分项】
- **可获取性与许可**：训练数据全部来自公开集（AFDB / ESMAtlas / Pinder / HumanPPI），可复现数据侧；但**代码与权重均未发布**，训练超参只给到步数/crop/batch，lr、优化器、loss 权重未给。【核实 + 未核实】

---

## 5. 宣称与可验证性

### 5.1 主结果（每靶约 1000 候选中的成功数）

| 靶标 | SeedProteo-R | BoltzGen | RFdiffusion3 |
| --- | --- | --- | --- |
| BHRF1 | 296 | 627 | 9 |
| SC2RBD | 92 | 0 | 0 |
| IL-7RA | 100 | 10 | 0 |
| PD-L1 | 380 | 17 | 1 |
| TrkA | 232 | 21 | 0 |
| Insulin | 303 | 99 | 0 |
| H1 | 16 | 1 | 0 |
| VEGF-A | 127 | 1 | 0 |
| IL-17A | 47 | 5 | 0 |
| TNFα | 1 | 0 | 0 |

【核实：§结果表 3；数字为表中"成功数"列】

读表要点：**BoltzGen 只在 BHRF1 一靶上大幅领先（627 vs 296），其余九靶落后**；RFdiffusion3 全场接近 0，说明该评测协议对 RFdiffusion 系不友好（可能是 atom14/序列重设计口径差异）。不能简单总结为"SeedProteo 全面最优"。【推断：基于表内分布】

### 5.2 宣称核查表

| 宣称 | 证据类型 | 核查状态 | 备注 |
| --- | --- | --- | --- |
| 全原子（atom14）设计 | 方法 + 图 | 已核实 | 逐原子扩散，非潜变量 |
| Design View 不用 MSA | 方法 | 已核实 | 与 AF3 预测范式区分 |
| MRF 序列解码 | 方法 + 公式 | 已核实 | 解决等排残基误判 |
| 10 靶 in-silico 成功率 | 表格 | 已核实（数值） | 每靶约 1000 候选 |
| 湿实验：PD-L1 8/10 结合、最佳 KD 56.9 pM | 表格 + SPR | 已核实（数值） | 2 个靶做了湿实验 |
| 湿实验：SC2RBD 7/10 结合、0.41 nM | 表格 + SPR | 已核实（数值） | 同上 |
| 无条件生成可达 1000 aa，>60% design success | 表格/图 | 已核实（数值） | 与 La-Proteina / Proteina / RFdiffusion 对比 |
| 扩散采样步数 | — | **作者未写** | 已核 PDF 全文、项目页、webserver |
| 噪声调度 / σ 参数 | — | **作者未写** | 同上 |
| 模型参数量 | — | **作者未写** | 同上 |
| 代码 / 权重开源 | — | **未开源** | 项目页仅 webserver |
| 评估器独立性 | — | 存疑 | 用同团队 SeedFold 评估，无跨评估器复核 |

- **可复现所需材料**：**不具备本地复现条件**。只能通过 `seedfold.io` 的 proteinDesign 入口使用；训练侧缺 lr、优化器、loss 权重、步数/调度等关键超参。【核实】
- **与官方文档 / 论文的一致性**：项目页与报告在"atom14、双视图、10 靶"上一致；项目页把 SeedProteo 与 SeedFold 并列呈现（预测 + 设计），容易让读者误以为二者是同一模型，需在笔记里区分。【核实】

---

## 6. 实践与应用

- **可用方式**：仅 webserver（`seedfold.io` → proteinDesign）。输入为靶标结构（或 PDB ID）+ 靶链/残基范围 + hotspot + binder 长度区间 + 可选 SS 条件；输出为候选序列/结构。
- **适用场景**：蛋白靶标的 de novo binder；需要控制二级结构组成的场景（选 R 或 D 变体）；长链无条件生成（≤1000 aa）。
- **不适用**：需要 contig 精确控制的任务；需要本地批量推理/微调（无代码无权重）；需要能量项引导或自定义 reward 的任务；需要严格 motif scaffolding 的任务（接口未提供）。
- **选型建议**：追求成功率选 SeedProteo-R（长 SS 段）；追求拓扑多样性选 -D。但注意两者都只在 10 个靶上验证过，外推到新靶标类（如膜蛋白、多聚体界面）无证据。【推断】

---

## 7. 疑问与讨论

- **Q**：它是论文还是技术报告？
  - **A**：技术报告。无同行评审记录，arXiv 上标注为 technical report。引用时应写"技术报告"而非"论文"。【核实】
- **Q**：为什么没有代码？
  - **A**：项目页只给 webserver。截至 2026-11-10 未见 GitHub / HF / NGC 任何链接。是否计划开源【未核实】。【核实：现状】
- **Q**：atom14 算"全原子"吗？
  - **A**：算"全重原子"，不含氢。若下游要做氢键网络或质子化态分析，需要自己加氢。论文标题用 "All-Atom"，按 AF 系惯例指重原子。【核实 + 推断】
- **Q**：用同团队 SeedFold 评估，结果可信吗？
  - **A**：这是本篇最大的方法论疑问。SeedFold 与 SeedProteo 同源，若两者共享训练数据或架构归纳偏置，误差可能相关。论文没有用 AF2/AF3/Boltz 复评。引用其成功率时应加"同源评估器"限定。【推断】
- **Q**：BoltzGen 在 BHRF1 上 627 vs 296，说明什么？
  - **A**：说明在该靶上 BoltzGen 明显更强，也说明 SeedProteo 并非全面领先。写对比时不能只取平均。【核实：表内数字】
- **Q**：扩散步数为什么重要？
  - **A**：它直接决定推理成本与可比性。没有这个数字，就无法与 La-Proteina（400 步）、AF3（200 步）做效率对比。这是本篇最实质的缺口。【核实：缺失】

---

## 8. 学习心得

- **"全原子"在不同论文里指的不是同一件事**：SeedProteo 是逐原子显式扩散（atom14），La-Proteina 是潜变量 + 解码，Complexa 是原子级条件 + 潜变量生成。三者都能在标题里写 all-atom，但计算性质完全不同。做横向表时必须把"生成端自由度"单列一栏。【核实 + 推断】
- **双视图是个被低估的设计**：Design View 负责"造"，Folding View 负责"验"，一套参数两种用法。比起外挂 ESMFold，它省了一次模型部署，但代价是评估独立性变差。这个取舍值得在自研方案里权衡。【推断】
- **MRF 解码解决的是一个真实痛点**：等排氨基酸误判在 binder 设计里会直接毁掉湿实验。多数模型靠外挂 MPNN 绕开，SeedProteo 选择在模型内解决。【核实 + 推断】
- **"查不到"要分类**：本篇的步数/参数量/σ 调度属于"作者没写"；代码缺失属于"确认为无"；训练集与靶标是否同源属于"作者没写但影响大"。三者性质不同，混成一句"未披露"会掩盖真正的风险点。

---

## 9. 参考资料

- 技术报告：Qu W., Ma Y., Ye F., Lu C., Zhou Y., Zhang K., Wang L., Gui M.（技术负责 Quanquan Gu）. *SeedProteo: Accurate De Novo All-Atom Design of Protein Binders.* ByteDance Seed, arXiv:2512.24192（v2, 2026-02-24）.
- 项目页：`https://seedfold.github.io`
- 在线服务：`https://seedfold.io`（proteinDesign）
- 本地材料：`d:\postdoc_work\文献学习\2026-11-10\laproteina-complexa-seedproteo-notes\work\seedproteo.pdf`（及 `seedproteo.txt`）
- 同谱系 / 同任务对照：[La-Proteina](../La-Proteina/README.md)、[Proteina-Complexa](../Proteina-Complexa/README.md)、[RFD4-Proteina](../RFD4-Proteina/README.md)、[Proteina](../Proteina/README.md)
