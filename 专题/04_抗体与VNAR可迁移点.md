---
topic: 抗体与 VNAR 可迁移点
axis: 抗体应用
related_models: [AlphaFold3, Mol-1, Boltz-1, Boltz-2, Chai-1, RFAA, Chroma, IgGM, AntiBMPNN, AbMPNN, AnewOmni, BoltzGen, APM, TorchCraft, SeedProteo, Proteina, Proteina-Complexa, SAbDab2, Pallatom2, OpenDDE, ATOMICA]
status: 初稿
verified_at: 2026-11-10
---

# 抗体与 VNAR 可迁移点

> **研究轴**：抗体/纳米抗体场景下的可复用模块、数据、指标与工程约束
> **关联模型篇章**：[01_全原子表示与生成范式](01_全原子表示与生成范式.md)、[02_训练数据构成与切分](02_训练数据构成与切分.md)、[03_评测体系与可复现性](03_评测体系与可复现性.md)；八篇模型篇章——[AlphaFold3](../模型篇章/AlphaFold3/README.md)、[Mol-1](../模型篇章/Mol-1/README.md)、[Proteina](../模型篇章/Proteina/README.md)、[AnewOmni](../模型篇章/AnewOmni/README.md)、[Pallatom2](../模型篇章/Pallatom2/README.md)、[OpenDDE](../模型篇章/OpenDDE/README.md)、[ATOMICA](../模型篇章/ATOMICA/README.md)、[RFD4-Proteina](../模型篇章/RFD4-Proteina/README.md)
> **记录人**：待填
> **最后更新**：2026-11-10

> 本篇原为主文档 §6「对 VNAR 全原子设计的可复用点与风险（压缩节）」的压缩回填，并补充主文档 §3.5 的抗体数据与 §5 准则 10。**2026-11-10 收口**：改为用八篇模型篇章的已核实结论写实，产出四份可执行清单（数据底数 → 角色分工 → 四个坑 → 下一步），不再做倡议式叙述。
> **不补造实验或数字**：凡原文标【推断】【未核实】【论文自称】的，此处保留同一标记；新增的量化只来自已入仓的模型篇章与 02／03 专题。

---

## 1. 专题目标

- 回答一个工程问题：现有全原子／抗体模型里，哪些部分可以迁移到 VNAR（单域抗体，shark），哪些必须重建。
- 约束前提：VNAR 的结构数据量只有 84 个（SAbDab2 v0.3.0），因此本篇讨论的是**迁移路径**，不是"训练一个 VNAR 专用模型"的可行性。
- 明确边界：本篇不重复 02 的训练数据细节与 03 的评测口径细节，只在需要支撑判断时引用其结论。

**一句话判断（原文档同款）**：现阶段的现实路线不是从零训练，而是在 AF3 系骨干上做域适配（继续训练／LoRA）＋把 CDR／框架写成条件接口；同时借用 RFAA 的元素／键级／手性 token 体系处理非经典残基与二硫键。【推断】

**收口结论（2026-11-10 补）**：在 84 个 VNAR 结构上，"域适配"也只能是**轻量**的——656M 参数级的全原子模型对 84 个样本做全参微调在统计上没有意义（§3.2）。所以可执行的起点不是训练，而是**先用通用权重把 84 个 VNAR 的 DockQ 基线打出来**（§7.3），再决定值不值得动权重。

---

## 2. 抗体 / 纳米抗体（VNAR）相关能力盘点

只列与抗体／单域抗体直接相关、且在 01 有对照数据的模型；能力描述沿用主文档口径，未做加强。

| 模型 | 抗体相关能力 | 关键限制 |
|---|---|---|
| AlphaFold3（2024） | 抗体-抗原共折叠；AF3 系 atom-token + 坐标扩散的精度基准 | 结构预测而非生成；抗体任务约需 1000 seeds 才饱和【论文自称】 |
| Boltz-1（2024） | 复现 AF3，支持 pocket 条件与 steering；pocket 为**单模型**实现（30% 迭代部分揭示），不需另训一套权重 | 无模板；训练分辨率论文 9 Å vs 代码 4.0 Å |
| Boltz-2（2025） | AF3 式骨干 + affinity 头 | 切点 2023-06-01，亲和力指标不可与 AF3 横比 |
| Chai-1（2024） | 单序列／ESM embedding／restraint，抗体场景可少依赖 MSA | 训练切点 2021-01-12【核实：论文三处】 |
| IgGM（2025） | 唯一开放权重、支持 nanobody 的抗体 CDR 共设计；抗原 + epitope + mask | 只生成 Cα + SO(3) frame，**无侧链**，不算全原子；抗原刚性假设 |
| AntiBMPNN／AbMPNN（2025） | 抗体逆折叠序列设计，recovery 高 | 主链级（N/CA/C/O）；训练集规模未披露【未核实】 |
| AnewOmni（2026） | graph prompt 支持纳米抗体／肽／小分子统一建模；报 KRAS G12D 纳米抗体湿实验 | 预印本；抗体仅 CDR／纳米抗体；低通量筛选；侧链为 block/VAE 解码 |
| BoltzGen（2025） | design as structure prediction；报纳米抗体湿实验 | 需靶标结构；各模态成功率差异大 |
| APM（2025） | 抗原→CDR-H3、受体→肽的逐链设计；残基 + 4 个侧链扭转角 | 无湿实验，指标全为计算代理 |
| TorchCraft（2026） | 冻结 AF3 做 test-time 反传优化，报 4/4 VHH 有结合 | 单靶耗时；置信度自洽偏差；无训练 |
| OpenDDE（2026） | AF3 家族全原子共折叠，JSON 支持任意链（含 VHH 案例）；另有 abag 专用 checkpoint | 非序列设计；abag 在纳米抗体上劣于 general（内部 3/3 seed）；VNAR 占比未披露，只能 zero-shot |
| SeedProteo（2026） | AF3 式 atom14 扩散 + MRF 解码，报 2 靶 pM 级 | 仅 2 靶湿实验，证据薄【未核实】 |
| Proteina-Complexa（2026） | 全原子 mini-protein／peptide binder，非 Ig | 不做抗体；不能直接当抗体模型用 |
| RFAA（2024） | 元素／键级／手性 token 体系可处理非标准残基、共价修饰 | 抗体不是其主战场；大体系算不动 |
| Chroma（2023） | 可组合能量 conditioner（对称性／子结构／形状），免重训切换条件 | 成功率约 3%；不支持小分子 |
| ATOMICA（2025） | 重原子界面表征，可把 VNAR-抗原直接当蛋白-蛋白界面编码，用 `z_interface` 做表位相似性检索、用 ATOMICAScore 做接触位点归因 | 非生成式；无氢／电荷／键级；CDR 柔性未建模；**Q-BioLiP 中抗体/纳米抗体占比未披露**，只能 zero-shot 用通用权重 |

---

## 3. 可复用模块与数据

### 3.1 可直接复用的四点（主文档 §6.1 原样）

1. **AF3 系 atom-token + 坐标扩散是当前最成熟的全原子骨干。** 对抗体，最现实的做法不是从零训练，而是在 AF3／Boltz 系骨干上做抗体域适配（继续训练或 LoRA），并把 CDR／框架作为条件接口。
2. **RFAA 的元素／键级／手性 token 体系可直接借用于非标准残基与二硫键。** VNAR 的非经典半胱氨酸、环化肽、修饰残基都需要这套表示；纯 20 氨基酸词表不够。
3. **Chroma／AnewOmni 的可组合条件接口适合 VNAR 的几何约束。** CDR3 长度、二硫键配对、框架锚定、表位残基，都可以写成约束而不是重训。
4. **SAbDab2 的 `ab-ag-split` 是目前最干净的抗体评测协议。** 做 VNAR 基准时应在此基础上再加"单域 + 物种（shark）"筛选。

### 3.2 抗体数据规模（主文档 §3.5，数字原样）

| 数据集 | 规模 | 日期 | 来源性质 |
|---|---|---|---|
| SAbDab | 10,840 结构；8,633 含配对 VH/VL；21,285 Fv；10,200 带抗原；735 有亲和力数据 | 2026-05-28 | 【核实：官网统计页】 |
| SAbDab-nano | 2,246 单域结构；4,684 Fv；2,114 带抗原；86 有亲和力数据 | 2026-05-28 | 【核实：官网统计页】 |
| SAbDab2 ML Dataset | 16,415 结构／8,992 PDB／5,556 unique antibodies | v0.3.0，2026-10-13 | 【核实：Zenodo】 |
| 其中 VNAR | 84 个结构（67 带抗原 + 17 无抗原），占 0.5% | 同上 | 【核实】 |
| 其中 VHH-like 单域 | 3,427 个结构（2,156 带抗原） | 同上 | 【核实】 |
| PLAbDab-nano | 4,913 条 VHH + VNAR 序列，来自 796 项研究 | 2024 | 【核实：NAR 论文摘要】 |
| OAS | >10 亿序列、>80 研究；2022 年论文报 3.5B 序列，配对仅 121,838 | — | 【官网／论文】 |

**三个硬约束（主文档 §3.5 原样）：**

1. **VNAR 只有 84 个结构。** 这不足以训练一个从零开始的结构生成器，也不足以做可靠的留出评估。任何"VNAR 专用全原子模型"的宣称，都必须说明它用了什么预训练迁移。
2. **OAS 只有序列，没有坐标。** 用它做结构监督必须经过预测结构，等于引入自蒸馏偏差。
3. **框架与抗原偏差。** 抗体结构高度偏向少数胚系（IGHV1-69、IGHV4-39 等）与免疫优势抗原；VNAR 的框架与 VHH 不同，且没有 CDR2、CDR3 更长、常含非经典半胱氨酸。【领域共识 + 推断】

**量化：参数量 vs 样本量（收口新增）**

| 模型 | 可核实参数量 | 表征类型 | 在 84 个 VNAR 上的可行操作 |
|---|---|---|---|
| ATOMICA | 8.4M | 判别式（非生成） | 冻结后只训任务头也可行，但没有 VNAR 界面标签可训 |
| Pallatom2（v1） | 65.9M | 重原子生成 | 不重训；只做 zero-shot 与 partial diffusion 局部重设计 |
| OpenDDE | 655.8M | 全原子共折叠 | 只做 zero-shot 推理；abag 专用权重在纳米抗体上已见退化（§5.2） |
| AlphaFold3 | 未披露 | 全原子共折叠 | 只做推理；抗体任务约需 1000 seeds（§5.1） |

参数量取自 [01 §6.2 覆盖矩阵](01_全原子表示与生成范式.md)（A 级证据），AF3 未披露。**656M 参数对 84 个样本，等于每参数 1.3×10⁻⁷ 个样本**；即便只训 1% 参数（LoRA 级），可训练维度仍然远多于样本数，验证集只能切出个位数抗原簇——这样的微调结果无法区分"学到 VNAR 先验"与"记住 84 个结构"。【推断，依据参数量与样本量】

**结论**：VNAR 上的权重动作只有三类——① zero-shot 推理；② 轻量条件化（把 CDR／框架／表位写成条件接口，权重不动）；③ 在此之上的局部重设计（§3.4）。全参微调不在可行域内；任何"VNAR 专用模型"的宣称，都应先出示预训练来源、冻结策略与留出抗原簇数量。

### 3.3 可借用的骨架与训练接口

- 骨干：AF3／Boltz 系（atom-token + EDM 扩散，200 步，x0 预测）；改动点放在抗体域适配与条件接口，而不是重造表征。
- 表征补丁：RFAA 的 46 种元素 token、键级 token、显式 R/S token（用于非经典 Cys 与修饰残基）。
- 条件接口：BoltzGen／Chai-1 的约束语言（pocket／SS／共价键／restraint）、AnewOmni 的 graph prompt（环化／motif／非天然残基／锚定）、IgGM 的表位 + 掩码模式。
- 评测协议：SAbDab2 `ab-split` / `ab-ag-split`（按克隆／抗原切分，而非结构随机切分）。

### 3.4 局部重设计路径（Pallatom2，本轮新增）

> 以下不在主文档 §6.1 的原始四点内，属本轮新增的可迁移路径；只列做法与边界，不复制模型篇章整节。

- **做法**：以已有 VNAR–靶标复合物为起点，`norepack` 锁住框架侧链、`redesign` 放开 CDR 区域，再做 partial diffusion（t=0.2 起步、t=0.3 加强）；筛选用 `min-ipAE + ptm-binder`，小批量 BLI 验证。条件系统把「二级结构」注入 residue 级、「固定侧链朝向」注入 atom pair 级，因此能同时约束框架与 CDR 的局部几何。【核实】
- **先例**：纳米抗体 PB-272（起始 KD 2,000 nM、无结合）→ 改造 CDR 界面生成 500 个候选 → 提名 10 个 → 7 个表达并上 BLI → PB-276 4.31 nM、PB-277 5.93 nM。单 scaffold、单案例，属**存在性证明**，不是可重复率。【核实：Pallatom2 技术报告 p.22–23】
- **两条禁止用法**：
  1. **不是 de novo VNAR 生成器**：训练语料是 AFDB／TED 的单域／多域球状蛋白，**不含 Ig 折叠**，也没有 VNAR 条件接口；84 个 VNAR 结构不足以从零训练或可靠留出。【推断，依据 02 的语料描述 + 本篇 §3.2】
  2. **不是 KD 预测器**：`ptm-binder` 是 binder **单链折叠置信度**，不是界面或亲和力置信度；界面不确定性要看 `min-ipAE`，亲和力只能靠湿实验。【核实】
- **完整论证与数据出处**：[模型篇章/Pallatom2](../模型篇章/Pallatom2/README.md) §4、§6。

### 3.5 八篇模型在 VNAR 流程里的角色分工（收口新增）

八篇覆盖四个不可互相替代的角色。**同一个模型只归一个角色**：能做预测的模型未必能评估，能生成的模型未必能做局部改造。

| 角色 | 模型 | 在 VNAR 流程里能做什么 | 不能做什么（硬边界） | 证据 |
|---|---|---|---|---|
| **预测器**（精度基准） | AlphaFold3 | 抗原–VNAR 共折叠；输出 pLDDT／PAE／ipTM，可转 DockQ | 不生成序列；抗体任务约需 1000 seeds 才饱和，成本见 §5.1 | AF3 篇章；§5.1 |
| **预测器**（本地可跑、可从 checkpoint 微调） | Mol-1 | AF3 系开源复现，JSON 模板驱动（仓库自带纳米抗体案例）；微调时权重严格加载、优化器与调度器重置 | "能微调"不等于"该微调"（§3.2 的样本量约束）；JSON 模板相对 MSA 的收益尚未量化，是本仓库的未决问题 | [Mol-1 篇章](../模型篇章/Mol-1/README.md) |
| **预测器**（条件接口最全） | OpenDDE | general 权重；序列必需、MSA 可关、模板／contact／epitope／movable_chains／共价键可选；可用掩码已知原子做局部约束 | abag 专用权重在纳米抗体上 3/3 劣于 general，默认只用 general | §5.2；[OpenDDE 篇章](../模型篇章/OpenDDE/README.md) |
| **评估器**（唯一，且是判别式） | ATOMICA | 把 VNAR–抗原当蛋白–蛋白界面编码；`z_interface` 做表位相似性检索，ATOMICAScore 做接触位点归因 | 不生成、不给亲和力；无氢／电荷／键级，CDR 柔性未建模；抗体占比未披露 | [ATOMICA 篇章](../模型篇章/ATOMICA/README.md) |
| **局部重设计**（唯一有湿实验闭环） | Pallatom2 | `norepack` 锁框架侧链 + `redesign` 放开 CDR + partial diffusion（t=0.2 起步、0.3 加强）；`min-ipAE + ptm-binder` 筛选；PB-272 → PB-276/277 是存在性证明 | 不是 de novo 生成器（语料不含 Ig 折叠、无 VNAR 条件接口）；`ptm-binder` 不是 KD 预测器 | §3.4 |
| **de novo 生成器** | AnewOmni | 原子 block + VAE 潜空间；graph prompt 支持纳米抗体／环化／非天然残基／锚定；报 KRAS G12D 纳米抗体湿实验 | 预印本；侧链为 block 解码；通量与成功率低，抗体只到 CDR／纳米抗体 | §2 表 |
| **de novo 生成器**（全原子，且自带局部重设计） | RFD4-Proteina | 残基级 32 维隐变量 + 130M 全原子解码器；仓库带 `examples/antibody_design/vhh/`（4 框架 × 2 靶）；`partial.yaml` 从第 35/60 步加噪，可锁 framework 只重画 CDR | 不能当预测器（无 pLDDT／PAE 头，评估必须外挂 AF3／AF2）；抗体／纳米抗体在训练集中的占比未披露；二硫键模式未显式建模；无 VNAR 湿实验 | [RFD4-Proteina 篇章](../模型篇章/RFD4-Proteina/README.md) §6.2 |
| **主链草稿**（不构成全原子路径） | Proteina | Cα flow matching 出主链，可接逆折叠补序列 | 无侧链、无抗体先验，在 VNAR 上无任何验证；不能算 VNAR 全原子生成方案 | [01 §6](01_全原子表示与生成范式.md)（粒度＝Cα 单点） |

**这四条角色的组合方式**（也是本篇的唯一推荐链路）：OpenDDE／AF3 预测 → ATOMICA 做界面评估与接触归因 → Pallatom2 对命中复合物做局部重设计 → 湿实验。AnewOmni 与 RFD4-Proteina 只在"新靶点、没有起始复合物"时才进入链路（前者偏小分子／肽的统一建模，后者偏全原子 de novo 且自带 partial diffusion 局部重设计）；Proteina 不进入 VNAR 链路，保留在 01 的谱系里说明 Cα 级方案的边界。

---

## 4. 可借用指标

| 指标 | 用途 | 注意事项（沿用 03 的口径要求） |
|---|---|---|
| DockQ / DockQ SR | 抗体-抗原界面质量 | 需注明 DockQ 版本与是否按抗原切分 |
| 界面 lDDT | 界面局部几何 | 与 DockQ 并列报告，不单独作为结论 |
| CDR-H3 RMSD | CDR 环几何 | 先声明是否只比主链（IgGM 无侧链） |
| 序列恢复率（recovery） | 逆折叠／序列设计 | AntiBMPNN / AbMPNN 口径需核对 |
| 物理有效性（手性、键长、clash、二硫键几何） | 抗体场景的硬约束 | AF3 自报 4.4% 手性违规；二硫键多数模型无显式监督 |
| 湿实验 hit rate | 最终判据 | 必须给每靶候选数与分母；不同工作的 hit 数字不可直接横比 |

**不能借用为"全原子"证据的指标**：仅主链／Cα 的 RMSD、仅 in-silico 的 designability（见 03 §3 与 §7 V8／V9 的判据设计）。

---

## 5. 工程与算力约束与四个坑

### 5.1 工程与算力约束

- **不能从零训练**：84 个 VNAR 结构不足以支撑从零训练，也不足以做可靠留出评估——先决条件是预训练迁移（继续训练／LoRA／条件适配）。
- **采样预算被抗体任务放大**：AF3 论文默认 5 seeds × 5 samples = 25 个样本取 top-1；抗体任务约需 1000 seeds 才饱和。【论文自称】这一点直接抬高湿实验前筛选成本一个量级；按 84 个结构 × 1000 seeds 估算，单模型一轮全量评测即 8.4 万次前向，须在动手前定好 seeds 上限与早停规则。【论文自称 + 推断】
- **推理期优化可作低成本替代**：TorchCraft 用冻结 AF3 做 test-time 反传，报 4/4 minibinder、4/4 VHH 有结合；代价是单靶耗时与置信度自洽偏差。
- **没有同一 VNAR 靶标的头对头湿实验**：现有通用全原子模型报 in-silico designability，抗体模型报 DockQ 与湿实验 hit rate，两者目前不可直接比较。
- **专用 checkpoint 未必迁移**：OpenDDE 的 `opendde_abag` 在内部纳米抗体＋MC4R 测试（3 seed，`cycle=10/step=200/use_msa=false`）上 3/3 劣于 general（pLDDT 63.56→50.04，ipTM 0.313→0.201）；提示按常规 Ab-Ag 界面强化的权重可能对 VNAR 过拟合，默认应先用 general，abag 只做 A/B。配方未披露，结论仅限该内部测试。【核实·内部记录，未复现】
- **上下文并行只救大体系**：OpenDDE 的 CP4（4 卡）在 N ≤ 900 token 时比单卡更慢、峰值显存更高（N=100 时 15,549 vs 4,959 MiB），N ≥ 1200 才占优，真实价值是让 N=2000 不 OOM。VNAR-抗原复合物多在数百 token，开 CP4 属负优化。【核实：Appendix C Table 3】

### 5.2 四个必须避开的坑（收口新增）

1. **abag 专用权重在纳米抗体上是负迁移。** 数据见 §5.1 末两条：`opendde_abag` 在内部纳米抗体测试上 3/3 劣于 general。**处理边界**：默认 general，abag 只做 A/B；且该结论只覆盖一个内部测试、配方未披露，不能外推成"abag 权重普遍无效"。【核实·内部记录，未复现】
2. **VNAR 的 MSA 稀缺会让 MSA 依赖型模型退化。** 鲨鱼序列在 UniRef／ColabFold 中代表性弱，MSA 深度与多样性均低于人源 VHH，AF3／Boltz 系的有效性随之下降；这也是 Chai-1 单序列模式与 OpenDDE `use_msa=false` 在 VNAR 上值得优先试的原因。**处理边界**：报结果必须同时给"有 MSA／无 MSA"两组，否则无法区分是模型不行还是 MSA 不行。【推断，依据 MSA 依赖机制 + Chai-1 单序列结论】
3. **非经典 Cys 与长 CDR3 落在多数模型的表达边界之外。** VNAR 无 CDR2、CDR3 更长、额外 Cys 形成非经典二硫键；而 01 §6 里逐原子模型的化学语义只到"元素 + 坐标"，Pallatom2 与 ATOMICA 明确无键级／无电荷。**处理边界**：三选一——用带键级／手性的表示（RFAA 的 46 元素 + 键级 + R/S token）、把二硫键写成显式共价约束（BoltzGen／Chai-1 的约束语言）、或在评测里单独统计二硫键几何与手性违规率（AF3 自报 4.4% 手性违规）。三者至少要有一项，否则"全原子"只是原子数够、化学不成立。
4. **切点泄漏会直接污染 AF3 系评测。** AF3／Boltz-1 训练切点 2021-09-30，Boltz-2 为 2023-06-01，Chai-1 为 2021-01-12、OpenDDE 未披露。SAbDab2 v0.3.0（2026-10-13）收录的 84 个 VNAR 中，**首次 release 早于对应切点的结构已被训练集见过**，在其上算 DockQ 会系统性偏高；晚于切点的才是干净子集。**处理边界**：按每条链的 PDB 首次 release date 分层，对每个模型按其切点分别报"切点前／切点后"两栏并给分母——切点后子集可能只有个位数，单栏报均值等于没有结论。【核实：各模型切点；推断：泄漏方向】

---

## 6. 模型对照表

| 模型 | 抗体相关能力 | 可复用模块 | 可借用指标 | 限制 |
| ---- | ---- | ---- | ---- | ---- |
| AlphaFold3 | 抗体-抗原共折叠；AF3 系骨干 | atom-token 扩散骨干；pLDDT／PAE／pTM／ipTM | DockQ、界面 lDDT、CDR RMSD | 预测非生成；抗体需 ~1000 seeds【论文自称】 |
| Boltz-1 / Boltz-2 | 复现 AF3；Boltz-2 带亲和力头 | `full.yaml` 可核实的训练配置；pocket 条件／steering（单模型，AF3 则需另训有／无 pocket 两套） | 同 AF3；亲和力指标仅 Boltz-2 | 切点 2021-09-30 / 2023-06-01 不同，不可混比 |
| Chai-1 | 单序列／restraint，适合设计闭环 | 单序列 + ESM embedding + restraint 接口 | PoseBusters 77%、CASP15 LDDT 0.849、DockQ 69.8%【论文自称】 | 训练切点 2021-01-12【核实：论文三处】 |
| RFAA | 非标准残基／共价修饰的表示 | 46 元素 token、键级、显式 R/S | 配体重原子 RMSD、手性合规 | 抗体非主战场；大体系算不动 |
| Chroma | 可组合条件（对称／子结构／形状） | 能量 conditioner，免重训切换条件 | 设计成功率、晶体 RMSD | 成功率约 3%；不支持小分子 |
| IgGM | 抗体 CDR 共设计（含 nanobody） | 抗原 + epitope + mask；consistency 降步数 | SAb23H2 RMSD 2.14 Å／lDDT 0.89／DockQ SR 0.467；H3 恢复 36% | 无侧链；抗原刚性 |
| AntiBMPNN / AbMPNN | 抗体逆折叠 | 主链图逆折叠 | recovery >80%；单点设计 75%【论文自称】 | 训练集规模未披露【未核实】 |
| AnewOmni | 纳米抗体／肽／小分子统一 block 表征 | graph prompt + CFG；block VAE | KRAS G12D 75% hit、Kd 587 nM；PCSK9 57%／30%【论文自称】 | 预印本；低通量；侧链隐式 |
| BoltzGen | design as prediction；纳米抗体湿实验 | 统一约束语言（位点／SS／共价键／构象） | 26 靶；纳米抗体 60% 靶有 hit【论文自称】 | 需靶标结构；模态差异大 |
| APM | 抗原→CDR-H3 | 残基 + 4 侧链扭转角；逐链条件 | RAbD 恢复／RMSD | 无湿实验；折叠精度不及专用预测器 |
| TorchCraft | 冻结 AF3 的 test-time 优化 | 序列 logits + CDR 限定；无需训练 | 4/4 minibinder、4/4 VHH | 单靶耗时；置信度自洽偏差 |
| OpenDDE | AF3 家族全原子共折叠（含抗体-抗原）；Ab-Ag benchmark 有 top-rank／oracle 双线 | structural token 角色分解；shape-complementarity loss；统一 masked 生成 | DockQ ≥0.23；top-rank 与 oracle 需同时报 | 非序列设计；abag checkpoint 在纳米抗体上劣于 general【内部记录】；preview 跨版本不复现 |
| SeedProteo | AF3 式 atom14 + MRF 解码 | 靶标口袋条件 | 2 靶 70–80% hit、pM 级【未核实】 | 仅 2 靶，证据薄 |
| Proteina-Complexa | 全原子 binder（非 Ig） | 潜空间 flow matching + test-time search | 19 个 in-silico 靶标（23 候选筛到 19）、桥接实验、无湿实验【核实：论文全文】；原「133 靶／>1M 候选／63.5% hit／碳水化合物 binder」一手论文查无，已撤【待溯源，见 `01` 表注 4】 | 不做抗体 |
| RFD4-Proteina | 全原子 de novo（残基级 32 维隐变量 + 130M 解码器）；自带 partial diffusion 局部重设计 | 双模态并行流匹配；`examples/antibody_design/`（VHH 4 框架 × 2 靶）；beam search + AF3/AF2 oracle | 蛋白 binder 10.6%／小分子 14.3%／AME 12.1%／DNA 3.5%【核实】；湿实验 158/188 | 无 pLDDT／PAE 头，不能当预测器；抗体数据占比未披露；搜索负收益（普通判据 naive 41.0 > temporal 25.1） |

---

## 7. 结论与开放问题

### 7.1 必须警惕的五点（主文档 §6.2 原样）

1. **VNAR 只有 84 个结构（SAbDab2 0.3.0）。** 任何"VNAR 专用全原子模型"的宣称，先问预训练来源与迁移方式；84 个结构不足以支撑从零训练。
2. **架构先验不匹配。** 现有抗体模型的 tokenizer 与环长先验按 VH／VHH 拓扑设计；VNAR 无 CDR2、CDR3 更长、无 VL 界面，直接套用会产生系统性偏差。
3. **二硫键几何没有监督信号。** 多数模型不显式建模二硫键；非经典 Cys 的配对模式在 VNAR 中高度多样，必须单独建模块或约束。
4. **"全原子"标签虚高。** IgGM／AntiBMPNN 是残基／主链级；La-Proteina／AnewOmni 的侧链是隐变量或 block 解码。选型时要按"是否生成侧链坐标"而不是按宣传词。
5. **评测不可比。** 通用全原子模型报 in-silico designability，抗体模型报 DockQ 与湿实验 hit rate；目前没有同一个 VNAR 靶标上的头对头湿实验。

### 7.2 开放问题

- 没有 VNAR 靶标上的头对头湿实验，因此"哪个模型对 VNAR 更好"目前无法回答。
- 84 个结构只够做零样本／少样本评估；`ab-ag-split` 叠加"单域 + shark"筛选后，可用于留出的抗原簇数量尚未核算（对应 03 §7 V9）。
- 二硫键几何、非经典 Cys 配对是否值得单独建模块，缺少可引用的对照实验。
- 抗体侧"全原子"的实际收益（侧链坐标对结合预测／设计的增量）没有独立验证。

### 7.3 一条可执行的下一步（收口新增）

**先做 zero-shot DockQ 基线，不训练任何权重。** 四步可落地：

1. **取集合**：SAbDab2 v0.3.0 的 84 个 VNAR 里取 67 个带抗原的复合物；按每条链的 PDB 首次 release date 打标，并按各模型的切点分层（§5.2 坑 4）。
2. **定运行**：AF3、Boltz-1、Chai-1（单序列模式）、OpenDDE（general 权重）四个预测器各跑一遍；seeds 上限先定 100（不是 1000），同时记录"有 MSA／无 MSA"两组（§5.2 坑 2），单样本成本按秒／结构记账。
3. **报四件事**：DockQ + 界面 lDDT（主指标）、手性／二硫键几何违规率（§5.2 坑 3）、切点前／切点后两栏分母、推理成本。
4. **判据**：若切点后子集上 OpenDDE-general 的 DockQ 中位数站得住其 Ab–Ag benchmark 的 0.23 量级，说明"通用权重 zero-shot 跑 VNAR"可用，再谈条件化与局部重设计；若站不住，先回头查数据与切点，不要动权重。

**这一步同时回答本仓库的两个悬空问题**：Mol-1 篇章留下的"JSON 模板驱动相对纯 MSA 在纳米抗体上的收益"（Mol-1 Q4），以及本篇 §7.2 列出的开放问题——两者都要靠这条基线，而不是靠再读一篇论文。

**不要做的事**：不要一上来做 LoRA／继续训练（§3.2）；不要用 abag 权重（§5.2 坑 1）；不要把 ATOMICA 的界面相似度当亲和力（§3.5）。

---

## 8. 参考

主文档与证据日志中的原始来源，本篇只列与本主题直接相关者：

- SAbDab 统计页：https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/stats/（更新于 2026-05-28）
- SAbDab-nano 统计页：https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/nano/stats/（更新于 2026-05-28）
- SAbDab2 ML Dataset（v0.3.0）：https://zenodo.org/records/23343734
- OAS：https://opig.stats.ox.ac.uk/webapps/oas
- PLAbDab-nano：https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkae881/7816862
- IgGM 论文（ICLR 2025）：https://proceedings.iclr.cc/paper_files/paper/2025/file/fa5b423e24b442180bcd4e13ae75a27f-Paper-Conference.pdf ；代码：https://github.com/TencentAI4S/IgGM ；权重：https://zenodo.org/records/16909543
- AntiBMPNN：https://pmc.ncbi.nlm.nih.gov/articles/PMC12463024/
- AbMPNN：https://arxiv.org/html/2310.19513v1
- AnewOmni：https://doi.org/10.64898/2026.03.12.711044 ；代码：https://github.com/bytedance/AnewOmni
- BoltzGen：https://doi.org/10.1101/2025.11.20.689494 ；代码：https://github.com/HannesStark/boltzgen
- APM：https://arxiv.org/abs/2504.13075
- TorchCraft：https://arxiv.org/html/2609.19770
- AlphaFold3：https://www.nature.com/articles/s41586-024-07487-w
- Boltz-1：https://www.biorxiv.org/content/10.1101/2024.11.19.624167v4
