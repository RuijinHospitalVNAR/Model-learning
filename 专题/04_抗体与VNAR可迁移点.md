---
topic: 抗体与 VNAR 可迁移点
axis: 抗体应用
related_models: [AlphaFold3, Boltz-1, Boltz-2, Chai-1, RFAA, Chroma, IgGM, AntiBMPNN, AbMPNN, AnewOmni, BoltzGen, APM, TorchCraft, SeedProteo, Proteina-Complexa, SAbDab2, Pallatom2]
status: 初稿
verified_at: 2026-11-04
---

# 抗体与 VNAR 可迁移点

> **研究轴**：抗体/纳米抗体场景下的可复用模块、数据、指标与工程约束
> **关联模型篇章**：[01_全原子表示与生成范式](01_全原子表示与生成范式.md)、[02_训练数据构成与切分](02_训练数据构成与切分.md)、[03_评测体系与可复现性](03_评测体系与可复现性.md)、[模型篇章/Pallatom2](../模型篇章/Pallatom2/README.md)
> **记录人**：待填
> **最后更新**：2026-11-04

> 本篇对应主文档 §6「对 VNAR 全原子设计的可复用点与风险（压缩节）」，并只补充主文档 §3.5 的抗体数据与 §5 准则 10。
> 原始素材对该主题的判断本身是压缩的、结论性的；**本篇保持这一密度，不扩写、不补造实验或数字**。凡原文标【推断】【未核实】【论文自称】的，此处保留同一标记。

---

## 1. 专题目标

- 回答一个工程问题：现有全原子／抗体模型里，哪些部分可以迁移到 VNAR（单域抗体，shark），哪些必须重建。
- 约束前提：VNAR 的结构数据量只有 84 个（SAbDab2 v0.3.0），因此本篇讨论的是**迁移路径**，不是"训练一个 VNAR 专用模型"的可行性。
- 明确边界：本篇不重复 02 的训练数据细节与 03 的评测口径细节，只在需要支撑判断时引用其结论。

**一句话判断（原文档同款）**：现阶段的现实路线不是从零训练，而是在 AF3 系骨干上做域适配（继续训练／LoRA）＋把 CDR／框架写成条件接口；同时借用 RFAA 的元素／键级／手性 token 体系处理非经典残基与二硫键。【推断】

---

## 2. 抗体 / 纳米抗体（VNAR）相关能力盘点

只列与抗体／单域抗体直接相关、且在 01 有对照数据的模型；能力描述沿用主文档口径，未做加强。

| 模型 | 抗体相关能力 | 关键限制 |
|---|---|---|
| AlphaFold3（2024） | 抗体-抗原共折叠；AF3 系 atom-token + 坐标扩散的精度基准 | 结构预测而非生成；抗体任务约需 1000 seeds 才饱和【论文自称】 |
| Boltz-1（2024） | 复现 AF3，支持 pocket 条件与 steering | 无模板；训练分辨率论文 9 Å vs 代码 4.0 Å |
| Boltz-2（2025） | AF3 式骨干 + affinity 头 | 切点 2023-06-01，亲和力指标不可与 AF3 横比 |
| Chai-1（2024） | 单序列／ESM embedding／restraint，抗体场景可少依赖 MSA | 训练切点未披露【未核实】 |
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

---

## 4. 可借用指标

| 指标 | 用途 | 注意事项（沿用 03 的口径要求） |
|---|---|---|
| DockQ / DockQ SR | 抗体-抗原界面质量 | 需注明 DockQ 版本与是否按抗原切分 |
| 界面 lDDT | 界面局部几何 | 与 DockQ 并列报告，不单独作为结论 |
| CDR-H3 RMSD | CDR 环几何 | 先声明是否只比主链（IgGM 无侧链） |
| 序列恢复率（recovery） | 逆折叠／序列设计 | AntiBMPNN / AbMPNN 口径需核对 |
| 物理有效性（手性、键长、clash、二硫键几何） | 抗体场景的硬约束 | AF3 自报 4.4% 手性违规；二硫键多数模型无显式监督 |
| 湿实验 hit rate | 最终判据 | 必须给每靶候选数与分母；"75%"与"63.5%"不可直接横比 |

**不能借用为"全原子"证据的指标**：仅主链／Cα 的 RMSD、仅 in-silico 的 designability（见 03 §3 与 §7 V8／V9 的判据设计）。

---

## 5. 工程与算力约束

- **不能从零训练**：84 个 VNAR 结构不足以支撑从零训练，也不足以做可靠留出评估——先决条件是预训练迁移（继续训练／LoRA／条件适配）。
- **采样预算被抗体任务放大**：AF3 论文默认 5 seeds × 5 samples = 25 个样本取 top-1；抗体任务约需 1000 seeds 才饱和。【论文自称】这一点直接抬高湿实验前筛选成本一个量级。【推断】
- **推理期优化可作低成本替代**：TorchCraft 用冻结 AF3 做 test-time 反传，报 4/4 minibinder、4/4 VHH 有结合；代价是单靶耗时与置信度自洽偏差。
- **没有同一 VNAR 靶标的头对头湿实验**：现有通用全原子模型报 in-silico designability，抗体模型报 DockQ 与湿实验 hit rate，两者目前不可直接比较。
- **专用 checkpoint 未必迁移**：OpenDDE 的 `opendde_abag` 在内部纳米抗体＋MC4R 测试（3 seed，`cycle=10/step=200/use_msa=false`）上 3/3 劣于 general（pLDDT 63.56→50.04，ipTM 0.313→0.201）；提示按常规 Ab-Ag 界面强化的权重可能对 VNAR 过拟合，默认应先用 general，abag 只做 A/B。配方未披露，结论仅限该内部测试。【核实·内部记录，未复现】
- **上下文并行只救大体系**：OpenDDE 的 CP4（4 卡）在 N ≤ 900 token 时比单卡更慢、峰值显存更高（N=100 时 15,549 vs 4,959 MiB），N ≥ 1200 才占优，真实价值是让 N=2000 不 OOM。VNAR-抗原复合物多在数百 token，开 CP4 属负优化。【核实：Appendix C Table 3】

---

## 6. 模型对照表

| 模型 | 抗体相关能力 | 可复用模块 | 可借用指标 | 限制 |
| ---- | ---- | ---- | ---- | ---- |
| AlphaFold3 | 抗体-抗原共折叠；AF3 系骨干 | atom-token 扩散骨干；pLDDT／PAE／pTM／ipTM | DockQ、界面 lDDT、CDR RMSD | 预测非生成；抗体需 ~1000 seeds【论文自称】 |
| Boltz-1 / Boltz-2 | 复现 AF3；Boltz-2 带亲和力头 | `full.yaml` 可核实的训练配置；pocket 条件／steering | 同 AF3；亲和力指标仅 Boltz-2 | 切点 2021-09-30 / 2023-06-01 不同，不可混比 |
| Chai-1 | 单序列／restraint，适合设计闭环 | 单序列 + ESM embedding + restraint 接口 | PoseBusters 77%、CASP15 LDDT 0.849、DockQ 69.8%【论文自称】 | 训练切点未披露【未核实】 |
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
| Proteina-Complexa | 全原子 binder（非 Ig） | 潜空间 flow matching + test-time search | 133 靶、>1M 候选、最高 63.5% hit【论文自称】 | 不做抗体 |

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
