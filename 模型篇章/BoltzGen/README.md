---
model: BoltzGen
aliases: [BoltzGen, HannesStark/boltzgen]
family: 复合物生成（binder design）/ 序列-结构设计
granularity: 全原子坐标 + 残基 token 双轨（AF3／Boltz 式原子槽位）；设计残基的序列身份以几何方式编码，采样输出中设计残基只有主链坐标，侧链由后续 inverse folding 补齐
backbone: 沿用 Boltz-2 架构（PairFormer 式 Trunk + Diffusion Module）；扩散模块为 3 atom-level + 24 token-level + 3 atom-level 层，atom 层用序列局部注意力，atom↔token 用平均／展开转换
generative: 生成式；全原子坐标扩散（沿用 Boltz-2 扩散头），把"设计"重写为"给定部分结构、预测其余部分"的同一去噪目标
conditioning: 设计规范语言（entities / 序列长度与设计区 / binding_types / secondary_structure / structure_groups + visibility / constraints 共价键与总长 / 靶标结构）；无 MSA 作为设计条件的证据，亦未见 CFG
data_scope: PDB 实验结构 + AFDB（AF2 自蒸馏）+ Boltz-1 自蒸馏（蛋白-配体、RNA、DNA-蛋白）；随机裁剪 + 随机选取设计区／条件区构造多任务；具体 cutoff、分辨率阈值、条目数在附录 B.1，本次未取到【未核实】
status: 学习中
verified_at: 2026-11-10
---

# BoltzGen

> **状态**：学习中（出处、架构、许可证、流水线与湿实验口径已核实；训练集 cutoff／规模、设计表征细节未核实）
> **记录人**：团队共同学习
> **最后更新**：2026-11-10

---

## 1. 学习背景与目标

- **为什么学**：BoltzGen 是 2025 年底"**把 de novo binder design 直接写成结构预测**"这条路线最完整的公开实现，且是**唯一同时放出训练代码、权重、数据和全部设计**的 binder 设计模型。对本仓库的意义有两点：① 它把"设计"与"预测"放进同一个扩散目标，等于用预测任务的监督去撑设计任务的泛化；② 它给了一套**可执行的约束语言**（结合位点／二级结构／共价键／靶标构象），这是 Chroma 的"能量项 + 分类器"之外的另一条条件化路线。
- **先确认出处（本笔记的第一个任务）**：
  - 论文真实存在：*BoltzGen: Toward Universal Binder Design*，bioRxiv **10.1101/2025.11.20.689494**，2025-11-24 发布；一作 Hannes Stärk（MIT CSAIL），通讯 hstark@csail.mit.edu。【核实：bioRxiv 元数据 + 官方 PDF 首页】
  - **与 Boltz 系列同源但不是 Boltz-2 的别名**：作者列表含 Jeremy Wohlwend、Gabriele Corso、Regina Barzilay、Tommi Jaakkola，机构标注含 "Boltz"；但它是**独立仓库**（`HannesStark/boltzgen`）、**独立权重**、**独立任务定位（binder design，而非结构／亲和力预测）**，架构上"沿用 Boltz-2 并做修改"。**把它当成 Boltz-2 的别名是错误的。**【核实：论文作者与机构标注 + 仓库 README】
  - 代码与权重：MIT；约 6 GB 权重自动下载，另需 ESMFold2（约 33 GB）。【代码／配置核实：仓库 LICENSE 与 README】
- **本次目标**：把三轴（粒度／生成范式／条件接口）按专题 [01 §6.1](../../专题/01_全原子表示与生成范式.md) 的判据填实；把"设计残基只有主链坐标"这一实现限制讲清楚；核实训练数据的真实构成，核不到的部分明确标注查过哪里。
- **一句话判断**：BoltzGen 的实质创新不在采样器（仍是 Boltz-2 的坐标扩散），而在**任务重写 + 设计规范语言 + 完整筛选流水线**——它用"随机掩掉结构的一部分让模型补全"把 folding、binder design、motif scaffolding、无条件设计统一成一个训练目标。【论文自称 + 代码／配置核实】

---

## 2. 核心概念

| 术语 | 含义 |
| ---- | ---- |
| design as structure prediction | 把 binder 设计表述为"目标结构已知、binder 结构未知"的部分结构预测：同一个扩散模型既补全已知复合物，也生成全新 binder。【论文自称】
| 纯几何的设计残基表征 | 设计残基不输入序列 one-hot，而用几何／原子占位表达"这里是一个待定残基"，使设计样本与预测样本共用同一输入范式。【论文自称；具体原子槽位编码细节本次未核到——见 §3.3】
| design specification（设计规范） | 一份 YAML，描述 entities、序列长度与设计区、结合类型、二级结构、结构分组与可见性、共价键与总长约束。是 BoltzGen 的条件接口本体。【代码／配置核实：`example/README.md`】
| `binding_types` | 逐残基标记 binder 与靶标的结合角色（如 `uuuuBBBuNNNuBuu`），本质是逐残基的界面约束。【代码／配置核实】
| `structure_groups` + `visibility` | 把若干链／片段编为一组并规定谁能"看到"谁，用于固定靶标构象、隐藏某些界面信息。【代码／配置核实】
| `constraints` | 成对残基约束：共价键（bonds）与总长（total length），以残基对形式写入。【代码／配置核实】
| intermediate_designs | 扩散采样直接产物；**设计残基只有主链原子坐标，侧链坐标为 (0,0,0)**。【代码／配置核实：README 原文】
| inverse folding | 把主链变成序列：默认 SolubleMPNN，也可用 BoltzIF（`protein-small_molecule` 协议下用 BoltzIF + Boltz-2 affinity）。【代码／配置核实】
| refolding / final_ranked_designs | 用预测器重折并打分、过滤、去冗余后输出可上机的少量候选。【代码／配置核实：输出目录结构】 |
| budget / num_designs | 采样预算。README 给出的量级是 1 万–6 万条设计，最终只取数十条验证。【代码／配置核实：`--num_designs` 示例】 |

---

## 3. 原理与推导

### 3.1 任务统一：为什么"设计"可以写成"预测"

- **形式化**：设复合物完整结构为 $x$，把它分成已知部分 $x_{\text{known}}$（靶标、可选 motif、可选固定残基）与待设计部分 $x_{\text{design}}$。普通预测是"给定序列／MSA 求 $x$"，binder 设计是"给定 $x_{\text{known}}$ 求 $x_{\text{design}}$ 及其序列"。BoltzGen 把两者都写成**在扩散去噪中只对 $x_{\text{design}}$ 施加监督／只对 $x_{\text{known}}$ 施加条件**，于是同一个去噪目标自动覆盖两种任务。【论文自称 + 推断】
- **训练时的做法**：论文原文说数据"被随机裁剪，并通过随机选择结构中被设计或被条件化的部分，用于一组多样化的任务"（Figure 10）。这句话是"多任务"的实现级描述：**不是**给每个任务单独建数据集，而是对同一条结构反复重采样"哪些残基被掩掉"。【论文自称：Methods "Training Data" 段】
- **它解决什么**：结构预测任务的监督信号量级远大于真实 binder 复合物，联合训练等于给设计任务加了强几何先验；论文也据此声称"单一模型同时达到 SOTA folding 性能"（Supplementary Fig. 15）。【论文自称】

### 3.2 架构：Trunk 一次，Diffusion 多次

- **沿用 Boltz-2**：Trunk（PairFormer 式，含 MSA／pair 表征）产出 token 与 pairwise 表征，作为条件送入 Diffusion Module；**trunk 每个样本只跑一次，扩散模块做多次去噪**。这是 Boltz 系列与 AF3 一致的"重骨干、轻采样"结构。【论文自称 + 代码／配置核实：README 架构段】
- **Diffusion Module 的层构成**（论文 Methods 原文）：3 个 atom-level 层 → 24 个 token-level 层 → 3 个 atom-level 层；atom 层用**序列局部注意力**；atom↔token 之间用平均／展开转换；条件信息在扩散模块内注入。【核实：论文 Methods "Diffusion Module" 段，经检索片段核对】
- **相对 Boltz-2 的修改**：本笔记能确证的只有"保留 Boltz-2 架构并做修改"这一句 + 上述扩散模块层构成；**具体改了哪些模块（是否改 token 化、是否去掉 affinity 头）本次未核到**【未核实】。写对比时不要假设它与 Boltz-2 逐层相同。

### 3.3 设计残基怎么表示（最容易误读的一节）

- **能确证的两条**：
  1. 论文把设计残基的表征称为**纯几何**的（不靠序列 one-hot 告诉模型"这里该是什么氨基酸"），从而使设计样本与预测样本可同时训练。【论文自称】
  2. **采样产物中，设计残基只有主链原子有坐标，侧链坐标恒为 (0,0,0)**；序列由后续 inverse folding 给出，再由 refolding 打分。【代码／配置核实：README 原文】
- **不能确证的一条**：二手解读普遍称 BoltzGen 用"**14 原子槽位**（4 主链 + 10 虚拟侧链原子，用虚拟原子落点编码氨基酸身份）"来填这个几何表征。该方案在 RFdiffusion3／Emyx 的技术材料里有明确描述，但 BoltzGen 原文方法节（§3.1–3.2）本次**没有取到全文**，无法确认两者是否同一实现。**因此本笔记不把它写成事实**，只标注为【二手解读，未核实】。
- **对使用的直接含义**：BoltzGen 不是"一步出全原子 binder"的模型。它出的是**主链 + 结合姿态**，侧链与序列在第二个阶段补。这决定了它的误差来源与 RFAA／AF3 那类"直接回归全原子坐标"的模型不同：**侧链错误不会被扩散过程纠正，只会被 inverse folding + refold 过滤掉**。

### 3.4 条件接口

| 条件 | 写入方式 | 是否需重训 | 代价／局限 |
| ---- | ---- | ---- | ---- |
| 靶标结构 | 在 YAML 中以 PDB／mmCIF 引入 entity | 否 | 必须有靶标结构；无结构目标不适用 |
| 结合位点 | `binding_types` 逐残基标记 binder／靶标角色 | 否 | 位点是残基级粒度，不能指定原子级化学 | 
| 二级结构 | `secondary_structure` 字符串（H/E/L） | 否 | 每残基一个字母，长度须与设计区一致 |
| 序列长度与设计区 | `sequence: 15..20AAAAAAVTTTT18PPP` 形式：数字段为长度采样，字母段为固定残基，设计区显式标出 | 否 | 长度分布是采样先验，不保证命中指定值 |
| 共价键／总长 | `constraints`：残基对 bonds、total length | 否 | 成对残基约束；复杂拓扑需手写残基索引 |
| 结构分组可见性 | `structure_groups` + `visibility` | 否 | 控制"谁看到谁"，用于固定构象／遮挡信息 |
| 序列（预测模式） | 直接给序列，走 folding 任务 | 否 | 同一套权重 |
| MSA | 未见作为设计条件的用法；trunk 是否用 MSA 本次未核实 | — | **不要把 BoltzGen 写成 MSA 条件模型** |

- **与 Chroma 的路线差异**：Chroma 的条件是**采样期能量项**（几何类按构造满足、语义类靠分类器）；BoltzGen 的条件是**输入规范 + 训练期随机掩码学出来的**——条件在训练分布内是"见过的题型"，不需要额外分类器，但**超出训练分布的条件组合没有理论保证**。【推断，依据两篇方法】

### 3.5 流水线与算力

```
design（扩散采样，1e4–6e4 条）
  → intermediate_designs/            # 设计残基仅主链有坐标
  → inverse folding                  # SolubleMPNN 或 BoltzIF
  → intermediate_designs_inverse_folded/
  → refolding（预测器重折）           # refold_cif/
  → 过滤 + 排序 + 去冗余              # final_ranked_designs/
```

- 协议：`protein-anything` / `peptide-anything` / `protein-small_molecule`（后者用 BoltzIF + Boltz-2 affinity 打分）。【代码／配置核实：README】
- **成本口径**：论文的 10 个新靶点是"每靶 60,000 条设计、取前 15 条上机"，也就是**命中率的分母是计算预算，不是上机条数**。引用它的 hit rate 时必须带上这个分母。【论文自称】

---

## 4. 训练数据构成与切分

- **数据源**（【论文自称：Methods "Training Data" 段】）：
  - **PDB 实验结构**（含 RNA、DNA、蛋白-小分子等复合物）；
  - **AFDB（AlphaFold2 自蒸馏结构）**；
  - **Boltz-1 自蒸馏结构**，用于蛋白-配体、RNA、DNA-蛋白（即 PDB 覆盖不足的模态）。
- **与 Boltz-2 的差异**（论文原话）："Our data also differs from Boltz-2 by not including the upsampled antibody and …"——即**去掉了 Boltz-2 对数据做的抗体上采样**（后文本次未取全）。这说明它在 Boltz-2 数据管线上做减法，而非另起一套。【论文自称】
- **多任务构造**：随机裁剪 + 随机指定"设计区／条件区"，同一结构可产出 folding、binder design、motif scaffolding、无条件设计等样本。【论文自称】
- **切分与泄漏控制**：
  - **cutoff 日期、分辨率阈值、条目数、聚类阈值：本次未核实。** 论文把细节放在 Appendix B.1，而我把能试的路都试过了：
    - 官方 PDF（hannes-stark.com/assets/boltzgen.pdf）：抓取在约 16.7 KB 处截断，只到正文第 6 页（Wetlab Results），Methods §3.3 与附录未进入可读范围；
    - PMC12697729：返回 HHS 漏洞披露提示页，正文不可读；
    - europepmc fullTextXML：返回空；
    - bioRxiv 全文页：返回空；
    - 仓库 README、`example/README.md`、jsDelivr 文件清单：只有推理用法，无训练数据说明。
  - **可做的推断（标【推断】）**：既然作者自述数据"与 Boltz-2 的差异仅在于去掉抗体上采样"，则 **cutoff 很可能沿用 Boltz-2 的 2023-06-01**；但这只是推断，**不能当作已核实事实**。
- **评测泄漏控制（这条反而核实得比较硬）**：10 个"新靶点"的入选标准是"**全 PDB 中不存在与该靶序列同一性 >30% 且处于结合态的蛋白**"，实际同一性 1%–27.9%；并用 6 Å 界面残基 + Folddisco IDF + PPIRef 计算**结合位点相似度**，用于说明"不是抄了已有界面"。【论文自称 + 方法可复核】
- **可获取性**：论文声明放出 training code、inference code、weights 与全部 designs，MIT。【核实：论文 Open Source Release 段 + 仓库 LICENSE】

---

## 5. 宣称与可验证性

| 宣称 | 证据类型 | 核查状态 | 备注 |
| ---- | ---- | ---- | ---- |
| 出处与团队 | 论文元数据 + 作者机构 | 已核实 | bioRxiv 10.1101/2025.11.20.689494；与 Boltz 团队同源，**非 Boltz-2 别名** |
| MIT 许可、权重与训练代码开放 | LICENSE + README | 已核实 | 代码 MIT（Copyright 2025 Hannes Stärk）；权重约 6 GB 自动下载，另需 ESMFold2 约 33 GB |
| 单一全原子扩散模型同时做 design 与 folding | 论文叙述 + 代码 | 论文自称 | 结构上确是同一套权重、同一扩散模块；"SOTA folding"仅见 Supplementary Fig. 15，未独立复现 |
| 26 靶 / 8 个设计 campaign 湿实验 | 表格 + 图 | 论文自称 | 目前是作者与合作湿实验室自测；部分数据作者称因合作方要求暂不公开 |
| 10 个新靶：60% 靶点有 hit（纳米抗体）、50%（蛋白） | 湿实验 | 论文自称 | **分母是每靶 60,000 设计、取前 15 上机**；"hit"含筛选阳性，作者自己区分 screening hit 与 confirmed binder |
| Rag GTPase 线性肽 88% / 二硫键环肽 28% | 湿实验 | 论文自称 | 低微摩尔级；样本量小 |
| 抗微生物肽 19.5% 设计抑制生长 >4× | 湿实验 | 论文自称 | 单一 campaign |
| 设计残基"全原子" | 代码 | **需限定** | 采样产物设计残基**只有主链坐标**，侧链为 0；全原子性来自 inverse folding + refold，不是扩散直接生成 |
| 训练数据构成（PDB + AFDB + Boltz-1 蒸馏） | 论文 Methods | 已核实 | 具体 cutoff／规模／阈值未核实（见 §4） |
| 设计规范语言可表达位点／SS／共价键／构象 | 代码／示例 | 已核实 | YAML 字段与示例齐全，可直接跑通 |

- **可复现所需材料**：MIT 代码 + ~6 GB 权重 + ESMFold2（~33 GB）+ 靶标结构；**算力门槛在采样预算**——论文量级是每靶 6 万条设计，不是几百条。
- **一致性检查**：专题 01 §6.2 与对比 CSV 把 BoltzGen 的"公开程度"记为 A（MIT 代码 + 权重），与仓库一致；但 CSV 的"表征粒度"写"design residue 同时出序列与全原子坐标"，与 README 的"设计残基只有主链坐标"冲突，需订正（见 work 下待订正清单）。

---

## 6. 实践与应用

- **适用**：蛋白／肽／纳米抗体／小分子靶标的 binder 设计；带位点、二级结构、共价键约束的设计；需要"设计—筛选—重折"完整闭环且要求全流程开源的场景。
- **不适用**：没有靶标结构的目标；要求扩散一步出全原子（含侧链）的场景；低算力快速试错（6 万条设计的预算不是小数目）。
- **最小用法**：

```bash
boltzgen run <design_spec>.yaml --num_designs 10000 --budget <N> --output ./out
```

  输出依次落在 `intermediate_designs/` → `intermediate_designs_inverse_folded/` → `refold_cif/` → `final_ranked_designs/`。【代码／配置核实：README】
- **关键旋钮**：`num_designs`（预算）、`budget`（最终保留数）、协议（`protein-anything` / `peptide-anything` / `protein-small_molecule`）、inverse folding 后端（SolubleMPNN vs BoltzIF）、refold 打分器。
- **与我们 VNAR 工作的接口**：把 FR 作为固定残基写进 `sequence` 的字母段、CDR 区写成长度采样段，用 `binding_types` 指定表位残基，走 `protein-anything`；产出主链后用 BoltzIF／SolubleMPNN 出序列，再用 Boltz-2 或 AF3 refold 打分。风险：① 设计分布里纳米抗体样本量取决于训练数据，作者已明确**去掉了 Boltz-2 的抗体上采样**，对抗体类可能不如通用 binder 友好【推断】；② 长 CDR3 的长度采样先验需自己设。

---

## 7. 疑问与讨论

- **Q**：BoltzGen 和 Boltz-2 到底是什么关系？
  - **A**：同实验室（MIT + Boltz）的后续工作，复用 Boltz-2 架构并修改，但**任务、权重、仓库都独立**。把它写成"Boltz-2 的 binder 设计版"是准确的；写成"Boltz-2 的别名"是错的。
- **Q**：它算不算"全原子生成"？
  - **A**：模型输入输出是原子级表示，但**设计残基的侧链不由扩散生成**（坐标为 0），全原子结构要靠 inverse folding + refold 补齐。按专题 01 §2.2 的五档判据，它是"原子级表示 + 两段式全原子"，不是"每个重原子都被扩散"。
- **Q**："design as prediction" 是营销词还是真机制？
  - **A**：有机制支撑——同一扩散目标、同一套权重、训练时随机掩码切换任务。但它不能推出"设计精度等于预测精度"：设计样本的序列是由模型自己定的，预测样本的序列是给定的，监督信号强度不同。【推断】
- **Q**：hit rate 能不能直接和 RFdiffusion / BindCraft 比？
  - **A**：不能直接比。BoltzGen 的分母是"每靶 6 万条设计取前 15 上机"，且靶点经过"全 PDB 无 >30% 同源结合结构"的筛选；与别的工作报告的 hit rate 在预算、靶点难度、hit 定义上都不同口径。【核实：论文方法 + 论文对 hit 的定义】

---

## 8. 学习心得

- 这篇最值得学的不是模型，是**评测设计**：先定义"什么算 hit"（screening hit vs confirmed binder），再把靶点难度用"全 PDB 结合态同源度"量化，最后把计算预算写进分母。这比多刷几个 benchmark 有用。
- "Design as prediction" 的价值在于**借用预测任务的监督规模**，代价是设计残基的序列信息必须被"几何化"才能进同一个扩散头——而这一步的具体实现，恰恰是论文里最容易被二手解读带偏的地方（14 原子槽位之说未见原文确证）。
- 一个诚实的工程取舍：它没有追求"一步出全原子"，而是接受"主链扩散 + 逆折叠 + 重折"的三段式。这说明在当前算力下，**侧链与序列仍适合交给专用模块**，全原子扩散并不是唯一解。

---

## 9. 参考资料

- Stärk H., Faltings F., Choi M., et al. *BoltzGen: Toward Universal Binder Design.* bioRxiv 10.1101/2025.11.20.689494 (2025-11-24).
- 代码与权重：https://github.com/HannesStark/boltzgen （MIT）
- 设计规范示例：仓库 `example/README.md`（entities / sequence / binding_types / secondary_structure / structure_groups / constraints）
- 论文方法片段（经检索核对）：Methods "Training Data"、"Diffusion Module" 段
- 仓库内相关笔记：[AlphaFold3](../AlphaFold3/README.md)、[RFAA](../RFAA/README.md)、[Chroma](../Chroma/README.md)、专题 [01 §2.2 / §6.1 / §6.2](../../专题/01_全原子表示与生成范式.md)
