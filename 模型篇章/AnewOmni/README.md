---
model: AnewOmni
aliases: [AnewOmni, Anew]
family: 复合物生成（全原子生物分子相互作用生成与序列-结构共设计）
granularity: block 级潜变量 + 逐原子解码（输出为全原子坐标；生成自由度为每化学 block 的 8 维隐变量 + 3 维潜坐标）
backbone: atom-to-block 分解 + CondIterAutoEncoderEdge（IterVAE，latent 8 / hidden 512 / edge 64 / k 9）+ 潜空间 E(3) 等变 EPT 去噪网络（6 层 / 8 头）
generative: 离散时间 DDPM 扩散（ε-预测，100 步，std 10.0），支持 DDIM 采样
conditioning: 口袋原子（10 Å）| hotspots（与 lig_chains 互斥）| 框架库（48 条，含 VHH）| 参考复合物（须预 docked）| 图 prompt（node-level + edge-level）| 无分类器引导（CFG）
data_scope: BioLiP2、PDBBind v2020（refined + other-PL）、CrossDocked2020（test split 仅作 in silico 评测）、SIU 合成集（>5.3M，每 epoch 采 5%）、PepBench、ProtFrag、SAbDab；40% 序列同一性截断；测试为 CrossDocked2020 test / PoseBusters / XDock（139 靶）/ LNR / RAbD；SAbDab 版本与截止日期未核实
status: 已完成
verified_at: 2026-11-04
---

# AnewOmni

> **状态**：已完成（主体已核实；未核实项集中在训练数据清洗细节、逐 benchmark 数值与蒸镏策略）
> **记录人**：个人文献学习
> **最后更新**：2026-11-04
> **证据标记**：`【核实】` 一手论文或仓库事实；`【论文自称】` 仅有作者陈述；`【代码／配置核实】` 仓库代码或配置直接可读；`【推断】` 由前两类推出；`【未核实】` 材料不可得。

---

## 1. 学习背景与目标

- **为什么学**：AnewOmni 是仓库「全原子模型」四路线中唯一把**小分子、肽、抗体**放进同一套表征与同一套采样器的模型（四路线自 2026-11-10 起转为 [01 §2.5](../../专题/01_全原子表示与生成范式.md) 的历史分类，按现行三轴它落在「块级 × 坐标扩散 × graph prompt」）。此前它在仓库里只作为对比表的一行出现，出处标为「待补」，抗体部分则被当作「和 IgGM 类似的多肽/抗体生成器」一笔带过。这两处都需要修正。
- **本次目标**：
  1. 把出处、版本、许可、开源状态查实，不留空；
  2. 明确回答「all-atom」标签在**输出层**与**采样层**是否都成立；
  3. 把抗体路径写到机制级：它做的是**框架固定的 CDR 局部修补**，不是全 Fv 共设计；
  4. 把训练数据与切分写到可指导复现的程度，官方未披露的明确写「未披露」；
  5. 给出 VNAR 场景下可直接借用与必然失效的模块边界。
- **一句话判断**：AnewOmni 的可核实贡献是**把「化学积木块 → 潜变量 → 一次扩散 → 逐原子解码」串成跨模态流水线**，并用**精确似然**替代监督置信头做排序；它的抗体能力来自**外循环共折叠 + 框架库**，而不是模型内部学会了抗原-抗体界面物理。后者是本文最容易被高估的部分。【推断，依据 §3.5、§5.3、§6.3】

---

## 2. 核心概念

| 术语 | 含义 |
| ---- | ---- |
| atom-to-block 分解 | 标准氨基酸与核碱基整块保留；其余分子按从 ChEMBL24 挖掘的频繁片段切分；稀有子结构退化为单原子。【核实：论文方法段；官网口径差异见 §5.2】 |
| block | 生成的基本单位，可以是一个残基、一个环系片段，或一个单原子。block 内部原子拓扑在解码时按模板恢复。【核实 + 推断】 |
| CondIterAutoEncoderEdge（IterVAE） | 把 block 编码为「隐变量 h + 潜坐标 x」的编解码器；`latent_size 8`、`hidden 512`、`edge_size 64`、`k_neighbors 9`；编解码器 6 层 / 8 头 / cutoff 10 Å。【代码／配置核实：`configs/train_vae.yaml`】 |
| 潜坐标 x | 每个 block 的 3 维空间位置；与隐变量 h 一起构成扩散的状态空间，因此去噪发生在**潜空间**而非原子坐标空间。【代码／配置核实：`models/LDM/diffusion/dpm_full.py`】 |
| 图 prompt（graph prompt） | 条件注入接口，分 node-level 与 edge-level，逐步注入去噪轨迹；训练时条件从真值结构随机采样，推理时配合 CFG。【核实 + 代码／配置核实】 |
| hotspot | 用户指定的界面残基集合；与 `lig_chains` 互斥。口袋定义为参考 binder 10 Å 内的残基，或 hotspots 10 Å 内的邻近残基。【代码／配置核实：`README.md`】 |
| framework_lib | 48 条抗体/单域抗体框架模板，条目名为 PDB-ID + 链标记（如 `8fk5_IGC_H_L`）；VHH 条目 `lightchain` 字段为空。**未见 VNAR 型条目**。【代码／配置核实：`api/templates/framework_lib.json`】 |
| CDR inpainting | 抗体设计的实际形态：原生 CDR 被替换为 GLY 假原子，只生成该区段，框架与抗原保持不动。【代码／配置核实：`api/templates/antibody.py`】 |
| 精确似然（exact likelihood） | 用 VAE 的显式似然（含 Jacobian 项）给候选打分，作为无监督、模态无关的排序准则。论文称其优于监督 confidence head，后者会过拟合。【核实：论文 Fig 2f】 |
| 复合评分（composite score） | 抗体外循环的排序键，混合表位覆盖、接触 CDR 比例、ipTM、binder pLDDT、归一化 ipAE、生成似然与归一化 scRMSD。【核实（已在机制专文核对）】 |
| cofold 外循环 | 每轮把候选交 AF3/Protenix/Boltz2 重新共折叠，按复合评分挑父本再生成；默认 `max_num_iterations 60`，默认后端 `protenix_mini_esm_v0.5.0`。【代码／配置核实；机制细节见 [推理时优化机制对比](../../联合分析/推理时优化机制对比-AnewOmni与RFD4-Proteina.md)】 |
| RAbD | 抗体设计标准测试集；本笔记只核实到「SAbDab 训练、RAbD 留作测试」，其版本与复合物数【未核实】。 |

---

## 3. 原理与推导

### 3.1 表征粒度：all-atom 在输出层成立，在采样层不成立

**分解与编码**

1. 输入结构先被切分为 block：标准氨基酸与核碱基不做拆分，整块保留；其余分子按从 ChEMBL24 挖掘的频繁片段切分；无法覆盖的稀有子结构退化为单原子。因此 block 的粒度是**化学惯用单元**，不是固定原子数。【核实：论文方法段】
2. 每个 block 经 IterVAE 编码为 `h ∈ R^8` 与 `x ∈ R^3`；解码时按 block 模板把 h 还原为**该 block 的全部原子坐标**，包括氨基酸侧链的重原子。【代码／配置核实：`latent_size 8`；解码器输出 `atom_coord` 损失项，权重 1.0】
3. 扩散在 `(h, x)` 上运行，不在原子坐标上运行。侧链原子坐标不是扩散变量，而是「block 潜变量 + 上下文」经解码器的确定性函数。【代码／配置核实：`configs/train_ldm.yaml` 无原子级扩散噪声项；`latent_deterministic: true`】

**结论（不用「取决于」）**

- 「逐原子侧链坐标有没有生成」：**有**。解码器输出的是逐原子笛卡尔坐标，不是扭转角，也不是隐式几何；`atom_coord` 损失与 `bond` / `local_distance` 损失共同约束这一层。【代码／配置核实】
- 「侧链构象是否被采样」：**没有**。一个残基的侧链自由度被压进 8 维隐变量，扩散只在这个隐变量上加噪与去噪；同一 block 潜变量对应一个（近似）确定的侧链构象。因此模型不会输出「同一序列的不同侧链 rotamer 分布」。【推断，依据潜变量维度与确定性解码】
- 所以「all-atom 标签名不副实」这一判断**部分适用**：说它「不生成侧链坐标」是错的；说它「像 AF3/RFAA 那样在原子坐标上做扩散、因而覆盖侧链构象多样性」也是错的。准确表述是**原子级输出、block 级生成**。【推断】

这一区分对抗体尤其重要：CDR-H3 的柔性既来自骨架二面角，也来自侧链堆积；后者在 AnewOmni 中被 8 维潜变量压扁。论文的 `normalized_scRMSD_cdr` 排序键能把明显不合理的构象筛掉，但不能恢复构象 ensemble。【推断】

### 3.2 主干：IterVAE 的编码目标与损失构成

- 编码器在 block 图上做消息传递（`k_neighbors 9`、`edge_size 64`、cutoff 10 Å），输出 h 与 x；解码器按 block 模板重建原子坐标。【代码／配置核实】
- 损失为多项加权：`Zh_kl 0.8`、`Zx_kl 1.0`、`atom_coord 1.0`、`block_type 1.0`、`contrastive 0.1`、`local_distance 0.5`、`bond 0.5`；`pocket_mask_ratio 0.05`、`decode_mask_ratio 0.0`。【代码／配置核实：`configs/train_vae.yaml`】
- `pocket_mask_ratio 0.05` 表示训练时随机遮蔽 5% 的口袋 block，迫使解码器在条件不完整时也能重建；`decode_mask_ratio 0.0` 表示解码阶段不额外遮蔽，即**训练与推理的解码条件分布尽量一致**。【代码／配置核实 + 推断】
- `latent_deterministic: true` 说明训练时不对潜变量做重参数化重采样，KL 项只作正则。这与「VAE 用于生成」的常见直觉相反：这里的 VAE 主要承担**可逆压缩 + 似然计算**，随机性交给下游扩散。【代码／配置核实 + 推断】

### 3.3 生成范式：潜空间 DDPM 与采样策略

- **去噪空间**：block 潜变量 `(h, x)`。`EpsilonNet` 的输入是「加噪潜变量 + 条件嵌入 + 时间嵌入」，预测噪声 ε。【代码／配置核实：`models/LDM/diffusion/dpm_full.py`】
- **生成过程**：离散时间 DDPM，`num_steps 100`，`std 10.0`；时间以 beta 时间戳嵌入；`construct_transition` 预计算转移核。【代码／配置核实：`configs/train_ldm.yaml`】
- **等变性与条件注入**：主干为 E(3) 等变 EPT（6 层 / 8 头 / cutoff 3.0 Å），Adapter 把 2D 条件与原子嵌入注入各层。等变性保证潜坐标 x 的旋转/平移一致性，但对 h 无约束。【代码／配置核实】
- **采样**：`sample()` 为完整 DDPM；`sample_ddim()` 用 `linspace(0, num_steps, ddim_steps+1)` 跳步，并要求步数可整除。默认 `n_samples 20`、`batch_size 8`（显存 <12 GB 时的建议值）、`patience 100`（连续生成 100 个后 top-k 未变即停）。【代码／配置核实：`README.md`、`api/generate_with_rank.py`】
- **排序**：`--rank_criterion` 可取 confidence / likelihood / normalized_likelihood / none，**默认 confidence**；论文主张 likelihood 更好，但代码默认值未随之改变。这是论文与配置的一处不一致，见 §5.2。【代码／配置核实】

### 3.4 条件接口：图 prompt 与 CFG

- 条件以图 prompt 形式注入：node-level 描述「哪些 block 属于口袋/表位」，edge-level 描述 block 间约束；注入沿去噪轨迹逐步进行，而非只在 t=0 拼接一次。【核实 + 代码／配置核实】
- 训练时条件从**真值结构**随机采样，包括随机遮蔽比例 `max_cond_ratio_2d/3d 1.0`、氨基酸腐蚀比例 `is_aa_corrupt_ratio 0.1`。这使模型见到「条件不完整/部分错误」的输入，推理时才能用 CFG 放大条件强度。【代码／配置核实 + 推断】
- 口袋条件与配体链条件**互斥**：给了 `lig_chains` 就不能再给 `hotspots`。这一约束在 README 中明确写出，并在 demo 中体现（`tgt_chains: [R]`、`lig_chains: [HL]`）。【代码／配置核实】
- hotspots 的语义是「用户认为重要的界面残基」，模型把它们同时并入口袋；README 警告 hotspots 应覆盖整个口袋，否则条件过窄。【代码／配置核实】

### 3.5 抗体：框架固定的 CDR 修补，而非全 Fv 共设计

**实际流程**

1. **框架选择**：从 48 条 `framework_lib` 中取一条模板（含 VHH，`lightchain` 为空）。框架原子全部冻结，只暴露一个 CDR 区段。【代码／配置核实：`api/templates/framework_lib.json`】
2. **CDR 掩蔽**：按 Chothia 编号定位 CDR；把原 CDR 残基改名为 GLY、原子替换为单碳原子，但**保留 atom id**，以保证化学键拓扑仍成立；两侧各保留 `fr_len=3` 个框架残基作为上下文；`ChainBreakFilter` 与 `LTypeAAFilter` 保证生成结果不产生链断裂与非法残基类型。【代码／配置核实：`api/templates/antibody.py`】
3. **表位条件**：`epitope_block_id` 与 heavy/light block ids 一起进入条件；口袋由参考 binder 10 Å 内残基定义。【代码／配置核实：`data/antibody.py`、`README.md`】
4. **长度**：README 明确写 "Change of the CDR lengths is also supported"；`length_ranges` 默认取原生长度，改为区间即允许长度变化。6 类 CDR（HCDR1/2/3、LCDR1/2/3）逐样本随机选一类，即单次生成只补一个 CDR。【代码／配置核实】
5. **前置条件**：抗体 demo 要求**已预 docked 的抗原-框架复合物**（`8u4r_chothia.pdb`，`tgt_chains [R]`、`lig_chains [HL]`）。模型不解决「框架如何结合抗原」这一层，只解决「在这个姿态下 CDR 该长什么样」。【代码／配置核实：`demo/antibody_single_cdr.yaml` 注释】
6. **外循环**：每轮 10 个父本（top 8 + 随机 2）、每父本 100 条 CDR、按 likelihood 取 top 5 → 50 个候选交 cofold；复合评分排序后重设父本，默认 60 轮，后端默认 Protenix、可换 Boltz2。【核实（机制专文已核）】

**这意味着什么**

- 抗体能力的一半在**外循环**：姿态修正由 AF3/Protenix/Boltz2 完成，AnewOmni 只负责 CDR 序列-构象提议。把 75% 成功率全部记在生成模型头上会高估它。【推断】
- 框架库固定为 48 条，且以 camelid VHH 为主；它给出的是**框架先验**，不是「任意抗体框架都能用」。换框架要先进库。【代码／配置核实】
- 预 docked 要求使它对**未知表位**的适用性受限：需要先有抗原-框架的合理起始姿态，否则 CDR 在一个错误界面上被优化。【推断】

**与三条对照路线的差别**

| 维度 | AnewOmni | AF3 | IgGM | BoltzGen |
| ---- | ---- | ---- | ---- | ---- |
| 任务 | 条件生成（序列-结构共设计），抗体为框架固定 CDR 修补 | 结构预测 | 抗体/纳米抗体序列-结构生成 | binder 从头设计 |
| 表征 | block 级潜变量 + 逐原子解码 | 残基 token + 配体每原子 token | 仅 Cα + SO(3) frame | 全原子扩散（细节本篇未核实） |
| 侧链 | 解码器显式输出，但不被采样 | 显式原子坐标，diffusion 采样 | 需后处理重建 | 未核实 |
| 条件 | 图 prompt + CFG + 口袋 10 Å + 框架库 + 预 docked 复合物 | MSA/模板/配体 CCD；口袋条件需微调 | 表位/框架条件（细节未核实） | 靶点条件（细节未核实） |
| 抗体自由度 | 一次一个 CDR，框架冻结 | 不生成 | 生成整个 binder | 从头生成 binder 链 |

- 与 AF3 的关键区别是**方向相反**：AF3 把条件当输入特征去做坐标回归，AnewOmni 把条件当去噪引导去做分布采样；AF3 的口袋条件需要额外微调，AnewOmni 原生支持 hotspots/lig_chains。【核实：AF3 篇 §3.3；代码／配置核实】
- 与 IgGM 的关键区别是**表征代价**：IgGM 只给 Cα 与每残基 frame，侧链必须外挂重建；AnewOmni 直接出全原子，但把侧链自由度压进 8 维潜变量。两者都不是「原子级构象采样」，差别在于压缩发生在哪一层。【核实 IgGM 表征为二手口径；AnewOmni 侧为代码／配置核实】
- 与 BoltzGen 的关键区别是**任务框定**：BoltzGen 从零生成 binder 链并与靶点共折叠；AnewOmni 的抗体路径是「已有框架 + 已 docked 抗原 + 补一段 CDR」。二者输出物不同，成功率分母也不同，不能横比。BoltzGen 侧细节本篇【未核实】。【推断】

### 3.6 排序：精确似然与置信头的取舍

- VAE 提供显式似然，可对任意模态（小分子、肽、抗体 CDR）给同一套分数，无需训练专门的打分头。论文报告监督 confidence head 在分布外候选上过拟合（Fig 2f），因此主张 likelihood。【核实：论文】
- 代码保留了 confidence 头（`configs/train_conf.yaml`：recycle 1、`use_vae_dec_embedding true`、`use_bonds false`），并把 `confidence` 设为默认排序准则；likelihood 需显式指定。**论文主张与代码默认不一致**，见 §5.2。【代码／配置核实】
- 似然分数的性质是「像不像训练分布」，不是「结合不结合」。用它排序候选可以提升 in-silico 指标，但不保证湿实验命中；论文的湿实验命中率（23%–75%）才是外部效度。【推断】

---

## 4. 训练数据构成与切分

- **数据源与规模**（论文口径）：BioLiP2；PDBBind v2020（refined + other-PL）；CrossDocked2020；SIU 合成集（蛋白-配体，>5.3M，每 epoch 采 5%，损失权重 0.05）；PepBench（肽）；ProtFrag（片段）；SAbDab（抗体）。【核实：论文数据段】
- **合成数据**：SIU 为合成复合物集合，用于扩充蛋白-配体覆盖；其生成方法、去重与物理合理性过滤【未核实】。
- **自蒸馏**：论文正文**未提**使用 AFDB、AF2 或 Boltz-1 的预测结构做训练。按「未披露即未披露」处理，不假定存在。【未核实】
- **清洗与去重**：可核实的只有 40% 序列同一性截断；分辨率阈值、缺失原子处理、链断裂过滤、配体过滤规则均未在可读正文出现。【核实 40%；其余未核实】
- **切分与泄漏控制**：
  - CrossDocked2020 的 test split 仅用于 in silico 评测，不参与训练。【核实】
  - 测试集为 CrossDocked2020 test、PoseBusters、XDock（139 个 RNA/DNA 靶）、LNR（肽）、RAbD（抗体）。【核实】
  - **抗体-抗原难子集是否独立于 RAbD**：未披露。论文以 RAbD 为唯一抗体测试口径，未说明是否另设难子集。【未核实】
  - **泄漏渠道**：BioLiP2 与 PDBBind 的 PDB 条目可能同时出现在 PoseBusters/XDock 的靶点同源家族中；论文未给时间切点，因此「测试结构是否早于训练」无法判断。【推断】
- **覆盖与偏置**：数据以蛋白-配体为主（BioLiP2 + PDBBind + SIU 三个来源都偏小分子），蛋白-核酸仅见于 XDock 评测侧；抗体侧只有 SAbDab，且框架库仅 48 条、以 camelid VHH 为主，**没有 VNAR 条目**。这意味着模型对鲨鱼源单域抗体的先验基本缺失。【代码／配置核实 + 推断】
- **可获取性与许可**：代码 MIT（Copyright 2026 ByteDance / Xiangzhe Kong），权重在 `releases/tag/init`。训练数据不可整体再分发（BioLiP2/PDBBind/CrossDocked 各自有许可约束），训练管线未公开。【核实 + 推断】

---

## 5. 宣称与可验证性

### 5.1 宣称—证据表

| 宣称 | 证据类型 | 核查状态 | 备注 |
| ---- | ---- | ---- | ---- |
| native protein–ligand interactions 回收 50–70% | 表格/正文 | 已核实 | 论文结果段明确给出区间。 |
| 跨模态联合训练额外回收约 10% | 曲线/消融 | 已核实（方向） | 有跨模态对照，但仅给约数，未给逐模态差值。 |
| 湿实验总成功率 23%–75% | 表格 | 已核实 | 论文摘要与正文一致；官网写 29%–75%（见 §5.2）。 |
| 纳米抗体 4 个合成 3 个结合（75%），最佳 Kd 587 nM | 表格 | 已核实（分母极小） | 分母为 4，不能作为稳定成功率估计。 |
| KRAS G12D 小分子 IC50 24 µM / 36 µM | 表格 | 已核实 | 3 个候选送 HTRF，2 个出数；最强候选为 4 个立体异构体混合物，效力可能被低估。 |
| 线性肽 7/30 在 100 µM 下 >50% 抑制（23%） | 表格 | 已核实 | 候选先经 100 ns MD 稳定性筛选。 |
| 环肽 7/20 IC50 < 100 µM（35%） | 表格 | 已核实 | 同上。 |
| PCSK9 正交位肽命中 57%，最佳 Kd 3.19 µM | 表格 | 已核实 | — |
| PCSK9 变构小分子命中 30%，最佳 Kd 2.72 µM | 表格 | 已核实 | 官网 highlight 写 2.98 µM（compound-6），口径冲突见 §5.2。 |
| 晶体结构 RMSD 0.92 Å | 表格 | 已核实 | — |
| 精确似然优于监督 confidence head | 曲线（Fig 2f） | 已核实（论文） | 但代码默认仍用 confidence，见 §5.2。 |
| all-atom 覆盖侧链构象多样性 | 仅文字 | **存疑** | 输出全原子成立，构象采样不成立（§3.1）。 |
| 抗体可由单模型端到端设计 | 仅文字/流程 | **存疑** | 实际依赖 60 轮 cofold 外循环 + 预 docked 复合物（§3.5）。 |
| 多数纳米抗体候选几何合理 | 第三方评测 | **存疑** | OFAntibody 预印本报告多数候选未过 Geometry Check（链断裂/严重位阻），被排除出 enrichment 分析；局部界面指标尚可但有穿模。属二手证据，样本筛选口径未读全。 |
| 训练条目数、参数量、自蒸馏策略 | 未披露 | 未核实 | 可读正文与仓库均无。 |
| SAbDab 版本与截止日期 | 未披露 | 未核实 | 只写库名。 |

- **可复现所需材料**：官方仓库（MIT）提供代码与 config，权重见 `releases/tag/init`；训练数据管线未公开——数据源只列库名，无条目数、无时间切点、无 SAbDab 版本。训练算力有披露（48×A800），但数据不可得，训练不可完全复现。【核实：仓库与论文方法段；训练不可复现为推断】
- **与官方文档／论文的一致性**：atom-to-block 拆解、latent 8、DDPM 100 步、框架库与 CDR inpainting 等主体描述与代码一致；`--rank_criterion` 代码默认 `confidence`，与论文 Fig 2f「精确似然优于监督 confidence head」的结论方向相反（见 §5.2）；官网数字与论文存在口径差（§5.2）。【代码／配置核实 + 核实：论文 Fig 2f】

### 5.2 论文与代码／二手来源的不一致（两边并列）

| 项 | 论文方 | 代码／二手方 | 处理 |
| ---- | ---- | ---- | ---- |
| 代码公开时点 | v1「will be made publicly available later」 | 仓库 2026-04-14 建、2026-06-06 最后推送，MIT + 权重 release | 按「已开源」陈述，注明论文文本滞后 |
| 排序默认值 | 主张 likelihood 优于 confidence | `--rank_criterion` 默认 `confidence` | 引用时写「论文主张 likelihood，代码默认 confidence」 |
| 成功率区间 | 23%–75% | 官网 29%–75% | 正文用论文区间 |
| block 片段来源 | 从 ChEMBL24 挖掘频繁片段，稀有子结构退化为单原子 | 官网称片段由 principal subgraph 算法识别 | 正文用论文口径，官网表述列为差异 |
| PCSK9 最佳 Kd | 2.72 µM | 官网 highlight 2.98 µM（compound-6） | 正文用 2.72 µM |
| 训练算力 | 方法段 48×A800 | `train_vae.yaml` 注释 8×A800 | 视为不同阶段，论文未拆分 |
| XDock 阈值逻辑 | 80% 能量阈值下 70%–80% 靶点达标；同段 50% 阈值下比例更低 | 阈值更严而达标比例更高，内部逻辑可疑 | **不复述为结论**，标存疑 |

### 5.3 「机制存在 ≠ 计算扩展收益」

推理时搜索的机制事实已在专文核实（10 父本 / 每父本 100 条 CDR / 每轮 50 个 cofold 候选 / 60 轮 / 复合评分 / 无收敛判据），此处不重写。本篇只补三点：

1. **挂在哪个主干上**：搜索期间生成主干（VAE + LDM + confidence）冻结，`--ckpt` 指向 `checkpoints/model.ckpt`；可训练的只有外循环的父本选择与候选池。没有按靶点微调的报告。【代码／配置核实】
2. **预算量级**：60 轮 × 50 候选 ≈ 3,000 次 cofold/靶；单次生成默认 20 样本（demo 抗体 50/父本、batch 16），即每轮约 1,000 条 CDR 被生成后只留 50 条进入 cofold。相对 AF3 抗体任务的 ~1,000 seeds，这是**同一量级**的计算开销，但预算分配在「生成—筛选—再生成」而非「单次多采样」。【代码／配置核实 + 推断】
3. **证据强度**：论文未给「轮数 vs 质量」曲线、未给「去掉外循环」的消融、未给停机判据；`max_num_iterations` 是预算上限而非收敛条件。因此可核实的是**机制结构**，不可核实的是**计算扩展的边际收益**。代理奖励（ipTM、scRMSD、似然的加权）本身有 Goodhart 风险：候选被优化到「像 AF3 的自信结构」，不等于结合更好。【推断；详见 [推理时优化机制对比](../../联合分析/推理时优化机制对比-AnewOmni与RFD4-Proteina.md) §5】

### 5.4 评测协议与成本

| Benchmark | 口径 | 指标 | 采样预算 |
| ---- | ---- | ---- | ---- |
| CrossDocked2020 test | 留出 split | 对接质量、相互作用回收 | 未逐项核实 |
| PoseBusters | 物理有效性 | 键长/夹角/clash/手性 | 未逐项核实 |
| XDock | 139 个 RNA/DNA 靶 | 核酸-配体对接 | 未逐项核实（阈值逻辑存疑） |
| LNR | 肽 | 肽-受体 | 未逐项核实 |
| RAbD | 抗体 | CDR 设计、界面 | 抗体 pipeline 60 轮 × 50 候选 |

- 汇总口径为**组合加权总分**（论文 Fig 2a），不是单一指标；逐 benchmark 的 seeds 数与候选数【未核实】。
- 训练：48×A800，AdamW lr 1e-4，epoch 500（VAE）/ 500（LDM）/ 100（confidence）。【核实：论文方法段】
- 推理：默认 batch 8（<12 GB）、20 样本、patience 100；抗体 demo batch 16、50 样本/父本。【代码／配置核实】
- 参数量：未披露。【未核实】

---

## 6. 实践与应用

### 6.1 适用场景与前提

- **适合**：给定口袋/表位与参考复合物，生成小分子、线性/环肽、双硫键肽、头尾环肽；给定已 docked 抗原-框架复合物，补单个或多个 CDR；给已有设计做似然排序。
- **不适合**：无参考姿态的抗原-抗体对接；需要侧链构象 ensemble 的体系；需要显式手性/二硫键约束保证的化学（模板里有 `DiSulfidePeptide`，但约束强度未核实）；训练分布外的罕见化学。
- **前提**：Chothia 编号的框架 PDB；预 docked 复合物；cofold 后端（Protenix/Boltz2/AF3）可用；显存 ≥12 GB 时可用更大 batch。

### 6.2 关键参数

| 参数 | 建议 | 依据 |
| ---- | ---- | ---- |
| `num_steps` | 100（默认）；DDIM 跳步须整除 | 代码／配置核实 |
| `n_samples` | 20（默认）；抗体 50/父本 | 代码／配置核实 |
| `rank_criterion` | 论文主张 likelihood；代码默认 confidence，需显式指定前者 | 代码／配置核实 + 论文 |
| `max_num_iterations` | 60（默认）；按预算下调 | 代码／配置核实 |
| `fr_len` | 3（默认）；增大可给更多框架上下文，代价是可变区被压缩 | 代码／配置核实 + 推断 |
| `length_ranges` | 默认原生长度；允许变化时给区间 | 代码／配置核实 |
| 口袋半径 | 10 Å | 代码／配置核实 |

### 6.3 VNAR 的可迁移点（本篇落点）

**能直接借**

1. **block 表征 + 逐原子解码**：VNAR 单域、无轻链，不需要 VH/VL 配对逻辑；把 VNAR 框架作为一个 block 序列输入即可，`framework_lib` 的 VHH 条目已经验证了 `lightchain` 为空的路径。【代码／配置核实】
2. **CDR inpainting 的条件接口**：`Antibody(cdr_type, fr_len=3)` 不假定有轻链；`epitope_block_id` 只依赖靶链残基。把 `cdr_type` 指向 VNAR 的 CDR3 区段即可复用掩蔽/假原子/链断裂过滤这一整套。【代码／配置核实 + 推断】
3. **长度可变**：VNAR 的 CDR3 常显著长于 VHH，`length_ranges` 机制正好承接；但长度先验来自训练分布，超长区段仍会退化。【代码／配置核实 + 推断】
4. **精确似然排序**：与模态无关、不需要为 VNAR 重训打分头；在候选极少时比 confidence 头更稳（论文的过拟合结论）。【核实 + 推断】
5. **外循环的骨架**：把 cofold 换成 AF3/Boltz2 对 VNAR-抗原复合物重折叠，保留「top-k + 随机探索」的父本更新与复合评分框架；评分项可按单域场景删去轻链相关项。【推断】

**会因缺轻链而失效 / 需改造**

1. **框架库覆盖**：48 条里以 camelid VHH 为主，**未见 VNAR 型条目**；VNAR 的框架二面角与 CDR1/CDR2 拓扑不同，直接借 VHH 框架会把非经典 Cys 位置弄错。【代码／配置核实】
2. **非经典 Cys 与二硫键**：VNAR 常见额外 Cys 对（CDR1–CDR3、CDR3 内部），模型侧没有显式的配对约束接口；`DiSulfidePeptide` 模板针对肽而非抗体，能否约束 VNAR 的 Cys 配对【未核实】。若不能，生成结果需要外部配对检查。【推断】
3. **轻重链配对项**：复合评分里的 `contact_cdr_ratio`、heavy/light block ids 在单域下退化为单链版本；不能原样照搬双链权重，否则轻链项恒为空、分母失真。【推断】
4. **表位覆盖定义**：`bs_overlap` 依赖 binder 与抗原的接触集合，单域界面更小，同一阈值下的覆盖率分布会偏移；阈值需重标定。【推断】
5. **侧链构象**：长 CDR3 的侧链堆积在 8 维潜变量下被压扁；对 VNAR 这种靠长 CDR3 插入型结合的模式，构象多样性缺失比 VHH 更致命。【推断】

**工程建议**

- 先把 VNAR 框架（含非经典 Cys）补进 `framework_lib`，再谈生成；否则条件接口再通用，框架先验仍是错的。
- 用 AnewOmni 做**提议**、AF3/Boltz2 做**裁决**、几何检查（链断裂/位阻/原子重叠）做**硬过滤**；第三方评测已显示单靠 cofold 置信度会放过几何不合理的候选。
- 做 VNAR 基准时，把 RAbD 换成独立的 VNAR-抗原测试集，并按 PDB 释放日期过滤，避免 SAbDab 泄漏。

---

## 7. 疑问与讨论

**Q1：它到底算不算 all-atom？**
- **A**：输出层算，采样层不算。侧链原子坐标由解码器逐原子给出（`atom_coord` 损失权重 1.0），但扩散的自由度是 block 潜变量（8 维 + 3 维坐标），且 `latent_deterministic: true`。因此不能声称它覆盖侧链 rotamer 分布。【代码／配置核实 + 推断】

**Q2：抗体是不是「共设计」？**
- **A**：不是全 Fv 共设计。框架来自 48 条库并冻结，抗原姿态要求预 docked，模型只补一个 CDR；「共设计」发生在 60 轮外循环里，由 cofold 模型完成姿态修正。【代码／配置核实】

**Q3：75% 成功率能信吗？**
- **A**：数字本身核实（4 个合成、3 个 BLI 结合），但分母是 4，置信区间极宽；且第三方评测报告多数候选未过 Geometry Check。合理表述是「在该小样本上观察到 3/4 结合」，不是「纳米抗体设计成功率达 75%」。【核实 + 二手证据】

**Q4：精确似然真比 confidence 好吗？**
- **A**：论文有 Fig 2f 曲线支持，但代码默认仍用 confidence；两者在仓库里同时存在。工程上应先跑 likelihood 排序，再用几何检查兜底，不要只看一个分数。【核实 + 代码／配置核实】

**Q5：60 轮迭代是不是「测试时计算扩展」？**
- **A**：机制上是的（生成—筛选—再生成），但没有轮数-质量曲线，也没有停机判据；`max_num_iterations` 是预算上限。不能把「有迭代」读成「加算力就涨点」。【推断；详见机制专文 §5】

**Q6：和 AF3 的关系是替代还是互补？**
- **A**：互补。AF3 提供条件外挂与高精度共折叠，AnewOmni 提供口袋条件生成与似然排序；AnewOmni 的抗体路径本身就把 AF3/Protenix 当外循环裁决器。【核实 + 推断】

**Q7：训练数据能复现吗？**
- **A**：不能完整复现。数据源清单可读，但条目数、清洗阈值、SAbDab 版本、是否自蒸馏都未披露；代码开源不等于训练可复现。【核实 + 推断】

---

## 8. 学习心得

- **最值得借的是「压缩—采样—解码」三段式**：先用化学惯用单元把原子空间压到 block 潜空间，再在潜空间做扩散，最后逐原子解码。它把「化学合法性」交给 block 模板与解码器，把「多样性」交给潜空间，这个分工对小分子和肽都成立。
- **最容易被高估的是抗体能力**。75% 的分母是 4；框架固定、需预 docked、外循环 60 轮，这三点决定了它不是「端到端抗体设计器」，而是「CDR 提议器 + 共折叠搜索」。第三方评测的几何检查结论是对这一点的外部佐证。
- **最反直觉的一处是 VAE 不作随机源**。`latent_deterministic: true` 与 `decode_mask_ratio: 0.0` 说明这里的 VAE 更像可逆压缩器 + 似然计算器，随机性完全交给下游 DDPM；把它当标准 VAE 理解会推错采样行为。
- **数据透明度是它的短板**。数据源列得比 AF3 清楚（BioLiP2/PDBBind/CrossDocked/SIU/PepBench/ProtFrag/SAbDab），但没有条目数、没有时间切点、没有 SAbDab 版本；「40% 同一性截断」是唯一可核实的清洗参数。
- **对 VNAR，可借的是接口不是先验**。条件接口（表位 block、长度区间、假原子掩蔽）与排序（似然）可以直接搬；框架库与非经典 Cys 约束必须自建，否则借来的是一套错误的几何先验。

---

## 9. 参考资料

### 论文与官方文档

- Kong, X. et al. Programming Biomolecular Interactions with All-Atom Generative Model. bioRxiv (2026). DOI: [10.64898/2026.03.12.711044](https://doi.org/10.64898/2026.03.12.711044)；v1 2026-03-15，v2 2026-06-04；CC BY。
- 官方仓库：https://github.com/bytedance/AnewOmni（MIT，Copyright 2026 ByteDance / Xiangzhe Kong；权重 `releases/tag/init`）
- 项目页：https://anewbt.com/research/anewomni/
- 模型卡（二手）：https://bio.rodeo/models/anewomni

### 代码／配置核实点

- `configs/train_vae.yaml`：`CondIterAutoEncoderEdge`、`latent_size 8`、`hidden 512`、`edge_size 64`、`k_neighbors 9`、损失权重、`pocket_mask_ratio 0.05`、`decode_mask_ratio 0.0`、8×A800 注释。
- `configs/train_ldm.yaml`：`LDMMolDesignClean`、`num_steps 100`、`std 10.0`、EPT 6 层/8 头/cutoff 3.0、`latent_deterministic true`、`is_aa_corrupt_ratio 0.1`、`max_cond_ratio_2d/3d 1.0`。
- `configs/train_conf.yaml`：`Confidence`、recycle 1、`use_vae_dec_embedding true`、`use_bonds false`。
- `models/LDM/diffusion/dpm_full.py`：`EpsilonNet`、`construct_transition`、`sample`/`sample_ddim`（DDIM 整除断言）。
- `api/templates/antibody.py`：`Antibody(cdr_type, fr_len=3)`、`add_dummy_lig`、`remove_ref_lig`、`ChainBreakFilter`/`LTypeAAFilter`。
- `api/templates/framework_lib.json`：48 条框架；VHH 条目 `lightchain` 为空；无 VNAR 条目。
- `api/tools/ab_design.py`：`max_num_iterations 60`、`cofold_model protenix`（`protenix_mini_esm_v0.5.0`）、`Chothia` 重编号。
- `api/generate_with_rank.py`：`--rank_criterion` 默认 `confidence`、`--ckpt checkpoints/model.ckpt`。
- `demo/antibody_single_cdr.yaml`：`8u4r_chothia.pdb`、`tgt_chains [R]`、`lig_chains [HL]`、`HCDR3`、batch 16、50 样本、预 docked 注释。

### 仓库内相关篇目

- [推理时优化机制对比：AnewOmni 与 RFD4 / Proteina](../../联合分析/推理时优化机制对比-AnewOmni与RFD4-Proteina.md)——n_beam、n_loser_up、checkpoint、continuation criteria 的机制级结论，本篇不重写，只补主干冻结、预算量级与证据强度。
- [全原子模型解题思路归类分析](../../联合分析/全原子模型解题思路归类分析.md)——AnewOmni 的路线归类；本篇补齐其「出处待补」。
- [AlphaFold 3](../AlphaFold3/README.md)——条件外挂路线的参照系与证据标记范式。

### 二手转述（不作为一手证据）

- OFAntibody 预印本（arXiv 2610.09548）：报告 AnewOmni 多数纳米抗体候选未通过 Geometry Check，被排除出 enrichment 分析；局部界面指标尚可但存在穿模/原子重叠、AF3 置信度偏低。
- bio.rodeo 模型页与 Anew 官网：成功率区间、PCSK9 best Kd、RMSD 等数字；与论文存在口径差异，已在 §5.2 并列。
