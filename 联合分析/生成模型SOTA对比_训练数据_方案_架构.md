# 生成模型 SOTA 横向对比：训练数据 · 方案 · 架构

> **范围**：RFD4-Proteina、Proteina、La-Proteina、Proteina-Complexa、Chroma、BoltzGen、SeedProteo、APM、IgGM、AnewOmni（主表 10 个）；TorchCraft、ESM3 作为边界案例单列（§7），不进主表。
> **材料**：全部取自本仓库 `模型篇章/<model>/README.md`，与篇章冲突时以篇章为准；不引入仓库之外的二手数字。各模型的一手出处、代码/权重状态见对应篇章，本篇不重复。
> **不涉及**：`专题/01`、对比 CSV/矩阵（另一会话在改）；`联合分析/结构预测SOTA对比_训练数据_方案_架构.md`（并发写入，不触碰）。
> 最后更新：2026-11-10

---

## 0. 读法与证据约定

### 0.1 证据标签

| 标签 | 含义 | 使用规则 |
| --- | --- | --- |
| 【核实】 | 论文正文/表格/图或代码配置直接支持，模型篇章已复核 | 可直接引用 |
| 【论文自称】 | 作者在正文/摘要中的宣称，未见独立复现 | 必须带"作者称"语境 |
| 【代码／配置核实】 | 仓库代码、config、权重页可复核 | 可引用，但注意代码默认值与论文口径可能相反 |
| 【推断】 | 由已核实事实推出，非原文直述 | 必须显式标注推理链 |
| 【未核实】 | 本次没有取到证据 | 不得写成"作者没写" |

### 0.2 三条判定约定

1. **"全原子"按输出判定，不按采样自由度判定。** 只要模型（含其声明或惯用的补全链，如 IF、MPNN+refold）能给出可重建全部重原子的坐标（或坐标+身份），就算全原子输出；同时注明两件事：全原子是生成器自带还是外挂补全，以及它的生成端到底是"直接原子"、"Cα+潜变量+解码"、"主链+逆折叠"还是"token"。这条约定直接决定 La-Proteina / Proteina-Complexa / Proteina / BoltzGen 的写法不能混。 但在本仓库口径下，**Proteina 的生成器输出仅为主链（Cα-only）**，“全原子”只出现在其评测链（IF + refold），因此篇章与矩阵按“仅主链”记录，不计作全原子输出。
2. **进主表要同时满足三条**：(a) 有可复述的训练目标与采样过程；(b) 输出至少是主链坐标；(c) 任务是结构生成本身，而不是"对一个冻结预测器做输入反演"或"只做序列/注释". 因此 TorchCraft、ESM3 只能做边界案例。
3. **空白分三种，不混写**：
   - **作者未写**：篇章明确说论文/附录没有该信息（例如多数模型的切分与泄漏声明）；
   - **本次未取到**：篇章注明"未核实/未取到"（例如 BoltzGen 的 Appendix B.1、Chroma 的条目数）；
   - **设计上就没有**：例如 APM 明确"不指定结合位点"，不是漏写。

### 0.3 本篇不回答

- 不做跨任务排行榜。不同任务的"成功率/designability/hit rate/能量"不是同一量纲，强行排序会制造伪 SOTA。
- 不重抄各模型的完整指标表；只保留会影响横向判断的数字，并标口径。
- 不评价代码工程质量与许可证，除非它影响可复现性判断。

---

## 1. 名单与一句话定位

| 模型 | 正典篇章 | 谱系位置 | 一句话定位 | 进主表 |
| --- | --- | --- | --- | --- |
| RFD4-Proteina | [README](../模型篇章/RFD4-Proteina/README.md) | Cα+32 维 latent 的界面补全分支 | 以 PDB 界面为主、物理自蒸馏扩数据，把 design 写成条件补全的联合生成器 | 是 |
| Proteina | [README](../模型篇章/Proteina/README.md) | Cα-only rectified flow 起点 | 纯 Cα 流匹配，序列靠外挂 MPNN，全原子只在评测链出现 | 是 |
| La-Proteina | [README](../模型篇章/La-Proteina/README.md) | Cα+8 维部分潜变量 | 单体全原子生成的主干，VAE 解码原子+身份，400 步双时间 SDE | 是 |
| Proteina-Complexa | [README](../模型篇章/Proteina-Complexa/README.md) | La-Proteina 的复合物分支 | 复用 La-Proteina VAE，加靶标条件与 test-time search 的 binder 设计器 | 是 |
| Chroma | [README](../模型篇章/Chroma/README.md) | 等变图网络 + 独立 design network | 可组合几何/语义条件的单体与 motif 生成器，序列与 χ 由 design network 出 | 是 |
| BoltzGen | [README](../模型篇章/BoltzGen/README.md) | Boltz-2 扩散头的 design 分支 | 把 folding/binder/motif 统一成随机 mask 的条件补全；设计残基只出主链，侧链靠 IF | 是 |
| SeedProteo | [README](../模型篇章/SeedProteo/README.md) | AF3 式 atom14 直接扩散 | 主表里唯一直接原子级扩散；Design/Folding 双视图 + MRF 序列解码 | 是 |
| APM | [README](../模型篇章/APM/README.md) | 残基 frame+侧链隐变量 | SE(3) 结构流 + 离散序列流，三模块迭代；链级条件，不做表位 | 是 |
| IgGM | [README](../模型篇章/IgGM/README.md) | 抗体专用三条扩散 | 抗原+表位+框架条件下联合生成抗体序列与复合物；SAbDab ≤2022-12-31 | 是 |
| AnewOmni | [README](../模型篇章/AnewOmni/README.md) | 块级 latent + 原子解码 | 化学积木块→潜变量→一次扩散→逐原子解码，抗体能力靠外循环共折叠 | 是 |
| TorchCraft | [README](../模型篇章/TorchCraft/README.md) | 预测器反演 | 冻结 AF3，只优化序列 logits；不训练生成器、不采样分布 | 否（§7.1） |
| ESM3 | [README](../模型篇章/ESM3/README.md) | 离散掩码语言模型 | 序列/结构/功能多轨 token 生成；结构轨只有主链，无侧链 | 否（§7.2） |

---

## 2. 训练数据

### 2.1 数据源、切点、规模、过滤、蒸馏

| 模型 | 主数据源与切点 | 规模 | 过滤/去重 | 蒸馏/合成 |
| --- | --- | --- | --- | --- |
| RFD4-Proteina | PDB 蛋白-蛋白界面，**截止 2024-01-01**；AFDB 单体补二级结构上下文；酶/配体/核酸界面专项【核实：§4】 | 界面条目数、AFDB 规模本篇未载【未核实】 | 界面定义/分辨率/聚类阈值未载；作者**未给序列同一性阈值**【核实：§4 未声明】 | **有**：物理自蒸馏（Rosetta 能量/氢键标签）+ 合成界面/对称样本【核实】 |
| Proteina | 主训练集是 **AFDB（AF2 预测结构）**，无 PDB 切点；PDB 只用于 LoRA 微调【核实】 | D_FS 588,318；D_21M 20,874,485（50% 聚类代表）【核实】 | PDB 可设计子集 90,423 / 114,076 链；AFDB 过滤细节未载【核实/未核实】 | AFDB 本身是 AF2 蒸馏产物；无额外自蒸馏【核实】 |
| La-Proteina | AFDB Foldseek 簇代表；主集 pLDDT≥80、32–512 aa；长蛋白子集专门扩到 4.69e7→4.04e6 簇【核实】 | 主集约 3.5e5 簇【核实】 | 主集 pLDDT≥80、32–512 aa；motif 集独立 Foldseek 簇，max 356 aa、pLDDT≥80【核实】 | AFDB 预测结构；无在线自蒸馏【核实】 |
| Proteina-Complexa | AFDB 单体 + **Teddymer 合成二聚体** + PDB 多聚体【核实】 | Teddymer 10,089,503→3,556,223 簇；PDB 45,856→110,976 链【核实】 | 链长 50–256；RMSD≤5.0 Å；loop<50%；Rg<3.0；界面侧链用最低能量构象、不调整【核实】 | **有**：Teddymer 是合成复合物；NGC 另称 500k 合成集，与论文口径冲突【未核实】 |
| Chroma | PDB 查询 **2022-03-20** + UniProt 2022_01 + Pfam 35【核实】 | 条目数本次未取到【未核实】 | 序列同一性聚类阈值本次未取到；条件接口为 5 类可组合 Conditioner（substructure/symmetry/shape/ProClass/ProCap）【核实/未核实】 | 未见自蒸馏/合成；纯数据库【核实：未见】 |
| BoltzGen | PDB + AFDB + **Boltz-1 自蒸馏**；相对 Boltz-2 **去掉抗体上采样**【核实/论文自称】 | cutoff/规模/阈值本次未核实（Appendix B.1 未取得）【未核实】 | 随机裁剪 + 随机指定设计区/条件区；其他过滤未核实【论文自称/未核实】 | **有**：AFDB（AF2）+ Boltz-1 自蒸馏（蛋白-配体、RNA、DNA-蛋白）【核实】 |
| SeedProteo | AFDB 单体 ~0.5M + ESMAtlas；Pinder ~50k 双链；HumanPPI ~0.1M DDI【核实】 | 合计 ~0.7M【核实】 | 50–768 aa；pLDDT>80；coil<50%；界面定义未载【核实】 | AFDB/ESMAtlas 预测结构；无自蒸馏【核实】 |
| APM | 单链 187,494（PDB 18,684 + Swiss-Prot 140,769 + AFDB 28,041）；多链 11,620【核实】 | 187,494 + 11,620【核实】 | 单链 pLDDT>85/95；多链**排除 SAbDab 的 PDB ID** 与 <30 aa 链【核实】 | AFDB/Swiss-Prot 预测结构；无自蒸馏【核实】 |
| IgGM | **SAbDab ≤2022-12-31**：6,448 paired + 1,907 single-chain；CD-HIT 95%→2,436 簇【核实】 | 8,355 条 → 2,436 簇【核实】 | CD-HIT 95%；长度/抗原类型等细节未载【核实/未核实】 | 无；实验结构【核实】 |
| AnewOmni | BioLiP2 + PDBBind v2020 + CrossDocked2020 + SIU 合成 >5.3M + PepBench + ProtFrag + SAbDab【核实】 | 合成 >5.3M；各集条目未载【核实/未核实】 | 40% 序列同一性截断；其他过滤未载【核实/未核实】 | **有**：SIU >5.3M 合成【核实】 |

### 2.2 抗体数据、复合物与切分

| 模型 | 抗体类数据（SAbDab 及版本） | 复合物/结合态处理 | 切分与泄漏控制声明 |
| --- | --- | --- | --- |
| RFD4-Proteina | **未披露**训练集中抗体/纳米抗体占比【核实：§4 未给】 | 以结合态 PDB 界面为主；界面就是训练信号，不是 caveat【核实】 | **有**：2024-01-01 硬切点 + benchmark 并集 + S8.4.1 相似性控制；具体 identity 阈值未载【核实/未核实】 |
| Proteina | 未说明；单体数据为主【核实：未见抗体专项】 | 无复合物；单体 AFDB【核实】 | **无**：无 held-out 生成测试；D_FS 兼作 novelty 参照；无时间切点【作者未写】 |
| La-Proteina | 未说明【核实：未见】 | 无复合物；单体 + motif scaffolding【核实】 | **作者未写**：无 identity split / 无泄漏分层；novelty 只报 vs PDB 0.75、vs AFDB 0.82【核实】 |
| Proteina-Complexa | 未披露【核实：§4 未给】 | Teddymer 合成二聚体（unbound-like）+ PDB bound multimers；靶标不进 VAE【核实】 | **作者未给**明确 identity split；与 RFD4 对照基准是否重叠未核实【作者未写/未核实】 |
| Chroma | 未说明；PDB 通用【核实：未见抗体专项】 | 复合物处理未说明；按 PDB 链/结构处理【未核实】 | **有**：train/val/test 的 PDB ID 清单公开于 Zenodo（10.5281/zenodo.8285077）；条目数与聚类阈值未取到【核实/未核实】 |
| BoltzGen | 训练占比未披露；**评测含纳米抗体靶点**【核实：§4】 | 蛋白-配体、RNA、DNA-蛋白、蛋白-蛋白；预测态与实验态混合【核实】 | 训练切分**未核实**；评测泄漏控制硬：10 个新靶全 PDB 同源 ≤30% 结合态，实测 1–27.9%，并用 6 Å 界面 + Folddisco/PPIRef 控界面相似度【论文自称/方法可复核】 |
| SeedProteo | 未说明【核实：未见】 | 有；Pinder 双链 + HumanPPI DDI；结合态与预测态混合【核实】 | **作者未写**：未见切分/泄漏控制声明【核实：三处均无】 |
| APM | 预训练**明确排除 SAbDab**；抗体 SFT 用 SAbDab 1,786/193/55（版本未载）【核实】 | 11,620 PDB biological assemblies；抗体界面只在 SFT 出现【核实】 | **部分**：SAbDab 排除 + SFT/测试分离（RAbD）；无时间切点【核实】 |
| IgGM | **核心数据**：SAbDab ≤2022-12-31；**无 VNAR/单域重链专项**【核实】 | 抗体-抗原复合物；bound-state【核实】 | **有**：CD-HIT 95% 簇划分 + 时间切分（训练 ≤2022；测试 SAb-23H2：60 抗体复合物 + 27 纳米抗体）【核实】 |
| AnewOmni | 有 SAbDab；**版本/截止日未核实**；框架库 48 条含 VHH，**无 VNAR 条目**【代码／配置核实】 | 蛋白-配体、肽、抗体-抗原；混合【核实】 | **部分**：40% identity 截断；测试集 CrossDocked2020 test / PoseBusters / XDock / LNR / RAbD；训练/测试切分细节未核实【核实/未核实】 |

### 2.3 读表要点

1. **真正把切分写死的只有四家**：RFD4（2024-01-01 硬切点）、Chroma（公开 PDB ID 清单）、IgGM（95% 簇 + 时间切分）、BoltzGen（训练切分未核实，但**评测**泄漏控制最硬）。APM 只做到"预训练排除 SAbDab + SFT/测试分离"，不算完整切分声明。
2. **"作者没写"是多数，不是少数**：Proteina、La-Proteina、Proteina-Complexa、SeedProteo 都没有可复核的 identity split。这不是本笔记没查，而是这些篇章在数据节明确找不到；必须与 BoltzGen 的"本次未取到"分开。
3. **数据规模不可直接比**：Proteina 的 2,087 万条是 AFDB 单体、La-Proteina 是 40 万簇、Complexa 是 1,009 万合成二聚体、IgGM 只有 8,355 条抗体复合物。规模差异首先反映任务与数据可得性，不反映模型能力。
4. **抗体覆盖是分层的**：
   - APM 在预训练阶段**主动排除** SAbDab，抗体能力完全来自 SFT；
   - IgGM 以 SAbDab 为核心，但 ≤2022-12-31 且**无 VNAR 专项**；
   - AnewOmni 有 SAbDab 与 48 条 VHH 框架，但**框架库无 VNAR**；
   - RFD4、Complexa、BoltzGen、SeedProteo 的抗体占比均未披露。
   对 VNAR 而言，这不是"数据多一点少一点"，而是"先验覆盖接近零"（IgGM 篇章原话：接口可迁移、先验不可迁移）。
5. **AFDB 是预测结构，不是实验结构**：Proteina、La-Proteina、Complexa、SeedProteo、APM 的主训练集都含 AF2/AFDB 预测坐标。用它训练出的几何先验会偏向"AF2 会输出的几何"，下游做 Rosetta/对接时值得单独验【推断】。

---

## 3. 方案

### 3.1 生成范式、训练目标、生成端自由度

| 模型 | 生成范式 | 训练目标/损失构成 | 生成端自由度 | 备注 |
| --- | --- | --- | --- | --- |
| RFD4-Proteina | 双模态并行流匹配（`bb_ca` + `local_latents`），非扩散【核实】 | flow matching 速度回归 + 物理自蒸馏；AE 解码器用 Rosetta 能量/氢键监督；**无 folding loss**【核实】 | 每残基 Cα (3) + **32 维 latent**；AE 解码为 37 槽原子 + 147 类身份 + presence mask【核实】 | 与 Proteina 同族，但 latent 更宽且编码身份 |
| Proteina | rectified flow / conditional OT【核实】 | `L_RF + λ·L_distogram`；无侧链损失；λ 未载【核实/未核实】 | **Cα only**（3/残基）；序列由外部 ProteinMPNN 出【核实】 | "全原子"只在 IF+refold 评测链出现，不是生成端 |
| La-Proteina | 部分潜变量流匹配 + 双时间 SDE【核实】 | VAE ELBO（重建+KL）+ flow matching 速度回归；无折叠模型监督【核实】 | Cα (3) + **8 维 latent/残基**；解码器出 36 槽原子 + 20 类身份【核实】 | 全潜变量消融 83.8→21.2，证明 Cα 必须显式【核实】 |
| Proteina-Complexa | 部分潜变量流匹配 + test-time search【核实】 | 复用 La-Proteina VAE + 靶标条件 denoiser；reward 来自 AF2-Multimer/RF3【核实】 | binder 侧 Cα + 8 维 latent；靶标 Atom37 显式、不进 VAE【核实】 | 唯一把 test-time compute 写进主结果的 binder 模型 |
| Chroma | DDPM x0-prediction（correlated noise）+ 自回归 design network【核实】 | 扩散 x0 MSE + design network 的序列/χ 交叉熵；classifier/energy guidance【核实】 | 主链 4 原子 + χ + 序列；序列与 χ 由独立 design network 采样【核实】 | 不是联合扩散；条件可组合 |
| BoltzGen | 坐标扩散（Boltz-2 diffusion head）+ design-as-prediction【核实/论文自称】 | 扩散坐标损失 + 随机 mask/裁剪多任务；inverse folding 单独训练/调用【核实】 | 设计残基**只有主链坐标**，侧链恒为 (0,0,0)；序列由 IF 补【核实】 | **不能写成一步全原子扩散** |
| SeedProteo | AF3 式 **atom14 直接扩散** + Design/Folding 双视图【核实】 | AF3 式扩散损失 + MRF 序列解码；具体损失项未全载【核实/未核实】 | **atom14/残基直接扩散**；序列 MRF 解码【核实】 | 主表里唯一直接原子级扩散；步数/参数量/σ **作者未披露** |
| APM | SE(3) 结构流 + 离散序列流 + sidechain/refine【核实】 | `L_discrete + L_SE(3)` + consistency 正则；Phase II `L_Seq&BB+L_Packing+L_Refine`【核实】 | 残基 frame + χ1–χ4 latent；可重建重原子，无氢【核实】 | 序列是显式离散流，不是 VAE 解码 |
| IgGM | 三条扩散（序列/CDR Cα/SO(3) 框架）+ consistency distillation【核实】 | 离散扩散 + Cα/框架损失 + 蒸馏；权重未载【核实/未核实】 | Cα + 主链 N/C/O + SO(3) frame；**无侧链**【核实】 | 抗体专用；表位条件 |
| AnewOmni | 离散时间 DDPM（block latent + latent coords）+ IterVAE【核实】 | DDPM ε-prediction + VAE 编解码；精确似然排序【核实】 | block 8 维 latent + 3 维潜坐标；逐原子解码【核实】 | 块级表征，全原子由 decoder 还原 |

### 3.2 采样步数与 σ / t 调度

| 模型 | 默认步数 | σ / t 调度 | 随机性来源 | 备注 |
| --- | --- | --- | --- | --- |
| RFD4-Proteina | **60 步**【核实】 | `vf_tsr` 随机采样器；σ 具体调度未载【核实/未核实】 | 随机采样器 + 外部 beam search | 60 步是报告口径，不是通用最优 |
| Proteina | **400 步** Euler–Maruyama【核实】 | 训练 uniform t；推理 log-spaced；γ 控制噪声【核实】 | SDE 噪声项 | 训练/推理调度不一致是刻意设计 |
| La-Proteina | **400 步**【核实】 | 双时间 SDE；σ 具体调度未载【核实/未核实】 | 双路去噪随机性 | 400 步是正文明确值 |
| Proteina-Complexa | **未载**；test-time 分叉 + reward【核实/未核实】 | 部分潜变量流匹配；σ 具体调度未载 | SDE + beam/SMC | 主结果依赖 test-time compute；默认积分步数篇章未给 |
| Chroma | **未披露**【未核实】 | DDPM correlated noise；低温柔性退火 | 扩散噪声 + guidance | 篇章只称"几分钟"，未给步数 |
| BoltzGen | **未核实**（Boltz-2 扩散默认）【未核实】 | 未载 | 扩散噪声 | 采样预算重点在 6 万条/靶，不在步数 |
| SeedProteo | **作者未披露**【核实：三处均无】 | 作者未披露 | 扩散噪声 | 步数/参数量/σ 是三项核心空白 |
| APM | **作者未写**默认步数【核实】 | flow matching 时间调度；具体未载 | 流匹配随机性 | consistency 是训练正则，不是蒸馏加速 |
| IgGM | consistency 后可用少步；**默认步数未载**【论文自称】 | 离散转移核 | 扩散噪声 | 少步会牺牲界面精度【论文自称】 |
| AnewOmni | **100 步**，`std 10.0`，支持 DDIM【代码／配置核实】 | 离散 DDPM beta 时间戳 | 扩散噪声 | 代码与论文一致 |

### 3.3 是否需要外部折叠模型做监督或打分

| 模型 | 训练监督 | 生成时打分/筛选 | 评测 oracle |
| --- | --- | --- | --- |
| RFD4-Proteina | **无折叠模型**；AE 用 Rosetta 能量/氢键物理项【核实】 | 可外挂 AF3/OF3 oracle 做 beam search；非生成必需【核实】 | AF3/OF3 重折 + TM-score/Foldseek |
| Proteina | 无【核实】 | 无 | ProteinMPNN + ESMFold（scRMSD）【核实】 |
| La-Proteina | 无；ESMFold/MPNN 只在 metrics 路径【核实】 | 无 | ProteinMPNN + ESMFold【核实】 |
| Proteina-Complexa | 无折叠监督【核实】 | **有**：AF2-Multimer / RF3 作为 test-time reward【核实】 | 同上；主结果依赖 reward |
| Chroma | 无【核实】 | 无 | 湿实验（40/42 可溶表达）+ 常规几何指标【论文自称】 |
| BoltzGen | **有**：Boltz-1 自蒸馏（折叠模型当教师）【核实】 | **有**：ESMFold2 refold/打分【核实】 | ESMFold2 + 湿实验 |
| SeedProteo | 无外挂；内部 Folding View【核实】 | 无 | SeedFold（同团队）【核实】 |
| APM | 无【核实】 | 无 | ESMFold 基线 + 能量/几何指标 |
| IgGM | 无【核实】 | 可选 AF3 初始化（非必需）【核实】 | AF3 / SAb-23H2 |
| AnewOmni | 无【核实】 | **抗体主结果用 AF3/Protenix/Boltz2 共折叠闭环**【核实】 | 外循环 cofold + PoseBusters【核实】 |

### 3.4 把 design 重写成条件补全：这一脉到底改了什么

BoltzGen 与 RFD4-Proteina 的共同点不是"都用了扩散"，而是**训练样本的构造方式**：同一结构随机指定"设计区/条件区"，模型学 `p(缺失区 | 上下文)`；推理时把靶标设为条件、把 binder 设为缺失。这样 folding、binder design、motif scaffolding、无条件生成被压成同一个补全目标【核实：BoltzGen §3；RFD4 §3】。

但两者在"序列从哪来"上分叉：

| 模型 | 结构怎么来 | 序列怎么来 | 实际含义 |
| --- | --- | --- | --- |
| BoltzGen | 扩散直接出设计区主链坐标 | **逆折叠单独补**（SolubleMPNN/BoltzIF） | 条件结构补全 + 序列装饰；序列不在扩散环内 |
| RFD4-Proteina | 流匹配出 Cα + 32 维 latent | latent 经 AE 的 147 类身份头**联合解码** | 条件结构-序列联合生成；序列在环内 |

这条差别比"层数/参数量"实质得多：

- **BoltzGen 的"全原子"是流水线属性，不是生成器属性**。设计残基侧链恒为 (0,0,0)，必须靠 IF + refold 才有全原子；因此它不能和 SeedProteo 的 atom14 直接扩散放在同一"生成自由度"刻度上。
- **RFD4 也不是"直接扩散到原子"**：它的 32 维 latent 同时编码身份与局部构象，AE 一次解出 37 槽原子。它和 La-Proteina/Complexa 同属"潜变量+解码"，只是 latent 更宽、训练数据更偏界面。
- **"从头生成"一脉（Proteina、La-Proteina、Chroma）学的是 `p(结构)` 或 `p(结构|motif)`**，条件通过 guidance/inpainting 注入；"条件补全"一脉学的是 `p(缺失|上下文)`，靶标结构直接进网络。两者都能做 binder design，但前者需要额外的条件机制，后者把条件写进了训练分布。**这不是同一任务的两种实现，而是两个任务被同一个 mask 目标近似了。**

---

## 4. 架构

### 4.1 主干、层数、宽度、参数量

| 模型 | 主干与规模 | 参数量 | 备注 |
| --- | --- | --- | --- |
| RFD4-Proteina | 14 层 transformer；token 768 / pair 256 / 12 头；AE 12 层 768【核实】 | AE ~130M；flow 主干参数量未载【核实/未核实】 | pair track 较轻；AE 训练时砍掉 pair |
| Proteina | 15/15/18 层，768/768/1024 维，12/12/16 头；pair bias，可选三角乘法，10 个 register token【核实】 | M_FS 208M；M_21M 397M；M_small 59M【核实】 | 不是等变网络 |
| La-Proteina | 16 个 block；pair-biased attention，**无三角乘法**【核实】 | denoiser 160M【核实】 | 与 Complexa 共享 VAE 谱系 |
| Proteina-Complexa | 14 层；继承 La-Proteina 主干【核实】 | denoiser 160M；合计 256M【核实】 | 靶标条件在 denoiser 内，不进 VAE |
| Chroma | 等变图神经网络；random graph，子二次复杂度【核实】 | **未披露**【未核实】 | 层数/宽度公开材料未见 |
| BoltzGen | Boltz-2 PairFormer trunk + 扩散头（3 atom + 24 token + 3 atom）【核实】 | **未披露**【未核实】 | 与 Boltz-2 同源，但数据做减法 |
| SeedProteo | AF3 式 Pairformer **12 层** + 扩散模块【核实】 | **作者未披露**【核实】 | 保留 MSA module |
| APM | ESM2-650M 冻结 + 残基 Transformer 384/192【核实】 | 127M（52M+22M+54M）【核实】 | 三模块：Seq&BB / Sidechain / Refine |
| IgGM | ESM-PPI 冻结 + Sgformer **16 层** + **8 个 IPA** 预测模块【核实】 | 未披露【未核实】 | 三条扩散共享主干 |
| AnewOmni | IterVAE（latent 8 / hidden 512 / edge 64 / k=9）+ 6 层 8 头 E(3) EPT【代码／配置核实】 | 未披露【未核实】 | 块级表征 |

### 4.2 表征粒度与"全原子"的实际位置

| 模型 | 生成侧表征 | 输出能否重建重原子 | 是否一步全原子 |
| --- | --- | --- | --- |
| RFD4-Proteina | Cα + 32 维 latent | 能（AE 37 槽 + presence mask） | 否；潜变量+解码 |
| Proteina | Cα only（3/残基，无 latent） | 能（Cα 骨架经 MPNN + refold 补全；生成端只出 Cα） | 否；Cα 流 + 外挂补全 |
| La-Proteina | Cα + 8 维 latent | 能（36 槽 + 20 类身份） | 否；潜变量+解码 |
| Proteina-Complexa | binder Cα + 8 维 latent；target Atom37 | 能（binder 侧走 La-Proteina 解码） | 否；潜变量+解码 |
| Chroma | 主链 4 原子 + χ + 序列 | 能（χ 显式） | 否；序列/χ 由 design network 单独出 |
| BoltzGen | 设计区主链坐标；侧链 (0,0,0) | 能（IF + refold 后） | **否；分段** |
| SeedProteo | atom14 | 能 | **是**（直接原子扩散） |
| APM | residue frame + χ1–χ4 latent | 能（无氢） | 否；frame+torsion 解码 |
| IgGM | Cα + N/C/O + SO(3) frame | 仅主链，无侧链 | 否；主链级 |
| AnewOmni | block 8 维 latent + 3 维潜坐标 | 能（逐原子解码） | 否；块级+解码 |
| ESM3（边界） | 离散 token | 仅主链轨 | 否；token 生成 |
| TorchCraft（边界） | 固定全原子结构 + 可变序列 logits | 结构继承自 AF3 | 不适用；非生成器 |

> 表注（"全原子"分三档）：**生成器自带**（La-Proteina / Complexa / RFD4 / Chroma / APM / AnewOmni / SeedProteo）→ **需外挂补全**（BoltzGen 主链+IF；Proteina Cα+MPNN/refold；输出可达重原子，但补全不在生成器内）→ **只有主链且篇章未声明补全路径**（IgGM；ESM3 的结构轨本身就是 token）。按 §0.2 的输出口径，第一档为全原子；第二档只是“经外挂补全可达重原子”，本仓库矩阵不据此改判粒度（Proteina 仍记仅主链 / Cα-only，见专题 01 §6.2 表注 2）；但"自带"与"需补全"在采样成本、侧链多样性、结构与序列是否同源上完全不同，不能因为都写"全原子"就并成一类。

### 4.3 条件接口与序列/逆折叠模块

| 模型 | 条件接口 | 独立序列/逆折叠模块 | 输出序列的方式 |
| --- | --- | --- | --- |
| RFD4-Proteina | 靶标绝对/内部坐标、hotspot、motif 索引、序列概率、CATH、对称性、partial diffusion、配体【核实】 | **无** | AE 的 147 类身份头联合解码 |
| Proteina | CATH 折叠类（CFG/autoguidance）+ motif 坐标/mask（inpainting）【核实】 | 无（评测用 ProteinMPNN） | 训练不生成序列；下游 MPNN |
| La-Proteina | 无条件 + atom-level motif（indexed/tip-atom/unindexed）【核实】 | **无** | VAE 解码器出 20 类身份 |
| Proteina-Complexa | 靶标 Atom37 + 链断点 + hotspot token + 催化残基片段 + binder 长度；无 contig/MSA/模板【核实】 | **无** | VAE 解码器出序列 |
| Chroma | substructure / symmetry / shape / ProClass / ProCap 五类可组合条件【核实】 | **有**（独立 design network） | design network 自回归出序列 + χ |
| BoltzGen | YAML 设计规范：实体、长度、设计区、结合类型、二级结构、结构组可见性、约束、靶标【核实】 | **有**（SolubleMPNN / BoltzIF） | 逆折叠补序列，扩散不生成序列 |
| SeedProteo | 靶标链 ID + 区间 + hotspot + binder 长度 + 二级结构；无 contig【核实】 | 无独立网络；**MRF 序列解码头** | MRF 联合解码，抑制等排残基误判 |
| APM | 链级：受体链/抗原链/固定链/待生成链；**不指定表位**【核实】 | 无 | 显式离散序列流 |
| IgGM | 抗原序列/结构 + 表位残基 + 框架 + 掩码 CDR + paired/single【核实】 | **无** | 序列扩散在同一主干内 |
| AnewOmni | 口袋原子（10 Å）/ hotspots / 48 条框架库（含 VHH）/ 参考复合物 / 图 prompt；无 CFG【代码／配置核实】 | 无 | 原子解码隐含序列；抗体侧靠框架库 |
| ESM3（边界） | 任意轨道组合掩码 | 无 | 序列轨直接生成 |
| TorchCraft（边界） | 靶标结构 + 可编辑 mask + hotspot + 序列先验 | 无（IgLM 先验） | 优化序列 logits，AF3 只读 |

### 4.4 读表要点

1. **层数/宽度不是可比量**：Proteina 的 15/18 层 transformer、SeedProteo 的 12 层 Pairformer、Chroma 的等变 GNN、AnewOmni 的 6 层 EPT，计算原语不同，层数不能当"容量"排序。
2. **"有没有独立序列模块"决定失败模式**：
   - 联合生成（RFD4/La-Proteina/Complexa/APM/IgGM/AnewOmni）：序列-结构耦合好，但序列多样性受结构先验限制；
   - 外挂 IF（Proteina/Chroma/BoltzGen）：序列可独立优化，但结构与序列可能失配，需要 refold 过滤；
   - 无侧链（IgGM/ESM3）：只能做界面/主链级判断，侧链packing 必须外接。
3. **条件接口的粒度差异很大**：IgGM 有表位残基、APM 只有链级、RFD4/Complexa 有 hotspot、BoltzGen 有 YAML 级设计规范、Chroma 有条件分类器。把"能条件化"当成同一能力是误读。

---

## 5. 分析：差异从哪里来

### 5.1 来自表征选择的差异

表征选择直接决定三件事：采样成本、侧链自由度、序列-结构耦合方式。

- **直接原子（SeedProteo，atom14）**：采样维度最高，侧链自由度最完整，但步数/算力成本也最高；作者未披露步数与参数量，无法判断它是否用更多步数换质量。
- **Cα + 每残基 latent + 解码（RFD4 32 维、La-Proteina 8 维、Complexa 8 维、AnewOmni block 8 维）**：把侧链与身份压进 latent，采样只在低维进行。四者的 latent 语义不同：La-Proteina/Complexa 的 8 维主要编码身份+局部构象；RFD4 的 32 维同时编码身份与锚点相对原子坐标；AnewOmni 的 8 维是化学积木块的块级隐变量。**不能因为都写"8 维"就当成同一表征。**
- **主链+扭转角（APM、Chroma）**：APM 用 residue frame + χ1–χ4 隐变量，Chroma 用 4 个主链原子 + 显式 χ；两者都能重建重原子，但 APM 的序列是显式离散流，Chroma 的序列由 design network 单独出。
- **主链级（Proteina、IgGM、ESM3）**：Proteina 是 Cα-only，IgGM 是 Cα+N/C/O+frame，ESM3 是 token 主链轨；三者**生成端**都不出侧链，重原子只能靠外接 MPNN/packing/refold 补全（Proteina 的评测链正是这样补齐的；ESM3 的"结构"本身只是 token 轨）。
- **分段（BoltzGen）**：设计区扩散只出主链，侧链置零，靠 IF 补。**它的"全原子"是后处理属性，不是生成属性。**

对 VNAR 的直接含义：长 CDR3 的侧链 packing 与框架二硫键模式是主要难点。Cα-only 与主链级模型（Proteina/IgGM/ESM3）无法自行解决；latent 模型（La-Proteina/RFD4/AnewOmni）能否覆盖非典型侧链构象，取决于训练分布；atom14（SeedProteo）自由度最完整，但抗体数据覆盖未知；APM 的 χ 显式建模在原理上更贴近侧链，但它不做表位条件。

### 5.2 来自任务定义的差异

**同一句"生成模型"下其实有四个任务原语**：

| 任务原语 | 代表 | 学的是什么 | 评测什么 |
| --- | --- | --- | --- |
| 无条件/条件单体生成 | Proteina、La-Proteina、Chroma | `p(结构)` 或 `p(结构\|motif/条件)` | designability、多样性、新颖性 |
| binder design | Complexa、SeedProteo、APM、IgGM、AnewOmni、BoltzGen、RFD4 | `p(binder\|靶标)` | hit rate、DockQ、湿实验结合 |
| motif scaffolding | La-Proteina、Chroma、RFD4 | `p(支架\|motif)` | motif RMSD、designability |
| folding | SeedProteo Folding View、BoltzGen | `p(结构\|序列)` | RMSD/lDDT |

"把 design 重写成条件补全"这一脉（BoltzGen、RFD4）之所以值得单列，是因为它**用同一个 mask 目标覆盖了上表前三个任务**：训练时随机决定哪段是条件、哪段缺失；推理时把靶标当条件、binder 当缺失。这带来两个实际后果：

1. **任务边界消失，评测边界不能消失**。模型可以在同一个权重上做 folding、binder、motif，但"它在 folding 上 SOTA"与"它在 binder design 上 SOTA"仍然是两个独立命题；BoltzGen 的 folding 主结果只在 Supplementary Fig. 15，不能拿来支撑 binder 结论。
2. **序列是否在环内，决定了它是不是"生成"**。BoltzGen 的扩散不生成序列，序列由 IF 出；RFD4 的 latent 解身份，序列在环内。把两者都叫"全原子生成模型"会掩盖这个差别。

"从头生成"一脉（Proteina/La-Proteina/Chroma）学的是蛋白折叠先验，条件通过 guidance/inpainting 注入；"条件补全"一脉学的是界面/复合物统计，靶标结构直接进网络。**前者不是"更通用的后者"，后者也不是"更强的前者"**：它们的数据分布、条件机制、失败模式都不同。

### 5.3 来自评测口径的差异

这一节是本篇最重要的"防误读"部分。

- **La-Proteina 的三个数不是同一指标**：**68.4%** 是 all-atom co-designability（100–500 aa，主模型）；**72.2%** 是 α-carbon co-designability（只评 Cα 的宽松口径）；**93.8%** 是 ProteinMPNN-8 口径。进表只能取一个。本篇在需要引用时取 **68.4%（all-atom co-designability）**，并在表注写明口径【核实】。
- **BoltzGen 的 hit rate 必须带分母**：每靶 **60,000 条设计**、取前 **15 条**上机；**60%**（纳米抗体）/ **50%**（蛋白）是"有 hit 的靶点比例"，且作者自己区分 **screening hit** 与 **confirmed binder**【论文自称】。不带分母的"60% hit rate"是误导。
- **APM 的"能量领先"只在特定协议下成立**：在"固定主链、仅侧链 relax"的 Table 10 协议下，APM 的能量指标领先；常规流程 Table 6 的 **E=137.74** 并不领先。不能写成普适结论【核实】。
- **IgGM 的抗体数字有明确测试集**：SAb-23H2（60 抗体复合物 + 27 纳米抗体），CDR-H3 序列恢复率 36%；纳米抗体 TM 低于 AF3，作者自己承认。**没有 VNAR 专项**，对 VNAR 的判断只能是"接口可迁移、先验不可迁移"【核实/论文自称】。
- **AnewOmni 的抗体成功率含外循环**：框架选择 + 外循环共折叠（AF3/Protenix/Boltz2）是主结果的一部分，不是模型单独生成的能力；把它当成"生成器内部学会了抗原-抗体界面物理"会高估【核实/推断】。
- **SeedProteo 的评测器是同团队的 SeedFold**：这不是错，但存在同源偏置，跨模型比较时要标注【核实】。
- **Complexa 的主结果含 test-time compute**：去噪轨迹分叉 + beam/SMC + AF2-Multimer/RF3 reward（默认积分步数篇章未给）。与单次前向的模型比"质量/算力"时，必须把搜索预算算进去【核实】。
- **RFD4 的通过率是 clustered pass rate**：不是单链 designability；与 La-Proteina 的 co-designability 不是同一指标【核实】。
- **Chroma 的主验证是湿实验**：42 个条件设计中 40 个（95%）可溶表达，无第三方复现；与 in silico designability 不可直接比【论文自称】。

---

## 6. 不可比的地方（单独列出，不混进结论）

1. **任务不同**：无条件单体生成（Proteina/La-Proteina/Chroma）vs binder design（Complexa/SeedProteo/APM/IgGM/AnewOmni/BoltzGen/RFD4）vs motif scaffolding（La-Proteina/Chroma/RFD4）vs folding（SeedProteo Folding View/BoltzGen）。跨任务的成功率没有共同分母。
2. **输出粒度不同**：Cα-only（Proteina）vs Cα+latent（La-Proteina/Complexa/RFD4）vs atom14（SeedProteo）vs frame+χ（APM）vs 主链+χ（Chroma）vs 主链+IF（BoltzGen）vs 主链级（IgGM）vs token（ESM3）。
3. **序列来源不同**：联合生成（RFD4/La-Proteina/Complexa/APM/IgGM/AnewOmni）vs 外挂 IF（Proteina/Chroma/BoltzGen）vs 给定序列（TorchCraft/ESM3 的部分模式）。"设计成功率"里序列的贡献无法拆分。
4. **评测 oracle 不同**：AF3、OF3、AF2-Multimer、RF3、ESMFold、SeedFold、Boltz-1/2、湿实验。同一设计在不同 oracle 下的排名可能翻转；同源 oracle（SeedProteo→SeedFold）还有循环性。
5. **算力预算不同**：BoltzGen 6 万条/靶 vs RFD4 200 条/靶 vs Complexa 多次搜索（默认积分步数篇章未给）vs Chroma 湿实验 42 条。不比预算的"质量"没有意义。
6. **数据 regime 不同**：AFDB 千万级合成/预测结构 vs SAbDab 8,355 条实验抗体复合物 vs Teddymer 1,009 万合成二聚体。数据规模首先反映可得性，不反映模型优劣。
7. **"全原子"定义不同**：生成器自带重原子（La-Proteina/Complexa/RFD4/AnewOmni/APM/Chroma）vs 直接原子扩散（SeedProteo）vs 需外挂补全才拿得到重原子（BoltzGen 主链+IF、Proteina Cα+MPNN/refold）vs 只有主链且未声明补全路径（IgGM/ESM3）。同名不同义；"能重建重原子"不等于"生成器会造重原子"。
8. **湿实验 vs in silico**：Chroma/BoltzGen/IgGM/AnewOmni 有湿实验或外循环；Proteina/La-Proteina 主要是 in silico designability。两者不能放在同一"成功率"列。
9. **评价指标定义不同**：designability、co-designability、scRMSD、DockQ、hit rate、confirmed binder rate、Rosetta 能量、TM-score。没有统一换算。
10. **训练/测试泄漏控制不同**：RFD4/Chroma/IgGM 有明确切分；BoltzGen 训练切分未核实但评测控制硬；Proteina/La-Proteina/Complexa/SeedProteo 未见 identity split。泄漏程度不同的数字不可直接比。

---

## 7. 边界案例：为什么它们不该进主表

### 7.1 TorchCraft：预测器反演，不是生成模型

TorchCraft 的做法是：冻结 AF3，把序列 logits 作为可优化变量，用 AF3 的置信度/接触/几何项 + IgLM 序列先验做梯度上升，输出**一条序列 + AF3 给出的结构**【核实】。它没有：

- 没有在结构数据上训练生成器；
- 没有定义 `p(结构)` 或 `p(序列|结构)` 的采样过程；
- 没有多步去噪/流积分；
- 没有分布级多样性，只有优化轨迹上的局部解。

因此它属于"**design by inversion**"：把设计问题转成对一个冻结预测器的输入优化。它可以和生成模型比"单位算力的设计成功率"，但不能比训练数据、生成范式、架构容量——这三轴对它不适用。它还有一个方法论风险：TorchScore 与 AF3 同源，评估自洽性会被高估【核实/推断】。

### 7.2 ESM3：离散掩码语言模型，结构轨只有主链

ESM3 是离散 token 上的掩码语言模型：序列、二级结构、三级结构坐标（VQ-VAE tokenizer，4096 codes）、功能注释等多轨 token，用任意 mask rate/order 训练【核实】。它的规模很大（2.78B 自然蛋白 → 3.15B 序列、236M 结构、539M 功能注释、771B unique tokens；48/96/216 层，最大 98B）【核实】，但：

- **结构轨只有主链**，没有侧链；不满足本篇"可重建重原子"的全原子输出判定；
- 生成是 token 级掩码补全，不是坐标扩散/流匹配；与主表的"生成自由度"不可比；
- 任务定位是序列-结构-功能联合建模，不是 binder design；单链偏置明显，抗体/复合物条件能力弱；
- 它的"结构生成"更接近"用结构 token 约束序列生成"，而不是"设计一个可折叠的架构"。

因此 ESM3 应作为**序列表征/先验基座**或**designability 基线**，而不是主表里的全原子生成器。把它放进主表会同时污染"表征粒度"和"任务"两列。

### 7.3 边界判定表

| 候选 | 有训练目标与采样？ | 输出可重建重原子？ | 任务是结构生成？ | 结论 |
| --- | --- | --- | --- | --- |
| TorchCraft | 否（无生成器训练） | 结构继承 AF3，序列可变 | 否（预测器反演） | 边界，§7.1 |
| ESM3 | 是（掩码 LM） | 否（仅主链） | 部分是（token 补全） | 边界，§7.2 |

---

## 8. 证据缺口：哪些格子留空，以及性质

| 模型 | 留空/未核实项 | 性质 | 对结论的影响 |
| --- | --- | --- | --- |
| RFD4-Proteina | 序列同一性阈值、抗体数据占比、界面条目数 | **作者未写/未披露** | 无法判断抗体先验覆盖；切分只能信 2024-01-01 硬切点 |
| Proteina | PDB 切点、抗体占比、AFDB 过滤细节 | **作者未写/未披露** | 无法做泄漏分层；只能确认"无 held-out" |
| La-Proteina | σ 具体调度、切分/泄漏声明 | **作者未写** | 400 步可核实；切分不可复核 |
| Proteina-Complexa | 抗体占比、identity split、500k 合成集口径 | **作者未写/口径冲突** | 与 RFD4 对照的可比性受限 |
| Chroma | 参数量、层数/宽度、采样步数、数据条目数、聚类阈值 | **本次未取到**（篇章明确） | 架构容量与采样成本无法量化 |
| BoltzGen | 训练 cutoff/分辨率/条目数/聚类阈值、采样步数、抗体占比 | **本次未取到**（Appendix B.1 未取得） | 训练数据不可复核；只能确认 AFDB+Boltz-1 蒸馏与"去掉抗体上采样" |
| SeedProteo | 扩散步数、参数量、σ 调度、切分 | **作者未披露/作者未写** | 无法判断采样成本与泄漏控制；atom14 优势无法量化 |
| APM | 默认采样步数、σ 调度、SAbDab 版本 | **作者未写/未核实** | 生成成本不可比；SFT 数据不可精确重建 |
| IgGM | consistency 后默认步数、SAbDab 精确快照 ID | **未记录/未核实** | 推理成本不可比；数据可重建性受限 |
| AnewOmni | SAbDab 版本/截止日、各数据集条目数 | **未核实** | 抗体先验覆盖只能确认"无 VNAR 框架条目" |
| ESM3 | 各层宽度、数据切分细节 | **本次未取到** | 不影响边界判定（仅主链已足够） |
| TorchCraft | — | 不适用 | 非生成模型，三轴不适用 |

**明确留空的格子**（表中用"未核实/作者未写"标出，不用数字填充）：

- Chroma：参数量、层数/宽度、采样步数、条目数；
- BoltzGen：训练 cutoff/规模/过滤、采样步数、抗体占比；
- SeedProteo：步数、参数量、σ、切分；
- APM：默认步数、σ、SAbDab 版本；
- IgGM：默认步数、快照 ID；
- AnewOmni：SAbDab 版本/截止日、各集条目数；
- ESM3：层宽度、切分细节；
- 其余模型：抗体占比、identity 阈值等"作者未写"项。

---

## 9. 三条最实质的差异（结论）

1. **"全原子"不是同一件事，真正的分界是生成端自由度。** SeedProteo 是 atom14 直接扩散；La-Proteina/Complexa/RFD4/AnewOmni 是"Cα/block + latent + 解码"；APM 是 frame+torsion；Chroma 是主链+χ+独立 design network；BoltzGen 是主链扩散+IF 补侧链；Proteina/IgGM/ESM3 生成端只有主链（Proteina 的全原子靠外挂补全）。即便输出都能"重建重原子"（含外挂补全），采样成本、侧链多样性、序列-结构耦合方式也完全不同。

2. **"design = 条件补全"与"从头生成"是两个任务，不是两种实现。** BoltzGen/RFD4 用随机 mask 把 folding、binder、motif 压成 `p(缺失|上下文)`；Proteina/La-Proteina/Chroma 学 `p(结构)` 再用 guidance/inpainting 注入条件。更细的分叉是**序列是否在生成环内**：RFD4 的 latent 解身份，BoltzGen 的序列靠 IF。把两者都叫"全原子生成"会掩盖最重要的差别。

3. **报告出来的 SOTA 差异，很大一部分来自数据 regime 与评测口径，而不是架构。** 数据侧：AFDB 千万级 vs SAbDab 8,355 条 vs Teddymer 1,009 万合成；评测侧：La-Proteina 的 68.4/72.2/93.8 是三个口径，BoltzGen 的 60% 带 60,000/15 的分母，APM 的能量领先只在固定主链 relax 协议下成立，AnewOmni 的抗体成功率含外循环共折叠。架构层数（12/14/15/16/18）在计算原语不同的情况下几乎不构成可比证据。

---

## 附：本篇引用的正典篇章

- [RFD4-Proteina](../模型篇章/RFD4-Proteina/README.md)
- [Proteina](../模型篇章/Proteina/README.md)
- [La-Proteina](../模型篇章/La-Proteina/README.md)
- [Proteina-Complexa](../模型篇章/Proteina-Complexa/README.md)
- [Chroma](../模型篇章/Chroma/README.md)
- [BoltzGen](../模型篇章/BoltzGen/README.md)
- [SeedProteo](../模型篇章/SeedProteo/README.md)
- [APM](../模型篇章/APM/README.md)
- [IgGM](../模型篇章/IgGM/README.md)
- [AnewOmni](../模型篇章/AnewOmni/README.md)
- [TorchCraft](../模型篇章/TorchCraft/README.md)
- [ESM3](../模型篇章/ESM3/README.md)
