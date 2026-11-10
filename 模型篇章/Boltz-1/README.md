---
model: Boltz-1
aliases: ["Boltz-1x（= Boltz-1 + 默认开启的 Boltz-steering，v1.0.0 tag 34cf560；是否换了权重未核实）"]
family: AF3 系（开源复现 + 工程重写 + 置信度模型重构），上游 AlphaFold3，下游 Boltz-2
granularity: 双尺度——聚合物每残基 1 token（蛋白氨基酸 / DNA、RNA 碱基），配体与修饰残基/碱基每个重原子 1 token；无 residue frame 输入（frame 仅用于置信度 loss 的 aligned error），有原子级 sequence-local cross-attention（32 query 原子 × 128 key 原子）
backbone: trunk = MSA 模块 4 block（msa_s 64，可整体跳过）+ PairFormer 48 block（token_s 384 / token_z 128，16 head，无三角乘法更新）；diffusion = token transformer 24 block × 384 通道 × 16 head、atom transformer 3+3 block × 128 通道 / atom_z 16 / 4 head、dim_fourier 256、cond transition 2 层；置信度模型 = 整个 trunk 复刻（48 层）+ 扩散轨迹激活聚合
generative: EDM 扩散（σ_data 16、σ_min 4e-4、σ_max 160、ρ 7、P_mean −1.2、P_std 1.5；γ0 0.8 / γmin 1.0、noise 1.003、step_scale 1.638）；CLI 默认 200 步、1 样本、recycling 3；论文评测 200 步 / 10 recycling / 5 输出 / MSA≤16384
conditioning: MSA（推理 YAML 可给 .a3m 路径或 empty）| 共价键（constraints.bond）| 口袋/表位（constraints.pocket，binder + contacts，蛋白或核酸 binder；训练期 30% 迭代、6 Å 部分揭示，故单模型可用、不需重训）| 无模板（论文明确删除）| 无抗体/CDR/亲和力条件
data_scope: PDB 全部 ≤2021-09-30 释放结构（论文写「at least 9 Å」，structure.yaml 9.0 / full.yaml、confidence.yaml 4.0），Biological Assembly 1，OpenFold 蒸馏约 270K 结构；68k 步 × batch 128，前 53k 步 crop 384 token/3456 atom，后 15k 步 512/4608（纯 PDB）；val 553 / test 593（≤4.5 Å、100–2000 残基、2023-01-13 后释放、序列与配体相似度隔离）
status: 代码 MIT（LICENSE 核实）；权重 MIT 仅 README 声明，HF 模型卡本次不可达（未核实）；Boltz-1 线冻结在 v0.4.1，v1.0.0 加 steering，主线已转 Boltz-2；论文 bioRxiv 10.1101/2024.11.19.624167（v4 2025-05-06），本篇代码锚定 tag v0.4.1 = 9461d02
verified_at: 2026-11-10
---

# Boltz-1

> **三句话速判**
> 1. Boltz-1 是 AF3 的**「复现 + 五处算法改动 + 大量工程优化」**，不是新范式：trunk 尺寸（4/48 block、384/128、MSA 64）逐项相同，改的是 MSA 算子顺序、Transformer 残差顺序、Kabsch 插值、扩散损失权重，以及把置信度模型从「4 层 PairFormer 头」换成「整个 48 层 trunk 复刻 + 扩散轨迹聚合」。
> 2. 相对 AF3 最实质的两处：**减掉模板输入**；**加上单模型口袋条件**（训练时 30% 迭代、6 Å 内几何分布揭示部分残基，替代 AF3 双模型）。
> 3. 最该记住的核查结论：**论文只写 9 Å，代码里 9.0 与 4.0 并存**——9.0 只在结构模型配置，4.0 在置信度与未文档化的 full 配置，论文从头到尾没有 4 Å；「9 Å 是爬取初筛」在代码里找不到对应物。

---

## 1. 定位与谱系

### 1.1 出处（一手）

| 项 | 值 | 依据 |
|---|---|---|
| 标题 | **Boltz-1: Democratizing Biomolecular Interaction Modeling** | p.1（注意：不是 "Prediction"） |
| 预印本 | bioRxiv `10.1101/2024.11.19.624167`，v4 = 2025-05-06（v1 = 2024-11-20） | p.1  |
| arXiv | **不存在**（2411.19169 是另一篇 HCI 论文 ComViewer，全文无 "Boltz"） | 自查  |
| 作者/单位 | Wohlwend, Corso, Passaro, Getz（共同一作）等 13 人；MIT CSAIL / Jameel Clinic / Genesis Research / CHARM Therapeutics | p.1  |
| 仓库 | `github.com/jwohlwend/boltz`；查 Boltz-1 须 `?ref=v0.4.1`（commit `9461d02`） | tags API  |
| 许可 | 代码 **MIT**（LICENSE 原文）；权重 README 声明 MIT，HF 模型卡未核实 | 见本篇 §9 |

版本差异：v1（2024-11）称"大置信度模型尚在训练"，结果仅一张主图；v4（2025-05）才补齐 Results、消融、Boltz-1x 与物理有效性。**本篇引"结果"一律指 v4**；"设计声明"已对照 v1/v4 一致（9 Å 与 token 定义措辞相同）。

### 1.2 先纠正一条流传的线索

任务线索称"arXiv 版本 2411.19169"。实测该 PDF 是 *ComViewer*（协同可视化，CSCW），与结构预测无关，故本篇**不引用任何 arXiv 版本**；二手信息里出现 "Boltz-1 arXiv 2411.19169" 可判为串号。

### 1.3 相对 AF3：逐项对照（这是本篇的主轴）

| 维度 | AlphaFold3（基线，本项目 AF3 笔记已代码核实） | Boltz-1 | 判定 |
|---|---|---|---|
| 输入 | 序列 + MSA + **模板** + 配体 | 序列 + MSA + 配体，**删掉模板**  | 改（减法） |
| token | 残基 1 token；配体每重原子 1 token；每 token **≤24 原子槽位** | 残基 1 token；配体每重原子 1 token ；**无槽位上限**（变长 `atom_num`，`tokenize/boltz.py:89,131`），见 §2.2 | 改（实现层） |
| residue frame | 已去掉，只在 loss 里 | 同；frame 仅用于置信度 loss 的 PAE（§2.3） | 同 |
| MSA 模块 | 4 block，c_m 64 | 4 block，msa_s 64，**可整模块跳过**（no_msa） | 同尺寸 + 开关 |
| PairFormer | 48 block，pair 128 / single 384，16 head | 48 block，token_z 128 / token_s 384，16 head  | 同 |
| 三角乘法（AF2 外积式） | 无 | 无 | 同 |
| MSA 算子顺序 | OuterProductMean → PairWeightedAveraging → MSATransition → TriangleUpdates → PairTransition | **PairWeightedAveraging → MSATransition → OuterProductMean → TriangleUpdates → PairTransition**  | **改** |
| Diffusion Transformer | 24 block，token 宽 **768**，atom 128 / atom pair 16，atom transformer 3 block 4 head | 24 block，token 宽 **384**，atom 128 / atom_z 16，encoder 3 + decoder 3 block 各 4 head，dim_fourier 256，cond transition 2 层 ；残差改为 `a ← a + AttnPairBias(a)`、`a ← a + CondTransition(a)`  | **改**（宽度减半 + 残差） |
| 原子 attention 局部性 | token 内 24 槽位 + 全局 atom attention | **序列局部**：32 原子 query 只 attend 序列最近 128 原子，块稀疏  | **改** |
| EDM 参数 | σ_data 16、σ_min 4e-4、σ_max 160、ρ 7 | 完全相同，另给 P_mean −1.2 / P_std 1.5  | 同 |
| 采样 | 200 步、5 样本、γ0 0.8 / γmin 1.0 / noise 1.003 / step 1.5 | 200 步（配置）、CLI 默认 **1** 样本、recycling **3**、step_scale **1.638**  | 改（更省的默认） |
| 扩散损失权重 | `(t²+σ²)/(t+σ)²` | `(t²+σ²)/(tσ)²`（EDM 原式） | **改** |
| 置信度模型 | **4 层** PairFormer + 4 头 | **整个 48 层 trunk 复刻** + 扩散轨迹时间条件聚合  | **改**（最大结构改动） |
| 口袋条件 | 两个模型（有/无 pocket），6 Å 全残基 | **单模型**，30% 迭代、6 Å、几何分布部分揭示、one-hot token 特征  | **改** |
| 数据 | PDB ≤2021-09-30，crop 384→640→768，batch 256，150k 步 | PDB + OpenFold 270K 蒸馏，crop 384/3456→512/4608，batch 128，68k 步  | 改（更小更省） |
| 训练目标 | 扩散 + distogram + confidence | 同四类，权重 4.0 / 3e-2 / 3e-3（论文未披露，仅配置） | 未披露 |

一句话：**结构骨架几乎逐位复刻 AF3，改动集中在「置信度头变成第二个 trunk」与「工程可扩展性」**。

### 1.4 谱系位置与许可证

- 上游：AlphaFold3（Nature 2024，权重未开放）。Boltz-1 是第一批把 AF3 级精度做成 **MIT 开源权重**的工作之一，与 Chai-1 同年（Chai-1 非商用）。演进：`v0.1.0`（2024-11 纯推理）→ `v0.4.1`（补数据管线、评测、口袋条件）→ `v1.0.0`（Boltz-1x）→ 主线转向 Boltz-2。
- Boltz-1x 的**定义论文内部即不一致**：p.3 说"Boltz-1 + steering 即 Boltz-1x"，p.18 说它 "an updated model"；代码侧 v1.0.0 默认 checkpoint 名仍是 `boltz1_conf.ckpt`，只多出默认开启的 steering 与 `--no_steering`。→ 倾向"同权重 + 推理期引导"，**权重是否被替换【未核实】**。
- 许可证分两层：**代码 MIT**（LICENSE 全文标准 MIT，无附加条款，可商用）【核实】；**权重**仅 README 声明 "code and weights under MIT"、论文亦称含数据集全 MIT，但 HF `boltz-community/boltz-1` 模型卡本次不可达，**原文【未核实】**；对外交付前应补存模型卡 LICENSE 凭证。

---

## 2. 表征粒度与表示层

### 2.1 token 的定义（论文原句）

> "Tokens are defined as amino acids for protein chains, nucleic acid bases for RNA and DNA, and individual heavy atoms for other molecules and modified residues or bases." 

对应代码（`src/boltz/data/tokenize/boltz.py:89,131`）：聚合物 token 的 `atom_num` = 该残基全部重原子数（变长），配体/修饰基团每个重原子单独成 token 且 `atom_num = 1`。故**配体确为"每重原子 1 token"**；修饰残基（如 MSE）拆成"主链残基 token + 修饰部分逐原子 token"。

### 2.2 单 token 原子槽位上限：Boltz-1 没有这个东西

- AF3 给每 token 固定 24 个原子槽位（不足 pad、超出截断），故需 `max_atoms_per_token` 常量；Boltz-1 **不设上限**：原子数直接等于该残基重原子数，配体恒为 1，整批原子平铺成变长数组、靠 `atom_pad_mask` 对齐（`featurizer.py:778`）。
- 计算量由 **`atoms_per_window_queries: 32` / `atoms_per_window_keys: 128`** 与 **`atom_feature_dim: 389`** 控制，而非"每 token 原子数"。
- 影响：大残基不被截断，是相对 AF3 的**隐性正确性提升**；代价是原子维不能按"token×24"整齐 reshape，需分段实现（§3.4 的 32×128 块稀疏正为此）。

### 2.3 有没有 residue frame / 扭转角？

- **输入里没有 frame，也没有扭转角**：featurizer 仍算 N–CA–C（核酸 C1'–C3'–C4'）骨架 frame 与 distogram 目标，但 frame 只在**置信度 loss** 里把误差投影到 token 局部坐标系（AF3 的 PAE），与 AF3「去掉 frame 输入」一致。
- 几何相关输入只有 `ref_pos`、token 代表原子坐标、`disto_coords`；**无 chi 角、无 sidechain torsion**。

### 2.4 原子级 cross-attention：有，而且是序列局部的

> "blocks of 32 atoms only attend to the 128 atoms that are closest to them in sequence space" 

即 AtomAttentionEncoder / Decoder 的 pair-biased transformer 不是"token 内 24 原子自注意力"，而是**沿序列铺开的 32×128 块稀疏注意力**：原子表示与原子对（`atom_z = 16`）互更新，再由 `token_to_rep_atom` 汇聚成 token 表示；映射预计算为"每 16 token 一块"，用稠密矩阵实现块稀疏。

### 2.5 表征小结表

| 项 | 值 | 来源 |
|---|---|---|
| 蛋白/DNA/RNA | 每残基 1 token | 见本篇 §9 |
| 配体 | 每重原子 1 token，`atom_num=1` |  + 代码 |
| 修饰残基/碱基 | 主链 1 token + 修饰部分逐原子 token | 见本篇 §9 |
| 每 token 原子槽位上限 | **不存在**（变长 `atom_num` + `atom_pad_mask`，`featurizer.py:778`） | 代码检索 |
| 原子特征维度 / 窗口 | 389；32 query / 128 key | 见本篇 §9 |
| frame / 扭转角输入 | 无（frame 仅用于 loss） | §2.3 |
| 原子级 cross-attention | 有（AtomAttentionEncoder/Decoder） | 见本篇 §9 |

---

## 3. 主干与生成范式

> 论文对架构只说「reproduce AlphaFold3」，σ 参数、通道宽、层数、头数一概未披露。故以下数字均来自 `v0.4.1` 的 `scripts/train/configs/full.yaml`（行号见 ）。

### 3.1 trunk（结构模块）

| 组件 | 值 | 来源 |
|---|---|---|
| MSA 模块 block / `msa_s` | 4 / 64 | `full.yaml:98,97`  |
| MSA 可否跳过 | **可**：`msa: empty` 走单序列，MSA 模块被跳过（`no_msa` 分支，`model.py:69,181,303`） | 代码  |
| PairFormer block 数 | **48** | `full.yaml:107`；论文 p.8 置信度语境也写 48 层  |
| pair / single 通道 | `token_z` 128 / `token_s` 384 | `full.yaml:81,80` |
| attention 头数 | 16 | `full.yaml:108` |
| dropout | msa 0.15 / z 0.25 / token 0.25 | `full.yaml:99,100,109` |
| distogram | 64 bins，2–22 Å | `full.yaml:71-73` |
| 推理期 MSA 深度 | CLI `max_msa_seqs` 默认 **4096**；论文评测 **16384**，消融 **4096** | 代码 +  |

### 3.2 diffusion module

| 组件 | 值 | 来源 |
|---|---|---|
| token transformer 层数 / 宽度 / 头数 | **24 / 384 / 16** | `full.yaml:118,119` + `:80` |
| atom transformer 宽度 / 原子对通道 | `atom_s` 128 / `atom_z` 16 | `full.yaml:78,79` |
| atom encoder / decoder 层数 | 3 / 3 | `full.yaml:116,120` |
| atom 侧头数 / k-v 通道 | 4 / 128 | `full.yaml:117,121` + `:85` |
| Fourier 维度 / cond transition 层数 | 256 / 2 | `full.yaml:115,122` |
| 原子 attention 窗口 | 32 query × 128 key | 见本篇 §9 |

> 实质差异：AF3 扩散 token 宽 768，Boltz-1 仅 **384**（与 trunk 共用宽度），去噪网络约为 AF3 一半【推断：依据 `token_s: 384` 与 AF3 笔记的 768】。

### 3.3 EDM 与采样预算（论文未披露 σ，全部来自配置）

| 参数 | 值 | 来源 |
|---|---|---|
| σ_data / σ_min / σ_max / ρ | 16.0 / 0.0004 / 160.0 / 7 | `full.yaml:114,171,172,174`  |
| P_mean / P_std | −1.2 / 1.5 | `full.yaml:175,176` |
| γ0 / γ_min | 0.8 / 1.0 | `full.yaml:177,178` |
| noise_scale / step_scale | 1.003（推理默认）/ 配置 1.0、**CLI 默认 1.638** | `BoltzDiffusionParams` / `full.yaml:180` + `main.py` |
| 默认采样步数 | **200** | `full.yaml:145`；CLI 同 |
| 输出样本数 | CLI **1**；验证配置 5；论文评测 5 | `main.py` / `full.yaml:147,167` /  |
| 训练期 multiplicity | 16 | `full.yaml:146` |
| recycling | CLI **3**（+1 前置 = 4 趟 trunk）；论文评测 **10** | `full.yaml:144` /  |

**这是第二个「论文 vs 代码」冲突点**：论文称所有方法都用「200 步 + 10 轮 recycling + 5 输出」评估、且 "matching our model"；但发布配置与 CLI 默认是 **3 轮 recycling、1 个样本**，文档甚至教用户"要模拟 AF3 默认请用 `--recycling_steps 10 --diffusion_samples 25`"——说明 10/25 是**模仿 AF3**的参数。消融也显示 recycling 3→10 只把 oracle LDDT 从 0.729 抬到 0.735（top-1 0.716→0.720）。→ **默认 3 轮是性价比选择，论文表格用 10 轮；正文未交代默认值是 3**。

### 3.4 与 AF3 的五处算法差异

1. **MSA 模块算子改序**：OuterProductMean 从第一位挪到 MSATransition 之后，让 transition 学到的单序列表示立刻经外积传播到 pair；作者称"对我们有利"。属**无成本结构性改动**，未给消融数字。
2. **Transformer 残差改序**：AF3 的 `a ← AttnPairBias(a) + CondTransition(a)` 既无残差、又使同一 block 内 attention 输出进不了 transition；Boltz-1 改为两步各自残差 `a ← a + …`。
3. **Kabsch 插值**：反向每步在 `x̂_t` 与 `x̂_{t-1}` 插值前先做刚体对齐；作者同时承认**最终模型基本把去噪结果投影回流形，Kabsch 并非关键**（仅子集数据训练时明显）。
4. **扩散损失权重**：改用 EDM 原式 `(t²+σ²)/(tσ)²`，AF3 用 `(t²+σ²)/(t+σ)²`。
5. **置信度模型 = 第二个 trunk**：AF3 只在 trunk 后接 **4 层** PairFormer + 4 头；Boltz-1 复用 AtomAttentionEncoder + MSAModule + **48 层** PairFormer，以 trunk 的 EMA 权重初始化，额外吃"扩散轨迹上各步 token 表示的时间条件聚合"，并把 token 特征**乘性**注入 pair。代价是置信度模块参数量与 trunk 同量级（论文未给参数量）。

### 3.5 工程侧（决定"能不能在单卡跑"）

- **trunk 只跑一次**：产出的 attention pair bias 与输入结构、扩散步**都无关**，可跨样本、跨去噪步共享——"5 个样本"几乎不增加 trunk 成本。
- **块稀疏原子注意力**：32×128 窗口，映射按 16 token 一块预计算。
- **chunking**：沿用 OpenFold 的 triangular attention 分块，并**扩展到 MSA 模块**（transition / pair-weighted averaging / outer-product）。
- **trifast**：自研 Triton kernel，带 inner bias 的 chunked online softmax，pair attention 显存从 O(n²) 降到线性级。
- **对称性校正**：链指派最多 100 种、配体原子最多 1000 种扰动（硬上限，非穷举）。
- **统一裁剪**：空间与连续裁剪间插值，邻域 **0–40 token** 均匀随机；配置 `min/max_neighborhood = 0/40`。

---

## 4. 条件控制

### 4.1 逐项判定表

| 条件 | 是否推理接口 | 接口字段（YAML） | 代码位置 | 是否需重训 |
|---|---|---|---|---|
| MSA | 是 | `sequences.protein.msa: <path>` / `empty`；或 `--use_msa_server` | schema.py、main.py | 否 |
| MSA 配对策略 | 是 | `--msa_pairing_strategy {greedy,complete}`，按 taxonomy 保密度配对 | main.py、Algorithm 3  | 否 |
| 模板 | **不支持** | —（无 `templates` 键） | 论文明确删除  | 需改代码 + 重训 |
| 口袋 / 表位 | 是（**单模型**） | `constraints.pocket: {binder: [chain,res], contacts: [[chain,res],…]}` | schema.py ；配方  | 否（能力已在权重里） |
| 共价键 | 是 | `constraints.bond: {atom1: [chain,res,atom], atom2: […]}` | schema.py  | 否 |
| 多链（同源/异源复合物） | 是 | `sequences` 下多条 `protein/dna/rna/ligand`，`id` 为链名 | 例子 YAML  | 否 |
| 多链/链间模板 | 不支持 | — | 同上 | 需重训 |
| 抗体 / CDR 条件 | **不支持**（全仓无 antibody/CDR 代码） | — | 见本篇 §9 | 需重训 |
| 亲和力 | **不支持**（属 Boltz-2） | — | v0.4.1 无 affinity 键 | 需换模型 |
| 部分指定残基 | 部分支持：pocket 的 contacts 天然支持"给一部分" | 同上 | 见本篇 §9 | 否 |

### 4.2 口袋条件的正确理解（最容易被误读的一点）

AF3 是**微调出第二个模型**（有/无 pocket），且要求给出 6 Å 内**全部**残基。Boltz-1 改为：

- **只训练一个模型**，30% 迭代里随机挑一个 binder 注入口袋信息；
- 口袋残基数量上限从**几何分布**抽取，再从"与 binder 任一重原子距离 ≤6 Å"的候选里随机选——训练时模型见过"只给一部分表位"的输入；
- 编码是一个额外的 **one-hot token 特征**（不是新的注意力偏置）。

因此**推理时给 1–2 个关键残基也能跑，且不需换 checkpoint**——对 VNAR 特别有用（只想说"结合在这个 CDR3 疏水 patch"时无需列全 6 Å 邻居）。但该能力的验证**只有 Figure 4 的定性展示**（无定量指标），"部分指定"的鲁棒性属【论文自称】。

### 4.3 哪些"条件"是假接口

模板（要加 = 改输入特征 + 重训，非 YAML 能解决）、抗体轻重链配对先验、亲和力梯度：**都没有**。

---

## 5. 训练数据构成与切分

### 5.1 数据表

| 项 | 值 | 来源 |
|---|---|---|
| 结构来源 | PDB（`Biological Assembly 1`，mmCIF 解析） | 见本篇 §9 |
| **PDB 切点** | **2021-09-30**（与 AF3 相同） | 见本篇 §9 |
| **分辨率阈值** | 论文 **"a resolution of at least 9 Å"**；代码 structure 9.0 / full、confidence **4.0** | 见本篇 §9 |
| 结构总数 | **未披露**（只给 val/test 数字） | 见本篇 §9 |
| 蒸馏数据 | **OpenFold 蒸馏约 270K 结构**（自带 MSA），前 53k 步与 PDB 等量混合；**不用 AF3 蒸馏** | 见本篇 §9 |
| MSA 生成 | ColabFold/MMseqs2，uniref30 2302 + colabfold envdb 202108，taxonomy 标注 | 见本篇 §9 |
| 配体构象 | RDKit ETKDGv3，每个 CCD 只预计算 **1 个**构象 | 见本篇 §9 |
| 清洗 | 沿用 AF3：配体排除表、最小解析残基数、去冲突链；共价配体去掉 leaving atoms | 见本篇 §9 |
| 长度 / 聚类 | val/test 100–2000 总残基，静态过滤器 4–5000；序列 40% identity（`mmseqs --min-seq-id 0.4`），小分子 Tanimoto ≤0.8 隔离 | 见本篇 §9 |
| 采样 | 空间↔连续插值裁剪，邻域 0–40 token 均匀随机 | 见本篇 §9 |
| 训练规模 | 68k 步 × batch 128；前 53k 步 crop 384 token/3456 atom（PDB+蒸馏各半），后 15k 步 512/4608（纯 PDB） | 见本篇 §9 |

### 5.2 【核心】分辨率阈值：论文 vs 代码，到底成不成立？

**先给结论：部分成立。**"论文 9 Å vs 配置 4.0"的对立**确实存在**，但讲成"论文写错/代码写错"都不准；准确说法是「**论文只描述了 9 Å 这一个训练集，代码里 9.0 与 4.0 并存、分属不同训练阶段，而论文对阶段划分与 4.0 只字未提**」。

（a）论文侧——只有一个训练数字，全文无 4 Å：

| 位置 | 原文 | 数值 |
|---|---|---|
| p.3 §2.1（训练数据） | "with a resolution of **at least 9˚A**" | **9 Å**（唯一训练阈值） |
| p.4 §2.2（验证/测试） | "Resolution is **below 4.5˚A**" | 4.5 Å（只管 val/test） |
| 其余位置 | 检索 `resolution` / `4˚` / `4.5` / `9˚` | **无其它命中**，**没有 4 Å** |

措辞本身有缺陷：英文 "at least 9 Å" 字面是「≥9 Å」（更差），同页 val/test 却写 "below 4.5 Å"；代码语义是 `resolution <= 9.0` 才保留，故应读作语病、**意图为 ≤9 Å**——但按字面读会得到相反结论。

（b）代码侧——一道过滤器、三种配置值：

| 文件 | 行 | 值 | 该配置的角色 |
|---|---|---|---|
| `scripts/train/configs/structure.yaml` | 42 | **9.0** | 结构模型（`confidence_prediction: false`） |
| `scripts/train/configs/confidence.yaml` | 45 | **4.0** | 置信度模型（`pretrained:` 载结构权重，`structure_prediction_training: false`） |
| `scripts/train/configs/full.yaml` | 44 | **4.0** | 任何文档都未提及的「结构+置信度一体」配置 |
| `resolution.py` | 8 / 34 | 默认 `9.0`；`return structure.resolution <= self.resolution` | 唯一一道过滤，True=保留  |

（c）"两段式"解释站得住一半：

- 站得住：9.0 与 4.0 确属**两个训练阶段**——`structure.yaml` 是结构模型阶段（9.0），`confidence.yaml` 是加载结构权重、只训置信度的阶段（4.0），`full.yaml` 合一并沿用 4.0，符合"先粗后细"。
- 站不住：`scripts/process/*` 完全无分辨率过滤，只有长度 4–5000、unknown、clash 三类静态过滤器；「9 Å 是爬取初筛」在代码里**找不到对应物**——两个值都只是 dataloader 的 dynamic filter。
- 也不成立：「论文笔误」——`structure.yaml` 确有 9.0，且 v1/v4 两版都写 9 Å。

（d）最可能的真相【推断】：① 结构模型主训练用 **9 Å**（即论文那段：68k 步、384/3456→512/4608）；② 置信度微调收紧到 **4 Å**，论文对该阶段数据**只字未提**（仅写 trunk EMA 初始化、末层零初始化）；③ 后放出的 `full.yaml` 沿用 4.0，推测为省算力/提质，**未同步论文**。

（e）仍无法判定：产出 `boltz1_conf.ckpt` 那次训练用 9.0 还是 4.0——仓库自称这些 yaml 是 "a template configuration file **analogous to** the one we used"；论文第一阶段 crop 384/3456 在全仓零命中，同样说明发布配置 ≠ 生产配置；S3 的 `rcsb_processed_targets` 实际阈值需下载统计，本次未做 →【未核实】。

一句话（可直接引用）：论文的 9 Å 对应结构模型阶段（`structure.yaml:42 = 9.0`），不是笔误；但置信度阶段与未文档化的 full 配置都收紧到 **4.0**（`confidence.yaml:45`、`full.yaml:44`），论文从未出现 4 Å。故"论文 9 Å、配置 4.0"**部分成立**，准确表述是「论文只写了结构阶段的 9 Å，未交代后续收紧到 4 Å」。

### 5.3 其他"论文 vs 代码"不一致（同一张表里看完）

| 项 | 论文 | 代码（v0.4.1） | 判定 |
|---|---|---|---|
| 训练分辨率 | 9 Å  | 9.0 / 4.0 / 4.0 三份配置  | **部分成立**（见 §5.2） |
| batch size | 128  | `batch_size: 1`，DDP，**无梯度累积** | **不一致**（除非 128 卡；论文未说卡数） |
| 第一段 crop | 384 token / 3456 atom  | 全仓零命中，均为 512/4608  | **不一致**（无法复现第一阶段） |
| 训练步数 | 68k，前 53k / 后 15k 分段  | 无 `max_steps`、无分段逻辑  | **不一致**（未发布分段脚本） |
| 训练 MSA 深度 | 未披露（评测 16384） | `max_seqs: 2048`  | 无可比 |
| 默认 recycling | 评测 10 轮，称 "matching our model"  | 3 轮 ；3 轮已饱和  | **不一致（措辞）** |
| 默认样本数 | 5  | CLI 1，验证 5  | **不一致** |
| 采样步数 | 200  | 200  | 一致 |
| 蒸馏数据 | OpenFold ≈270K  | 清单在 S3，**未核实** | 未核实 |
| 卡数 / GPU 小时 | 未披露，仅"AF3 约 4×"  | 无 | 未披露 |

### 5.4 测试集与隔离（这块论文写得最扎实）

- val 553 / test 593；**test 只取 2023-01-13 后释放**的结构，蛋白序列不与"2021-09-30 前训练簇"重合（40% identity），小分子相似度 ≤0.8 Tanimoto，总残基 100–2000。
- 训练切点 2021-09、测试起点 2023-01，中间留 15 个月缓冲；CASP15 的 76 个 target 因 OOM/失败剔到 66 个。
- **注意**：test 集仅 **541** 个结构计入指标，小分子子集更小，LDDT-PLI / L-RMSD 的置信区间偏宽——论文自称与 AF3 的差距 "well within the confidence intervals"。

---

## 6. 宣称与可验证性

### 6.1 宣称三列表

| 宣称 | 证据类型 | 核查状态 |
|---|---|---|
| "达到 AF3 级精度" | 柱状图（Fig 5/7），**正文无基线数值** | 【论文自称】；仅核到"差异在置信区间内"，**无法复算**（图无数值；仓库有输入/输出/聚合脚本，本次未跑） |
| 计算效率明显优于 AF3 | 仅文字 + 图 4/6 | 【论文自称】，无墙钟/显存表；架构上有支撑（trunk 只跑一次 + bias 缓存、trifast） |
| 训练只需 AF3 的 ~1/4 算力 | 仅文字（68k×128 vs 150k×256） | 【核实：算术可推】步数×batch 比 ≈ 0.11–0.23，量级支持"约 4×"，**无 GPU 小时** |
| 开源可得（code/weights/data/benchmark 全 MIT） | 声明 + LICENSE + 仓库 | 代码**【核实】**；权重**【未核实】**（HF 不可达）；数据集为 S3 大包（未下载） |
| 置信度重建带来更好校准 | 无消融、无数值 | **无证据**（仅结构描述与 Fig 3） →【论文自称】 |
| 单模型口袋条件优于 AF3 双模型 | 仅 Fig 4 定性 | **无定量指标** →【论文自称】 |
| Boltz-1x 解决物理有效性问题 | 不通过率 57% → 通过率 97% | 【核实：文字有数】，但基线方向写反（Chai-1/AF3 给通过率），**跨模型需换算** |
| "移除模板不影响大模型性能" | 无消融 | 【论文自称】，无对照实验 |
| 默认参数即可用（3 recycling / 1 样本） | Table 1 消融 | 【核实】，但正文未点明默认 3，且评测用 10 |

### 6.2 我是拿什么核的（方法论透明）

- 论文侧：bioRxiv v4 PDF（24 页）全文 + v1 JATS XML 对照；页码为 PDF 物理页。代码侧：`v0.4.1` zipball 逐文件读并记行号，另取 `v1.0.0` 查 steering。
- **没有**核的：HF 权重、S3 数据包、论文图（仅位图）、benchmark 复跑。

### 6.3 【核心】论文 vs 代码：冲突汇总表

| # | 冲突点 | 论文怎么说 | 代码怎么做 | 严重度 / 建议采信 |
|---|---|---|---|---|
| 1 | 训练分辨率 | 9 Å（唯一一处，p.3） | structure 9.0 / confidence 4.0 / full 4.0  | **高**。采信"论文只描述结构阶段，4.0 属未文档化阶段"——两者都不算错，**但论文漏写** |
| 2 | 默认 recycling | 评测 10 轮，称 "matching our model" | CLI 3 | 中。采信代码；10 轮是**评测预算** |
| 3 | 默认样本数 | 5 | CLI 1；验证 5 | 中。同上 |
| 4 | batch size | 128 | 1/卡 + DDP，无梯度累积 | 中。要么 128 卡，要么 128 是等效 batch；论文未交代 |
| 5 | 第一段 crop | 384/3456 | 零命中（只有 512/4608） | 中。无法复现第一阶段 |
| 6 | 训练步数/分段 | 68k = 53k+15k | 无 `max_steps`、无分段逻辑 | 中。需人工改配置 |
| 7 | 架构超参 | "复现 AF3"，未给宽度/层数/σ | 全在 full.yaml（48/4/384/128/64/24/3…，σ=16/4e-4/160/7） | 低（信息缺失） |
| 8 | Boltz-1x 是什么 | p.3 "同一模型 + steering"；p.18 "an updated model" | v1.0.0 同 checkpoint 名 + steering 默认开 | 中。倾向"同权重 + 推理引导"，权重是否替换【未核实】 |
| 9 | 物理有效性数字方向 | Boltz-1 给"不通过率 57%"，Chai-1/AF3 给"通过率" | 无可对照代码 | 低但**易误读**：换算后 ≈43% 通过 < AF3 ≈58% |
| 10 | 评测可复现性 | "公开所有 inputs/outputs/scripts" | eval 脚本只覆盖 4 类指标，**无 PoseBusters** | 中。第 5 类需自建 |

---

## 7. 评测协议与成本

### 7.1 benchmark 一览

| benchmark | 规模 | 是否公开 | 来源 |
|---|---|---|---|
| 自建 test set（PDB 2023-01-13 后） | 593 结构，计入指标 **541** | 论文称公开输入输出（S3 大包，本次未下载） | 见本篇 §9 |
| CASP15 | 76 → 计入 **66** | 同上 | 见本篇 §9 |
| 消融 | test set 上 9 组（recycling × 采样步） | Table 1 全文给出 | 见本篇 §9 |

**没有用 CASPR、也没用 PoseBusters 官方套件排名**：物理有效性是"按 PoseBusters 规则自己实现"的通过率，仓库里连这部分脚本都没放。

### 7.2 指标与实现

1. all-atom LDDT（全复合物）；
2. DockQ 成功率（DockQ > 0.23）；
3. 蛋白-配体界面 LDDT（LDDT-PLI）；
4. 配体 pocket-aligned RMSD < 2 Å 的比例；
5. PoseBusters 规则通过率（六类物理检查）。

全部用 **OpenStructure 2.8.0** 计算；仓库 `scripts/eval/run_evals.py` 用 `ost compare-structures --lddt --bb-lddt --qs-score --dockq`，配体另加 `--lddt-pli --rmsd`。报 oracle（5 样本中最好）与 top-1（按置信度选）两档。

### 7.3 数值：唯一可抄的表是消融表

Table 1（test set，MSA ≤4096，200 步）：

| recycling | 步数 | LDDT oracle | LDDT top-1 | DockQ oracle | DockQ top-1 | LDDT-PLI oracle | LDDT-PLI top-1 | L-RMSD oracle | L-RMSD top-1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 200 | 0.698 | 0.681 | 0.579 | 0.544 | 0.573 | 0.530 | 0.582 | 0.541 |
| 1 | 200 | 0.718 | 0.702 | 0.656 | 0.635 | 0.623 | 0.573 | 0.588 | 0.535 |
| 2 | 200 | 0.726 | 0.710 | 0.651 | 0.632 | 0.616 | 0.581 | 0.587 | 0.546 |
| **3** | 200 | **0.729** | **0.716** | 0.654 | 0.625 | 0.621 | 0.580 | 0.581 | 0.545 |
| 6 | 200 | 0.732 | 0.714 | 0.644 | 0.635 | 0.630 | 0.593 | 0.595 | 0.555 |
| 8 | 200 | 0.733 | 0.717 | 0.644 | 0.633 | 0.630 | 0.584 | 0.588 | 0.545 |
| 10 | 200 | 0.735 | 0.720 | 0.644 | 0.631 | 0.619 | 0.577 | 0.575 | 0.541 |
| 3 | 50 | 0.727 | 0.710 | 0.645 | 0.627 | 0.621 | 0.579 | 0.586 | 0.540 |
| 3 | 20 | 0.720 | 0.693 | 0.615 | 0.592 | 0.577 | 0.547 | 0.550 | 0.532 |

读法：**oracle 单调涨、top-1 在 3 轮后基本平**（0.716→0.720）；采样步 20→50 涨得最猛（0.693→0.710），50→200 只再涨 0.006。→ `--recycling_steps 3 --sampling_steps 50` 即接近满配。

### 7.4 基线数值与成本

- **AF3 / Chai-1 的数值未以数字形式出现**，只在 Figure 5/7 柱状图里 → 记「未披露（仅图）」；任何"Boltz-1 比 AF3 高/低几个百分点"的二手说法都无法从正文复核。
- 物理有效性：Boltz-1 top-1 **57% 不通过**（≈43% 通过）；Chai-1 ≈27% 通过、AF3 ≈58% 通过；Boltz-1x ≈97%。→ 零样本物理质量**不如 AF3**，这是最明显短板，也是 Boltz-1x 存在的理由。
- 硬件：**A100 80GB**，OOM 结构对所有方法一律剔除；用时未披露。训练成本：**未披露 GPU 小时**，只给"AF3 约 4×"。

---

## 8. 对我们的可复用点（VNAR 专列）

### 8.1 能不能直接当 VNAR 预测器？

**能，零样本可用，但没有抗体特化。** VNAR/VHH 是单域 ~120 aa，约 120 个 token + 少量配体/糖基 token，远低于 `max_tokens 512`；32/128 序列局部窗口对这么短的链**几乎等于全局注意力**。单链无配对需求 → 不触发 MSA 配对算法（Algorithm 3）。但论文 benchmark **完全没有抗体/纳米抗体子集**（test set 只按分辨率、长度、日期、相似度筛），仓库也无 antibody/CDR 代码，故"它在 VNAR 上有多准"**没有一手证据**，只能用我们自己的 held-out 集量。

### 8.2 MSA 依赖有多强？

- 强依赖：MSA 是 trunk 主干（MSA 模块 4 block + 配对），ColabFold 检索是官方路径。
- 但**可退化为单序列**：`msa: empty` 让 MSA 模块走 `no_msa` 分支。对 VNAR 关键——IgNAR 种系在 UniRef 覆盖稀薄、深 MSA 常拿不到，此时要么接受单序列的精度损失，要么用 IMGT/自定义库造 MSA。
- 推理默认 `max_msa_seqs = 4096`（CLI），比论文评测的 16384 更省；对 ~120 aa 的 VNAR 已远超需要。
- 实务：`--use_msa_server` 走 ColabFold 公共服务会限流/排队，建议本地部署 MMseqs2 + uniref30。

### 8.3 能否微调？代码里有入口，但没有配方

- **入口存在**：`scripts/train/train.py` 的 `pretrained`（L66）在 L128–149 把预训练权重灌进模型，`confidence.yaml`/`full.yaml` 都用它指向上阶段 checkpoint，故"从 `boltz1_conf.ckpt` 继续训"是**代码支持**的。
- 但需自备：① `scripts/process` 处理过的 target 清单（`rcsb_processed_targets` 级，数十 GB 起）；② 每个 target 的 ColabFold MSA；③ 1–8 张 A100 级卡（`batch_size: 1` + DDP，论文的 128 是等效大 batch）。
- **没有**的：LoRA/冻结层开关、抗体专用 loss、CDR 加权。只想调置信度头可以——`full.yaml` 里设 `structure_prediction_training: false`、`confidence_prediction: true`（即 `confidence.yaml` 的形态）。
- 建议：**先零样本 + 多 seed 采样**评估基线，再决定是否微调；VNAR 仅 ~120 aa，微调收益不确定而成本明确。

### 8.4 硬边界（逐条）

| 边界 | 实情 | 依据 |
|---|---|---|
| 许可 | 代码 MIT 可商用；权重 MIT 仅 README 声明、HF 条款未核实 | 见本篇 §9 |
| 显存 | 评测用 A100 80GB；VNAR 量级单卡 24 GB 应可跑【推断】；官方无显存表 | 见本篇 §9 |
| 抗体能力 | 无 CDR/epitope 专用条件、无轻重链配对先验、无亲和力 | 见本篇 §9 |
| 物理质量 | 零样本 top-1 仅 43% 通过物理检查，低于 AF3 的 ~58% | 见本篇 §9 |
| 模板 | 不支持（论文主动删除） | 见本篇 §9 |
| 版本冻结 | Boltz-1 线停在 v0.4.1，修复与新特性都进 Boltz-2 | 见本篇 §9 |

### 8.5 给 VNAR 项目的落地建议

1. 用 `v0.4.1`（或 `v1.0.0` 加 `--no_steering` 做对照）跑零样本基线：`msa` 用本地库或 `empty`，`--diffusion_samples 5 --recycling_steps 3 --sampling_steps 50`（50 步已近 200 步）。
2. 关心"像不像真抗体"就用 Boltz-1x 的 steering（v1.0.0 默认开启：`fk_lambda=4.0`、3 粒子、每 3 步重采样、16 步梯度下降）——物理通过率 43%→97% 而精度不掉。
3. 表位定向用 `constraints.pocket`（binder 填抗原链、contacts 填 VNAR 表位残基），**只给关键残基即可**。
4. 必须微调时优先只训置信度（`structure_prediction_training: false`），并降低 `max_seqs` 省显存。

---

## 9. 疑问、心得与参考

### 9.1 仍未解决的问题

1. **4.0 到底用没用**：`confidence.yaml` 的 4.0 是模板值还是生产值？需下载 S3 的 `rcsb_processed_targets` 统计分辨率分布才能定论【未核实】。
2. **batch 128 怎么凑**：配置 `batch_size: 1` 且无梯度累积，若真是 128 卡 DDP，"训练只需 AF3 1/4 算力"要按 128 卡重算【卡数未核实】。
3. **论文第一阶段 384/3456 的配置从未发布**，复现"前 53k 步混合蒸馏集"只能自写分段脚本。
4. **置信度重建有没有用**：全文无"4 层 vs 48 层"消融——最大的"宣称无证据"缺口。
5. **Boltz-1x 是否换了权重**：checkpoint 名未变、HF 不可达，无法比对哈希。
6. **pocket 条件的定量效果**：只有 Figure 4 定性图，无"部分指定 vs 完整指定"曲线。
7. 物理有效性为何仅 43%：论文归因于"扩散模型的固有冲突"，无诊断实验。

### 9.2 写作/核查心得

- **配置名比数值更容易骗人。** 只看 `full.yaml`（4.0）会得出"论文写错"；同时打开 `structure.yaml`（9.0）并确认 `resolution.py` 只有一道过滤器，才得到"分阶段收紧"。→ **先枚举同目录所有配置再下判断。**
- **别信"arXiv 版本号"这类二手线索**：本次预下载的 PDF 是另一篇论文，不查正文就会写错整篇。
- 仓库文档的 "analogous to" 与残留旧模块名说明发布配置是"脱敏模板"，不能当生产快照；论文自己也承认"AF3 原文可能就有 reporting mistakes"——读 AF3 系论文的默认心态应是：**凡"复现 AF3 但改了 X"，先怀疑 AF3 原文写错了 X**。

### 9.3 参考资料

**（A）论文（一手）**

- Wohlwend J., Corso G., Passaro S., Getz N., et al. *Boltz-1: Democratizing Biomolecular Interaction Modeling*. bioRxiv 2024.11.19.624167, **v4 (2025-05-06)**。页码为 PDF 物理页：p.1 题名/许可、p.3–4 数据与训练、p.5 裁剪/配对/口袋、p.6–8 架构改动、p.9–11 原子注意力与工程优化、p.14–18 评测与消融。v1 JATS XML 用于确认 9 Å 与 token 定义两版未变。

**（B）代码/配置（路径 + 行，均指 tag `v0.4.1` / `9461d02`）**

- `src/boltz/data/filter/dynamic/resolution.py:8,14,17,34`（默认 9.0、`<=` 语义、唯一一道）；`training.py` 的 `all(f.filter(record))`（True=保留）；`scripts/process/rcsb.py` 只有长度/unknown/clash 三类静态过滤器。
- `scripts/train/configs/structure.yaml:39,42,50,52,124`（2021-09-30、**9.0**、crop 512/4608、`confidence_prediction: false`）；`confidence.yaml:16,42,45`（`pretrained:`、**4.0**）；`full.yaml:16,41,44,52-54,59,71-73,78-85,90,97-98,107-108,114-122,144-150,165-180,188-189`（**4.0** + 全部架构/EDM 超参）。
- `src/boltz/model/modules/{trunk,diffusion,confidence,encoders}.py`、`model.py` —— 48/4 block、24 层扩散、置信度双路径（`imitate_trunk` 时 48 层，否则 4 层）；`tokenize/boltz.py:89,131`（变长 `atom_num` / 配体 1）；`parse/schema.py`、`yaml.py`（`constraints.bond`/`pocket`、`msa: empty`）。
- `scripts/train/train.py:66,128-149,192-198`（`pretrained`、DDP、无梯度累积）；`scripts/eval/run_evals.py`（`ost compare-structures`，无 PoseBusters）；`docs/training.md:44`（"analogous to"）；`docs/prediction.md`（默认参数与"模仿 AF3 用 10/25"）；`LICENSE`/`README.md`（MIT）。
- `v1.0.0`（`34cf560`）：`main.py` 的 `--no_steering` 与 `BoltzSteeringParams`（fk_lambda 4.0、3 粒子、每 3 步重采样、16 步 GD）。

**（C）二手转述（不作一手证据）**

- "Boltz-1 是 AF3 开源平替"是定位性描述（方向对、无数值）；"Boltz-1 有 arXiv 2411.19169"**已被本篇否证**。社区对"9 Å vs 4.0"的讨论方向与本篇一致，但**未区分 structure/confidence 两阶段**；本篇以配置与代码语义为准。
