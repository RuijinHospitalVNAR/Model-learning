---
model: AlphaFold 3
aliases: [AF3, AlphaFold3]
family: 结构预测
granularity: 全原子坐标（聚合物残基 token + 配体每原子 token）
backbone: Pairformer trunk（48 blocks，pair 128 / single 384）+ 4-block MSA 模块 + atom-level diffusion module（代码默认 24 个 token block）
generative: 扩散（EDM 噪声调度，x0 参数化，默认 200 步）
conditioning: MSA | 模板 | 输入序列/聚合物类型 | 配体/离子/修饰残基（CCD）| CCD 共价键；口袋条件需单独微调（待核实）
data_scope: PDB ≤2021-09-30；AF2-Multimer v2.3 蒸馏；MSA：UniRef90/UniProt/Uniclust30+BFD/Reduced BFD/MGnify（RNA：Rfam/RNACentral/Nucleotide collection）；测试：Recent PDB 8,856、PoseBusters 428（2019-09-30 重训模型）、CASP15 RNA 10 个靶
status: 学习中
verified_at: 2026-11-04
---

# AlphaFold 3

> **状态**：学习中（主体已核实，待核实项集中在训练数据清洗与部分指标精确值）
> **记录人**：团队共同学习
> **最后更新**：2026-11-04

---

## 1. 学习背景与目标

- **为什么学**：AF3 是本仓库“全原子模型”四路线分类的参照系。后续 RFAA、Boltz-1/2、Chai-1、Protenix、BoltzGen、AnewOmni 的对比，都要先回答“相对 AF3，差在表征、架构、条件接口，还是数据与评测口径”。
- **本次目标**：
  1. 把 AF3 的表征、trunk、diffusion、置信头拆到“可复述”的程度；
  2. 把训练数据与切分写到能指导复现的程度，而不是停留在“用了 PDB”；
  3. 建立一张“宣称—证据—核查状态”表，区分论文原文、代码配置、二手转述与推断；
  4. 给出 VNAR 场景下“能直接用/不能直接用”的边界。
- **一句话判断**：AF3 的核心变化不是“原子级 token 让骨架更准”，而是**去掉 residue frame 与侧链扭转角、用 token 化原子承接任意化学、把 AF2 的 MSA 堆栈压缩成 4 个 block 的 pair 通道**；精度提升的归因在论文里没有被拆成“表征 vs 架构 vs 数据”的消融，**不能把化学覆盖面的扩大误读成骨架精度的机制**。【推断，依据 §3.1 与 §5.3】

---

## 2. 核心概念

| 术语 | 含义 |
| ---- | ---- |
| token | AF3 的输入/输出基本单位。聚合物（蛋白、DNA、RNA）**每个残基 1 个 token**；配体、离子、小分子**每个原子 1 个 token**。【核实：`features.py` L334；Nature 正文】 |
| atom slot | 一个 token 内最多容纳的原子槽位。代码默认 `max_atoms_per_token=24`；残基原子数超过上限时截断并告警，不足时补空位。【代码／配置核实：`pipeline.py` L71、L118、L129】 |
| atom cross-attention | 先在 token 内部做原子级局部注意力，再在 token 级做全局注意力；避免对全部原子做二次复杂度注意力。 |
| Pairformer | AF3 的 trunk 主体，48 个 block，pair 表示 128 通道、single 表示 384 通道；每个 block 参数独立。【核实：Nature Fig. 2a 图注；`modules.py` L396-448】 |
| MSA 模块 | AF3 仍保留 MSA 处理，但压缩为 4 个 block，用 pair-weighted averaging 把 MSA 信息压到 pair 表示；MSA 表示本身不保留。【核实：Nature L69；`evoformer.py` L201-222】 |
| EDM diffusion | 对原子坐标加高斯噪声、网络预测干净坐标（x0）的扩散范式。AF3 用 `σ_data=16`、`σ_min=4e-4`、`σ_max=160`、`ρ=7`，默认 200 步。【代码／配置核实：`diffusion_head.py` L34、L73-77、L101-104】 |
| x0 参数化 | 网络输出的是去噪后的坐标，而不是噪声；配合 EDM 的 `c_skip/c_out` 预条件做坐标回归。【代码／配置核实：`diffusion_head.py` L258-260】 |
| pLDDT | 每个原子的局部置信度，50 个 bin；反映局部几何（侧链、局部堆积）而非全局 domain 朝向。【代码／配置核实：`confidence_head.py` L44、L217-227】 |
| PAE | Predicted Aligned Error，残基对级误差；回答“以残基 i 对齐时，残基 j 的位置误差多大”，对 domain/链间朝向最有用；64 个 bin。【代码／配置核实：`confidence_head.py` L34-46】 |
| PDE | Predicted Distance Error，原子对级距离误差；64 个 bin，`max_error_bin=31 Å`；对界面、配体接触更直接。【代码／配置核实：同上】 |
| pTM / ipTM | 由 PAE 的 softmax 经 TM-score 权重积分得到的全局/界面置信度；用于排序，不是独立训练的 head。【代码／配置核实：`confidence_head.py` L200-213、L245-287】 |
| CCD | Chemical Component Dictionary；AF3 的配体、离子、修饰残基、共价键信息都从这里取。【核实：SI L110；`evoformer.py` L116-159】 |
| ref_pos | CCD 参考构象的原子坐标，作为输入特征；它不是刚性 frame，也不是手性约束。【核实：Nature L175；代码特征】 |
| cross-distillation | 用 AF2-Multimer v2.3 的预测结构作为额外训练数据；论文称显著抑制长无序区的“幻觉”。【核实：Nature L76-82】 |
| PoseBusters | 配体共折叠的物理有效性 + RMSD 基准；AF3 论文用 428 个 2021 年后结构，并用 2019-09-30 切点重训的模型评测。【核实：Nature L96、L215】 |
| DockQ | 蛋白-蛋白界面质量指标；acceptable 阈值通常取 0.23。AF3 正文给的是 P 值，不是百分比。【核实：Nature 正文；百分比缺失见 §5.1】 |
| seed / sample | 一个 model seed 可采多个 diffusion sample；AF3 默认 5 seeds × 5 samples = 25 个样本；抗体任务可到 1,000 seeds。【核实：Nature L59、L155、L175】 |
| 手性违规 | 生成结构与输入参考构象的局部手性不一致；AF3 无显式手性约束，论文自报 4.4% 违规率。【核实：Nature L140】 |

---

## 3. 原理与推导

### 3.1 表征：混合 token 方案

**具体做法**

1. **聚合物残基 = 1 个 token。** 一个氨基酸/核苷酸的全部原子（N、CA、C、O、CB、侧链原子等）打包进同一个 token 的 atom slot；token 内原子数可变，用 mask 标记有效原子。【代码／配置核实：`features.py` L147、L310-334；`pipeline.py` L71、L118】
2. **配体/离子/小分子 = 每原子 1 个 token。** 因此配体大小变化体现为 **token 数变化**，而不是同一 token 内 atom slot 变化；配体的元素、电荷、键级、芳香性等从 CCD 读取。【核实：`features.py` L334；Nature 正文】
3. **原子槽位上限 24。** `max_atoms_per_token=24`；残基原子数超过 24 时截断并告警，少于 24 时补空位；无效槽位由 atom mask 屏蔽。【代码／配置核实：`pipeline.py` L71、L118、L129；`diffusion_head.py` L198、L259-260】
4. **坐标张量形状。** diffusion head 在 `(n_token, max_atoms_per_token, 3)` 上工作；局部原子注意力在 token 内，全局注意力在 token 间。【代码／配置核实：`diffusion_head.py` L202-252；专题 01 §2】

**为什么这样设计能扩大化学覆盖**

- 残基级 frame 表示要求“每个残基有固定的 N–Cα–C 骨架和侧链扭转角”，这对标准 20 种氨基酸很自然，但对配体、金属、非天然残基、共价修饰、糖基化非常别扭；AF3 直接把“任意原子”变成 token，**不需要为每种化学定义一套 frame/扭转角**。【核实：Nature L61、L71】
- 配体每原子一个 token，使得原子类型、键、电荷可以逐原子输入；金属离子就是单原子 token；修饰残基通过 CCD 进入聚合物 token。这确实扩大了化学覆盖面。【核实 + 推断】
- 代价是：模型不再有显式 SE(3) 等变结构模块，只能靠**随机旋转/平移增强 + 坐标回归**学几何；每个采样步都做一次随机增强。【代码／配置核实：`diffusion_head.py` L286-289；Nature L74】

**为什么“扩大化学覆盖”不必然“提高骨架精度”**

- 骨架精度主要由 **trunk 的 pair 表示 + MSA/模板信息 + 训练数据与蒸馏** 决定；diffusion head 只负责在给定 trunk 条件下把坐标去噪出来。
- 论文没有给出“atom-level vs residue-level”的同架构消融；AF3 相对 AF2 的精度提升，不能归因于 token 化本身。论文甚至引用了 AF2 的观察：**“removing most of the complexity of the structure module had only a modest effect on the prediction accuracy”**（这是 AF2 的观察，不是 AF3 的消融）。【核实：Nature L71】
- 因此，正确的读法是：**原子级表征是“化学接口”的升级，不是“骨架精度”的开关**；它让 AF3 能覆盖配体/离子/修饰残基，但骨架精度仍要靠 trunk、MSA 和训练数据。【推断，依据上述引文与缺失消融】

**与 RFAA 的对照**

| 维度 | AF3 | RFAA |
| ---- | ---- | ---- |
| 聚合物 | 残基 token，原子打包进 token | 残基 + Cα frame |
| 配体/金属 | 每原子 token，CCD 元素/键 | 原子级，46 种元素 token、键级 token |
| 手性 | 无显式约束；参考构象 + 排序惩罚；4.4% 违规 | 显式 R/S token |
| 等变 | 无显式等变；随机旋转增强 | SE3Transformer |
| 侧链 | 显式原子坐标 | 显式重原子 |

【核实：RFAA 论文与专题 01 §2；AF3 见 Nature L140、L74】

### 3.2 Trunk：Pairformer 与 MSA 模块的演进

**结构**

- **输入嵌入**：聚合物残基 token、配体原子 token、原子类型/电荷/键特征、参考位置 `ref_pos`、MSA、模板。
- **MSA 模块（4 block）**：保留 Evoformer 风格的 MSA 表示与 outer product mean，但只用来把 MSA 信息压进 pair 表示；MSA 表示不向后传。【核实：Nature L69；`evoformer.py` L201-222】
- **Pairformer（48 block）**：每个 block 包含 triangle multiplication（outgoing/incoming）、triangle attention、pair transition、single attention、single transition；pair 128 通道、single 384 通道；每个 block 独立参数。【核实：Nature Fig. 2a；`modules.py` L396-448】
- **模板**：模板嵌入直接加到 pair 表示；最多 20 个命中，取前 4 个；**不从同一个 PDB 文件里选模板**。【核实：SI L145-151；`evoformer.py` L160-186】

**与 AF2 的演进关系**

- AF3 把 AF2 Evoformer 的 **MSA stack 从 48 层压到 4 层**，并用 pair-weighted averaging 替代昂贵的 MSA attention；MSA 信息只通过 outer product mean 进入 pair，之后全部在 pair 上处理。
- **pair 处理本身与 AF2 基本一致**，48 个 block 没有大改；论文原话是 “The pair processing and the number of blocks (48) is largely unchanged from AF2”。【核实：Nature L69】
- 这意味着 AF3 的 trunk 创新主要是**“MSA 压缩 + pair 通道统一”**，而不是把 pair 表示重新发明；真正的结构生成被下放到 atom-level diffusion head。【推断，依据 L69、L71】

### 3.3 Diffusion module：24 个 token block、EDM、x0

**模块组成**

- **24 个 token 级 block**（代码默认值；论文正文/图注没有标注这个数字）。【代码／配置核实：`diffusion_transformer.py` L167-169；论文该数字未核实】
- 每个 token block 内含 **3-block atom transformer**（4 heads，key/value dim 128）；先做 token 内局部原子注意力，再做 token 间全局注意力，最后原子解码。【核实：Nature Fig. 2a 图注；`atom_cross_attention.py` L25-33；`diffusion_transformer.py` L234-236】
- token 级注意力 16 heads；token 表示 768 通道、原子表示 128 通道。【代码／配置核实：`diffusion_transformer.py` L111；`atom_cross_attention.py` L25-33】
- 条件化：single（384）+ target features、pair（128）+ 相对位置编码，各自过 transition 后注入；**不使用 frame、不使用侧链扭转角**。【核实：Nature L61、L71；`diffusion_head.py` L144-177】

**EDM 与 x0 参数化**

AF3 用 EDM 的噪声调度与预条件：

```text
σ(t) = (σ_max^(1/ρ) + t · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ
x0_hat = c_skip(σ) · x_σ + c_out(σ) · F_θ(x_σ, σ)
c_skip(σ) = σ_data² / (σ² + σ_data²)
c_out(σ)  = σ · σ_data / sqrt(σ² + σ_data²)
```

- 常数：`σ_data=16.0`（代码注释称“对 multimer 训练集平均测得”）、`σ_min=4e-4`、`σ_max=160`、`ρ=7`。【代码／配置核实：`diffusion_head.py` L34、L73-77】
- 网络输出 `position_update`，再与 noisy 坐标做 skip/out 组合，等价于预测 x0；不是预测噪声。【代码／配置核实：`diffusion_head.py` L258-260】
- 推理默认 **200 步**、每个 seed **5 个 sample**；论文评测用 **5 seeds × 5 samples = 25 个样本**。【代码／配置核实：`diffusion_head.py` L101-104；Nature L175】
- 每个采样步都做随机旋转/平移增强，替代显式等变。【代码／配置核实：`diffusion_head.py` L286-289】

**为什么选 x0 而不是噪声**

- 坐标尺度固定，x0 回归可以直接接 RMSD/FAPE 类损失；EDM 预条件把不同噪声尺度的梯度归一化，训练更稳。这是 EDM 的标准做法，不是 AF3 独创。【推断：EDM 标准式；代码实现核实】

**采样成本的含义**

- 默认 25 个样本已经是 200 步 × 25 次前向；抗体任务论文说排序在 **1,000 个 model seeds** 上仍在提升，Fig. 5a 最多到 1,200 seeds。这意味着抗体场景的采样预算比默认高约 40 倍。【核实：Nature L59、L144、L155】

### 3.4 置信头：pLDDT / PAE / PDE / pTM / ipTM 各估什么

- **结构**：4 个 PairFormer block 的 mini-trunk；pLDDT 每原子 50 bin；PAE 64 bin；PDE 64 bin（`max_error_bin=31 Å`）；pTM/ipTM 由 PAE 的 softmax 经 TM-score 权重积分得到。【代码／配置核实：`confidence_head.py` L34-46、L39-43、L200-213、L245-287】
- **训练方式**：用 diffusion 的 mini-rollout 生成带噪结构，再让置信头预测误差；主干梯度被停止（Fig. 2c 的 stop sign）。正文没有用 “trunk frozen” 这个词，但图注的 stop sign 表达同一件事。【核实：Nature Fig. 2c；`confidence_head.py`】

| 头 | 估计对象 | 主要用途 | 盲区 |
| ---- | ---- | ---- | ---- |
| pLDDT | 每个原子的局部环境置信度 | 判断侧链/局部几何是否可信 | 对 domain 朝向、链间界面不敏感 |
| PAE | 残基对 (i,j) 的对齐误差 | 判断 domain/链的相对朝向、复合物整体排布 | 对局部侧链精度不直接 |
| PDE | 原子对距离误差 | 判断界面接触、配体-蛋白距离 | 仍是模型自评，不是实验误差 |
| pTM / ipTM | 全局/界面 TM-score 风格置信度 | 排序、筛 top-1 | 与物理有效性（手性、clash）无关 |

- **排序策略**：全局排序混用 pTM/ipTM，并加入减少 clash、增加 disorder 的项；配体排序用 ipTM 聚合，若配体有手性错误或 clash，则分数再除以 100。【核实：Nature L177、L217】
- **关键提醒**：置信头是“模型自洽度”，不是“实验正确率”。AF3 自己报告 4.4% 手性违规，说明高 pLDDT/ipTM 不等于物理正确。【核实：Nature L140】

### 3.5 条件接口

| 条件 | AF3 的处理 | 证据/边界 |
| ---- | ---- | ---- |
| MSA | 必需输入；MSA 经 4-block 模块压进 pair；最多 16,384 行 | 【核实：Nature L122；SI L143】 |
| 模板 | 可选；PDB mmCIF 模板，最多 20 命中、取前 4；不从同一 PDB 文件选模板 | 【核实：SI L145-151】 |
| 配体/离子/修饰残基 | CCD 驱动；配体每原子 token；离子单原子 token；修饰残基进聚合物 token | 【核实：SI L110；`features.py` L334】 |
| 共价键 | CCD 的 covalent bond 信息进入 pair 的 polymer–ligand / ligand–ligand 接触矩阵 | 【核实：`evoformer.py` L116-159】 |
| 任意用户自定义共价键 | 正文只承诺 CCD 内的 modifications；非 CCD 共价键是否支持未核实 | 【未核实/推断】 |
| 二硫键 | 没有显式二硫键约束接口；是否从几何中学到未核实 | 【推断】 |
| 手性 | 无显式约束；参考构象 `ref_pos` 输入正确手性；只在排序中惩罚 | 【核实：Nature L140】 |
| 口袋条件 | 团队底稿称需要单独微调一个 pocket-conditioned 模型；正文未逐项核实 | 【未核实（团队底稿转述）】 |

**MSA 的必要性**

- 论文明确称：**AF3 对 MSA 深度的依赖与 AlphaFold-Multimer v2.3 非常相似**。【核实：Nature L122】
- 讨论部分有一句“跨实体进化信息的缺失不是实质障碍”（L159），但这不能读成“AF3 不需要 MSA”；正文同时给出了 MSA 深度依赖的定量表述。正确口径是：**MSA 仍是默认精度路径，低 MSA 场景会退化，论文没有给出完整的“无 MSA”消融表**。【核实 + 推断】
- 对 VNAR：鲨鱼源序列在 UniRef/PDB 中稀少，MSA 深度大概率低；这会直接触发上述退化。详见 §6.3。【推断，依据数据库构成】

---

## 4. 训练数据构成与切分

> 这是本仓库当前最大的空白。下面把“已核实”“论文未披露”“团队底稿转述”分开写，避免把“PDB 全量”当成一个已核实的规模数字。

### 4.1 数据源与规模

- **主训练集**：PDB，**截止 2021-09-30**。【核实：Nature L171；专题 02 §2.1】
- **训练条目数**：论文正文与可读 SI 没有给出总结构/复合物数；团队底稿写的“PDB 全量”是定性描述，不是规模数字。【未核实】
- **数据混合权重（可读部分）**：Weighted PDB 0.5；disordered PDB distillation 0.02；protein monomer distillation 0.495。【核实：SI L157-159（表被截断，其余权重未读到）】
- **蒸馏**：加入 AF2-Multimer v2.3 的预测结构；论文称长无序区在预测结构里常被拉成长延伸环，训练会“教会”AF3 模仿这种行为，**cross-distillation 显著减少了幻觉**。【核实：Nature L76-82】
- **裁剪与 batch**：crop 384 → 640 → 768 tokens 三阶段；batch 256；256×48=12,288 个 diffusion 样本（微调阶段 8,192）。【核实：Nature L171】
- **MSA 数据库**：蛋白用 UniRef90、UniProt、Uniclust30+BFD、Reduced BFD、MGnify；RNA 用 Rfam、RNACentral、Nucleotide collection；MSA 最多 16,384 行。【核实：SI L122-143】
- **模板**：PDB mmCIF，最多 20 命中、取前 4；不从同一 PDB 文件选模板。【核实：SI L145-151】

### 4.2 清洗与去重

| 项目 | AF3 口径 | 状态 |
| ---- | ---- | ---- |
| 分辨率过滤 | 论文 SI 有 “Structure filtering” 小节，但可读版本被截断；**没有拿到具体阈值** | 【未核实】 |
| 链长/缺失原子/altloc | 未在可读正文/SI 中逐项披露；AF3 用 atom mask 与 CCD 处理缺失/非标准原子 | 【未核实 + 代码部分核实】 |
| 配体过滤 | 未披露 | 【未核实】 |
| 训练集聚类阈值 | 未披露 | 【未核实】 |
| 测试集同源排除 | **>40% 序列同一性剔除** | 【核实：Nature L193】 |
| 测试集短肽 | 测试集剔除 <16 残基的肽段（论文测试集构造） | 【核实：Nature L193 附近；具体上下文见 §5】 |

**证据纪律提醒**：不要把 Boltz-1 的 “9 Å / 4.0 Å” 口径套到 AF3。AF3 的分辨率阈值没有在可读材料中出现；任何写“AF3 也用 9 Å”的说法都应标为未核实。

### 4.3 切分与泄漏控制

**切点的含义**

- **2021-09-30 是训练/评测的时间边界**：训练只用该日期前释放的 PDB 结构；主评测用该日期后的结构，或单独重训更早切点的模型。AF3、Boltz-1、Protenix v1 都用这个切点，所以三者有“同切点对比”的基础。【核实：Nature L171；专题 02 §2.1】
- **PoseBusters 的特殊处理**：AF3 为 PoseBusters 单独训练了一个 **2019-09-30 切点**的模型，而不是用主模型；PoseBusters 测试集是 428 个 2021 年后释放的结构。【核实：Nature L96、L215】
- **Recent PDB**：8,856 个复合物，发布日期 **2022-05-01 至 2023-01-23**，<5,120 tokens；另有 40% 同源剔除的低同源子集。【核实：Nature L189、L193】
- **CASP15**：AF3 论文评估的是 **10 个公开 RNA 靶标**，不是蛋白/抗体 CASP15 全集。【核实：Nature L211】
- **CASP16**：晚于 AF3 发布，不在 AF3 训练集内；AF3 论文没有报告 CASP16。任何“AF3 在 CASP16 上如何”的说法，要么是第三方复现，要么是未核实。【核实：时间线 + 推断】

**仍然存在的泄漏渠道**

1. **模板日期未显式控制**：AF3 用 PDB mmCIF 模板，模板本身可能来自训练切点之后释放的同源结构；论文只限制了“不从同一 PDB 文件选模板”，没有说模板发布日期必须早于切点。对 benchmark 而言，这是一个隐性泄漏通道。【推断，依据 SI L145-151】
2. **40% 同源阈值仍允许 30–40% 相似度的结构**：对近距离同源物，40% 阈值可能不够严格；抗体/纳米抗体尤其如此。【推断】
3. **同一 PDB 条目的多版本/多构象**：同一实验多次 deposition，按结构切分会把同一克隆分到训练和测试；AF3 论文没有逐条说明如何处理。【推断，依据 PDB 常见情况】
4. **MSA 数据库的序列时间**：UniRef/BFD/MGnify 的构建时间与 PDB 切点不完全同步；如果测试蛋白的序列已进入 MSA 库，等于序列层面的信息提前注入。论文没有给出 MSA 库的时间戳控制。【推断/未核实】
5. **PoseBusters 用重训模型**：论文的 PoseBusters 数字来自 2019-09-30 重训模型；开源权重是主模型（2021-09-30 切点）。用开源权重复现 PoseBusters，可能得到不同数字。【核实 + 推断】

**对 VNAR 的直接后果**

- SAbDab2 的 84 个 VNAR 结构里，很可能有相当一部分早于 2021-09-30；如果直接拿这 84 个做 AF3 零样本评测，训练/测试重叠风险很高。必须按 PDB release date 过滤，或使用 2021-09-30 之后新释放的 VNAR 结构。【推断；发布日期未逐条核对】
- 84 个结构也不足以做可靠的留出评估；任何 VNAR 基准都应该按抗原相似度/克隆去重后再切分。【核实数字；切分建议为推断】

### 4.4 覆盖与偏置

| 维度 | 情况 | 状态 |
| ---- | ---- | ---- |
| 配体/离子/修饰残基 | CCD 覆盖；离子单原子 token；修饰残基进聚合物 token | 【核实】 |
| 金属/罕见元素 | 覆盖取决于 PDB/CCD 出现频率；没有按元素分层的误差统计 | 【未核实】 |
| 抗体/纳米抗体 | AF3 不是抗体专用模型；抗体-抗原精度低于其他复合物；抗体任务需约 1,000 seeds | 【核实：Nature L59、L155】 |
| VNAR | PDB/SAbDab2 中仅 84 个高分辨率 VNAR 结构；框架、CDR3 长度、非经典 Cys 与 VH/VHH 不同 | 【核实数字；架构先验差异为推断】 |
| 构象多样性 | AF3 输出单一静态构象；无序区通过蒸馏与置信头处理，不是 ensemble | 【核实蒸馏；ensemble 缺失为推断】 |
| 物种/框架偏置 | PDB 本身偏向少数物种、少数抗体胚系；AF3 没有做抗体域重平衡 | 【推断】 |

### 4.5 可获取性与许可

- **代码**：官方仓库 `google-deepmind/alphafold3`，Apache-2.0。【核实：仓库 license】
- **权重**：受 `WEIGHTS_TERMS_OF_USE` 约束，不是无条件开放；商用/再分发需读条款。【核实：仓库文档】
- **训练数据**：PDB 本身公开，但 AF3 的完整清洗管线、聚类文件、蒸馏结构没有全部开放；MSA 数据库（BFD/MGnify 等）体量巨大，重建成本高。【推断，依据仓库内容】
- **可复现边界**：推理可复现（Docker + 权重 + MSA 管线）；训练不可完全复现（数据管线与算力未披露）。【推断】

---

## 5. 宣称与可验证性

### 5.1 宣称—证据—核查状态

| 宣称 | 证据类型 | 核查状态 | 备注 |
| ---- | ---- | ---- | ---- |
| PoseBusters 成功率约 76% | 图 1c + 二手转述 | **待核实（精确值）** | 正文只给统计显著性，未给百分比；76% 来自二手来源与团队底稿；原文精确值需读图 1c。【未核实】 |
| 手性违规率 4.4% | 正文明确数字 | **已核实** | “we still observe a chirality violation rate of 4.4% in the benchmark”（Nature L140）。 |
| 蛋白-蛋白/抗体 DockQ 优于 AF2 | 正文 P 值 + 图 | **部分核实** | 正文给 P=1.8×10⁻¹⁸、6.5×10⁻⁵；没有给 DockQ 成功率百分比；百分比在图 1c，未逐项读。【核实 P 值；百分比未核实】 |
| MSA 依赖与 AF2-Multimer v2.3 相似 | 正文 | **已核实** | Nature L122。 |
| 默认 25 个样本（5 seeds × 5 samples） | 正文 + 代码 | **已核实** | Nature L175；`diffusion_head.py` L101-104。 |
| 抗体任务约 1,000 seeds | 正文 + 图 | **已核实** | Nature L59、L155；Fig. 5a 最多 1,200 seeds。 |
| 训练切点 2021-09-30 | 正文 | **已核实** | Nature L171。 |
| PoseBusters 用 2019-09-30 重训模型 | 正文 | **已核实** | Nature L215。 |
| 训练结构总数 | 未披露 | **未核实** | 论文/可读 SI 没有给总条目数；“PDB 全量”不是规模数字。 |
| 训练分辨率过滤阈值 | SI 5.1 | **未核实** | SI 可读部分被截断；不要假设 9 Å。 |
| 任意共价键支持 | 正文/代码 | **存疑** | 只核实到 CCD 内的 covalent bond 信息；非 CCD 自定义共价键未核实。 |
| 显式手性约束 | 正文/代码 | **已核实（无）** | 参考构象输入正确手性，但无显式约束；4.4% 违规 + 排序惩罚。 |
| “原子级表征带来精度提升” | 无消融 | **存疑/常见误读** | 论文没有 atom-level vs trunk 的定量消融；见 §5.3。 |
| 训练算力/参数总量 | 未披露 | **未核实** | 正文/可读 SI/代码文档均无；SI 只有推理时间表（16×A100）。 |
| CASP15 10 个 RNA 靶 | 正文 | **已核实** | Nature L211；注意不是蛋白/抗体 CASP15 全集。 |

- **可复现所需材料**：权重（受 `WEIGHTS_TERMS_OF_USE` 约束）、Docker 推理镜像与 MSA 管线；训练数据管线与算力未公开，训练不可完全复现（详见 §4.5）。【核实：仓库文档；训练不可复现为推断】
- **与官方文档／论文的一致性**：代码形态与论文 Fig. 2a 一致（48-block Pairformer、4-block MSA、atom cross-attention）；代码默认 24 个 token block 未见于论文正文（见 §3.3、§7 Q1）。【代码／配置核实】

### 5.2 评测协议与成本

**AF3 自己用的 benchmark**

| Benchmark | 规模/口径 | 指标 | 采样预算 |
| ---- | ---- | ---- | ---- |
| PoseBusters | 428 个 2021 年后结构；2019-09-30 重训模型 | 配体 RMSD（对称校正）、物理有效性（键长/夹角/clash/手性） | 25 样本（5×5）【核实：L175、L215】 |
| Recent PDB | 8,856 个复合物，2022-05-01–2023-01-23，<5,120 tokens；低同源 40% 子集 | LDDT、DockQ、RMSD、界面指标 | 25 样本【核实：L189、L193】 |
| CASP15 RNA | 10 个公开 RNA 靶 | RNA 结构指标 | 25 样本【核实：L211】 |
| 蛋白-蛋白/抗体 | 论文子集；抗体到 1,000 seeds | DockQ（acceptable 0.23）、界面 LDDT | 抗体约 1,000 seeds【核实：L59、L155】 |

**成本**

- 推理：每个靶 200 步 × 25 样本；抗体到 1,000 seeds 时是默认的约 40 倍采样。SI 有“16×A100 推理时间”表，但具体数值未读到。【核实：200 步/25 样本；推理时间表未读】
- 训练：论文未给 TPU/GPU 时长与总样本数；只能确认 crop/batch/样本数（12,288 diffusion 样本）。【未核实】
- 参数总量：未披露。【未核实】

**与 RFAA / Boltz-1 / Chai-1 对比为什么必须重新校准**

1. **切点**：AF3 2021-09-30；Boltz-1 同切点；Boltz-2 2023-06-01；Chai-1 未披露；RFAA 未明示统一时间切点。不同切点意味着测试结构可能已进训练集。【核实：专题 02 §2.1】
2. **采样预算**：AF3 25 样本；Chai-1 默认 5 样本；RFAA 代码示例 T=100/200；Boltz 200 步 × 5 验证样本。样本数不同，top-1 精度不可直接比。【核实：专题 01 §2.4】
3. **hit rate 分母**：AF3 是结构预测，分母是 benchmark 复合物数（如 PoseBusters 428）；RFAA/BoltzGen/AnewOmni 是设计/湿实验，分母是“每靶候选数 × 靶数”，命中定义、表达系统、筛选通量都不同。75% 与 63.5% 不能横比。【推断，依据专题 03 §2.3 与 §6】
4. **指标实现**：PoseBusters 的 RMSD 是否对称校正、DockQ 版本、LDDT 对齐方式、界面残基定义，都会改变数字；必须统一实现。【推断】
5. **模型版本**：AF3 的 PoseBusters 用 2019 重训模型，开源权重是 2021 切点主模型；直接拿开源权重跑 PoseBusters，可能复现不出论文数字。【核实 + 推断】
6. **MSA 管线**：各模型的 MSA 生成/配对策略不同；比较时应统一 MSA 或统一单序列模式，否则测的是数据管线不是模型。【推断；与专题 03 §7 V1 同结论】

**最小重校准协议（建议）**

- 测试集：只取 2023-01-13 之后释放的复合物（晚于 AF3/Boltz-1 切点，早于 Boltz-2 切点则对 Boltz-2 不公平，需分层报告）。
- 输入：同一份 MSA（或统一单序列模式）、同一模板策略、同一配体质子化/CCD 版本。
- 预算：统一 25 样本 × 200 步；另跑 5 样本与 100 样本做预算曲线。
- 指标：配体 RMSD（对称校正）、LDDT、DockQ、手性违规率、clash 数、键长/夹角违规率。
- 报告：分子集（蛋白-蛋白、蛋白-配体、蛋白-核酸、抗体-抗原）分别报，不混总分。

### 5.3 “原子级表征带来精度提升”的误读

**实际发生的事**

- AF3 的表征变化解决的是**化学覆盖面**：配体、离子、修饰残基、非标准原子都能进入同一个 token 空间；不需要为每种化学写 frame/扭转角。【核实：Nature L61、L71；`features.py` L334】
- 骨架精度的提升没有同架构消融支撑。论文没有报告“把原子 token 换回残基 frame、其他不变”的对照；也没有报告“去掉 atom cross-attention、只留 token 注意力”的对照。【核实：论文无该消融】
- 论文确实引用了 AF2 的观察：简化 structure module 对精度只有 modest effect；这句话不能反过来证明“原子级表征是精度来源”。【核实：Nature L71】
- 因此，以下说法都过强：
  - “AF3 因为原子级 token 所以骨架更准” → **存疑**；
  - “AF3 的 4.4% 手性违规说明原子级表征解决了手性” → **错误**；论文明确说没有显式约束，违规仍存在。【核实：L140】
  - “AF3 的全原子 = 所有原子都高精度” → **错误**；配体/侧链/长 CDR3 的局部精度仍取决于训练覆盖与置信度。【推断】

**可以安全主张的版本**

> AF3 的原子级 token 化扩大了可建模的化学空间，并用 atom cross-attention 在 token 内做局部原子注意力；它去掉了 AF2 的 frame/扭转角，使结构生成更统一。骨架精度的提升主要来自 trunk（Pairformer + MSA）、训练数据（PDB + AF2-Multimer 蒸馏）与训练流程；论文没有把增益单独归因于原子级表征。【推断】

---

## 6. 实践与应用

### 6.1 适用场景与前提

- **适合**：给定序列/配体/离子/修饰残基，预测复合物结构；给实验结构做独立复核；给设计模型做 self-consistency 评估（AF3 重折叠）；给 Boltz/Chai 做交叉对照。
- **不适合**：从零生成序列/新 binder；需要多构象 ensemble 的柔性体系；需要显式二硫键/手性约束的化学；没有 MSA 且没有模板的孤儿体系（精度会退化）。
- **前提**：MSA 管线（jackhmmer/MMseqs 等）、CCD 版本、模板库、GPU/TPU；权重条款允许的使用场景。

### 6.2 关键参数与调优经验

| 参数 | 建议 | 依据/性质 |
| ---- | ---- | ---- |
| diffusion steps | 200（默认） | 【代码／配置核实】 |
| samples per seed | 5 | 【代码／配置核实】 |
| seeds | 普通复合物 5；抗体/纳米抗体 100–1,000 | 【核实：L59、L155】 |
| 排序 | pTM/ipTM + clash/disorder 惩罚；配体看 ipTM 与手性/clash | 【核实：L177、L217】 |
| 模板 | 有同源模板时开启；注意模板日期与同源性 | 【核实模板机制；日期控制为推断】 |
| MSA | 尽量深；低 MSA 时预期退化 | 【核实：L122】 |
| 置信阈值 | pLDDT>70/80、PAE<5–10 Å、ipTM>0.7/0.8 是常用经验阈值，不是论文标准 | 【经验/推断】 |
| 物理复核 | 单独跑 PoseBusters/手性/clash/二硫键几何，不只看置信度 | 【推断；依据 4.4% 手性违规】 |

### 6.3 VNAR 场景：AF3 能直接做什么、不能做什么

**能直接做**

1. **VNAR-抗原复合物结构预测**：VNAR 是单域，不需要轻链；把 VNAR 序列 + 抗原序列作为两条聚合物链输入即可。AF3 不要求 VH/VL 配对。【核实：AF3 输入是聚合物链；单域无特殊限制】
2. **给设计做独立评估**：用 AF3 重折叠 VNAR 设计，看 ipTM/PAE/界面 LDDT；但必须用独立模型交叉复核，避免 self-consistency 偏差。【推断】
3. **有同源模板时提精度**：SAbDab2 的 84 个 VNAR 结构可作为模板来源；模板能部分弥补 MSA 不足。【推断】
4. **作为对比坐标系**：AF3 的 25 样本默认预算、PoseBusters/Recent PDB 口径，可以作为 VNAR 基准的“参照列”。【推断】

**不能直接做 / 明显短板**

1. **不能生成 VNAR 序列或 de novo 设计**：AF3 是结构预测，不是序列/结构生成器；它不能替代 IgGM、AntiBMPNN、BoltzGen、AnewOmni 的设计功能。【核实：模型类型】
2. **长 CDR3 与构象多样性**：VNAR 的 CDR3 通常更长、更灵活；AF3 输出单一静态构象，长环常出现低 pLDDT/高 PAE，且没有 ensemble。不能把 top-1 当作唯一构象。【推断，依据 AF3 单构象输出】
3. **MSA 稀缺**：VNAR 序列在 UniRef/PDB 中稀少；SAbDab2 仅 84 个结构，PLAbDab-nano 也只有 4,913 条 VHH+VNAR 序列（未拆分 VNAR）；OAS 主要是人源，鲨鱼源覆盖低。AF3 对 MSA 深度依赖与 AF2-Multimer 相似，因此 VNAR 的 MSA 深度低会直接导致退化。【核实数字；退化机制为核实+推断】
4. **非经典半胱氨酸/二硫键**：VNAR 常有非经典 Cys，二硫键配对多样；AF3 没有显式二硫键约束接口，CCD 只覆盖标准修饰；不能保证配对正确。【推断；CCD 共价信息为核实】
5. **抗体-抗原精度天然低于其他复合物**：论文自己显示抗体-抗原是难子集，且需要 1,000 seeds 才饱和；VNAR 只有单域，界面更小，难度不会更低。【核实：L59、L155；VNAR 更难为推断】
6. **框架先验不匹配**：AF3 的训练分布以常规蛋白为主，没有 VNAR 专用 tokenizer/环长先验；直接套用会有系统性偏差。【推断】

**工程建议**

- 把 AF3 当作 **VNAR 的评估器/参照系**，不是设计引擎；设计端用 IgGM/BoltzGen/AnewOmni，评估端用 AF3 + Boltz-2/Chai-1 交叉。
- 做 VNAR 基准时，按 SAbDab2 的 `ab-ag-split` 再加“单域 + 鲨鱼物种”筛选，并按 PDB release date 过滤到 2021-09-30 之后。【核实协议存在；组合为推断】
- 若要微调 AF3/Boltz 骨干做 VNAR，优先补：非经典 Cys/二硫键约束、长 CDR3 的构象多样性、VNAR 框架的 MSA 增强；不要只做全量微调。【推断】

---

## 7. 疑问与讨论

**Q1：论文说 diffusion module 有 24 个还是 30 个 block？**
- **A**：代码默认 **24 个 token 级 block**（`diffusion_transformer.py` L167-169）；论文正文与 Fig. 2 图注没有标注这个数字。写笔记时以代码为准，并注明“论文未给数字”。不要凭记忆写 30。【代码／配置核实；论文数字未核实】

**Q2：原子级 token 是不是 AF3 比 AF2 准的原因？**
- **A**：不是可证实的因果。论文没有 atom-level vs residue-level 的同架构消融；AF3 的提升还叠加了 MSA 压缩、训练数据、AF2-Multimer 蒸馏与训练流程。原子级 token 的主要可核实贡献是**化学覆盖**与**去掉 frame/扭转角**。【推断，依据 §5.3】

**Q3：AF3 没有 MSA 还能用吗？**
- **A**：论文说 AF3 对 MSA 深度的依赖与 AF2-Multimer v2.3 相似；讨论里“跨实体进化信息缺失不是实质障碍”不能被读成“完全去 MSA”。没有完整的无 MSA 消融表。工程上应按“低 MSA → 精度退化”预期。【核实：L122、L159；消融缺失为核实】

**Q4：手性违规 4.4% 怎么理解？**
- **A**：AF3 没有显式手性约束；输入参考构象有正确手性，但模型仍会在 4.4% 的 benchmark 结构里产生手性错误；排序时会对配体手性错误做惩罚（分数除以 100）。这说明“全原子”不等于“手性正确”。【核实：L140、L217】

**Q5：PoseBusters 76% 能直接用开源权重复现吗？**
- **A**：不一定。论文的 PoseBusters 数字来自 2019-09-30 切点重训模型；开源权重是 2021-09-30 主模型。复现时要固定模型版本、样本数、对称校正与物理检查实现。【核实：L215；复现差异为推断】

**Q6：和 Boltz-1 / Chai-1 比，AF3 到底强多少？**
- **A**：不能只看论文的“接近/超过”。三者切点、采样预算、MSA 管线、指标实现都不同；Boltz-1 还有论文 9 Å vs 代码 4.0 Å 的口径冲突。必须在同一测试集、同一 MSA、同一 25 样本预算下重跑。【核实：专题 02 §2.1；重校准协议见 §5.2】

**Q7：VNAR 只有 84 个结构，AF3 能直接用于 VNAR 吗？**
- **A**：能做“预测/评估”，不能做“设计”。VNAR 单域不需要轻链，这是 AF3 的便利；但 MSA 稀缺、长 CDR3、非经典 Cys、抗体-抗原难子集、1,000 seeds 成本，决定了它只能作为参照/评估器，不能作为 VNAR 生成器。【核实数字；结论为推断】

**Q8：AF3 的“全原子”与 RFAA 的“全原子”是同一个意思吗？**
- **A**：不是。RFAA 是残基 frame + 非聚合物原子，显式 46 元素/键级/R-S；AF3 是残基 token + 配体原子 token，无显式手性、无 frame。二者都覆盖全原子，但表示哲学不同：RFAA 保留几何归纳偏置，AF3 用数据增强替代等变。【核实：专题 01 §2；AF3 L74、L140】

---

## 8. 学习心得

- **AF3 最值得复用的不是“24 个 diffusion block”，而是三件事**：① 残基 token + 配体原子 token 的混合表征；② token 内 atom cross-attention + token 间全局注意力的两段式；③ 置信头（pLDDT/PAE/PDE + pTM/ipTM）与排序惩罚。这三件事可以直接搬到 VNAR 的评估/微调管线。
- **AF3 最容易被误读的是“原子级 = 更准”**。真正的信息是：原子级 token 把化学接口打开了，但骨架精度仍由 trunk、MSA、数据和蒸馏决定；论文没有做能证明“表征换精度”的消融。
- **数据透明度是 AF3 的最大短板**。训练条目数、分辨率过滤、训练聚类阈值都没有在可读材料中出现；复现推理容易，复现训练几乎不可能。写 benchmark 时不要把“PDB 全量”当成已核实规模。
- **对 VNAR，AF3 是坐标系，不是发动机**。它能告诉你“这个设计在 AF3 眼里像不像真的”，但不能替你生成 VNAR；VNAR 的 84 个结构、长 CDR3、非经典 Cys 与 MSA 稀缺，决定了必须做域适配/约束，而不是直接套用。
- **下一步最值得做的实验**：按 [专题 03 §7](../../专题/03_评测体系与可复现性.md) 的 V1/V4/V9，用同一 MSA、同一 25 样本预算，在 2023 年后结构上重跑 AF3/Boltz-1/Chai-1，并单独报手性/clash/二硫键几何。这比继续读论文更能判断“AF3 系到底谁可用”。

---

## 9. 参考资料

### 论文与官方文档

- Abramson, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* **630**, 493–500 (2024). DOI: [10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)；正文: https://www.nature.com/articles/s41586-024-07487-w
- 补充材料（可读部分）: https://pmc.ncbi.nlm.nih.gov/articles/instance/11168924/bin/41586_2024_7487_MOESM1_ESM.pdf
- 官方仓库: https://github.com/google-deepmind/alphafold3
- 权重条款: 仓库内 `WEIGHTS_TERMS_OF_USE`；代码 Apache-2.0

### 代码/配置核实点（仓库路径 + 行）

> 行号来自 `main` 分支 raw 文件；Nature 行号是页面 markdown 提取行，不是 PDF 页码。

- `src/alphafold3/model/pipeline/pipeline.py` L71、L118、L129、L208：`max_atoms_per_token=24`、atom slot 注释、传入 features。
- `src/alphafold3/model/pipeline/features.py` L147、L310-334：token 化；配体每原子 1 token；超上限截断/告警。
- `src/alphafold3/model/network/evoformer.py` L116-159、L160-186、L201-222、L279-298：CCD 共价接触矩阵、模板嵌入、4-block MSA、48-block Pairformer。
- `src/alphafold3/model/network/modules.py` L396-448：Pairformer block 组成。
- `src/alphafold3/model/network/diffusion_transformer.py` L111、L167-169、L234-236：16 heads、24 token blocks、3-block atom transformer。
- `src/alphafold3/model/network/atom_cross_attention.py` L25-33：atom transformer 4 heads、kv dim 128。
- `src/alphafold3/model/network/diffusion_head.py` L34、L73-77、L101-104、L144-177、L198、L258-260、L286-289：σ 常数、200 步/5 样本、条件化、atom mask、x0 预条件、随机增强。
- `src/alphafold3/model/network/confidence_head.py` L34-46、L39-43、L200-227、L245-287：4-block mini-trunk、50/64 bins、pTM/ipTM。

### 团队内部底稿（已整合进本仓库）

> 正文中“团队底稿 §x”即指下列专题汇总之源文档；仓库内不再保留本地绝对路径。

- [专题 01 全原子表示与生成范式](../../专题/01_全原子表示与生成范式.md)：AF3 表征、生成范式与覆盖矩阵。
- [专题 02 训练数据构成与切分](../../专题/02_训练数据构成与切分.md)：PDB 切点、去重协议与泄露渠道。
- [专题 03 评测体系与可复现性](../../专题/03_评测体系与可复现性.md)：PoseBusters/Recent PDB 口径与 V1/V4/V9 复现设计。
- [专题 04 抗体与VNAR可迁移点](../../专题/04_抗体与VNAR可迁移点.md)：VNAR 数据规模与可迁移点。

### 二手转述（不作为一手证据）

- PoseBusters “AF3 76% / Chai-1 77%” 的二手来源（PMC12424684 等）；本笔记将其标为未核实，不作为结论。
