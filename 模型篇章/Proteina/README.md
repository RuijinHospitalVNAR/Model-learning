---
model: Proteina（Proteína）
aliases: [Proteina, Proteína]
family: 序列/结构设计（主链无条件生成；扩展到 motif scaffolding 与长链）
granularity: 主链显式（仅 Cα，每残基 1 个三维点，即 3L 维；无 N/C/O，无残基 frame，无侧链，无 latent）
backbone: 非等变 transformer（15/15/18 层，768/768/1024 维，12/12/16 头；pair bias 偏置注意力，可选三角乘法；10 个 register token）
generative: 流匹配（rectified flow / conditional optimal transport，线性插值；Euler–Maruyama 概率流 SDE，默认 400 步）
conditioning: CATH 折叠类标签（C/A/T 三类，CFG 或 autoguidance）｜motif 坐标 + mask（inpainting）；无 MSA、无模板、无能量项、无受体/配体
data_scope: AFDB 合成结构为主（D_FS 588,318；D_21M 20,874,485，MMseqs2 50% 聚类后按簇采样）；PDB 114,076 链仅用于 LoRA 适配（designable 子集 90,423）；无 held-out 生成测试集，D_FS 兼作 novelty 参考集；PDB 截止日期未披露
status: 学习中
verified_at: 2026-11-04
---

# Proteína（Proteina）

> **状态**：学习中（表征、采样、数据已核实；时间切分与算力未核实）
> **记录人**：团队共同学习
> **最后更新**：2026-11-04

---

## 1. 学习背景与目标

### 1.1 命名沿革与对象界定

团队旧称 "RFD4-Proteina" 在仓库里被用来指代"锚原子 N/CA/C/O + 侧链 latent + Rosetta 能量监督 + reward guided search"这一整套东西。逐条回查后，这套描述无法落在 arXiv 2503.00710 上，因为它指向的是 2026 年的另一个模型。

| 名称 | 身份 | 与本篇的关系 | 证据 |
| --- | --- | --- | --- |
| Proteína（Proteina） | arXiv 2503.00710，ICLR 2025 Oral，NVIDIA；自述为 "large-scale flow-based protein **backbone** generator" | 本篇对象 | 【核实：摘要与 §3.2】 |
| La-Proteina | arXiv 2507.09466，ICLR 2026；Cα 显式 + 每残基 8 维 latent + VAE 解码 | 同一条线的**下一步**，不是同一个模型 | 【核实：La-Proteina §3.1、§3.2.1】 |
| Proteina-Complexa | arXiv 2603.27950，ICLR 2026；全原子 binder 设计 + 推理期搜索 | 同一条线的**分支**（binder 任务） | 【核实：Complexa 摘要、§3.1】 |
| RFD4-Proteina | `RosettaCommons/RFD4-Proteina`（`rfd4-proteina.org`），Baker Lab × NVIDIA GenAIR，"next generation sequence structure co-design model"，覆盖 protein/RNA/DNA/小分子复合物；完整技术仓库（1317 blob，训练＋推理＋search，BSD-3-Clause），仓库内托管 54 MB 技术报告 PDF（CC BY 4.0） | 旧称实际所指；接的是 La-Proteina 的 partial-latent 线，与 Proteina 无直接技术血缘；主基线为 Proteina-Complexa | 【核实：RosettaCommons 仓库、技术报告 PDF、`concepts/metric-guidance`、`concepts/search`】 |

因此 `aliases` 只收录真正等价的名字（`Proteina`、`Proteína`），不把 `RFD4-Proteina` 记为别名——它指代的是上表最后一行那个 2026 年模型，只是旧笔记的错误挂靠。正文提到旧称时一律写明指代。凡旧笔记中"32 维侧链 latent""0.82 Å 侧链误差""Rosetta 能量监督""beam search / oracle reward"这几条，均不得记在 Proteina 名下（归属裁决见 §3.7 与 §5.3）。其中"32 维"成立但为**残基级**隐变量（非侧链专属），"0.82 Å"查无此数——实际是 PLACER 的 AUROC 0.82，PackBench 真实口径为 MolProbity 1.70 vs LigandMPNN 1.99。【核实：RFD4-Proteina 技术报告 §4.3 与 PackBench 段；全仓库 `grep "0.82"` 无命中】

### 1.2 本次目标

1. 判定表征粒度：它是否可被称为"全原子"，边界落在哪里。
2. 拆开生成范式：流匹配作用在什么空间，步数与采样器为何这样选，与 AF3 的 EDM x0 200 步能否比。
3. 查清条件控制：无 MSA 是设计选择还是妥协；"能量项"是否进入网络。
4. 补齐训练数据与切分——这是仓库最缺、也最容易靠印象填错的一块。
5. 建立"宣称—证据—核查状态"表，并给出 VNAR 场景下的可用边界。

### 1.3 一句话判断

Proteina 的贡献不是"原子级建模"，而是**把主链 Cα 生成器的容量与数据规模同时推高，并把"分布级"评测引入主链生成**：它只生成每残基一个 Cα 点，连主链 N/C/O 都不生成，也不生成序列；设计可设计性靠外挂 ProteinMPNN + ESMFold 评估。【核实：§3.2、App. F.1】

---

## 2. 核心概念

| 术语 | 本篇含义 |
| --- | --- |
| rectified flow / conditional OT | 用线性插值 `x_t = t·x_1 + (1-t)·ε` 定义概率路径，回归速度场；与"加噪扩散"是不同的路径构造。【核实：App. I.1】 |
| Cα-only 表征 | 每个残基只保留一个 Cα 坐标，`x ∈ R^{3L}`；没有 N/C/O，也没有残基局部坐标系。【核实：§3.2】 |
| 非等变 transformer | 网络不含 SE(3) 等变层，靠训练数据与随机旋转增广获得旋转鲁棒性；作者明言回避 rotation manifold。【核实：§3.2】 |
| pair bias | 用相对位置编码的 pair 表示给注意力加偏置；可选三角乘法，但默认模型（M_no-tri）关掉。【核实：§3.3；`configs/model/multi_level_flow.yaml`】 |
| register token | 10 个可学习的全局 token，用来承载与残基无关的全局信息。【核实：§3.3】 |
| CFG / autoguidance | 条件引导的两种取法：CFG 用无条件模型当对照（α=0），autoguidance 用训练不足的弱模型当对照（α=1）；两者被写成同一公式的两端。【核实：§3.4 Eq.4】 |
| self-conditioning | 把上一步预测的 x̂₁ 作为额外输入；论文称 M_FS 不用、M_no-tri 用，因为它能缩小训练/推理差距。【核实：§3.5】 |
| designability（scRMSD） | 每 backbone 用 ProteinMPNN 生成 8 条序列（T=0.1），ESMFold 重折叠，取最低 scRMSD；<2 Å 记为可设计。【核实：App. F.1】 |
| best-of-8 | 上述协议取 8 条里最好的一条，等价于 best-of-8 选择，不是单序列命中率。【推断：依据 App. F.1 的 "lowest RMSD"】 |
| diversity (TM) / diversity (cluster) | 两两 TM-score 的均值；以及在 Foldseek TM 0.5 下的簇数除以样本数。【核实：App. F】 |
| novelty | 与参考集的最大 TM-score 均值；越低越新。参考集是 PDB，或训练用的 D_FS。【核实：Table 1 表注】 |
| FPSD / fS / fJSD | 作者新引入的三个分布级指标：折叠类分布与参考集的 Fréchet 距离、按 C/A/T 分层的折叠类占比差、折叠类分布的 Jensen–Shannon 散度（表中 ×10）。【核实：App. G.1–G.2】 |
| Euler–Maruyama 概率流 SDE | 在 rectified flow 的速度场上加一个随时间衰减的噪声项，用 SDE 积分；γ 控制噪声尺度。【核实：App. I.4】 |
| inpainting / motif scaffolding | 固定 motif 残基的坐标与类型，只对其余位置加噪生成。【核实：App. B.1】 |
| （对照）partially latent | La-Proteina 的表述：Cα 显式 + 每残基隐变量承载序列与非 Cα 原子；**Proteina 没有这一层**。【核实：La-Proteina §3.1】 |

---

## 3. 原理与推导

### 3.1 表征：单点 Cα，且刻意不用 frame

- 生成变量是**每残基一个 Cα 坐标**，`x ∈ R^{3L}`；论文的图与公式都只画/写 Cα。【核实：§3.2、Fig. 2】
- 论文明确说，采用 Cα-only 是为了**避开旋转流形**：如果把 N/C/O 放进残基局部坐标系，就要在 SO(3) 上做流匹配；作者选择只对点云建模。【核实：§3.2】
- 否定性检索：全文 `latent`、`autoencoder`、`decoder`、`torsion`、`chi angle`、`side-chain` 均为 0 次命中。【核实：PDF 全文检索】
- 代码侧同构证据：Atom37 张量只是格式容器，`trans_nm_to_atom37` 只把 Cα 写进 index 1，其余 36 个槽位全置零。【代码／配置核实：`proteinfoundation/utils/coors_utils.py`】

**边界结论（直接回答"是否全原子"）**

Proteina 不是全原子模型，甚至不是"主链模型"的完整形态：它只保留了主链上的一个点。所谓"主链显式 + 侧链 latent"的划分，在 Proteina 里不存在——没有侧链，也就没有 latent 维度、没有解码器、没有"解码器是谁"这个问题。

因此仓库里"all-atom 标签名不副实"这条判断，**对 Proteina 不适用**：论文从未自称全原子，摘要第一句就是 backbone generator。若仓库曾把它列进 all-atom 组，那是仓库归类错误，不是模型的自我宣称问题。该判断对 La-Proteina **部分适用**（它确实用隐变量承载非 Cα 原子，全原子由 VAE 解码器给出，解码精度需单独评），但不属于本篇。

### 3.2 网络：非等变 transformer，靠数据而非等变性获得旋转鲁棒性

| 组件 | 配置 | 出处 |
| --- | --- | --- |
| 主干 | 15 层（200M）/ 18 层（400M）；768 / 1024 维；12 / 16 头 | §3.3、Table 16 |
| 参数量 | M_FS 208M；M_no-tri 191M；M_21M 397M；M_small 59M | Table 9 |
| 注意力 | pair bias 偏置；可选三角乘法（M_FS 用，M_no-tri 不用） | §3.3 |
| 三角层参数 | 15M（Table 16）与 ≈17M（§3.3）**不一致** | 两边并列 |
| 全局 token | 10 个 register token | §3.3 |
| 等变性 | 无显式等变层，作者自述为非等变网络；靠随机旋转平移增广补偿 | §3.2 |

**为什么这仍然能work**：作者的解释是数据规模与增广替代了架构归纳偏置；这与"非等变一定更差"的直觉相反，但论文没有给出"等变 vs 非等变"的同数据消融，只有容量消融（M_FS / M_21M / M_no-tri）。【推断：依据 §3.3 与 App. C 缺失对照】

### 3.3 训练目标：rectified flow 为主，distogram 为辅

```text
路径：      x_t = t · x_1 + (1 − t) · ε,   ε ~ N(0, I),  t ~ U[0,1]
主损失：    L_FM = E ‖ v_θ(x_t, t, c) − (x_1 − ε) ‖²
辅助项：    L_distogram（主链 Cα–Cα 距离分箱的交叉熵）
总损失：    L = L_FM + λ · L_distogram
```

- 目标速度是 `x_1 − ε`，即"干净坐标减噪声"，等价于指向数据的方向；这与 EDM 的 x0 参数化是同族但不同参数化。【核实：App. I.1】
- distogram 辅助项只作用在 Cα–Cα 距离上，可视为对局部几何的弱正则；论文报告它带来小幅提升，但**没有给出单独的消融表**。【论文自称；消融缺失为核实结论】
- 训练时 `t` 均匀采样；`t` 的采样分布与推理噪声调度**不是同一套**（训练均匀、推理非均匀 log-spaced）。这一点在跨模型比较时容易读错。【核实：App. I.4】

### 3.4 采样：400 步 Euler–Maruyama SDE，末段切 ODE

```text
dt = 0.0025  ⇒  400 步
g(t) = 1 / (t + 0.01),  t ∈ [0, 0.99]
t ∈ (0.99, 1] 时 g(t) = 0  ⇒  退化为 ODE
γ ∈ {0.30, 0.35, 0.45, 0.50, 0.60}（按模型分别调）
```

- 加噪声的理由是**提升多样性**；代价是 t 接近 1 时 SDE 不稳定，作者的处理是把最后 1% 的时间切成 ODE。【核实：App. I.4、App. H.2】
- 每一步都做随机旋转平移增广，因为网络不等变。【核实：§3.2；`coors_utils.py` 的 `centre_random_augmentation`】
- 推理配置与主表预算一致：`nres_lens` 50–500、每长度 100 个样本、`max_nsamples: 5`。【代码／配置核实：`configs/experiment_config/inference_base.yaml`】

**步数选择没有依据可查**：论文没有报告"步数—质量"曲线，也没有给出少步数（如 50/100）的对照，因此无法判断 400 是不是过采样。【核实：缺失】

### 3.5 条件控制：折叠类标签，不是 MSA

- 条件只有两类：(i) CATH 折叠类标签（C / A / T 三个层级的组合）；(ii) motif 的坐标与 mask（inpainting）。【核实：§3.4、App. B.1】
- 引导写成统一式：

```text
v̂ = v_θ(x_t, t, c) + ω · ( v_θ(x_t, t, c) − v_ref(x_t, t) )
v_ref = α · v_θ(x_t, t, ∅) + (1 − α) · v_weak(x_t, t, c)
α = 0 → CFG；α = 1 → autoguidance
```

- 定量后果（Table 2）：ω 从 1.0 → 2.0，designability 91.4 → 89.2 → 83.8；novelty vs PDB 0.77 → 0.77 → 0.78（越低越好，即变差）。引导换来的是折叠类符合度，代价是样本质量与新颖性。【核实：Table 2】
- 折叠类条件的效果（Table 4）：无条件 96.4% designability、β 含量 6.9%；"mainly β" 条件下 designability 90.0%、β 含量 33.3%。条件能拉动二级结构组成，但同时压低可设计性。【核实：Table 4】

**无 MSA 是设计选择，不是妥协**

训练目标是单体结构的无条件分布 `p(x)`，输入里没有 query 序列，MSA 在生成侧没有对应的条件变量。论文的处理是把"序列信息"整体交给下游的 ProteinMPNN。这与 AF3 的路线正好相反：AF3 把 MSA/模板当作**外挂条件**注入 trunk，用来消解"同一序列可能对应多个构象"的不确定性；Proteina 则是先采一个构象，再让逆折叠模型去解释它。两者不是同一类任务，不能把"没有 MSA"读成"能力不足"。【核实：§3.1 "without sequence inputs"；对照为推断】

### 3.6 与 AF3 的对照：200 步 EDM x0 vs 400 步 rectified flow

| 维度 | Proteina | AF3 | 是否可比 |
| --- | --- | --- | --- |
| 任务 | 无条件主链生成 | 给定序列/MSA 的结构预测 | 否 |
| 状态空间 | 3L 个 Cα 点 | 每残基最多 37 原子 + 配体原子（token 化） | 否，维度差一个量级 |
| 路径 | rectified flow，线性插值 | EDM，σ 参数化，`σ_data=16`、`σ_min=4e-4`、`σ_max=160`、`ρ=7` | 否 |
| 参数化 | 速度 `v = x_1 − ε` | 干净坐标 `x0`，配 `c_skip/c_out` 预条件 | 否 |
| 步数 | 400（无步数消融） | 200（代码默认） | 数字不可直比 |
| 采样预算 | 5 长度 × 100 样本 = 500 | 5 seeds × 5 samples = 25 | 否 |
| 随机性来源 | SDE 噪声项 + 随机旋转增广 | 随机旋转平移增广 + 噪声 | 部分可比 |
| 评测指标 | designability / diversity / novelty / FPSD | LDDT / DockQ / RMSD / 物理有效性 | 否 |

【核实：AF3 数字取自仓库 `AlphaFold3/README.md` §2、§3.3 与 `diffusion_head.py` 的既有记录；Proteina 数字见上】

**真实差别**：不是"400 步比 200 步慢一倍"。每步成本由状态维度决定——Proteina 每步只更新 3L 维，AF3 每步要在 token 化的全原子上做注意力。步数更多但每步更轻，方向相反。要比较效率，只能比"生成一条可用主链的 wall-clock"，而论文的推理成本表（Tab. 8/9/10）是图表，本次未取到数值。【未核实】

**没有同数据对照**：Proteina 的 benchmark 是自生成分布，AF3 的 benchmark 是实验结构预测，两者没有共享测试集，也没有共享指标。任何"Proteina 比 AF3 快/准"的说法都缺对照组。【核实：两篇评测章节】

### 3.7 "Rosetta 能量项"的归属核查

三条否定证据，逐条列出：

1. 全文 "Rosetta" 仅 1 次，出现在基线名 "RoseTTAFold"；`PyRosetta`、`energy function`、`force field`、`ref2015` 均 0 次。【核实：PDF 全文检索】
2. 仓库 `proteinfoundation/` 包内 `rosetta` 0 命中；唯一 python 侧能量代码是 vendored OpenFold 的 AMBER 最小化器（`openfold/np/relax/amber_minimize.py`），未被 designability 链路调用。【代码／配置核实】
3. 设计可设计性链路用的是 HuggingFace ESMFold，`is_amide_bond` 只用于判定肽键，不涉及能量打分。【代码／配置核实：`proteinfoundation/metrics/designability.py`】

因此：**Proteina 没有能量项，没有打分器梯度，没有能量 conditioner，也没有能量后处理**。

该说法在家族中的真实位置：

- 旧称所指的 2026 年模型（旧笔记称 "RFD4-Proteina"，即 `RosettaCommons/RFD4-Proteina` / `rfd4-proteina.org` 上 Baker Lab × NVIDIA GenAIR 的那个模型）：微调阶段在**解码器**（自编码器）上引入 Rosetta 能量损失，经 TMol 对解码后的坐标求导，**不是 denoiser 的条件项**；每残基能量 −0.18 → −1.25 kcal/mol（降幅约 1.07）、严重键角偏差减少 62%（相对只做重建的微调）。这是"可微能量损失"，属于训练期监督，不是 conditioner。【核实：仓库内技术报告 S4.3.1 与 Fig 3；2026-11-10 已通读全文】
- Proteina-Complexa：把 Rosetta 定义的氢键能量作为**推理期可选 reward**（Tmol 实现，配置默认整块注释关闭）。这是"搜索目标"，不是训练损失。【核实：Complexa 仓库 `search/reward/rosetta_hbond.py` 与 config】

三条路——训练期可微损失、推理期 reward、后处理过滤——都不在 Proteina 里。若沿用"Proteina 解码受 Rosetta 监督"的说法，会把一个 2026 年模型的设计选择误记到 2025 年模型上。

---

## 4. 训练数据构成与切分

### 4.1 数据源与规模

| 数据集 | 规模 | 用途 | 过滤条件 | 出处 |
| --- | --- | --- | --- | --- |
| AFDB（AlphaFold DB） | ≈214M 结构 | 合成结构池 | — | §3.1 |
| D_FS | 588,318 | **主训练集** | 32–256 aa；pLDDT ≥80；Foldseek 簇代表；MMseqs2 50% 序列同一性 | §3.1；App. M.2 |
| D_21M | 20,874,485 | 第二训练集 | ≤256 aa；pLDDT 均值 ≥85、std ≤15；coil ≤50%；Rg ≤3 nm；MMseqs2 50% 聚类后按簇均匀采样 | §3.1 |
| PDB | 114,076 链 | 仅 LoRA 适配 + 评测参考 | 50–256 aa；分辨率 <5 Å；无非常规残基；coil ≤0.5；Rg ≤3.0 nm | App. M.1 |
| PDB designable 子集 | 90,423（79.26%） | LoRA 训练 | scRMSD <2 Å | App. M.1 |
| 磁盘 | >20 TB，≈214M 个文件对象 | — | — | App. M.2 |

**这里有一个反直觉的数据选择**：主训练集不是 PDB，而是 AFDB 的**预测结构**。PDB 只用来做 LoRA 适配。作者的理由是 PDB 规模与多样性不足，而 AFDB 覆盖 214M 条序列。代价是训练分布里混入了 AlphaFold2 的系统性偏差（例如对长无序区的拉伸、对某些折叠类的偏好），论文没有讨论这部分偏差如何影响生成分布。【核实：§3.1、§4 关于 LoRA 的动机；偏差为推断】

**PDB 截止日期：未披露**。全文与仓库 MODELCARD 都没有给出快照日期或版本号；检索 `cutoff|snapshot|as of|PDB version|release date` 无日期句。这一点直接导致无法判断时间切分，也无法判断测试结构是否已进入训练集。【未核实】

### 4.2 清洗与去重

| 项目 | 口径 | 状态 |
| --- | --- | --- |
| 分辨率 | PDB <5 Å | 【核实：App. M.1】 |
| pLDDT | AFDB ≥80（D_FS）；≥85 均值且 std ≤15（D_21M） | 【核实：§3.1】 |
| 链长 | 32–256 aa（D_FS）；≤256（D_21M）；50–256（PDB） | 【核实】 |
| 非标准残基 | PDB 侧剔除 | 【核实：App. M.1】 |
| 线圈比例 | coil ≤50%（D_21M）；≤0.5（PDB） | 【核实】 |
| 回转半径 | Rg ≤3 nm（D_21M）；≤3.0 nm（PDB） | 【核实】 |
| 序列聚类 | MMseqs2，50% 同一性 | 【核实】 |
| 结构聚类 | Foldseek，取簇代表 | 【核实】 |
| 配体/离子/核酸 | **全部剔除**（纯蛋白单体） | 【核实：数据构造仅蛋白链】 |
| 缺失原子处理 | 未披露具体策略（Cα-only 使得缺失非 Cα 原子不影响输入） | 【未核实/推断】 |

### 4.3 切分与泄漏控制

**核心事实：没有 held-out 生成测试集。**

- MODELCARD 把 D_FS 同时列为训练集与 novelty 的参考集；Table 1 的 "Novelty (AFDB)" 一列用的正是 D_FS。【核实：MODELCARD；Table 1 表注】
- 论文没有报告训练/验证/测试比例，也没有说明按 Foldseek 簇还是按序列相似度隔离。【未核实】
- 主表 Table 1 的表注写明"所有模型都为最大化 designability 调过参"。这意味着该表是**调参后的最优配置**，不是 held-out 泛化评测。【核实：Table 1 表注】

**泄漏渠道清单**

1. 训练集兼作 novelty 参考集：novelty 越低表示越"新"，但参考集本身就在训练里，所以这个数字衡量的是"离训练分布簇中心有多远"，不是"离自然蛋白世界有多远"。【推断：依据 MODELCARD 与 Table 1 表注】
2. PDB 日期未知：若测试用 PDB 链晚于训练快照，数字会偏高；论文没有给出日期，无法排除。
3. AFDB 与 PDB 重叠：AFDB 的输入序列大量来自 PDB 已解结构，D_FS 里可能包含 PDB 链的同源结构；两套数据没有做跨库去重说明。【推断】
4. LoRA 适配用了 PDB 的 designable 子集（90,423），而评测的 designability 也用 PDB 参考分布；适配集与评测分布同源。【推断】
5. 折叠类分类器（FPSD/fS/fJSD 的实现）训练于 214,564 条 CATH 标注链，与 D_FS 来源同库，可能共享同源结构；论文未做去重说明。【核实数字；泄漏为推断】

**对跨模型比较的直接后果**：Proteina 的 novelty 数字不能与"在 PDB 上训练、在 PDB 后释放结构上评测"的模型并列。任何"Proteina 更新颖"的结论都需要先统一参考集与去重协议。

### 4.4 覆盖与偏置

| 维度 | 情况 | 状态 |
| --- | --- | --- |
| 复合物 | 不支持；无链间界面、无配体、无核酸 | 【核实：数据构造】 |
| 长度 | 训练 ≤256；长链模型扩到 ≤800（M_long 在 512/768 两阶段续训） | 【核实：§4、Table 16】 |
| 折叠类 | CATH 三标签覆盖 D_FS 99.9%、D_21M 69.7% | 【核实：MODELCARD】 |
| 抗体/纳米抗体 | 未做分层统计；AFDB 以 UniProt 单体为主，单域抗体占比极低 | 【推断：依据 AFDB 构成】 |
| 物种偏置 | 继承 AFDB 与 PDB 的物种分布，未重平衡 | 【推断】 |
| 构象多样性 | 每序列一个静态构象（AF2 预测），无 ensemble | 【核实：数据源性质】 |

### 4.5 可获取性与许可

- 代码：`NVIDIA-BioNeMo/proteina`，Apache-2.0；含 `MODELCARD.md`、`LICENSE`、10 个实验 config。【核实：仓库】
- 权重：随仓库发布（2025-02-28），加载方式见 `MODELCARD`；商用条款需另读 NVIDIA 许可，`LICENSE` 只覆盖代码。【核实：仓库结构；商用条款未逐条读】
- 数据：AFDB 公开可下（≈20 TB），但论文的过滤清单与 Foldseek/MMseqs2 聚类产物未发布。【推断：依据仓库内容】
- 可复现边界：推理可复现；**训练不可完全复现**（切分、日期、GPU 型号均缺）。【推断】

### 4.6 Teddymer 与本篇的关系

Teddymer 不是 Proteina 的数据集，而是 Proteina-Complexa 的预训练集：用 TED 结构域注释把 AFDB50 单体切域，再拼成合成二聚体，链条为 47M（AFDB50∩TED）→ 123,606,001 个候选二聚体 → CATH 过滤后 10,089,503 → Foldseek-Multimer 聚类 3,556,223 簇 → 训练用 510,454 个簇代表、7,112,609 个数据点。【核实：Complexa §3.1、App. C】

它属于**合成结构**（由单体拼装），不属于**自蒸馏**（没有用模型自身输出再训练）。Complexa 全文 "distill" 0 命中。因此：

- "Proteina 用了合成数据"——成立，指 AFDB（AF2 预测结构）。
- "Proteina 用了 Teddymer"——不成立。
- "Proteina 用了自蒸馏"——不成立；仓库若这样写，属误标。

---

## 5. 宣称与可验证性

### 5.1 宣称—证据—核查状态

| 宣称 | 证据类型 | 核查状态 | 备注 |
| --- | --- | --- | --- |
| 规模化的主链生成器，参数量为前作 5× | 表格 + 文字 | **部分核实** | 绝对值：M_21M 397M；"5×" 的锚点（RFDiffusion 参数量）未给。【核实：Table 9；锚点未核实】 |
| 主表 designability 最高 99.0% | 表格 | **已核实** | M_21M, γ=0.3；注意是 best-of-8 协议。【Table 1；App. F.1】 |
| diversity 与 novelty 优于基线 | 表格 | **已核实（有条件）** | 优于 RFDiffusion/Chroma，但 diversity(cluster) 落后 M_21M 与 M_LoRA 自身；novelty 的参考集在训练内。【Table 1】 |
| 新指标（FPSD/fS/fJSD）能反映分布相似度 | 表格 + 曲线 | **已核实（作者自评）** | 只有作者自己的实现与参考集，无第三方复现。【App. G】 |
| motif scaffolding 全面优于基线 | 表格 | **已核实（有边界）** | 26 个 task 中 8 个唯一最佳（Genie2 为 5 个）；"极难"档仍失败。【Table 6、App. B.2】 |
| 可生成 800 残基长链 | 表格 | **已核实** | 但长链模型的训练数据构成与 300 步内的质量曲线未细读。【§4.2】 |
| 折叠类条件可控 | 表格 | **已核实** | β 含量 6.9% → 33.3%；代价是 designability 96.4% → 90.0%。【Table 4】 |
| 训练用 AFDB 主训练、PDB 仅 LoRA | 文字 + MODELCARD | **已核实** | — |
| 训练数据有明确时间切分 | — | **未核实（无披露）** | 无日期、无切分比例。 |
| 400 步是最优采样步数 | 无消融 | **存疑** | 无步数—质量曲线。 |
| "全原子"或"侧链 latent" | — | **不成立** | 全文 0 命中；代码只写 Cα。【§3.1】 |
| "解码受 Rosetta 能量监督" | — | **不成立（对象错置）** | 见 §3.7。 |

- **可复现所需材料**：推理可复现——官方仓库（Apache-2.0）提供代码与 config，权重经 NGC 获取（`MODELCARD.md` 记录推理测试于 A100）；训练数据管线、PDB 截止日期、切分文件、GPU 型号与算力均未公开，训练不可完全复现（详见 §5.4）。【核实：仓库 `MODELCARD.md`、`configs/`；训练不可复现为推断】
- **与官方文档／论文的一致性**：表征（Cα 单点）、网络层数与宽度、rectified flow 400 步等主体描述与论文及 `configs/` 一致；论文 Table 16（≈15M）与 §3.3（≈17M）三角层参数不一致，`self_cond` 默认值配置与论文表述方向一致但默认关闭（见 §5.3）。【核实：论文 §3.2–§3.5、Table 16；代码／配置核实】

### 5.2 评测协议、采样预算与口径陷阱

| Benchmark | 规模/口径 | 指标 | 采样预算 |
| --- | --- | --- | --- |
| 无条件生成（主表） | 5 个长度（50–250） | designability、diversity(TM/cluster)、novelty(PDB/AFDB)、FPSD/fS/fJSD、二级结构 | 每长度 100 个 **designable** 样本 = 500【Table 1】 |
| 全分布评估 | 40 个长度（60–255，step 5） | designability、diversity、novelty（按长度展开） | 40 × 125 = 5,000，不作过滤【App. G.2】 |
| Motif scaffolding | 26 个 task | unique successes、designability、motif RMSD | 每 task 1000 backbones【App. B.1】 |
| 长链 | 300–800，每 100 一个长度 | designability、diversity、novelty | 每长度 100 个【§4.2】 |
| 稳定性 | 5 seeds | 均值/方差 | 仅此实验报告 seeds【App. H.3】 |

**四个会让跨模型比较失真的口径**

1. **designability 是 best-of-8**：每个 backbone 生成 8 条序列，取最低 scRMSD 判 <2 Å。若对手用 best-of-1 或取平均，两者不可比；差的不是模型而是选择协议。【推断：依据 App. F.1】
2. **主表只统计 designable 样本**："5 长度 × 100 designable"意味着先按 designability 筛再统计，因此主表的 diversity/novelty 是**条件于可设计性**的分布，而 Table 3 的全分布数字（designability 19.0–44.2%）才是无过滤口径。两个表不能混用。【核实：Table 1 表注与 Table 3 表头】
3. **主表是调参后的最优配置**：表注写明所有模型都为最大化 designability 调过参；这不是 held-out 排名。【核实：Table 1 表注】
4. **hit rate 不适用于本篇**：Proteina 没有 binder 任务，也没有"每靶命中率"。仓库若出现"Proteina hit rate 63.5%"，那是 Proteina-Complexa 的旧数字——**该数字在一手论文中查无，已全仓撤下并标【待溯源】**（唯一登记处：`专题/01` 表注 4）；Complexa 项目页虽给出过单靶（PDGFR）122/191 的送检命中数，但论文正文无此口径，不能作为可引用指标。需要 hit rate 时一律转 `专题/03` §4 的可核实列。【待溯源；冲突裁决见本篇 §3.7、§5.1】

- **"unique successes" 的分母**：motif 表中 `0.82 (121)` 这类括号内数字是**该 task 的 unique successes 数**，unique 指 Foldseek TM 0.6 单连接聚类后的簇数，分母是每 task 的 1000 条 backbone。它同时做了"按靶"与"去重"两层归一，与其他模型的"每候选命中率"不可直接换算。【核实：App. B.2】

### 5.3 论文与代码／配置不一致（两边并列）

| 项目 | 论文 | 代码／配置 | 处理 |
| --- | --- | --- | --- |
| 三角层参数 | §3.3 写 ≈17M | Table 16 写 15M | 两边并列；总量 397M 以 Table 9 为准 |
| self-conditioning | §3.5：M_FS 不用、M_no-tri 用（可关闭以省成本） | `inference_base.yaml`：`self_cond: False`；`inference_ucond_200m_notri.yaml`：`self_cond: True` | 与论文一致，但**默认配置是不开的** |
| `sc_scale_noise` | 未给统一值 | base 0.4；notri 0.45 | 按配置文件区分，不合并 |
| 引导权重默认 | 主表扫 ω∈{1.0,1.5,2.0}；autoguidance 2.0 | base 1.0（等价无引导） | 开箱即用的是无引导版本 |
| 采样步数 | 400 | `dt: 0.0025` ⇒ 400 | 一致 |
| 训练 GPU | 未披露 | MODELCARD 仅写推理测试于 A100 | 不能反推训练卡型号 |

### 5.4 可复现边界

- **可复现**：推理（权重 + config + `nres_lens`）。
- **不可完全复现**：训练（无 PDB 日期、无切分文件、无 GPU 型号、无 wall-clock；折叠类标签生成脚本未见发布）。
- **不可复现的评测**：FPSD/fS/fJSD 依赖作者训练的那个 fold classifier；权重未见独立发布说明，第三方无法原样复算。【未核实】

---

## 6. 实践与应用

### 6.1 适用场景与前提

**适用**：需要"先出骨架、再定序列"的无条件主链生成；给定 motif 的 scaffolding；按折叠类采样做分布覆盖实验；作为主链先验给下游逆折叠/侧链打包。

**不适用**：binder/表位设计；任何需要侧链化学的任务；任何需要序列输出的任务；任何需要复合物（蛋白-蛋白、蛋白-配体、蛋白-核酸）的任务。

### 6.2 关键参数与调优

| 参数 | 默认 | 经验 |
| --- | --- | --- |
| 采样步数 | 400（`dt=0.0025`） | 无消融；先按默认跑，做成本敏感实验时自行扫 100/200/400 |
| γ | 0.3–0.6，按模型 | 大模型用更小的 γ（M_21M 0.3）；γ 越大越多样但结构越松 |
| guidance ω | 1.0（关闭） | 1.5–2.0 提升折叠类符合度，designability 与 novelty 下降 |
| 折叠类标签 | 可选 C/A/T | β 类条件会显著改变二级结构组成，但压低可设计性 |
| 每条 backbone 的 MPNN 序列数 | 8 | 下游若要公平可比，须统一 best-of-N |

### 6.3 VNAR / 单域抗体的可迁移点

**能借的模块**

1. **AFDB 预训练 → 小数据 LoRA 适配的配方**。Proteina 在 PDB 的 90,423 条 designable 链上只训 11K 步、rank 16、scaling 32、约 7M 可训练参数就完成适配。这对 VNAR 这类只有百条量级结构的数据集是有吸引力的范式：先在大规模单体分布上学骨架先验，再用 LoRA 拉向目标子分布。【核实：App. M.1、Table 16；对 VNAR 的适用性为推断】
2. **Cα-only 的极简接口**。如果流水线设计是"骨架生成 → ProteinMPNN 定序列 → 侧链打包"，Cα 表示让上游与下游解耦，接口只有一个 `[L,3]` 数组，最容易替换与拼接。【核实：代码接口】
3. **分布级指标而非 top-1 指标**。FPSD/fS/fJSD 衡量"生成分布与参考分布在折叠类上差多少"，适合小样本场景下判断"是否只生成了两三种框架"。VNAR 的框架多样性远低于常规球蛋白，这类指标比单一 designability 更能暴露塌缩。【核实：App. G；迁移为推断】
4. **末段切 ODE 的稳定化技巧**。SDE 在 t→1 不稳定，把最后 1% 置零即可；实现成本极低，可直接搬到任何 rectified-flow 采样器。【核实：App. I.4】

**能借的指标**

- designability（scRMSD <2 Å）——但必须同时报告 best-of-N，否则与别家不可比。
- diversity(cluster)（Foldseek TM 0.5 去重后的簇密度）——对小数据集比 TM 均值更稳。
- novelty(TM vs PDB)——只有在参考集与训练集做了去重之后才有意义；Proteina 自己没做，直接借用会重复同样的泄漏。

**明显的坑**

1. **Cα-only 看不见 paratope**。VNAR 的结合特异性几乎完全由 CDR 侧链决定；Proteina 不生成侧链，也不生成序列，因此它产出的"骨架"对结合设计没有判别力，只能当形状先验。
2. **训练分布外**。主训练集是 AFDB 单体、≤256 aa、无复合物；AFDB 的单域抗体覆盖极低（PDB 侧高分辨率 VNAR 结构仅约 84 条，此数为仓库既有口径，未回查 SAbDab2 原始统计）。把 Proteina 直接用于 VNAR 骨架，等于在分布边缘采样。【推断】
3. **没有 held-out**。它的 novelty/designability 是在"训练集兼参考集"的条件下得到的；若用它论证"能生成新型 VNAR 框架"，证据链不成立。【推断：依据 §4.3】
4. **不能做表位条件**。折叠类标签只控制全局拓扑，不控制界面化学；表位特异设计需要 target 侧原子条件（那是 Proteina-Complexa 的 Atom37 + hotspot token 接口，不是本篇）。【核实：Complexa §3.2；对照为推断】
5. **长度上限**。VNAR 单域约 110–130 aa 落在训练区间内；但带 linker/标签的构建会越界，长链模型（≤800）是另一套权重与另一套训练阶段，不能默认继承。【核实：§4.2】

---

## 7. 疑问与讨论

**Q1：为什么只做到 Cα，而不是完整的 N/C/O 主链？**

A：为了回避旋转流形。若把 N/C/O 放进残基局部坐标系，流匹配就要在 SO(3) 上定义路径与噪声；作者选择只对点云建模，把"主链几何"降为一个点序列问题，代价是无法保证键长键角与手性。论文没有报告键长/键角合格率，这类物理有效性只能在下游打包后评估。【核实：§3.2；物理有效性缺失为核实结论】

**Q2：400 步 SDE 是不是比 AF3 的 200 步 EDM 更贵？**

A：不能这样比。步数只是乘数之一：每步成本由状态维度与网络宽度决定，而两者的状态空间差一个数量级（3L 点 vs 每残基最多 37 原子 + 配体）。且任务不同（生成 vs 预测），预算不同（500 vs 25 样本），没有共享测试集。可比较的量只有"生成一条可用主链的 wall-clock"，论文给了图表但本次未取到数值。【核实：两篇方法与评测；成本数值未核实】

**Q3：无 MSA 是妥协吗？**

A：不是。生成侧没有 query 序列，MSA 没有对应的条件变量；序列信息整体下放给 ProteinMPNN。代价是"同一序列多构象"的不确定性无法在生成期消解，只能靠下游重折叠筛掉。这与 AF3 把 MSA/模板作为外挂条件注入 trunk 的路线是两种任务设定，不构成同一维度上的强弱。【核实：§3.1；对照为推断】

**Q4：为什么用 AFDB 而不是 PDB 做主训练集？**

A：规模与覆盖。PDB 只有 11.4 万条可用链，AFDB 有 2.14 亿条预测结构。代价是把 AF2 的系统性偏差写进生成分布，而论文没有量化这部分偏差。另一个后果是"新颖性"评测失去外部参照——参考集就是训练集。【核实：§3.1、MODELCARD；后果为推断】

**Q5：能不能拿它做 VNAR 的 binder 设计？**

A：不能。它没有 target 条件、没有序列输出、没有侧链。VNAR 的可用路径是把它当骨架先验，再串 ProteinMPNN 与侧链打包；若要表位条件，需要换到 Complexa 一类带 Atom37 target 接口的模型。【核实：本篇能力边界；替代路径为推断】

**Q6：旧称 "RFD4-Proteina"（指 2026 年 `rfd4-proteina.org` 上的那个模型，不是本篇的别名）是怎么来的？**

A：可以复原出一条合理链路：团队先接触到 `rfd4-proteina.org` 的工具（Baker Lab × NVIDIA，2026），它的接口里同时有 `bb_ca` 与 `local_latents`，文档里有 reward guided search 与 oracle refold，于是"Proteina + latent + 搜索 + Rosetta"被打包成一个名字；随后这个名字被回溯套到 2025 年的 Proteina 论文上。证据是：旧笔记里的每一条机制都能在 `rfd4-proteina.org` 找到对应页面，而在 arXiv 2503.00710 里一条都找不到。【推断，依据两处一手来源逐条比对】

**Q7：仓库里"all-atom 标签名不副实"这条判断，适用吗？**

A：对 Proteina 不适用。它从未自称 all-atom，摘要首句即 backbone generator；真正的问题是仓库可能把它归进了 all-atom 组。该判断对 La-Proteina 部分适用——它用 8 维/残基的隐变量承载非 Cα 原子，全原子由 VAE 解码器给出（重建全原子 RMSD ≈0.12 Å），但"生成的全原子是否物理合理"仍需独立评测，不能由重建精度代替。【核实：La-Proteina §3.1–§3.2.1、§4.3】

---

## 8. 学习心得

- **"主链生成"这个词在这条线上要拆成三档**：Cα 单点（Proteina）→ Cα + 隐变量解码全原子（La-Proteina）→ 全原子显式（Complexa 的 binder 分支）。三档的评测协议完全不同，用同一张表排名会得出错误结论。
- **数据规模换来的不是精度，而是分布的覆盖**。Proteina 从 D_FS（58 万）扩到 D_21M（2,087 万）后，主表 designability 从 96.4% 升到 99.0%，但 novelty 反而下降、diversity(cluster) 从 305 掉到 150——扩容让分布更"像"训练集，而不是更"新"。【核实：Table 1】
- **引导是一个零和旋钮**。CFG/autoguidance 提升的是条件符合度，代价是 designability 与 novelty；把它当"免费提升"是常见误读。【核实：Table 2、Table 4】
- **没有 held-out 的生成论文，其 novelty 数字只能当内部诊断**。这不是 Proteina 独有的问题，但它是判断"能否用于新家族（如 VNAR）"时必须先扣掉的分。
- **旧称是最容易污染证据链的东西**。这次核对里，四条"公认"的属性（32 维 latent、0.82 Å 侧链、Rosetta 监督、beam search）没有一条能在 Proteina 原文或代码里找到；它们全部来自一个 2026 年的后继模型。名字一旦混用，后面所有对比都会错位。

---

## 9. 参考资料

**一手**

- Geffner et al., *Proteína: Scaling Flow-based Protein Structure Generative Models*, arXiv:2503.00710（ICLR 2025 Oral）。HTML: https://arxiv.org/html/2503.00710 ；PDF: https://arxiv.org/pdf/2503.00710
- 官方代码与权重：https://github.com/NVIDIA-BioNeMo/proteina （Apache-2.0；`MODELCARD.md`、`configs/`）
- 项目页：https://research.nvidia.com/labs/genair/proteina/

**边界对象（仅用于分清归属）**

- La-Proteina, arXiv:2507.09466 — https://arxiv.org/html/2507.09466
- Proteina-Complexa, arXiv:2603.27950 — https://arxiv.org/html/2603.27950 ；项目页 https://research.nvidia.com/labs/genair/proteina-complexa/
- 旧称所指的 2026 年模型（旧笔记称 "RFD4-Proteina"，非本篇别名）：代码仓库 https://github.com/RosettaCommons/RFD4-Proteina （BSD-3-Clause，训练＋推理＋search 完整技术仓库）；项目文档 https://rfd4-proteina.org/latest/ （`concepts/metric-guidance`、`concepts/search`、`concepts/metrics`）；技术报告《Biomolecular design through latent reasoning with RFD4-Proteina》（Corley、Didi 等，2026 预印本，CC BY 4.0）PDF 随仓库发布于 `website/assets/paper/RFD4-Proteina.pdf`（54 MB，2026-11-10 已通读）

**仓库内参照**

- `模型篇章/_TEMPLATE.md`（结构规范）
- `模型篇章/AlphaFold3/README.md`（证据标记与"待核实"处理范式；AF3 的 EDM/x0/200 步/25 样本口径）
- `联合分析/推理时优化机制对比-AnewOmni与RFD4-Proteina.md`（旧称来源）
- `联合分析/全原子模型解题思路归类分析.md`（仓库既有归类）
