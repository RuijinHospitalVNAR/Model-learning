---
model: RFD4-Proteina
aliases:
  - "RFD4-Proteina（Corley, Didi et al., 2026 preprint）"
  - "RFD4 / RFdiffusion4-Proteina"
  - "rfd4-proteina.org"
family: "Baker Lab（UW IPD）× NVIDIA GenAIR；RFdiffusion 全原子线 × 部分隐变量线（La-Proteina）的合流"
granularity: "残基级 token：Cα/C1′ 锚点坐标 x∈R³ + 32 维 local latent ℓ∈R^32；latent 由 130M 自编码器解码为 37 槽位原子"
backbone: "14 层 transformer（token 768 / pair 256 / 12 heads），内部 recycling 60 步 × 1–6 次 trunk（120/180/300 次 trunk pass）"
generative: "双模态并行流匹配（bb_ca 坐标 + local_latents 隐变量），vf_tsr 随机采样器，60 步；外部 checkpoint beam search"
conditioning: "靶点绝对坐标 / 内部坐标、hotspot、motif 索引、序列概率、CATH fold 类、对称性、部分扩散、ligand 构象固定/共生成"
data_scope: "PDB 界面（截止 2024-01-01）+ AFDB 单体 + 物理方法自蒸馏数据 + 酶/小分子/核酸"
status: "已完成"
verified_at: "2026-11-10"
---

# RFD4-Proteina

> **状态**：已完成
> **记录人**：团队共同学习
> **最后更新**：2026-11-10

> 三句话速判：**它能把"生成什么分子"和"长成什么构象"放进同一个 32 维局部隐变量里联合生成**，再用 120/180/300 次 trunk pass 的"隐空间推理"和可选的 checkpoint beam search 换取质量；它在 binder／小分子／酶骨架／DNA 四类任务上同时给出 pass rate 与 GPU-hour 吞吐，并附带 158/188 的湿实验命中率；但它的搜索收益是**有条件**的（普通判据下反而降低每小时产出），且它不是预测器、没有自带置信度头——评估必须外挂 AF3/AF2。

---

## 1. 学习背景与目标

### 1.1 为什么学它

- 它是本仓库「模型篇章」的第八篇。前七篇（AlphaFold3、Mol-1、Proteina、AnewOmni、Pallatom2、OpenDDE、ATOMICA）分别落在"预测／Cα 生成／潜空间生成／全原子扩散／共折叠／界面表征"几端，缺的正是**"部分隐变量 + 测试时计算"这一支**。
- 本仓库此前多次以"2026 年 `rfd4-proteina` 模型"指代它，但只有项目页与文档，没有一手技术报告。本轮仓库公开发布后，这条缺口补上了——**本篇的任务之一就是把旧判断订正回来**（见 §1.2、§1.3）。

本次学习想达成三个目标：

1. 核实身份与谱系：它是谁的后继，和 Proteina 是什么关系；
2. 核清四条在团队内流传的旧说法（32 维侧链 latent / 0.82 Å / Rosetta 监督 / beam search）；
3. 给出 VNAR 场景下的可用性判断（能当生成器？能当预测器？能做局部重设计？）。

### 1.2 出处与身份结论

| 项 | 内容 | 标记 |
|---|---|---|
| 仓库 | `https://github.com/RosettaCommons/RFD4-Proteina`，public，默认分支 `main`，License **BSD-3-Clause**，仓库创建 2026-09-25，约 572 star | 【核实】 |
| 项目页 | `rfd4-proteina.org`，与仓库 `homepage` 字段一致 | 【核实】 |
| 技术报告 | 《Biomolecular design through latent reasoning with RFD4-Proteina》，Nathaniel R. Corley、Kieran Didi 等，Baker Lab（UW IPD）× NVIDIA，2026 预印本，**PDF 直接放在仓库** `website/assets/paper/RFD4-Proteina.pdf`（54 MB，CC BY 4.0，正文标注 not peer reviewed） | 【核实】 |
| 权重 | README 指向 HuggingFace `nvidia/RFD4-Proteina`，约 1.7 GB（AE + flow + LoRA 打包） | 【论文自称】；HF 页面本次网络无法直连，**下载可用性【未核实】** |
| 仓库形态 | 1317 个 blob；`src/rfproteina/` 含 generation／nn／training／search／rewards／metrics／oracles／datasets／utils；另有 `configs/`、`tests/`、`docs/`、`examples/antibody_design/` | 【核实】 |

**身份结论：RosettaCommons 仓库就是 rfd4-proteina.org 的那个模型。** 三条独立证据：

1. 仓库 `homepage` 字段 = `https://rfd4-proteina.org`【核实】。
2. README 首页横幅与正文把项目页、作者列表、模型卡描述作为同一份材料呈现【核实】。
3. 仓库里直接托管了该模型的 54 MB 技术报告 PDF【核实】。

此前核查记录说"只有 model card、无技术报告"——**该记录已过时**。2026 年 9 月建仓后，报告随项目页一起发布了。

所以：之前记录的"`bb_ca` + `local_latents`、reward guided search、AF3/OF3 oracle refold、微调期 Rosetta 能量损失"这一组特征，**就是本仓库这一个模型**，不是巧合同名。此前把它们判给"2026 年的 rfd4-proteina 模型"是对的，当时缺的只是技术报告这一环。

### 1.3 谱系：Proteina → La-Proteina → RFD4-Proteina，中间不经过 "Proteina 血统"

这是本篇最容易写错的地方。仓库自己的参考文献把谱系讲得很干净：

| 模型 | 文献 | 生成空间 | 与本文关系 |
|---|---|---|---|
| **Proteina** | Geffner et al., ICLR 2025, arXiv 2503.00710（ref [23]） | Cα 单点流匹配，`x∈R^{3L}`，Atom37 仅 index 1 非零；**无侧链 latent、无解码器、无 Rosetta、无搜索** | **不是本文**；是同组的"纯结构流"前身，**不含 latent** |
| **La-Proteina** | Geffner et al., ICLR 2026, arXiv 2507.09466（ref [12]） | 原子级**部分隐变量**流匹配（partial latent） | **本文直接继承其"部分隐变量"表述**（论文原文：extending the partially latent formulation introduced in La-Proteina）|
| **Proteina-Complexa** | Didi et al., ICLR 2026, arXiv 2603.27950（ref [7]） | 全原子 binder 设计 + **test-time compute**（搜索） | **本文的主要对照基线**，也是"搜索"思想的直接来源 |
| **RFD4-Proteina** | 本文（2026 preprint） | 双模态：锚点坐标 + 32 维 local latent，联合去噪；内部 reasoning + 外部搜索 | 全部特征 |

一句话：**Proteina 是"Cα 骨架流"，RFD4-Proteina 是"残基级部分隐变量流 + 全原子解码"**。两者唯一共享的是"蛋白质生成"这个大帽子，以及"名字里都有 Proteina"这个历史包袱。任务里要求核实的四条旧说法（32 维侧链 latent / 0.82 Å / Rosetta 监督 / beam search），**全部属于本文（RFD4-Proteina），没有一条属于 Proteina**——逐条证据见 §5.3。

再往上，`RFD4` 前缀来自 RFdiffusion 线：RFD3 是"全原子生成但不定序列"（ref [6]，Science 2026, eaeh2982），RFD4-Proteina 补上"联合设计高质量序列"这一块。论文自己就是这么定位的：RFD3 支持多分子上下文的全原子生成，但**不为生成的结构设计高质量序列**，这限制了同时优化结合与催化【核实】。

### 1.4 许可证与可用性

- **代码**：BSD-3-Clause，可商用、可修改【核实】。
- **权重**：README 声明公开下载、无需申请；许可证随 HF 模型卡（本次未取到，【未核实】）。
- **技术报告**：CC BY 4.0，可转载【核实】。
- **依赖**：Python 3.11+、PyTorch、Hydra 配置体系；`uv sync` 安装；推理需 CUDA GPU（显存需求论文未给，【未核实】）。

---

## 2. 核心概念

| 术语 | 含义 |
| ---- | ---- |
| anchor（锚点）`x_i ∈ R³` | 每个 token 保留的一个全局位置，蛋白取 Cα、核酸取 C1′；小分子按原子逐个 token，锚点即原子坐标 |
| local latent `ℓ_i ∈ R^32` | 与锚点配对的 32 维隐变量，编码该残基的**身份与局部结构**（不是"侧链专属"） |
| local_latents | 代码／配置里这条隐变量流的名字；与 `bb_ca` 并列为两道去噪对象 |
| 自编码器（AE） | 130M 解码器，把 (锚点, latent) 解成 37 槽位 Atom37 坐标 + 147 类身份 + presence mask |
| presence mask | 37 维 0/1，表达 Atom37 槽位里哪些原子"存在"，用来覆盖修饰残基的离去原子 |
| latent reasoning | 论文对内部 recycling 的叫法：同一步去噪内反复跑 trunk，让隐状态多想几拍 |
| trunk pass | 一次主干前向；三档预算 low=120 / medium=180 / high=300，比"步数"更能标定算力 |
| checkpoint beam search | 外部搜索：在指定积分步分叉，用 reward 打分后剪枝 |
| clustered pass rate | 主指标：通过的设计按 Foldseek TM 0.5 聚类后，通过簇数 / 200 次尝试 |
| explorative modeling | 训练期在同一时间点加噪 K=4 个候选，只对损失最低者反传 |

**两层表示的整体图景**：生成阶段只在 (Cα 锚点, 32 维 latent) 上做去噪；全原子坐标只出现在**解码之后**。所以"all-atom"体现在 (a) 解码器覆盖全部重原子；(b) 条件端可接受全原子 motif/ligand 坐标——而生成端的自由度是 **3+32 维/残基**，不是 `3×N_atom`。这正是它能省算力的原因【推断，基于 S3.1+S3.2】。

---

## 3. 原理与推导

### 3.1 两层表示：锚点 + 局部隐变量

论文 Methods S3.1 原文定义（关键句，直译）：

> 每个 token 保留一个**锚点** `x_i ∈ R³` 和一个**隐变量** `ℓ_i ∈ R³²`。锚点保留该残基的全局位置，隐变量编码它的**身份与局部结构**。

代码侧完全对应：`configs/nn/local_latents_score_nn_160M_atom37.yaml` 里两道并行的去噪对象就是 `bb_ca`（3 维）和 `local_latents`（32 维）【代码／配置核实】。

- 蛋白质/核酸的**每个残基=一个 token**，锚点取 Cα（蛋白）或 C1′（核酸）。
- 小分子/配体**不是按残基切**，而是按原子逐个 tokenize，锚点取原子坐标【核实】。
- 因此序列长度 = 残基/配体原子数，变长体系天然支持（mask + 索引），无需 padding 成固定原子数。

### 3.2 `local_latents` 到底是什么：不是"侧链 latent"，是"残基 latent"

这是本篇对旧说法 #1 的修正核心。

- **维度 32**：Table S13 明确 "Latent dimension: 32 per token"；AE 配置文件字段 `latent_z_dim: 32`，HF tag 直接叫 `z32`【核实】【代码／配置核实】。
- **语义**：它同时编码**身份（化学种类）**和**局部构象（锚点相对原子坐标）**，不是只编码侧链。解码时由 130M 自编码器一次解出三件东西（S3.2）：
  1. **身份**：147 类（20 标准氨基酸 + 修饰氨基酸 + 糖 + 核苷酸 + 配体原子类型等）；
  2. **锚点相对坐标**：37 个残基特异槽位 × 3；
  3. **原子存在性掩码**：37 维 0/1。
- 所以准确说法是"**残基级 32 维局部隐变量**"，"侧链"只是它解码出来的一部分。把它叫"侧链 latent"会让人误以为序列是另一条通道给的——实际序列和侧链都从这 32 维里解出来。

### 3.3 解码器：130M 自编码器 + 37 槽位 + 存在性掩码

- 自编码器配置 `configs/nn_ae/nn_130m_unified.yaml`：12 层、768 隐藏，约 130M【代码／配置核实】。
- 原子表示沿用 **Atom37 残基特异槽位**：每个残基固定 37 个槽位，不同残基类型只用其中一部分（例如 Gly 只占少数槽位）。**变长原子数 = 37 槽位 + presence mask**，而不是动态原子列表【核实】。
- 三个独立解码头：残基身份、锚点相对原子坐标、**原子存在性**。存在性头专门用来表达"离去原子"（leaving atoms）的缺位，这是 37 槽位方案能覆盖修饰残基的关键【核实：S3.2.1】。
- 训练 AE 时砍掉了 pair track（比 flow 主干更轻），用 256-token 裁剪、全局 batch 512，约 **4,000 H200 GPU-h**【核实：S3.2.1 + S4.2】。
- 解码是**一步完成**（非自回归），所以"生成序列"不是逐残基采样，而是一次性从全序列 latent 解码。

可变原子数与非天然残基的处理：

| 情形 | 处理方式 | 标记 |
|---|---|---|
| 不同残基原子数不同 | 37 槽位 + presence head；不存在的槽位 mask 掉 | 【核实】 |
| 修饰氨基酸 / 非天然单体 | 身份类别扩到 147 类，纳入统一的 shared latent space | 【核实】 |
| 糖、DNA、RNA | 同一 AE（`unified_biomolecule`），不同 id 类别 | 【核实】 |
| 配体 | 逐原子 token，锚点=原子坐标，latent 编码原子类型与局部几何 | 【核实】 |
| 多链 / 复合物 | 单序列多 token，靠 chain index + relative position 编码，无独立链间 attention 层 | 【核实】 |

### 3.4 生成范式：双模态并行流匹配

- 对两个模态**各自**定义一条 flow：坐标 `bb_ca` 用一套噪声日程，latent `local_latents` 用另一套；同一个 denoiser 同时预测两者的速度场 `v`【代码／配置核实：`noise_schedules` 下 `bb_ca` 与 `local_latents` 分别配置】。
- 不是"先骨架后序列"的两阶段，而是**同一步里联合去噪**——这正是 La-Proteina 的"partial latent"思路：把难生成的连续原子坐标压进低维 latent，让 flow 在更光滑的流形上学【核实】。
- 训练目标是 flow matching 的速度回归；采样用 **TSR（time-shifted reverse）随机采样器**，配置名 `vf_tsr`【代码／配置核实】。

### 3.5 主干规格与采样预算

| 项 | 数值 | 标记 |
|---|---|---|
| 层数 | 14（默认 160M 配置）；12 层为 400M 覆盖版的另一组 | 【代码／配置核实】 |
| token 维度 | 768 | 【核实】 |
| pair 维度 | 256（400M 版 384） | 【代码／配置核实】 |
| attention heads | 12 | 【核实】 |
| 条件维度 | 256 | 【核实】 |
| 参数量 | 约 160M（配置名 `160M`；另有 `400M` override） | 【推断】——论文正文未给总参数量，只能由配置名与层宽反推 |
| 位置编码 | 局部 frames + relative position + chain index | 【核实】 |
| 输出头 | 速度场 v（双模态）+ 序列/motif 映射预测头 | 【核实】 |

> 注意：**没有 pLDDT/PAE 置信度头**。全仓库 `plddt` 只出现在 oracle 代码里（`metrics/oracles/af2_oracle.py`、`af3_oracle.py`），模型自身不输出置信度【代码／配置核实】。这是 §6 的硬边界之一。

**采样是 60 步，不是 1000 步**：默认 `nsteps: 60`、`sampler: vf_tsr`【代码／配置核实】。60 步是"积分步数"，不是总计算量——每一步内部还要做 1–6 次 trunk pass（见 3.6）。所以论文用"**trunk pass 总数**"而不是"步数"来标定算力，这点比多数论文诚实。另支持部分扩散：从第 35/60 步恢复噪声（noise level 0.7），用于局部重设计【代码／配置核实：`examples/antibody_design/partial.yaml`】。

### 3.6 内部 recycling：五个"thinking"档位，120/180/300 次 trunk pass

论文把这叫 **latent reasoning**——同一个去噪步里反复跑 trunk，让隐状态"想一想"再走下一步。

- 配置目录 `configs/recycle_strategy/{low,medium,high}/{front_loaded,midpeak,flat}.yaml`【核实】。
- 调度是"按步分段给 R_k 次 pass"，例如 `medium/front_loaded`：

```yaml
n_recycle_schedule: [[5, 6], [10, 5], [5, 4], [10, 3], [20, 2], [10, 1]]
```
  即前 5 步每步 6 次、接着 10 步每步 5 次……合计 60 步、**180 次 trunk pass**【代码／配置核实】。
- 三档总 pass 数：**low=120、medium=180、high=300**（论文正文与配置一致）【核实】。论文还给了 `thinking_low/medium/high` 这组名字，本质就是这三档。
- 收益单调但递减：蛋白 binding 的 clustered pass rate 随档位 7.7% → 9.7% → 10.6%；但吞吐同步下降（low≈2388 designs/GPU-h，high≈986）【核实】。

### 3.7 外部 checkpoint beam search：真实现了，还是文档里的规划？

**真实现了，有代码、有配置、有消融表。** 证据链：

1. 代码：`src/rfproteina/search/beam_search.py`，`class BeamSearchStrategy`，在指定的**积分步**（checkpoint）分叉，用 reward 打分后剪枝【代码／配置核实】。
2. 配置：`configs/inference_experiment/inference_beam_example.yaml`、`inference_beam_regression.yaml`、`ppi_novelty_of3_search.yaml`、`sm_binder_hbond_search.yaml`【核实】。
3. 文档：`docs/how-to/search.md` 给出完整 CLI 流程（生成 → 打分 → 保留 top-k → 从 checkpoint 续跑）【核实】。
4. 奖励：`src/rfproteina/rewards/reward_model.py` 把多个 metric 输出**加权、裁剪、归一化后求和**成标量 reward（`search.reward`）【代码／配置核实】。
5. 消融：Table S18/S22，见 §5.4。

所以"reward guided search"不是规划，是 shipped feature。**但**它的收益高度依赖判据，见 §5.4。

### 3.8 训练期的"隐空间推理"：探索式多候选

除推理期 recycling，训练时还做 **explorative modeling**（ref [24]）：在同一采样时间点独立加噪 **K=4** 个候选，只对"坐标 + latent 联合 flow-matching 损失最低"的那一个反传【核实：S4.2】。K=1 的开发版约 4,000 GPU-h，K=4 发布版约 16,000 GPU-h（成本随 K 线性增长）+ 2,000 GPU-h LoRA。这与推理期 beam search 是同一思想的两个训练面。

### 3.9 条件控制

| 条件 | 注入方式 | 必要性 | 标记 |
|---|---|---|---|
| 靶点坐标 | 绝对坐标（fixed）或仅内部坐标（internal coords） | binder 任务必需 | 【核实】 |
| hotspot 残基 | 逐残基 37 维指示向量 | 可选，用于指定表位 | 【核实】 |
| motif 位置 | `index_group` 索引 + 序列概率 | 酶骨架/motif 任务必需 | 【核实】 |
| 序列条件 | 147 类 one-hot / 概率分布；`cond_residue_type` 对生成位置强制置零以防泄漏 | 可选 | 【代码／配置核实】 |
| fold 类 | CATH code embedding（96 维） | 可选，用于拓扑约束 | 【核实】 |
| 对称性 | cyclic / dihedral 对称算子 | 可选 | 【核实】 |
| 部分扩散 | 指定步恢复噪声，冻结其余 | 局部重设计 | 【核实】 |
| ligand 构象 | 固定 / 共生成（两种模式） | 小分子任务必需其一 | 【核实】 |
| 靶点侧链柔性 | 允许 target sidechain 一起生成 | 可选 | 【核实】 |
| DNA 构象柔性 | 允许 DNA 双链构象可变 | 可选 | 【核实】 |
| 分类器自由引导 | `guidance_w` 权重 | 可选，提升条件遵从 | 【代码／配置核实】 |

**能量项不是条件，是"训练信号 + 搜索奖励"**——需要区分三件事：

1. **Rosetta 能量作为训练监督**（AE 微调期）：见 §5.3 第 3 条，**成立**。但它作用在**自编码器的解码器**上，不是作为 denoiser 的条件输入。
2. **TMol 能量作为搜索奖励**：S7.1.4 用可微能量（bond-angle/键长等）给 beam search 打分，属于推理期 steer【核实】。
3. **oracle refold 作为筛选**：AF3/AF2 refold 后按 RMSD/ipAE/pLDDT 判 pass，是**评估**不是生成条件【核实】。

把"能量项"当成模型的条件输入是常见误读——它没有走 cross-attention，而是走损失和奖励。

**条件消融（论文自报，可直接引用的因果证据）**

| 改动 | clustered pass rate 变化 | 标记 |
|---|---|---|
| ligand 构象由"固定"改为"与蛋白共生成" | 8.1% → **14.3%** | 【核实】 |
| 允许靶点侧链随设计改变构象 | 9.6% → **10.6%** | 【核实】 |
| 允许 DNA 构象柔性 | 2.0% → **3.5%** | 【核实】 |
| 只给靶点**内部坐标**、不给全局坐标系 | 10.6% → **11.8%** | 【核实】 |

这四行说明：**"让上下文一起动"和"去掉全局坐标系"都是实打实的增益**，不是锦上添花。做 VNAR 时，把靶点侧链放开、只给内部坐标，大概率是免费的质量提升。

**条件泄漏的处理**：配置里 `cond_residue_type` 与 `cond_a37coors_nm` 有一个明确开关——"在训练期把**生成位置的**真值特征置零，只给条件位置真值，防止模型抄答案"【代码／配置核实】。这是全原子生成模型里容易被忽略、但决定"条件控制是否真的有效"的细节。

另一个细节：motif 与生成 token 的对应不是靠坐标邻近，而是用 final single 表示算 correspondence score `m_ij`，再做**逐链 Hungarian 分配**（S3.1.5）【核实】。这意味着 motif 可以在空间上被"挪到别处"再匹配，不受绝对坐标束缚——做酶骨架重设计时这点很关键。

---

## 4. 训练数据构成与切分

### 4.1 数据源与规模

| 来源 | 用途 | 标记 |
|---|---|---|
| PDB 蛋白–蛋白界面（截至 **2024-01-01**） | binder 主训练集 | 【核实】 |
| AFDB 单体结构 | 单体覆盖 | 【核实】 |
| 物理方法自蒸馏数据 | 补足 PDB 稀缺化学型（酶/配体/核酸） | 【核实】 |
| 酶–配体、核酸复合物 | 扩展化学空间 | 【核实】 |
| 合成/生成数据 | 覆盖欠代表化学型 | 【核实】 |

- 结构截止日期 2024-01-01 是硬切分点【核实】；这样 2025–2026 年新解析的复合物可自然作为测试。
- 论文 Supplementary Table S1 给出各子集的混合比例（本篇不重复具体百分比，避免与表格数值不一致；取数请直接看 S1）【未核实具体数值】。

### 4.2 清洗与去重

- 论文正文未给明确的序列同一性阈值（如 30% identity）；AE 侧清洗细节在 S1/S3，本次未逐字符核对【未核实】。
- 可核实的部分在**评测侧**：通过的设计按 Foldseek TM-score 0.5 聚类后再计数（见 4.3）。

### 4.3 切分与泄漏控制

- **聚类切分**：评价时对通过的设计按 **Foldseek TM-score 0.5** 聚类，报"**distinct passing clusters / 200 attempts**"，即 clustered pass rate【核实】。
- **测试集隔离**：蛋白 binder benchmark 是 Proteina-Complexa 与 RFdiffusion3 benchmark 的并集（24 targets），小分子用 STUDIO-170 子集（77 targets，来自 DISCO），AME 用 37 targets，DNA 用 RFdiffusion3 预印本的 3 个 target【核实】。
- **相似性控制**：S8.4.1 专门做"降低 binder–target 结构相似度"的对照，防止模型靠记忆靶点取胜【核实】。

### 4.4 覆盖与偏置

- 数据配方对"稀缺化学型"是显式补的（自蒸馏 + 合成），说明作者知道 PDB 的化学覆盖不足；但**抗体/纳米抗体占比未披露**，VNAR 相关的覆盖无法前置判断【未核实】。
- 小分子、核酸、DNA 各自有独立评测集（见 4.3），至少在这三类上有可观测的覆盖，而不是只报蛋白 binder。

### 4.5 可获取性与许可

- 代码 BSD-3-Clause；技术报告 CC BY 4.0；权重经 HF `nvidia/RFD4-Proteina` 分发（可用性【未核实】）。
- **训练数据本身不随仓库发布**，只有处理管线与配置；因此"同数据复现"不可行，只能"同代码换数据"。

### 4.6 算力（训练）

| 阶段 | 算力 | 标记 |
|---|---|---|
| 自编码器 | 约 **4,000 H200 GPU-h** | 【核实】 |
| Flow 预训练 | 约 **16,000 H200 GPU-h**（64 节点 × 8 = 512×H200） | 【核实】 |
| LoRA 微调 | 约 **2,000 H200 GPU-h** | 【核实】 |
| 消融（Table S11 参照） | 约 1,200 H200 GPU-h | 【核实】 |

---

## 5. 宣称与可验证性

### 5.1 主表：作者宣称 → 证据类型 → 核查状态

| # | 作者宣称 | 证据类型 | 核查状态 |
|---|---|---|---|
| 1 | 蛋白 binder：clustered pass rate **10.6%**（high thinking），高于 BindCraft 5.9% | Fig 2a + Table S11 | 【核实】数字一致 |
| 2 | 小分子 binder：**14.3%**，约为 RFD3（2.4%）的 6.1 倍 | Fig 2b | 【核实】 |
| 3 | AME 酶骨架：**12.1%**，约为 AlphaProtein-novo（5.7%）的 2.1 倍 | Fig 2c | 【核实】 |
| 4 | DNA binder：**3.5%**，高于 BoltzGen 0.8%、RFD3 0.5% | Fig 2d | 【核实】 |
| 5 | 计入 AF3 评估后，每小时通过簇仍高 5.0×/10.4×/6.0×/5.5× | Fig 2 右栏 + Table S29 | 【核实】 |
| 6 | 侧链打包质量：PackBench MolProbity **1.70**（vs LigandMPNN 1.99），diversity 0.31→1.20 Å | Fig 3a + S10.1 | 【核实】 |
| 7 | Rosetta 能量微调把每残基能量 **−0.18 → −1.25 kcal/mol**，bond-angle 偏差降 **62%** | S4.3.1 + Fig 3 | 【核实】 |
| 8 | 湿实验命中 **158/188** | 实验节 | 【核实】（但属"作者自测"，无第三方复现）|
| 9 | 钙调蛋白 binder：12/18 钙依赖，KD 69 pM–37 nM | 实验节 | 【核实】 |
| 10 | pMHC 结合：340 / 710 / 1600 pM 三档 | 实验节 | 【核实】 |
| 11 | TrkA 物种/旁系特异性：抗 mTrkA 与 TrkB/C 无交叉结合 | Fig 4e | 【核实】 |
| 12 | 搜索能提升"通过簇/GPU-h" | Table S18/S22 | **部分成立**：仅在特定判据下成立，普通判据反而下降（见 §5.4）【核实】 |

### 5.2 证据强度分级

- **强**（有一手数值 + 可复算口径）：1–7、11。benchmark 口径写得很细（200 attempts/target、Foldseek 0.5、含失败）。
- **中**（自测、单实验室、无独立复现）：8–10。158/188 是作者自己的湿实验，基线对比也是作者自己跑的。
- **弱/有条件**：12。搜索的收益被论文自己诚实地画成"条件性收益"。

### 5.3 四条旧说法的逐条核实（本篇核心）

| 旧说法 | 结论 | 一手依据 |
|---|---|---|
| **① 32 维侧链 latent** | **维度成立，语义需修正** | Table S13 "Latent dimension: 32 per token"；`configs/nn_ae/nn_130m_unified.yaml: latent_z_dim: 32`；HF tag `z32`。但它是**残基级** latent（身份 + 全部非锚点原子），不是"侧链专属"【核实】 |
| **② 0.82 Å 侧链打包误差（PackBench）** | **不成立 / 查无此数** | 全文与全仓库检索 `0.82`，只命中 PLACER 的 AUROC 0.82，**没有任何"0.82 Å 侧链 RMSD"**。PackBench 实际口径是 **MolProbity 质量分 1.70 vs 1.99** 与 **diversity（平均成对侧链 RMSD）0.31→1.20 Å**，不是"0.82 Å 误差"【核实：`PackBench` 段 + Fig 3a；`grep -r "0.82"` 无命中】 |
| **③ Rosetta 能量监督** | **成立** | S4.3.1：AE 微调加入 Rosetta 能量项（经 TMol 可微化），每残基能量 −0.18→−1.25 kcal/mol，bond-angle 偏差 −62%【核实】。注意：作用在**解码器**，不是 denoiser 条件 |
| **④ beam search / reward guided search** | **成立（代码级实现）** | `src/rfproteina/search/beam_search.py`（`BeamSearchStrategy`）+ `src/rfproteina/rewards/reward_model.py` + `configs/inference_experiment/inference_beam_example.yaml` + `docs/how-to/search.md` + Table S18/S22 消融【代码／配置核实】。**但收益有条件**（§5.4） |

> 一句话：四条里 **三条成立、一条（0.82 Å）是把 PLACER 的 AUROC 误当成了侧链 RMSD**，且"32 维侧链 latent"应改称"32 维残基 latent"。

### 5.4 搜索的真实收益：不是免费的午餐

**Table S18（peptide–DNA 切键酶单靶点，三种策略各固定 24 GPU-h）**

| 策略 | 完成设计数 | 通过数 | 唯一通过设计 | 唯一通过/GPU-h |
|---|---|---|---|---|
| Naive（跑满轨迹再筛） | 23,316 | 3 | 3 | 0.13 |
| Constant search（固定判据） | 10,676 | 0 | 0 | 0.00 |
| **Temporal search**（步 13/37 分叉 bf=2，步 45 bf=4，beam width 1）| 4,057 | 35 | 9 | **0.38** |

→ 搜索把"唯一通过/GPU-h"提到 **3×**，但**生成数少了 5.7 倍**；而同一套骨架换成"固定判据"搜索直接归零（0 通过）。注意 Table S18 是**单靶点**（正文称 "our example bond-cleaving enzyme"，补充材料称 "peptide–DNA target"，同一任务），不是 6 靶点【核实：Table S18 与 §S7.1.1】。

**Table S22（6 个 MCSA 酶骨架靶点：M0500/M0375/M0092/M0078/M0870/M0552）**——搜索换一个判据就翻车：

| 判据 | Naive | Temporal | Constant |
|---|---|---|---|
| Regular | **41.00** | 25.13 | 26.63 |
| Burial >85% | 7.38 | **9.04** | 6.88 |

→ **普通判据下，naive 采样每小时产出最高（41.0），搜索反而降到 25.1**；只有在 burial>85% 这种更严判据下搜索才 +22.6%【核实】。

另外，蛋白 binder 高 thinking 档：只算生成是 **120.7 clusters/GPU-h**，加上搜索跌到 **8.8**【核实】。

**结论**：搜索是"用吞吐换质量"的旋钮，且只在判据严格时才划算。任何"搜索一定能提升效率"的转述都是错的。

### 5.5 评测协议与成本口径

| 任务 | 规模 | Pass 判据 | 标记 |
|---|---|---|---|
| 蛋白 binder | 24 targets | AF3 refold；binder RMSD ≤ 阈值、ipAE、pLDDT | 【核实】 |
| 小分子 binder | 77 targets（STUDIO-170 子集） | AF3 refold + 口袋一致性 | 【核实】 |
| AME 酶骨架 | 37 targets | motif all-atom RMSD ≤1.5 Å + 全局 RMSD ≤2.0 Å | 【核实】 |
| DNA binder | 3 targets | AF3 refold + 界面指标 | 【核实】 |
| PackBench | 114 targets × 8 样本 | MolProbity（质量）× diversity（多样性） | 【核实】 |
| 逆折叠 | 156 native backbones + 340 RFD3 enzyme backbones | 序列恢复 / AF3 refold pass | 【核实】 |

- **主指标**：clustered pass rate = 通过的结构簇数 / 200 次尝试（含失败），跨 target 平均；簇用 Foldseek TM-score 0.5【核实】。
- **效率指标**：designs/GPU-hour、passing clusters/GPU-hour（**显式计入 AF3 评估的 GPU 时间**）【核实】。
- **基线**：BindCraft2、BindCraft、BoltzGen、ESMFold2、Genie3、ODesign、RFD3、Protein-Hunter、Proteina-Complexa【核实】。

**基线数值（clustered pass rate）**

| 任务 | RFD4-Proteina（high） | 最强基线 | pass rate 倍数 | 吞吐倍数 |
|---|---|---|---|---|
| 蛋白 binder | 10.6% | BindCraft 5.9% | 1.8× | **42×** |
| 小分子 binder | 14.3% | RFD3 2.4% | 6.1× | 3.3×（77/77 靶点 vs 64/77）|
| AME 酶骨架 | 12.1% | AlphaProtein-novo 5.7% | 2.1× | 5.9× |
| DNA binder | 3.5% | BoltzGen 0.8% / RFD3 0.5% | 4.4× / 7× | — |

（数值来自 Fig 2 与正文；【核实】。注意：这些数都是"**生成时不用结构预测器反馈**"的前提下取得的。）

**成本口径是否把 oracle refold 算进去？**

- **算进去了**，而且分得很细：Fig 2 右栏的"mean target-level passing clusters per GPU-hour, **including downstream AF3 evaluation**"【核实】。
- 但**两套口径并存**：一套只算生成（120.7 clusters/GPU-h），一套含 AF3（8.8）。论文没有把两者混为一谈，这点值得学【核实】。
- AF3 单次 refold 的绝对开销：Table S29 给了按 GPU 折算的对照，但未给"每设计秒数"的细表【未核实绝对值】。
- **显存占用**：论文与配置均未给 per-sample VRAM【未核实】。

**可复现所需材料**：代码（BSD-3-Clause）+ 配置 + 权重（HF，可用性【未核实】）+ 技术报告。缺"训练数据本体"与"单样本显存"，故只能做推理复现，不能做训练复现。

---

## 6. 实践与应用

### 6.1 结论先行

| 用途 | 能不能用 | 硬边界 |
|---|---|---|
| **de novo VNAR 生成** | **能，且是该模型最合适的用法之一** | 无轻链，属单域；需自己给靶点/hotspot；质量靠 AF3 判 |
| **CDR 局部重设计** | **能** | 用 partial diffusion + framework 内部坐标固定；CDR 长度需采样 |
| **预测器** | **不能** | 无 pLDDT/PAE 头，不输出置信度；必须外挂 AF3/AF2 |
| **评估器** | **半能** | 只能当"生成候选器"，评估仍靠 AF3；可借其 oracle/reward 管线做打分基础设施 |
| **序列设计（逆折叠）** | **能（S10.2）** | 156/340 backbone 上有 pass rate，不是 SOTA 逆折叠器 |

### 6.2 VNAR 专列：四问四答

**Q1：能当 de novo 生成器吗？**
能。仓库直接带 `examples/antibody_design/vhh/`：4 个框架 × 2 个靶点，用 `CDR1/2/3` 作 scaffold、framework 内部坐标作条件、CDR 长度作采样维度【代码／配置核实】。VNAR 与 VHH 同为**单域、无轻链**，接口几乎可以直接套用；差别只在 VNAR 的 CDR1 更长、framework 更偏（camelid/shark 的 FR2 替换），需要**用自己的 VNAR 框架重做内部坐标条件**。

**Q2：能当预测器/评估器吗？**
**不能当预测器**。无置信度头（§3.5 已核）。想拿它当评估器，只能间接：把它当"条件生成器"产出候选，再用 AF3 打分——这正是它自己 benchmark 的做法，等于没有独立评估能力。

**Q3：能做局部重设计吗？**
能，而且是它相对 AF3/RFdiffusion 系列的差异点：`partial.yaml` 从第 35/60 步加噪，冻结非 CDR 区域，等价于"固定 framework、只重画 CDR"【代码／配置核实】。对 VNAR 的意义：在已有一株 VNAR 上做 affinity maturation / 人源化，不必从头生成。

**Q4：de novo 生成的硬边界在哪？**
- **表位控制靠 hotspot**，不是靠自然语言；hotspot 给错，生成的 binder 会跑到别处。
- **CDR 长度分布**来自训练集，长 CDR（VNAR 常见）可能欠采样。
- **糖基化/环化/非天然修饰**：147 类里有修饰氨基酸，但 VNAR 特有的二硫键模式未被显式建模【推断】。
- **没有湿实验闭环**：生成完必须走 AF3 → 表达 → BLI/ELISA，模型不提供任何"可表达性"预测。

### 6.3 可直接搬的三件基础设施

1. **`reward_model.py` 的加权-裁剪-归一化聚合器**：任何"多指标 → 标量奖励"的搜索都能复用，逻辑干净、有边界检查。
2. **`beam_search.py` 的 checkpoint 分叉协议**：把"在哪些步分叉、bf 多大"做成配置，而不是写死，这套抽象可以直接移植到别的 flow 模型。
3. **双模态噪声日程分离**（`bb_ca` 与 `local_latents` 各一套）：如果我们要做"骨架 + 侧链 latent"的联合生成，这是现成的工程范式。

### 6.4 不该学的两点

- **不要把它当"全原子生成"来宣传**：生成端是 3+32 维/残基，全原子只发生在解码后。
- **不要默认搜索一定提效**：Table S22 的反例说明，判据不严时搜索是负收益。

### 6.5 工程约束

- Python 3.11+、PyTorch、Hydra；`uv sync` 安装；推理需 CUDA GPU。
- 默认 160M 主干 + 130M AE；400M override 另有配置。**发布权重具体对应哪一档未核实**（见 §7 Q1）。
- 单样本显存未披露；做吞吐规划时只能按"trunk pass 总数 × 60 步"的相对量估算【推断】。

---

## 7. 疑问与讨论

- **Q1：160M 还是 400M 是"发布版"？**
  - **A**：配置里默认 160M，另有 400M override；HF 权重对应哪一个，本次因网络无法直连未核实【未核实】。

- **Q2：AE 的 130M 是否含 encoder+decoder 全部参数？**
  - **A**：配置名 `nn_130m_unified` 与 Table S13 的 12 层/768 一致，但论文未给逐模块参数量【推断】。

- **Q3：训练数据去重阈值是多少？**
  - **A**：正文未给明确 identity 阈值，只在 benchmark 侧给了 Foldseek 0.5 聚类【未核实】。

- **Q4：OF3 oracle 是否随仓库发布？**
  - **A**：代码里只有 `af2_oracle.py`/`af3_oracle.py`，论文/文档提到 OF3 refold，但仓库未见独立 `of3` 模块【核实：目录内无 of3 文件】。

- **Q5："147 类"身份空间的具体构成？**
  - **A**：论文说含标准/修饰氨基酸、糖、核苷酸、配体原子类型，但未给完整 147 项清单【未核实】。

- **Q6：它和 Proteina 到底是什么关系？名字这么像。**
  - **A**：没有技术血缘。Proteina 是 Cα 单点流（`x∈R^{3L}`，Atom37 只有 index 1 非零），无 latent、无解码器、无搜索；RFD4-Proteina 继承的是 **La-Proteina 的 partial latent**，主基线是 **Proteina-Complexa**。四条旧说法全部属于 RFD4-Proteina，没有一条属于 Proteina（§1.3、§5.3）。

- **Q7：既然有 beam search，为什么还说它"不含评估能力"？**
  - **A**：搜索用的是外部 oracle（AF3/AF2 refold、TMol 能量）打分，模型自身没有置信度头。把 oracle 拿掉，它就没有任何自评能力；而 oracle 正是别人家的模型。

- **Q8：搜索既然会降吞吐，为什么作者还做？**
  - **A**：因为判据严格时它确实提效（burial>85% 下 +22.6%）。它的定位是"当你能定义严格判据时，用算力换质量"，不是"默认开启的加速器"。

---

## 8. 学习心得

- **"local_latents" 这个名字是全篇的钥匙**：local = 相对锚点，latents = 身份+几何的压缩。理解了这一点，"32 维侧链 latent"的旧说法自然就修正成"32 维残基 latent"。
- **它诚实得少见**：把搜索的负收益（Table S22）也放进正文，而不是只挑好看的数。这反而让它的正面结论更可信。
- **它和 Proteina 的关系是"同名不同宗"的教科书案例**：Proteina = Cα 单点流；La-Proteina = 部分隐变量；Proteina-Complexa = 全原子 + 搜索；RFD4-Proteina = 这条链的工程合流。写综述时按这条链分，比按名字分安全得多。
- **"0.82 Å"这类数字是最危险的一类错误**：它来自另一个指标（PLACER AUROC），却因为"看起来像 RMSD"而被传成了侧链误差。核查这类数字，唯一可靠的办法是回到一手表格逐列对。
- **"有代码"和"有收益"是两件事**：beam search 的代码、配置、文档、消融全都在，但它是不是提效，要看你用什么判据——这是本仓库"机制存在 ≠ 计算扩展收益"这条结论的第二个实例。

---

## 9. 参考资料

**一手来源（本篇全部数字的出处）**

1. Corley N.R., Didi K., et al. *Biomolecular design through latent reasoning with RFD4-Proteina.* 2026 preprint（仓库 `website/assets/paper/RFD4-Proteina.pdf`，CC BY 4.0，not peer reviewed）。
2. 仓库 `https://github.com/RosettaCommons/RFD4-Proteina`：`README.md`、`configs/nn/local_latents_score_nn_160M_atom37.yaml`、`configs/nn_ae/nn_130m_unified.yaml`、`configs/generation/inference_base.yaml`、`configs/recycle_strategy/*`、`configs/inference_experiment/*`、`src/rfproteina/search/beam_search.py`、`src/rfproteina/rewards/reward_model.py`、`docs/how-to/search.md`、`docs/concepts/search.md`、`examples/antibody_design/*`。
3. 论文内引：RFdiffusion3（ref [6], Science 2026, eaeh2982）；La-Proteina（ref [12], ICLR 2026, arXiv 2507.09466）；Proteina-Complexa（ref [7], ICLR 2026, arXiv 2603.27950）；Proteina（ref [23], ICLR 2025, arXiv 2503.00710）；PackBench（ref [25]）。
4. HuggingFace 权重：`nvidia/RFD4-Proteina`（约 1.7 GB；本次未能直连，可用性【未核实】）。

**同线对照**

- 本仓库《Proteina 模型笔记》（2026-11-04）——四条旧说法的最初来源与"归于 2026 模型"的判定。
- 本仓库《OpenDDE / ATOMICA / Pallatom2 模型笔记》（2026-11-06）——模板与证据标记规范。
- 本仓库专题 [01 表示与生成范式](../../专题/01_全原子表示与生成范式.md)、[02 训练数据](../../专题/02_训练数据构成与切分.md)、[03 评测与复现](../../专题/03_评测体系与可复现性.md)、[04 抗体与 VNAR](../../专题/04_抗体与VNAR可迁移点.md)。

---

*核查日期：2026-11-10。仓库快照：`git clone --depth 1`，HEAD `68058693cb88762a8aaf806571e13b291ce6318a`，1317 个 blob。所有【核实】项均可回溯到具体文件或论文页码/表格。*
