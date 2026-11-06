---
model: Pallatom2
aliases: [Pallatom2, Pallatom-2]  # 同系列但非同模型：一代 Pallatom（bioRxiv 2024.08.16.608235，代码与权重公开）与 Pallatom-Ligand（ICLR 2026 投稿版）
family: 全原子蛋白结构生成（Atom14 / 原子坐标扩散）
granularity: Atom14（每残基 14 个重原子槽位，含虚拟原子补齐；不含氢）
backbone: AtomPairFormer(384ch) + Atom-to-Atom Transformer + residue-guided sparse all-atom attention（65.9M）
generative: 迭代去噪（v1 核实为 EDM 扩散；自由度 = L×14 笛卡尔原子坐标；v2 schedule／步数未披露）
conditioning: hotspot/coldspot/binderspot/flexspot/unindex + motif/redesign/norepack/slevel/sse/seq_cond/hb_bond；靶标–表位；D-form
data_scope: AFDB + TED（内部处理版，非官方 TED）；保留原始序列，无序列重设计；规模／切分未披露
status: 已完成（技术报告 2026-11-06，58 页；v2 代码与权重未公开；v1 代码公开）
verified_at: 2026-11-06
---

# Pallatom2

> **状态**：已完成
> **记录人**：团队共同学习
> **最后更新**：2026-11-06
> **取证基线**：`levinthal/Pallatom2-techreport` 的 `Pallatom2-Technical-Report.pdf`（58 页，2026-11-06 版）＋ `levinthal/Pallatom` v1 公开代码。详细取证记录留存于项目底稿（未纳入本仓库）。
> **证据标记**：【核实】＝有可指认出处；【论文自称】＝作者文字主张；【代码／配置核实】＝从公开代码／默认参数读出；【推断】＝由已知事实外推；【未核实】＝找不到来源或无法验证。

---

## 1. 学习背景与目标

**一句话**：Pallatom2 是一个 65.9M 参数的**全重原子（Atom14）笛卡尔坐标扩散生成器**，把「backbone ＋ 侧链」放进同一个 `[L,14,3]` 状态里从噪声迭代去噪，再用 12 种条件接口把生成结果推向「指定表位结合／指定二级结构／指定侧链几何」的设计意图；它在 9 个靶标的 in-silico binder 筛选和 10 个靶标的 BLI 湿实验上给出了同类工作中较完整的一条证据链。【核实】p.1、p.5、p.31

本篇章是全原子模型专题的第 20 个条目（[01 专题 §6](../../专题/01_全原子表示与生成范式.md) 覆盖矩阵加入后共 20 行）。学习目标有三：① 判明它到底是不是「逐原子生成」；② 判明它的跨模型排名能不能直接引用；③ 判明它对 VNAR／单域抗体工程有哪些可迁移点与禁区。

### 1.1 真实仓库与版本（先纠正路径）

- 起始 URL `github.com/levinthal/Pallatom2-techreport` **就是真实仓库**，且它只是**报告仓**，不是代码仓：根目录仅 `Pallatom2-Technical-Report.pdf`、`README.md`、`assets/`。【核实】
- 最新技术报告 = `Pallatom2-Technical-Report.pdf`，50,351,453 B，58 页；封面日期 "November 6, 2026"；PDF metadata `CreationDate = 2026-11-06 04:05:25 UTC`；仓库最新 commit `b0493a0` "Update technical report with smaller PDF"（2026-11-06T05:12:58Z）。【核实】
- 仓库**没有** `levinthal/Pallatom2` 代码仓；v2 代码与权重未公开。【核实】
- 前作 `levinthal/Pallatom` 公开了代码、权重（`params_Pallatom.npz`，约 71 MB）与数据管线 `db_scripts/`，对应 bioRxiv 2024.08.16.608235（JAX 0.3.25 / Python 3.7）。【核实】
- 同系列还有 `levinthal/Pallatom-Ligand`（ICLR 2026 投稿版，向小分子／配体原子扩展）；是否接收**未核实**。**一代 Pallatom 与 Pallatom-Ligand 属于同系列工作，不与 Pallatom2 混为同一模型。**
- 作者单位为 Hangzhou Levinthal Biotechnology Co., Ltd.（杭州莱文塔尔生物）。【核实】

> 结论：本篇章的「代码核实」几乎全部来自 **v1**，v2 只能靠报告文字。凡涉及采样器细节，必须把「报告怎么说」和「v1 代码怎么做」并列，不能合并成一句。

### 1.2 一代 → 二代改了什么

| 维度 | v1 Pallatom | v2 Pallatom2 | 证据 |
|---|---|---|---|
| 训练语料 | AFDB（`pipeline.py` 输出文件名 `AFDB_xxxxx_yyyy.pdb`，编码即证明） | AFDB + TED（内部处理版），保留原始序列，明确不用 ProteinMPNN 重设计 | v1 `db_scripts/pipeline.py:144`；p.29–30 |
| 数据过滤 | 50–128 aa、Rg ≤ 20、packing density ≥ 0.30、SSE ≥ 4、avg degree ≥ 20、loop content ≤ 0.5、loop length ≤ 15；foldseek 去重 TM 0.8 / coverage 0.9 / e-value 0.001；final cluster TM 0.6 | 报告未披露任何阈值 | v1 `pipeline.py:180–201`；p.29–30 |
| 任务范围 | 短域单体（默认 ≤128 aa 训练域） | 单体 500–1,600 aa、多域 2–6 域、同源多聚体 C3–C10 / 600–2,000 aa | p.1、p.10–13 |
| 表征 | Atom14，导出 atom37；EDM 全原子坐标 | Atom14 贯穿 + residue-guided sparse all-atom attention，AtomPairFormer 384 通道 | v1 `sampling.py:69,80`；p.5 |
| 条件系统 | 代码中未见统一条件接口 | 5 种界面条件 + 7 种结构条件 | p.31 Tab. S1/S2 |
| 规模 | 未披露 | 65.9M | p.1 |
| 评测 | 未见完整 benchmark | AF3 refold designability + binder screen + BLI 湿实验 | p.9–25 |
| 开放性 | 代码＋权重＋数据管线全开 | 仅报告 | 仓库现状 |

**关键判断**：v2 的「升级」主要不是换生成范式，而是**把 v1 的原子坐标扩散扩展到长链／多域／多聚体，并补上条件接口与湿实验闭环**。原子层面的状态表示（Atom14、无氢、Cartesian）两代一致。【推断，依据 v1 代码 + v2 p.5】

### 1.3 与既有 19 个全原子模型的定位差异

按 [01 专题 §6](../../专题/01_全原子表示与生成范式.md) 覆盖矩阵中既有的 19 个条目分组对照（本篇章为该矩阵新增的第 20 行）：

| 组 | 代表 | 表示层核心 | 与 Pallatom2 的关系 |
|---|---|---|---|
| AF3 系（预测器） | AlphaFold3 / Boltz-1/2 / Chai-1 / Protenix / OpenFold3 | MSA + template → pair/atom 表征 → **单一结构**；Atom14/atom37 常量同源 | Pallatom2 借用了 AF 的 Atom14 常量与 atom transformer 思路，但**删掉 MSA/template/evoformer**，输入是噪声而不是序列；AF3 在本文里只当 **refolder/评估器**，不是被排名的基线 |【核实】p.5、p.9；【推断】架构血缘 |
| 全原子扩散／流（frame 系） | RFdiffusion-AA(RFAA) / RFD4-Proteina / Chroma / Proteina | 以 residue frame + torsion / rigid-body backbone 作中间变量，再展开到原子 | Pallatom2 **不做 frame/torsion 分解**，直接在 L×14 笛卡尔坐标上扩散；好处是侧链与 backbone 天然同步，代价是缺刚性约束、长链更吃注意力设计 |【推断，基于两者公开表示层描述】 |
| 部分 latent 系 | La-Proteina | per-residue latent + 原子解码，latent 上做 flow | Pallatom2 **无 latent**，状态即坐标；因此不存在「latent→构象」的解码不确定性，但也失去 latent 压缩带来的长链可扩展性 |【推断】p.4 无 latent 表述 |
| 复合物／抗体专用 | Proteina-Complexa / IgGM / AntiBMPNN / AbMPNN / BoltzGen / APM / AnewOmni | 针对 binder–target 或抗体配对做专门表征 | Pallatom2 用**统一 Atom14 + 条件接口**覆盖单体与 binder，不设专门抗体链配对模块；对抗体场景属「通用条件的垂类应用」，不是原生抗体模型 |【核实】p.6–8；【推断】 |
| 其他 | ESM3 / SeedProteo / TorchCraft | 序列-结构联合 token / 不同骨架 | 与 Pallatom2 的「坐标优先、序列为副头」路线相反或部分相反 |【推断】 |

**它唯一真正稀缺的位置**：在「**无 frame、无 latent、无 MSA 的纯笛卡尔全重原子扩散**」这一格里，同时给出**长链到 1,600 aa、多聚体到 C10、以及湿实验亲和力**的完整证据链。RFAA 有原子级但以 frame 为主；Proteina/La-Proteina 有规模但表示层不是纯坐标；AF3 系不是生成器。Pallatom2 的差异化不在「全原子」这个词，而在**表示层选择 + 条件粒度 + 湿实验闭环**三件事同时成立。【推断】

---

## 2. 核心概念

| 术语 | 含义 |
|---|---|
| Atom14 | 每残基固定 14 个重原子槽位，坐标张量 `[L,14,3]`；覆盖 39 种氨基酸（20 L + 19 D），重原子不足 14 个时用虚拟原子补齐 |
| all heavy atoms | 本工作的「all-atom」准确含义；**不含氢**，无质子化／互变异构状态 |
| 虚拟原子（virtual atom） | 补齐槽位的占位原子，坐标通常与真实原子重合，因此 14×L 个点不等于 14×L 个独立自由度 |
| residue-guided sparse all-atom attention | 先由 residue-level 相关性构建残基关系图，再由该图决定哪些原子对参与精细注意力，把原子级注意力从 O(N²) 压到残基尺度 |
| partial diffusion | 对已有结构做部分加噪（t=0.2／0.3）再降噪，用于局部重设计与亲和力成熟 |
| `norepack` / `redesign` / `unindex` | 结构条件：锁住指定侧链构象／指定可重设计区域／给定侧链几何 |
| `min-ipAE` / `ptm-binder` | binder screen 的两项指标：跨链对齐不确定性的最小值／binder 单链折叠置信度 |
| designability | 生成序列 → AF3 refold → all-atom RMSD ≤2.0 Å 且 mean pLDDT ≥80.0 的通过率 |
| apparent KD | BLI 单／多浓度拟合得到的表观解离常数，非热力学 KD |
| D-form | 非天然 D-氨基酸手性版本；本文用「AF3 预测 L-form → 手工镜像坐标」实现，不是模型原生手性标签 |

### 2.1 表征粒度与表示层

- 每个残基固定 14 个 atom slot，坐标张量 `[L, 14, 3]`；覆盖 **39 种氨基酸**（20 种 L + 19 种 D）。【核实】p.5 §2.1.2
- 因此「all-atom」在本工作中的准确含义是 **all heavy atoms**：**不含氢**。全文检索无 hydrogen／质子化相关表述。【核实（缺失即证据）】
- v1 代码同构：`atom_fix_size = fix_size*14`，扩散张量形状 `(atom_fix_size, 3)`；最终用 AF 的 `atom14_to_atom37` 导出。【代码／配置核实】v1 `sampling.py:69,80`、`ref_features.py:153`

**虚拟原子带来的两个读表注意点**：

1. 虚拟原子坐标通常被放在真实原子上（重合），因此「14×L 个点」不等于「14×L 个独立自由度」；有效自由度按残基类型不同（Gly 5、Ala 5、Trp 14 等）。【推断，依据 Atom14 常量的 `atom14_atom_exists` 掩码】
2. 任何「all-atom RMSD」若把虚拟原子计入，会人为压低误差；**报告未说明 RMSD 的原子集合与掩码方式。【未核实】**p.9 仅给阈值。

### 2.2 参考特征不等于生成状态（重要）

v1 的参考几何来自 `residues_sdf/<RES>_ideal.sdf`，以 CA 为原点；而 `get_entangled_ref_features` **只用 ALA 的理想几何**再 tile 到所有残基。【代码／配置核实】v1 `ref_features.py:35,108`

这意味着：

- `ref_pos / ref_mask / ref_element / ref_atom_name_chars` 是**原子身份与理想几何的锚**，用于让网络知道「第 k 个槽位是什么原子」；
- **真正被扩散、被更新的仍是 `[L,14,3]` 的绝对坐标**，不是这套 ALA 模板；
- 读报告时不要把 "reference frame / ideal geometry" 误读成「生成被约束在 ALA 构象」。【推断】

---

## 3. 原理与推导

### 3.1 主干：AtomPairFormer + 原子-原子 Transformer

- pair 表征 **384 通道**，坐标敏感：同时编码原子间**距离、方向、化学键连、手性**。【核实】p.5 §2.1.3
- **residue-guided sparse all-atom attention**：先由 residue-level 相关性构建 **residue relationship graph**，再由该图决定哪些原子对参与精细注意力；据此把原子级注意力从 O(N²) 压到可控。【核实】p.5、p.26
- residue-level 与 atom-level 表征**双向交换**：全局折叠与局部原子细节在同一主干里协调，而不是「先 backbone 后 side-chain」两段式。【核实】p.5
- 训练技巧：recycling + self-conditioning。【核实】p.6
- 规模：65.9M；对比 RFD4-Proteina 435.6M、BoltzGen 496M、BindCraft(AF2) 93M，分别约为其 15.1% / 13.3% / 70.9%。【核实】p.32 Fig. S2

**为什么这个设计能撑到 1,600 aa**：注意力只在「残基图选出的原子对」上做，成本随残基数而非原子数平方增长；这是它相对「稠密全原子注意力」的核心工程取舍。【推断，依据 p.5 + p.26 的表述】

### 3.2 生成范式：报告原话与 v1 代码并列

报告原话（逐字）：

> "…… iterative denoising … starting from an initial noise state … Successive sampling steps … update the coordinates as the noise level decreases."【核实】p.4

报告全文**没有**出现：`diffusion` / `flow matching` / `noise schedule` / `sampling steps = N` / `latent` / `token` 这些词（检索无命中）。【核实】因此 **v2 的采样器机制未披露**：是否仍为 EDM、步数、噪声 schedule、是否引入 latent，报告均未说明。

v1 代码实际做法（可核实部分）：

| 要素 | v1 实现 | 出处 |
|---|---|---|
| 采样器 | EDM（Karras 式），`num_step=T` | `diffusion.py:19–32` |
| 自由度 | 全部 `fix_size*14` 个笛卡尔坐标（原子级），**不是** frame、torsion、latent、token | `sampling.py:69,80,144` |
| 噪声参数 | `sigma_data=16.0`、`psigma_mean=-1.2`、`psigma_std=1.5` | `diffusion.py:21` |
| 采样默认 | `T=200`、`step_scale=2.25`、`gamma=0.2`、`t_min=0.01`、`t_max=1.0` | `pallatom.py` CLI |
| 序列 | 独立 `seq_logits` 头；最终 `argmax(seq_logits[:,-1,:])`，softmax 温度 0.1 | `pallatom.py:64`、`sampling.py:195` |
| 导出 | atom14 → atom37 | `ref_features.py:153`、`pallatom.py:71` |
| 增强 | `centre_random_augmentation` + `add_additional_noise(gamma,t_min,t_max)` | `sampling.py` SampleIter |

### 3.3 它到底是不是「逐原子生成」？

**是，但要加三个限定。**

1. **是逐（重）原子坐标生成**：输出层给出的就是每个残基 14 个槽位的 xyz；backbone 与侧链在**同一个**扩散状态中同时更新，**没有「先生成 backbone、再由解码器确定侧链」的两段式结构**。【核实】p.5 + v1 `sampling.py:80`
2. **不含氢**：Atom14 = 14 个重原子；没有氢坐标、没有质子化／互变异构状态。所谓「全原子」在跨模型比较时**不等价于含氢全原子**。【核实】
3. **序列不是采样出来的，是 argmax 点估计**（v1 代码）。**报告未说明 v2 是否仍为 argmax。【未核实】**

**因此准确的表述是**：Pallatom2 **生成全部重原子坐标的样本**（含侧链），侧链构象由扩散过程本身建模，而非后置解码器；但作者**没有**给出侧链／χ 角构象分布的校准评测，序列侧也是点估计。**「生成坐标」成立，「已校准构象分布」不成立。**差别在于：它的侧链确实在生成状态里，问题出在**评测层**没有分布级证据，而不是**架构层**用了确定性解码器。

**报告与代码对不上的地方（并列）**：

- 报告只说 "update the coordinates"，代码是 **EDM 的 `reverse()` 用 score=(x_t−x0)/t_hat 做 ODE step**——是扩散反向过程，不是普通迭代精修。【代码／配置核实】
- 报告说 "initial noise state"，代码对应 `z = normal * t_hat`（t_hat 由 EDM 的 noise level 决定），且 **`SampleReference` 用 `t=1.0−t_perturb`**，局部重设计时并非从纯噪声起步。【代码／配置核实】
- 报告把 "all-atom" 与 "Atom14" 并用，但两者在本工作中是同一件事（重原子），不是「含氢全原子」。【核实】

### 3.4 生成范式的边界

- **无 MSA、无 template、无进化信息**：训练语料是 AFDB/TED 的**单结构**，不是 MSA。【核实】p.29–30
- **无 latent**：状态即坐标，因而没有「latent 先验 + 解码器」的容量压缩，也就没有 latent 坍塌风险；代价是长链显存／注意力压力要靠稀疏化解决。【推断】
- **self-conditioning + recycling**：让每一步去噪能看到上一步的 x0 估计，属于 EDM／一致性模型的常规增强。【核实】p.6

### 3.5 条件控制与注入粒度

**界面条件（binder–target）**【核实】p.31 Tab. S1

| 条件 | 设计意图 | 典型用法 |
|---|---|---|
| Hotspot | 必须作用于目标表位 | 强制埋入／识别功能区 |
| Coldspot | 禁止接触的目标区域 | 避开保守位点、糖基化位点、脱靶表面 |
| Binderspot | binder 侧必须进入界面的残基 | 指定 binder 接触残基 |
| Flexspot | 允许目标局部柔性 | 允许 loop／侧链重排 |
| Unindex | 给定侧链几何 | 把已验证的原子相互作用「搬」进新 binder |

**结构条件**【核实】p.31 Tab. S2

| 条件 | 控制对象 | 作用 |
|---|---|---|
| motif | 结构 motif / backbone pattern | 指定要保留的结构元件 |
| redesign | 可设计残基／区域 | 定义哪些部分可重设计 |
| norepack | 指定侧链构象 | 保护侧链不被重排 |
| slevel | 全局／片段二级结构比例 | 设定 SS 组成 |
| sse | 指定位点的二级结构状态 | 指定 α/β/loop |
| seq_cond | 指定位点的残基身份 | 条件序列 |
| hb_bond | 氢键几何 | 指定原子间氢键关系 |

**注入粒度（它区别于「prompt 拼接」的关键）**：

- **二级结构条件注入到 residue-level single 表征**；
- **固定侧链朝向注入到 atom-level pair 表征**。【核实】p.6 §2.2

也就是说，条件不是简单拼一个 embedding，而是**按物理粒度落到对应表征层**：全局／残基级意图进 residue 通道，原子级几何意图进 atom pair 通道。这解释了为什么 `norepack` / `hb_bond` 这类原子级条件能生效。【核实】p.6

**D-form 手性控制**：对 IL-7RA、IL-17A、VEGF-A、PD-L1 生成 D-form binder，先用 AF3 预测 L-form counterpart（pLDDT 92.6 / 91.9 / 91.0 / 94.5），再**手工镜像反转**坐标，backbone RMSD 0.86 / 0.72 / 0.95 / 0.65 Å。【核实】p.16–17

注意：这里的 D-form 不是模型原生输出手性标签，而是**坐标镜像 + AF3 复核**的工程闭环；能说明「生成的手性可控」，但不能说明「模型显式建模了 D-氨基酸的构象分布」。【推断】

---

## 4. 训练数据构成与切分

- **数据源与规模**：**AFDB + TED**；TED 为作者**内部处理版**，非官方 TED；TED 本身派生自 AFDB。【核实】p.29–30 §S1。domain 中位长度：单域 **119 aa**、多域 **131 aa**，随域数在 119–138 aa 变化。【核实】§S1。Fig. 9b 统计的 **17,990,544 binder residues** 是生成样本的残基统计，**不是训练集规模**。【核实】p.16 Fig. 9b
- **清洗与去重**：报告**未披露**任何阈值（分辨率、Rg、packing、SSE、loop、clash；TM-score／seq identity／coverage／e-value）。【未核实】p.29–30
- **切分与泄漏控制**：**未披露**——时间切点、聚类切分（Foldseek／MMseqs 阈值、cluster 数）、held-out／test set、AFDB 快照版本（v4? v5?）与日期，全文均无。**这是本工作最大的可复现性缺口。【未核实】**
- **覆盖与偏置**：保留**原始序列**，明确**不做 ProteinMPNN 或其他序列重设计**。【核实】p.20、p.29–30。作者自述：Pallatom2 选择「纯 AFDB + 原始序列」以换取更接近自然蛋白的氨基酸组成，代价是**放弃蒸馏带来的 designability 增益**，且侧链 rotamer 的「真实分布」只被 AFDB 预测构象监督，没有被实验构象监督。【核实】p.29–30；【推断】
- **可获取性与许可**：v2 训练数据／处理脚本未公开；v1 有公开数据管线 `db_scripts/`。【核实】

**一个必须点破的口径问题**：报告说 "trained without distilled data"，同时说语料是 AFDB。**AFDB 本身就是 AlphaFold2 的预测产物**，属模型生成数据。因此该句只能读作「**没有再叠加 ProteinMPNN／自身重蒸馏**」，不能读作「未使用任何模型生成结构」。【论文自称】＋【核实】p.20 vs p.29–30

**为什么这条对 02 专题有意义**：它把「数据蒸馏取舍」写成了一个可对照的实例——用蒸馏（ProteinMPNN 重设计 + refold 一致性过滤）能抬高 in-silico co-designability，但会把训练序列分布拉向 ProteinMPNN 偏好（如 Glu/Lys 富集）；Pallatom2 反向选择原始序列，因此 designability 数值（54.2–83.3%）低于同期蒸馏路线**不是模型能力差**，而是作者主动不做数据蒸馏。【推断】详见 [02 专题 §2.2](../../专题/02_训练数据构成与切分.md)。

**v1 的阈值只能作「旁证」**：v1 `db_scripts/pipeline.py` 提供一套完整阈值（50–128 aa、Rg ≤ 20、packing ≥ 0.30、SSE ≥ 4、avg degree ≥ 20、loop ≤ 0.5／≤ 15；foldseek 去重 TM 0.8／cov 0.9／e-value 0.001；final cluster TM 0.6）。【代码／配置核实】`pipeline.py:180–201`。但这是 **v1 的 AFDB-only 管线**，v2 明说改用 "internally processed AFDB + TED"，**不能把这些阈值外推到 v2**。【推断】＋【未核实】

---

## 5. 宣称与可验证性

| 宣称 | 证据类型 | 核查状态 | 备注 |
|---|---|---|---|
| 65.9M 参数达到 SOTA in-silico 筛选 | 表格（Fig. 9a／Fig. S3b）+ 文字 | 数值已核实；排名不可作 matched-protocol 引用 | 非同协议、非同预算；"SOTA" 只能读作描述性比较（§6.2） |
| 9 个靶标上跨方法领先 | 表格（每靶标成功率） | 【核实】 | Pallatom2 35.6–98.5%，逐靶标均高于同图对手；BCMA 被排除 |
| 210 个 BLI 中 112 active（53.3%），10 靶标全有 binder | 表格（Fig. 12d）+ 湿实验 | 【核实】 | 284 constructs = 112 + 98 + 74 nonexpressed；BLI 原始曲线见 Fig. S5–S16 |
| 6 个靶标 pM 级（3.96 pM–828 pM） | 表格（Fig. 12a）+ 传感图 | 【核实】 | 为 **apparent KD**（单／多浓度拟合口径未统一说明），非热力学 KD |
| 长链单体 500–1,000 aa designability 54.2–83.3% | 曲线（Fig. 4a） | 【核实】 | 83.3%@500 → 56.3%@1000；AF3 refold 判据 |
| 多域 designability 58.0–69.5%，cluster rate 39.5–52.5% | 曲线（Fig. 4c） | 【核实】 | 63.5%@500 → 69.5%@600 → 58.0%@1000 |
| 同源多聚体 C3–C10 all-atom RMSD 0.84–1.59 Å、pLDDT 93.3–96.07 | 图（Fig. 7）+ 文字 | 【核实】 | 作者自述「未从这些挑选案例估计总体成功率」（p.13）→ 不能当成功率 |
| 条件系统可精细控制界面与构象 | 表（Tab. S1/S2）+ 用例 | 【核实】 | 条件存在且有 3 个实证用例；**无消融**证明每个条件的独立贡献 |
| 局部重设计可做亲和力成熟（2.93×–7.57×） | 表格（BLI KD） | 【核实】 | TNFα t=0.2／0.3 数据；样本 10–12，**无重复测量／无置信区间** |
| 可 rescue 已有纳米抗体（2,000 nM → 4.31 nM） | 单案例 | 【核实】 | 单案例、500 candidates、单 scaffold → 属**存在性证明**，非可重复率 |
| D-form binder 可行 | 图（Fig. 10）+ AF3 复核 | 【核实】 | 4 例；镜像 + AF3 闭环，非湿实验 |
| "trained without distilled data" | 仅文字 | 【论文自称】＋口径冲突 | AFDB 本身是 AF2 产物；p.20、p.29–30 |
| 紧凑模型即可支撑困难设计任务 | 图（Fig. S2）+ 文字 | 【核实】 | 参数对比存在；**无同算力／同时延对照**，「compact 更优」未证明 |
| 优于 Claude 的设计 campaign | 表格（Fig. 12a）+ 文字 | 【核实】 | 6 共享靶标 5/6 median 更低、4/6 min 更低；但**算力差 1–2 个数量级**（728–2,400 vs 20k–200k） |

**总体可验证性评价**：这是同类工作中少见的「in-silico + 湿实验 + 序列／结构相关性分析」三层证据；但**缺消融、缺独立 test split、缺算力口径**，因此更适合当「工程可行性证据」，不适合当「方法学归因证据」。【推断】

**可复现所需材料**：

- **权重**：v2 未公开；v1 `params_Pallatom.npz`（约 71 MB）公开。【核实】
- **代码／配置**：v2 无代码仓；v1 公开 `modules/sampling.py`、`modules/diffusion.py`、`modules/ref_features.py`、`pallatom.py`、`db_scripts/pipeline.py`。【核实】
- **数据**：v2 的 AFDB 快照与内部处理版 TED 均未发布处理脚本；v1 有 `db_scripts/` 数据管线。【核实＋未核实】
- **算力**：训练与推理成本均未披露（全文无 GPU／seconds／throughput）。**未披露**

**与官方文档／论文的一致性**：

- 报告写 "iterative denoising / update the coordinates"，v1 代码是 EDM 反向 ODE step；报告未给 v2 schedule／步数 → **v2 采样器机制未披露**。
- 报告写 "initial noise state"，v1 `SampleReference` 用 `t=1.0−t_perturb`，局部重设计并非从纯噪声起步。
- 报告 "all-atom" 与 "Atom14" 并用，本工作中两者都是重原子（不含氢）。
- 训练语料 "AFDB + TED" 与 "trained without distilled data" 存在口径冲突（AFDB 是 AF2 产物）。

**四类必须保留的「未披露／【未核实】」**：

| 项 | 状态 |
|---|---|
| v2 采样器机制（diffusion／flow／schedule／步数／latent） | **未披露**（报告全文无这些字样） |
| 训练集规模与切分（结构数、清洗／去重阈值、时间切点、held-out） | **未披露** |
| AF3 refold 版本与 MSA 配置 | **【未核实】**（报告未说明） |
| all-atom RMSD 的原子集合与对齐方式（是否含虚拟原子、按残基还是 CA） | **【未核实】** |

---

## 6. 实践与应用

### 6.1 三个有实证的条件用法

1. **局部重设计／亲和力成熟（partial diffusion）**：对已有结合物做 t=0.2／t=0.3 的部分加噪再降噪；TNFα 上 t=0.2 得 10/12 active、median 63.315 nM；t=0.3 得 11/11 active、median 20.1 nM，最佳 PB-146 4.56 nM（2.93×）、PB-153 10.41 nM（7.57×）。【核实】p.21–22
2. **起始界面引导（BCMA）**：有起始界面引导 5/6 active，de novo 3/6；与起始序列 identity 仅 12.5–37.3%，PB-006 达 252.1 pM。【核实】p.21（注：BCMA 不在 Fig. 9a 的跨模型排名里）
3. **纳米抗体 rescue**：PB-272 起始 KD 2,000 nM 且无结合 → 改造 CDR 界面生成 500 candidates → 提名 10 个、7 个表达并上 BLI → PB-276 4.31 nM、PB-277 5.93 nM；与 5M2I 比 CDR1 identity 29.5%／22.5%，全长 identity 47.3%／50.0%，最近公开同源 nanobody 的 CDR1 最高 40.0%。【核实】p.22–23

### 6.2 评测协议与成本

**A. 单体 designability（无条件）**【核实】p.9 §3.1

- 流程：生成序列 → **AlphaFold3 refold** → 与生成结构比；
- pass：`all-atom RMSD ≤ 2.0 Å` **且** `mean pLDDT ≥ 80.0`；
- designability = pass 数 / 被评估样本数；
- diversity：对 **passing designs** 按 `TM-score 0.5` 聚类；`cluster rate = 聚类数 / 全部被评估样本数 × 100%`。

注意两个分母不同：designability 与 cluster rate 的分母都是全部样本（不是 passing 数）。报告原文明确 "divided by the total number of samples evaluated"。【核实】

**B. Binder in-silico screen（有条件）**【核实】p.15 §3.2

- 三条全过才算 pass：`allatom_rmsd ≤ 2.5 Å`、`min-ipAE ≤ 1.5`、`ptm-binder ≥ 0.8`；
- 分别对应：生成 vs 重预测的结构一致性、跨链对齐不确定性、binder 折叠置信度。

**C. 湿实验**：BLI（Octet R8），284 constructs 表达，74 不表达（26.1%），210 上机，112 active。【核实】p.19–20

**Hit rate 的分母**：Fig. 9a 的成功率，分母 = **同一 target–epitope case、同一二级结构类的 400 个候选**（"400 candidates per secondary-structure class per target–epitope case"）。【核实】p.16 Fig. 9 caption。**不是**「每靶标一个二值」，也**不是**「每靶标全部候选合并」；epitope case 的具体个数报告未逐靶标列出 →【未核实】。湿实验的 53.3% 分母是 **210 个上机 construct**，与上面完全不同，**不可混引**。【核实】

**跨模型排名的三项校准**（切点／预算／分母，归入 [03 专题 §6](../../专题/03_评测体系与可复现性.md) 的既有框架）：

| 校准项 | 是否统一 | 证据 |
|---|---|---|
| **切点**（pass 阈值） | ❌ 未统一 | 各方法按各自 protocol／自筛流程；"under the reported protocols"【核实】p.16 |
| **采样预算** | ❌ 未统一 | Pallatom2 400/类/例；对手预算未在同一表中给出；报告明确 "not a matched computational cost"【核实】p.14–16 |
| **分母** | ❌ 未统一 | 各方法候选集合与筛选层级不同；BCMA 缺席、Fig. S3b 为 10 个 case 而 Fig. 9a 为 9 靶标【核实】 |

**结论**：Fig. 9a 的 "state-of-the-art" **不能**当作 matched-protocol 排名引用；正确引用方式是「在作者协议下，Pallatom2 的逐靶标成功率高于同图列出的方法」。若要在自己的报告里做跨模型表，**必须另建统一切点 + 统一预算 + 统一分母**的实验，或明确标注「非校准」。【推断】

**与 AF3 的可比性**：AF3 在本文中**不是竞争对手**，而是**评估器**（refold）和**辅助验证器**（D-form L-form 预测），因此不存在「与 AF3 切点可比」的问题——AF3 根本没有被排名。【核实】p.9、p.16。真正需要小心的是：**AF3 refold 的 pass 率本身依赖 AF3 版本／采样预算**，报告未披露 AF3 配置，因此 designability 的绝对值不可跨论文直接比。【未核实】

**成本**：推理与训练成本均**未披露**（全文无 GPU 型号、秒／结构、显存、throughput）。可间接读出的只有：`T=200` 步（**仅 v1**）、稀疏注意力（p.5）、65.9M 参数（p.1）。因此本篇章**不给出任何「比 X 快／便宜」的结论**。【核实（缺失）】

**湿实验统计口径**：112/210 = 53.3% active；284 constructs 中 74 不表达。不表达关联分析：n=248，hydrophobicity 每增加 1 SD，OR=2.07（95% CI 1.42–3.00，q=0.000889）。【核实】p.20。**284 vs 248 的排除规则未说明 →【未核实】**。KD 为 **apparent KD**（BLI 单／多浓度拟合），跨靶标比较时应注明方法差异（Tab. S3 显示外部参照分别用 SPR／BLI／HTRF）。【核实】p.35 Tab. S3

### 6.3 对 VNAR／单域抗体工程的可复用点

**先算清约束**：SAbDab2 中 VNAR 高分辨率结构仅 **84 个**（口径见 [04 专题 §3.2](../../专题/04_抗体与VNAR可迁移点.md)）。因此：

- 65.9M 全原子模型**不能**在 84 个结构上从头训练；微调也只能做轻量 adapter／LoRA，且极易过拟合到 84 个 scaffold 的 CDR3 长度／构象。【推断】
- 训练语料是 AFDB/TED 的**单域／多域球状蛋白**，不是免疫球蛋白折叠；VNAR 的 **β-sandwich + 长 CDR3 + 无轻链** 属明显分布外。【推断】
- 定位应是 **「在已有 VNAR 上做局部重设计／亲和力成熟」**，而不是「de novo 生成 VNAR」。【推断】

**三种角色判定**：

| 角色 | 能否用 | 理由／证据 |
|---|---|---|
| **预测器**（给 VNAR+靶标打分） | ❌ 目前不能 | v2 权重未公开；即便公开，它是生成器不是亲和力预测器，`ptm-binder`／`min-ipAE` 是**自洽性**指标，不是 KD 预测【核实】p.15 |
| **评估器**（in-silico 筛候选） | ⚠️ 协议可借，阈值不可直接借 | 三阈值（2.5 Å／1.5／0.8）是在 **minibinder** 上校准的；VNAR 的 CDR3 更长、界面更小，阈值需重标定【推断】p.15 |
| **设计引擎**（局部重设计／成熟） | ✅ 最有价值 | partial diffusion（t=0.2／0.3）+ `redesign`／`norepack`／`unindex` 的组合，正是「固定框架、只动 CDR／界面」的工具；纳米抗体 rescue 单案例（2,000 nM → 4.31 nM）是直接先例【核实】p.21–23 |

**四条可落地点**：

1. **VNAR 亲和力成熟流水线**：以现有 VNAR–靶标复合物为起点 → `norepack` 锁框架 + `redesign` 放开 CDR1/2/3 → t=0.2 起步、t=0.3 加强 → 用 `min-ipAE + ptm-binder` 筛 → 小批量 BLI。**预期不是「必出 pM」，而是「在 84 结构的约束下提高命中率」**。【推断，依据 p.21–23】
2. **表位定向**：`Hotspot`／`Coldspot` 可直接用于「避开 VNAR 的框架表位、只打靶标特定 loop」，这对只有单域的 VNAR 尤其重要（没有轻链帮忙遮避）。【核实】p.31 Tab. S1
3. **构象锁定的界面移植**：`Unindex` 把已知 VNAR–抗原的侧链接触几何「搬」到新框架；这是 84 结构条件下最划算的知识复用方式。【核实】p.31 Tab. S1
4. **评估协议移植**：把 §6.2 的三条阈值 + AF3 refold 流程直接搬来做 VNAR 候选的**内部排序**，但必须先用 10–20 个已知 VNAR 复合物重标定阈值，否则会把「长 CDR3 的高 ipAE」误杀。【推断】

**不建议做的事**：

- ❌ 用 Pallatom2 直接 de novo 生成 VNAR 骨架：训练分布不含 Ig 折叠，且无 VNAR 条件接口。
- ❌ 把 Fig. 9a 的 400/类/例预算直接套到 VNAR：VNAR 的候选空间与 minibinder 不同，400 个可能远不够覆盖 CDR3 长度多样性。
- ❌ 把 `ptm-binder ≥ 0.8` 当成「结合置信度」：它是 binder 单链折叠置信度，不是界面置信度；界面要看 `min-ipAE`。【核实】p.15

---

## 7. 疑问与讨论

- **Q：v2 采样器是否仍是 EDM Cartesian？**
  - **A**：报告只写 "iterative denoising"，全文无 diffusion／flow／schedule／steps／latent 字样；v1 代码是 EDM（`T=200`、`sigma_data=16.0`）。**v2 机制未披露**，只能并列陈述，不能合并成一句。若 v2 改成 flow matching 或引入 latent，「纯坐标扩散」的定位需要改写。→ 需作者放出 v2 代码／配置或补充材料。

- **Q：训练集到底多大、怎么切分？**
  - **A**：只披露 AFDB + TED（内部处理版）、保留原始序列、不做 ProteinMPNN 重设计；结构数、清洗阈值、去重阈值、时间切点、聚类切分、held-out、AFDB 快照版本**全部未披露**。没有这些，designability 的 54.2–83.3% 无法判断是「泛化」还是「记忆」。这是本工作最大的可复现性缺口。

- **Q：侧链构象分布有没有被评测？**
  - **A**：没有。全文没有 χ 角／rotamer 指标；designability 的 all-atom RMSD 是对 **AF3 refold** 比，而 AF3 的侧链本身也是预测。→ 需要 rotamer recovery／χ1–χ4 误差表。

- **Q：AF3 refold 的配置是什么？**
  - **A**：版本、MSA 配置、采样数均**未披露**，导致 designability 绝对值不可跨论文比较。→ 需补充方法学细节。

- **Q：`all-atom RMSD` 的原子集合与对齐方式是什么？**
  - **A**：是否含虚拟原子、是否按残基对齐、是否只算 CA，**均未说明**。→ 直接影响 2.0 Å 阈值的含义。

---

## 8. 学习心得

- **「全原子」是一个表示层声明，不是一个能力声明**。Pallatom2 的 Atom14 无氢、含虚拟原子；它能生成侧链坐标，但没有证明侧链构象分布正确。读任何「全原子生成」工作，先问三件事：含不含氢、侧链在不在生成状态里、有没有 χ 角评测。
- **跨模型表格要先做三项校准**（切点、预算、分母）。本文的 Fig. 9a 三项都不统一，所以它是有价值的「工程可行性展示」，不是「方法排名」。
- **65.9M 的稀疏全原子注意力是这篇最可偷的工程点**：用 residue relationship graph 决定原子对注意力，把成本从原子平方降到残基尺度，才撑得到 1,600 aa。这比它的条件系统更适合复用到我们自己的模型。

---

## 9. 参考资料

1. Pallatom2 Technical Report, 2026-11-06, 58 pp. `github.com/levinthal/Pallatom2-techreport`（p.1 摘要；p.5 §2.1；p.9 §3.1；p.15 §3.2；p.16 Fig. 9；p.29–30 §S1；p.31 Tab. S1/S2；p.32 Fig. S2；p.33 §S3.2；p.35 Tab. S3/S4；p.48–58 Tab. S5）
2. Pallatom v1 代码与权重：`github.com/levinthal/Pallatom`（`modules/sampling.py`、`modules/diffusion.py`、`modules/ref_features.py`、`pallatom.py`、`db_scripts/pipeline.py`）
3. Pallatom v1 预印本：bioRxiv 2024.08.16.608235
4. Pallatom-Ligand：`github.com/levinthal/Pallatom-Ligand`（ICLR 2026 投稿）
5. 本仓库配套：[01 专题 §6 覆盖矩阵](../../专题/01_全原子表示与生成范式.md)、[02 专题 §2.2 数据配比与自蒸馏](../../专题/02_训练数据构成与切分.md)、[03 专题 §6 评测口径对照表](../../专题/03_评测体系与可复现性.md)、[04 专题 §3.2 VNAR 结构规模](../../专题/04_抗体与VNAR可迁移点.md)、[对比表 CSV](../../联合分析/assets/全原子模型对比表_2026-11-04.csv)
6. 评测所依赖的外部模型：AlphaFold3（refold／D-form 复核）、RFdiffusion-AA／RFD4-Proteina、Proteina、La-Proteina、BoltzGen、BindCraft、AlphaProteo、PXDesign、Claude（对比 campaign）
