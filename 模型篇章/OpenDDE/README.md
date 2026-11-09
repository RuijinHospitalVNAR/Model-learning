---
model: OpenDDE（Open Drug Discovery Engine）
aliases: [OpenDDE, OPENDDE, OpenDDE-Preview, "arXiv:2607.03787"]
family: 结构预测（全原子生物分子共折叠，AF3 家族；非序列生成）
granularity: 全原子坐标（atom37 + 配体／离子／核酸原子；采样在连续三维坐标空间）
backbone: Pairformer trunk（48 blocks，c_s = c_z = 384）+ structural-token refiner（4 blocks／8 heads）+ diffusion transformer（24 blocks）；655,791,538 参数
generative: 连续坐标扩散（AF3／EDM 式并行去噪；非自回归、非离散扩散、无码本）；发布默认 10 recycle × 200 step × 5 sample
conditioning: 序列（必需）+ MSA（可关）+ 模板（默认关）+ RNA MSA（默认关）+ 结构约束（contact／epitope／movable_chains／covalent_bonds；其余 constraint 键被拒绝）
data_scope: PDB（截止 2021-09-30）+ AFDB-multimer + Teddymer（复现）+ MGnify 长／短单体 + Swiss-Prot + Disordered motifs + SAbDab；各子集条数与训练去重阈值未披露
status: 已完成（arXiv 2607.03787v1，2026-07-04；代码／权重 Apache-2.0；preview 阶段，已发 6 个版本）
verified_at: 2026-11-06
---

# OpenDDE

> **状态**：已完成（含 6 项未核实交接，见 §9.3）
> **记录人**：团队共同学习
> **最后更新**：2026-11-06

> **速判（三句话）**
> 1. OpenDDE = **Aureka** 的 **Open Drug Discovery Engine**，是 AF3 家族的**全原子共折叠预测器**，不是离散 token 模型。
> 2. 本仓库旧归类把它放在「路线 ③：离散 Token」是**误分类**；按专题分类定义，它在**路线 ①（直出坐标）**上（订正记录见 [归类分析](../../联合分析/全原子模型解题思路归类分析.md)）。
> 3. 它**不是序列设计引擎**：没有残基类型解码器，"de novo design" 只是已知坐标条件下的掩码扩散补全。

---

## 1. 学习背景与目标

### 1.1 为什么学它

本仓库的[全原子模型解题思路归类分析](../../联合分析/全原子模型解题思路归类分析.md)曾把 OpenDDE 列在**路线 ③（离散结构 Token 化）**，且出处标注为"待补"。这两个状态同时存在时，任何基于该行的比较都缺一手依据。本次学习先解决三件事：

- **钉出处**：确认 OpenDDE 指向单一模型，而不是团队内部简称或别的项目的别名；
- **判路线**：用代码／配置／论文逐项核对它的采样空间，判断它属于 ①／②／③ 中的哪一条；
- **判可用性**：对 VNAR／单域抗体，它能承担什么角色、不能承担什么角色。

### 1.2 目标

1. 把出处、版本、许可钉到一手来源（arXiv、GitHub、Hugging Face、PyPI、官方新闻稿）；
2. 用"采样发生在什么空间"这一判据，给出路线归属与否决证据；
3. 给出对 VNAR 的 zero-shot 使用协议与禁止用法；
4. 明确列出未披露／未核实项，不用推测值填表。

### 1.3 出处（已钉死）

| 项 | 值 | 证据 |
|---|---|---|
| 全名 | Open Drug Discovery Engine（OpenDDE） | 【核实】论文摘要；仓库 README 首段 |
| 机构 | Aureka AI Research（Laguna Hills, CA + 上海） | 【核实】arXiv 署名 `OpenDDE Project, Aureka AI Research`；新闻稿抬头 |
| 论文 | arXiv:2607.03787v1，*Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine* | 【核实】arXiv abs 页，`[v1] Sat, 4 Jul 2026`，CC0 |
| 代码 | `github.com/aurekaresearch/OpenDDE` | 【核实】README；GitHub API |
| 权重 | `huggingface.co/aurekaresearch/OpenDDE`（revision `eddd563…`） | 【核实】模型卡；`model_manifest.json` |
| 包 | PyPI `opendde`（首个可查版本 1.0.3） | 【核实】pypi.org/project/opendde/1.0.3 |
| 版本时间线 | v1.0.1（2026-07-16）→ v1.0.2（07-17）→ v1.0.3（08-05）→ v1.1.0（08-16）→ v1.1.1（09-02）→ v1.2.0（10-08） | 【核实】GitHub Releases API 的 `tag_name` / `published_at` |
| 对外发布 | 2026-07-06 官方新闻稿 | 【核实】aurekabio.com/news/60 |

结论：**不是团队内部简称，指向单一模型，一手来源齐全。** 旧状态"出处待补"可以直接补上。

### 1.4 与 IsoDDE 的关系（最容易混淆的一处）

- **IsoDDE** 是 **Isomorphic Labs** 的闭源 *Drug Design Engine*（技术报告 2026-02-10）。
- **OpenDDE** 是 **Aureka** 的独立开源模型，把 IsoDDE 作为**对标上限**，不是它的子模型、不是它的开源版、不共享权重。
- 两者共享的只是 "DDE" 这个缩写（Drug Discovery／Design Engine）。**【核实】**这一点很重要：把 OpenDDE 读成 "Discrete Diffusion Engine"，是旧归类误判最可能的来源。

### 1.5 技术谱系

- 直接沿用 AF3 的三段式：**Pairformer trunk → diffusion module → confidence／distogram heads**，输出容器是 atom37。
- 组件与数据协议依赖 Protenix、OpenFold3-Preview、ColabFold、ESMFold2，以及 NVIDIA Fold-CP（上下文并行）。
- 相对 AF3 的**四项明确改动**（论文 §2.1 原话）：
  1. Pairformer 隐藏维 128 → **384**（参数约 ×3，计算约 ×9）；
  2. 增加 **atomic-level structural-token reasoning**（§2）；
  3. 增加 **shape-complementarity loss**（界面互补）；
  4. **结构预测与 de novo design 统一为一个条件去噪问题**。

---

## 2. 核心概念

| 术语 | 含义 |
|---|---|
| OpenDDE / OPENDDE / OpenDDE-Preview | 同一模型；`OPENDDE` 是论文正文全大写写法，`OpenDDE-Preview` 是 GitHub 仓库首版标题 |
| IsoDDE | Isomorphic Labs 闭源 Drug Design Engine，仅作对标，不是同源模型 |
| residue token（`S^r`） | 以残基为单位的连续 token 表示，`c_token = 384`；不是词表 ID |
| structural token（`S^s`） | 按化学角色拆分的更细节点，状态 `s_u ∈ R^c`，属**连续隐节点**，不是离散 ID |
| atom37 | 蛋白原子输出容器（每残基 37 槽位），配体／离子／核酸原子另计 |
| EDM | AF3 使用的连续坐标扩散形式；`sigma_data / rho / s_max / gamma` 等超参与 AF3 同构 |
| `N_cycle` / `N_step` / `N_sample` | 采样循环的三层：recycle 次数 × 扩散步数 × 样本数 |
| TFG guidance | 推理期用约束梯度引导（contact／epitope 等） |
| DockQ | 复合物界面质量指标；`≥ 0.23` 为 acceptable，另有 medium／high 档 |
| top-rank / oracle | 用模型自己的 ranking score 选一个 vs 用真值 DockQ 选最优；两者差距反映排序器能力 |
| preview | 官方声明：CLI 参数、I/O JSON 字段、checkpoint 可能跨版本变化，预测不保证跨版本可复现 |
| general / abag checkpoint | 发布的两个权重：`opendde_v1`（通用）与 `opendde_abag`（抗体-抗原强化）；后者训练配方未披露 |

**一句话定位**：它是一个"**表征层做隐空间推理、采样层做坐标扩散**"的混合结构。术语里的 token 只是 embedding 的分组单位，不是离散状态。

---

## 3. 原理与推导

### 3.1 三层表示（论文 Fig. 7a / §3.1）

```
(序列, 原子, MSA, 模板, 约束)
        ↓
残基 token 表示 S^r  +  配对表示 Z^r        （Pairformer trunk, 48 blocks, c=384）
        ↓
结构 token 表示 S^s  +  配对表示 Z^s        （Structural Refiner, 4 blocks / 8 heads）
        ↓
全原子坐标 X ∈ R^{N_atom×3}                 （diffusion coordinate generator）
```

- **残基 token**：`c_token = 384`，输入拼接后 `c_s_inputs = 449`（`= c_token + 32 + 32 + 1`，代码注释原文）。【代码／配置核实】`opendde/config/model_base.py` L21–L24。
- **结构 token**：把每个残基／分子单元按化学角色拆成更细的节点，角色数 `n_roles = 7`（蛋白主链、侧链、核酸主链、碱基、配体／原子等）。【核实】Table 1(a) "Structural token roles 7"；【代码／配置核实】`model_base.py` L169–L171。
- **原子层**：`c_atom = 128`，`c_atompair = 16`，`max_atoms_per_token = 24`（注释："DNA G max_atoms = 23"）。【代码／配置核实】`model_base.py` L22–L26。

### 3.2 "离散 token"问题：直接回答

**否。OpenDDE 的采样层不在离散空间。**

| 检查项 | 结论 | 证据 |
|---|---|---|
| 是否有词表／码本（VQ、codebook） | **没有** | 【核实】论文 §3.1、Appendix A.1；【代码／配置核实】`opendde/model/` 下无 codebook／quantizer 类文件，`model_base.py` 只有 `n_roles`、`max_atoms_per_token`，无 vocab 尺寸 |
| 结构 token 是离散 ID 吗 | **不是**，是连续隐节点 | 【核实】Appendix A.1：结构 token 状态 `s_u ∈ R^c`，由残基 token 状态加角色嵌入 `e_ρ(u)` 初始化 |
| 解码器是确定性还是生成性 | **生成性**（扩散） | 【核实】§3.1 "diffusion-based coordinate generator denoises atom positions"；`sample_diffusion` 超参齐全（`sigma_data`／`gamma0`／`step_scale_eta`） |
| 输出是全原子吗 | **是**（atom37 + 配体／离子／核酸原子） | 【核实】Appendix A.1.3；`max_atoms_per_token = 24` 覆盖 DNA 最大基团 |
| 采样自由度 | 每个原子的 3 个连续坐标 | 【核实】`X ∈ R^{N_atom×3}`；噪声为高斯，不是 categorical |

**换句话说**："输出层是全原子"与"采样层在连续坐标空间"这两件事同时成立，且**后者才是判据**——所以它不是路线 ③。

### 3.3 报告原话 vs 代码／配置：并列写出

| 维度 | 论文说法 | 代码／配置说法 | 处理 |
|---|---|---|---|
| "token" 的含义 | 通篇用 token（"residue-token"、"structural-token"） | 配置里没有词表；token 只是 embedding 分组单位 | 【推断】术语债，不是离散化 |
| 采样步数 | Table 1(b)：`N_step = 20`，`N_cycle = 4` | `model_base.py` L64 `N_step: 200`、L134 `N_cycle: 10` | 【核实】口径冲突，见 §5.7 |
| 模板能力 | `supported_models.md` 标 Template ✓ | `inference_defaults.py` 默认 `use_template: False` | 【核实】默认关闭，能力与默认值不是一回事 |
| 结构 token 是否"离散" | 未在任何处声称离散 | 无码本、无离散采样 | 【核实】旧"路线 ③"归类缺证据 |

### 3.4 生成范式

- **并行去噪**（diffusion），**不是**自回归、**不是**掩码语言建模、**不是**离散扩散。
- 属于 **EDM／AF3 式连续坐标扩散**：`sigma_data = 16.0`、`gamma0 = 0.8`、`gamma_min = 1.0`、`step_scale_eta = 1.5`、`s_max = 160`、`rho = 7`。【核实】Table 1(b)；【代码／配置核实】`model_base.py` L54–L63。
- 采样循环是 AF3 的 **recycle × diffusion step × sample**：`N_cycle × N_step × N_sample`。
  - 论文口径：`4 × 20 × 5`；
  - 发布默认：`10 × 200 × 5`（默认 5 个 sample）。【代码／配置核实】`model_base.py` L64／L134。
- **码本大小：不适用**（无码本）。
- **温度：不适用**；多样性来自随机种子与初始高斯噪声，不是温度缩放。

### 3.5 主干规模

| 组件 | 规格 |
|---|---|
| Pairformer | 48 blocks，c_s = c_z = 384 |
| MSA 模块 | 4 blocks，c_m = 128，最大深度 2,048，dropout 0.2 |
| Structural Refiner | 4 blocks，8 heads |
| Diffusion transformer | 24 blocks（diffusion 内 c_token = 768） |
| Atom encoder／decoder | 各 3 blocks，c_atom = 128 |
| Distogram | 96 bins，[2.25, 25.75] Å |
| 总参数 | 655,791,538（约 656 M） |
| 权重体积 | 2,625,249,069 B（约 2.6 GB，fp32） |

参数与结构数字来自【核实】Table 1(a)；参数量与文件大小来自【核实】`model_config.json` 的 `parameter_count`／`size_bytes`。

### 3.6 一句话判断

这是一个"表征层做隐空间推理、采样层做坐标扩散"的混合结构。它借用"token"这个词描述表示分组，但**从未在离散状态空间上采样**。把它当路线 ② 也不准确：路线 ② 的判据是"采样在连续隐空间"，这里采样仍在坐标空间，隐空间只承担推理。

### 3.7 条件控制

| 条件 | 支持情况 | 证据 |
|---|---|---|
| 序列 | 必需 | 【核实】§3 "Given sequence, atom, MSA, template, and constraint features" |
| MSA | 支持，可关闭 | 【核实】`supported_models.md` ✓；内部运行曾 `use_msa=false` |
| 模板 | 支持，**默认关闭** | 【核实】`supported_models.md` ✓；`inference_defaults.py` `use_template: False` |
| RNA MSA | 支持，**默认关闭** | 【核实】同上 `use_rna_msa: False` |
| contact 约束 | 支持（TFG guidance） | 【核实】`docs/supported_models.md`；`docs/infer_json_format.md` |
| epitope 约束 | 支持（TFG guidance） | 【核实】同上 |
| movable_chains | 支持 | 【核实】同上 |
| 共价键 | 支持 | 【核实】`covalent_bonds` |
| 其他约束键 | **拒绝** | 【核实】`supported_models.md`："Other `constraint` keys are rejected" |

### 3.8 条件生成的真实边界（重要）

论文 §3.3 把"结构预测"和"de novo design"写成同一个问题的两个特例，靠 **known／target 原子掩码** 实现：

```
p_θ(X_target | I, X_known)
```

- 结构预测 = `m_known = 0`（全部原子都是 target）；
- 条件设计 = 固定一条或多条链、motif、pocket、分子语境，生成其余原子的**坐标**。

**关键限制（论文未强调、但对我们是决定性的）：**

- 生成的是 **X（坐标）**，条件 `I` 里包含**序列**；
- 论文与代码都**没有残基类型解码头**，也没有 inverse folding 模块；
- 因此它**不能生成序列**。所谓 de novo design 是 **structure inpainting／motif scaffolding**，不是 binder 序列设计。

【推断】基于 §3.3 公式与仓库 `opendde/model/` 无序列输出头；【核实】论文全文未出现 "sequence design" 或 "inverse folding"。

---

## 4. 训练数据构成与切分

### 4.1 数据源与规模

【核实】论文 §2.2 原话列出的来源：

1. **Weighted PDB**（权重方案沿用 Protenix）；
2. **AFDB-multimer**（Han et al.）；
3. **Teddymer**（按 DiDi et al. 复现的数据集）；
4. **MGnify long／short monomer**（AF3 提出、OpenFold3-Preview 发布）；
5. **Swiss-Prot**（AlphaFold DB）；
6. **Disordered motifs**（AF3 提出、OpenFold3-Preview 发布）；
7. **SAbDab** 抗体结构（ESMFold2 报告、SAbDab 采集）。

**各子集样本条数：未披露**——论文只给百分比。

### 4.2 分阶段配比（Table 2）

| 数据源 | Warmup | Ⅰ | Ⅱ | Ⅲ | Ⅳ(a) | Ⅳ(b) |
|---|---:|---:|---:|---:|---:|---:|
| Weighted PDB | 70% | 50% | 35% | 40% | 50% | 40% |
| AFDB-multimer | 4% | 5% | 11% | 9% | 5% | 9% |
| Teddymer（复现） | 4% | 5% | 11% | 9% | 5% | 9% |
| MGnify long monomer | 10% | 28% | 30% | 25% | 20% | 20% |
| MGnify short monomer | 4% | 5% | 4% | 4% | 4% | 4% |
| Swiss-Prot | 6% | 5% | 4% | 4% | 5% | 4% |
| Disordered | 2% | 2% | 2% | 1% | 1% | 1% |
| **SAbDab** | **0%** | **0%** | **3%** | **8%** | **10%** | **13%** |

【核实】Table 2。**对 VNAR 最重要的一行是最后一行**：抗体数据在最后两个阶段才被大幅加权（8% → 13%），说明 Ab-Ag 能力是**后期专门强化**的，而 SAbDab 以常规抗体为主。

### 4.3 清洗与去重

- **训练集去重阈值：未披露**。论文只给 benchmark 侧的 MMseqs2 easy-cluster（同一性 ≥40%、覆盖 ≥80%），那是**评测聚类**，不是训练去重。
- 分辨率、缺失原子、链断裂、配体过滤等清洗阈值：**未披露**。

### 4.4 切分与泄漏控制

- **PDB／模板库截止日：2021-09-30**（论文写 "2021-09"，manifest 写精确到日）。【核实】§2.2；`model_manifest.json` → `data_cutoff`。
- 与 AF3 采用**同一截止日**，这点对"和 AF3 横比"很关键（见 §5.5）。
- **训练／验证／测试切分方式：未披露**（论文未给序列相似度聚类阈值或时间切点以外的隔离规则）。
- **held-out 集合**：PXMeter-AB、FoldBench-AB、2026ARK-AB 三个抗体-抗原 benchmark。其中 2026ARK-AB 是自建，论文称 "not available to earlier released models"；**其收录日期窗口未披露**。

### 4.5 覆盖与偏置

- SAbDab 在最终阶段占 13%，但 **SAbDab 中 VHH／VNAR 的具体占比未披露**；论文从未给出该子集内的抗体类型分布。
- 因此对 VNAR 只能是 **zero-shot 外推**，不能称为"同分布泛化"。
- 其余覆盖偏置（物种、构象、罕见化学）论文未系统披露。

### 4.6 训练日程与算力

- 5 个阶段：warmup（lr 0.0012，10k steps；每 60k steps ×0.95）→ Ⅰ → Ⅱ → Ⅲ → Ⅳ(a)／Ⅳ(b)。
- Ⅳ(a)：冻结主干，只训 confidence head；Ⅳ(b)：全模型在 Hopper 上继续训，crop 从 544 提到 768。【核实】§2.3、Table 2 表注。
- 算力：预处理 5 周 × 10×8 张 80GB；训练 50×8 张 80GB Ampere 6 周 + 8×8 张 141GB Hopper 1 周；在线验证 5×8 张 80GB；合计 **≈414K GPU-hours**（≈单卡 54 年）。【核实】§2.4；新闻稿同口径。

### 4.7 可获取性与许可

- 代码 Apache-2.0；权重在 Hugging Face 公开（revision `eddd563…`）；PyPI 可安装。
- **preview 声明**：CLI 参数、I/O JSON 字段、checkpoint 都可能在版本间变化，且预测不保证跨版本可复现。
- 做对比实验必须**锁 revision** 并记录 checkpoint SHA-256。

---

## 5. 宣称与可验证性

### 5.1 主要宣称

| 宣称 | 证据类型 | 核查状态 | 备注 |
|---|---|---|---|
| 三个 Ab-Ag benchmark 上 top-rank 最高（51.0／70.0／66.4） | 表格 | 已核实（论文口径） | 权重＋代码＋benchmark 公开，但**无第三方复现** |
| oracle 选择下更高（65.9／81.9／80.1） | 表格 | 已核实（论文口径） | 同上；oracle 用真值 DockQ 选最优 |
| "achieve IsoDDE-level co-folding accuracy" | 仅文字（对标声明） | 待核实 | IsoDDE 闭源，训练配方／数据／推理均不可见，**不可归因** |
| 发现两条 scaling law 方向 | 曲线 | 部分可验证 | Fig. 4／5 的横轴是估算量，非实测 token 数 |
| 开源可复现 | 工程主张 | 已核实（版本内） | 跨版本不作保证（README 明确警告） |

### 5.2 IsoDDE 对比的可信度

论文 §5 自己承认：没有 IsoDDE 的训练配方、数据配比、推理流程、后训练策略与工程优化，因此**无法归因差距来自架构、数据、蒸馏还是推理**。新闻稿的措辞也更保守（"narrows the gap"）。**【核实】**

### 5.3 路线判定：③ 不成立、① 成立、② 仅在表示层

**按本仓库专题分类的定义（③ = 离散 Token），结论是：在这项工作上不成立。**

三条否决证据：

1. **无离散状态空间**：结构 token 是 `R^c` 连续隐节点，采样在 `R^{N_atom×3}` 连续坐标上。【核实】Appendix A.1、§3.1。
2. **无码本／无 tokenizer**：仓库与配置中不存在 codebook、VQ、quantizer；`n_roles = 7` 是语义角色数，不是词表大小。【代码／配置核实】`model_base.py` L171。
3. **无对照实验**：论文没有做"离散 token vs 连续坐标"的消融，也没有与任何离散 token 基线比较。所以即使有人主张 ③ 有结构性优势，**这项工作也没有提供实验证据**。

与 ①／② 的关系：

- **路线 ①（直出坐标）：成立。** 这是它的采样层本质。
- **路线 ②（连续隐空间）：只在表示层部分成立。** 隐空间用于"推理"（structural token refiner），不用于"采样"；判据是采样空间，故不应归入 ②。
- **路线 ③（离散 Token）：不成立。**

**结构性优势是否被证明？** 没有。③ 若主张"离散化能压缩序列长度、复用 LLM 式建模、降低采样自由度"，本文既未论证也未实验。与之相反，OpenDDE 明确选择了**连续坐标扩散**，并在 §3.1 把"reasoning before coordinate generation"作为卖点——这恰恰是"连续隐空间 + 坐标扩散"的组合，而不是离散 token。

> 若旧归类那一行 "OpenDDE" 实际指的不是本模型，则该行**没有任何一手出处可挂**（原本就是"待补"），应当删除或重新挂到正确对象；两种情况都指向同一动作：**订正该行，而不是保留"③ + 待补"的组合**。订正已落在 [归类分析](../../联合分析/全原子模型解题思路归类分析.md)，原有错误表述与证伪路径一并保留。

### 5.4 Benchmark 与结果

| Benchmark | 规模／来源 | 指标 | OpenDDE top-rank | OpenDDE oracle |
|---|---|---|---|---|
| PXMeter-AB | 沿用 PXMeter 协议 | DockQ ≥ 0.23 成功率 | 51.0% | 65.9% |
| FoldBench-AB | 抗体-抗原子集 | 同上 | 70.0% | 81.9% |
| 2026ARK-AB | 自建：164 个 PDB 复合物／159 个界面簇 | 同上 | 66.4% | 80.1% |

2026ARK-AB 上同口径基线：ESMFold2 51.0%、Protenix-v1 50.0%、OpenFold3 21.9%、Boltz-1 25.0%、Chai-1 16.7%。【核实】§1.1。

**2026ARK-AB 的构造**：按 PXMeter 协议；界面用残基空间邻近定义、SAbDab 注释；MMseqs2 easy-cluster（≥40% 同一性、≥80% 覆盖）；每个界面簇 = 两个互作实体簇的无序对。【核实】§1.1。**收录日期窗口未披露。**

### 5.5 指标定义与分母（必须先校准再排名）

- **切点**：`DockQ ≥ 0.23` 为 acceptable，另分 medium／high；高保真档用 `≥ 0.8`。
- **分母**：按 **target／interface** 计（不是按候选计）。2026ARK-AB 有 164 个复合物、159 个簇，**分母到底用 164 还是 159 论文未写明**。【未核实】
- **采样预算**：论文 Table 1(b) 为 `N_sample = 5`；测试时扩展实验则从 1 个 seed 扫到数百个。**各 baseline 的版本、seed 数与 step 数均未披露**。
- **top-rank 选择**：用模型自己的 ranking score；**oracle** 用真值 DockQ 选最优。
- **跨模型可比性三项校准**：
  1. 切点——一致（都用 0.23）；
  2. 采样预算——**不一致／未披露**：baseline 的 seed 数、step 数、版本均未给；
  3. 分母——**未明示**。

  因此 66.4% vs 51.0% 的排名**只能算"论文口径下的排名"，不能当作严格可比的 SOTA 结论**。【推断】

### 5.6 一个反直觉的强信号：排序器是瓶颈

- 2026ARK-AB：top-rank 66.4%，oracle **80.1%**；
- FoldBench-AB：top-rank 70.0%，oracle **81.9%**；
- 测试时扩展：oracle 从 ~66–67%（1 seed）升到 **>90%**（数百 seed），但 ranked 只从 ~28%→34%（FoldBench-AB）、35%→38%（2026ARK-AB）。

【核实】§1.1、§1.3。**结论**：多采样确实能造出好构象，但**当前 confidence／ranking 选不出来**。真正的工程收益在**重排序／置信度校准**，而不是继续加采样。这一条比"模型更强"更值得我们复用。

### 5.7 复现口径的三个坑

1. 论文 20 步／4 recycle vs 发布默认 200 步／10 recycle（§3.3 已列）。
2. 模板／RNA-MSA 能力可用但**默认关闭**。
3. `preview` 声明：CLI 参数、I/O JSON 字段、checkpoint 都可能在版本间变化，**且预测不保证跨版本可复现**。

### 5.8 可复现所需材料

| 材料 | 状态 |
|---|---|
| 权重 | 公开（HF，revision `eddd563…`） |
| 代码 | 公开（Apache-2.0） |
| 数据 | 多数为公开数据集的派生；各子集条数与清洗细节**未披露** |
| 算力 | 论文报告 ≈414K GPU-hours；推理侧见 §6.2 |

---

## 6. 实践与应用

### 6.1 能当什么用（先给结论）

| 角色 | 可否 | 前提 |
|---|---|---|
| **预测器**（VNAR-抗原共折叠） | **可以**，但需 zero-shot 验证 | 单域抗体就是"一条蛋白链"，JSON 支持任意链；论文案例含 VHH（8Z8V，2024-09） |
| **评估器**（对候选打分／排序） | **可以，且是最稳的用法** | 有参考复合物时用 DockQ；无参考时只能做跨引擎一致性，**不能把 pLDDT／ipTM 当精度** |
| **结构补全器**（固定框架、生成 CDR 坐标） | **理论可以** | 需 chain／motif 级 mask；**残基级 mask 是否暴露未核实** |
| **序列设计引擎** | **不可以** | 无残基类型解码器、无 inverse folding |
| **亲和力预测** | **不可以** | 论文 §5 明确划为 "future extensions" |

### 6.2 推理成本（Appendix C Table 3）

| N（token 长度） | 模式 | Forward (s) | 端到端 (s) | 峰值显存 (MiB) |
|---|---:|---:|---:|---:|
| 100 | single | 14.0 | 161.2 | 4,959 |
| 100 | CP4 | 400.8 | 557.4 | 15,549 |
| 250 | single | 36.2 | 180.8 | 10,619 |
| 250 | CP4 | 393.7 | 543.7 | 17,353 |
| 500 | single | 184.4 | 330.8 | 50,971 |
| 500 | CP4 | 462.6 | 612.4 | 30,279 |
| 900 | single | 468.4 | 615.2 | 56,855 |
| 900 | CP4 | 766.4 | 920.2 | 56,443 |
| 1200 | single | 982.2 | 1133.8 | 67,105 |
| 1200 | CP4 | 969.9 | 1127.3 | 49,273 |
| 1400 | single | 1636.7 | 1784.4 | 67,665 |
| 1400 | CP4 | 1259.2 | 1398.3 | 59,237 |
| 2000 | single | **OOM** | — | — |
| 2000 | CP4 | 2568.2 | 2714.1 | 80,899 |

【核实】Table 3。**反直觉**：CP4（4 卡上下文并行）在 N ≤ 900 时**更慢、峰值显存更高**（N=100 时 15,549 vs 4,959 MiB），只有 N ≥ 1200 才开始占优；它的真实价值是**让 N=2000 不 OOM**，不是加速小体系。【推断】

### 6.3 VNAR／单域抗体：把 84 个结构的约束算进去

**约束条件**：SAbDab2 中 VNAR 高分辨率结构约 **84** 个（项目内给定，【未核实】原始出处；SAbDab2 官方站确认 `VNAR` 是独立抗体类型）。

三条推论：

1. **不能微调。** 656 M 参数对 84 个结构，参数量／样本量 ≈ 780 万:1，任何全参微调都会过拟合。可行的是**冻结评估**或极轻量 adapter，且必须留出独立 VNAR 测试集（按界面簇切分，不能按 PDB ID 切）。
2. **训练分布里 VNAR 近似为零。** SAbDab 在最终阶段占 13%，但 SAbDab 以常规抗体为主；**论文从未披露 VHH／VNAR 占比**。【未核实】因此对 VNAR 只能是 **zero-shot** 外推，而不是"同分布泛化"。
3. **abag checkpoint 未必更好。** 项目内部记录（2026-09-19，nanobody + MC4R，3 个 seed，`cycle=10/step=200/use_msa=false`）：`opendde_abag.pt` 在 **3/3 seed** 上均劣于 general（pLDDT 63.56→50.04，ipTM 0.313→0.201，ranking 0.348→0.245）。【核实】·内部记录（未复现）。这与"专用 checkpoint 更强"的直觉相反，提示 abag 权重可能对**常规抗体界面几何**过拟合；**`opendde_abag.pt` 的训练配方未披露**。

### 6.4 建议的 VNAR 使用协议

1. **默认用 general checkpoint**（`-n opendde_v1`）；abag 只在常规 Ab-Ag 复现实验中使用，并在 VNAR 上做 A/B。
2. **先开 MSA**；模板在 VNAR 同源结构足够时再开（注意 `use_template` 默认 False）。MSA 对 CDR3 的帮助需单独验证。
3. **评测只认 DockQ**：与实验结构比；没有实验结构时，与 AF3／Protenix 在同一候选上做**跨引擎一致性**，或与"亲本 VNAR-抗原复合物"比。pLDDT／ipTM 仅作筛选用。
4. **报告 top-rank 与 oracle 两条线**：否则会把"排序器差"误读成"模型差"（§5.6）。
5. **采样预算固定后再比模型**：例如 8 seeds × 200 steps × 10 cycles，并在论文口径（20 步／4 recycle）下另跑一组做交叉验证。
6. **分工定位**：OpenDDE 放在"**结构实现 + 可计算评估**"环节；序列生成交给专门的 inverse folding／抗体语言模型。二者串联，而不是指望一个模型全包。

### 6.5 工程上可直接借用的三个点

- **structural token 的"角色分解"**：把 atom37 按化学角色（主链／侧链／碱基／配体）分组再精修，是一个便宜且可迁移的中间层设计。
- **shape-complementarity loss**：对界面"贴合"的显式监督，比单纯 LDDT 更贴近抗体-抗原的失败模式。
- **统一 masked 生成**：预测与"给定框架生成其余坐标"共用一套权重，部署上只需一套推理栈。

---

## 7. 疑问与讨论

- **Q1：为什么会被归成"离散 Token"？**
  - **A**：最可能是把 "DDE" 展开成 "Discrete Diffusion Engine"。实际是 Drug Discovery Engine。**术语缩写是这次误分类的唯一线索**，而"出处待补"让错误一直没被证伪。教训：**缩写不能当证据**。
- **Q2："token" 这个词是不是该改？**
  - **A**：论文用 token 描述连续 embedding 的分组，读者极易读成离散词表。这是**术语债**，不是技术矛盾。笔记里应显式写 "continuous latent token"。
- **Q3：650 M 在 2026 年算小模型，为什么还能打？**
  - **A**：因为大头在**数据与算力**（414K GPU-hours、5 周预处理），而不是参数。这与论文"scaling 是基础设施问题"的判断一致。
- **Q4：它到底离 IsoDDE 多远？**
  - **A**：不可归因。论文自己承认信息不足。任何"已追平／仍差距 X%"的说法都超出证据。
- **Q5：开源了，就能复现吗？**
  - **A**：版本内可以；跨版本不行（preview 声明）。做对比实验必须**锁 revision**（`eddd563…`）并记录 checkpoint SHA-256。

---

## 8. 学习心得

- **先钉出处，再谈机制。** 这次如果直接按"离散 Token"写九节，会产出一篇看起来完整、但根子错了的笔记。
- **"能生成坐标"≠"能设计序列"。** 这是当前全原子模型最容易被夸大的边界。
- **排序器比生成器更值得投入。** oracle 与 top-rank 之间 13–14 个百分点的差距，是现成的工程机会。

---

## 9. 参考资料

### 9.1 一手来源

1. OpenDDE Project, Aureka AI Research. *Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine.* arXiv:2607.03787v1, 2026-07-04. <https://arxiv.org/abs/2607.03787>
2. 代码仓库：<https://github.com/aurekaresearch/OpenDDE>（Apache-2.0）
3. 权重：<https://huggingface.co/aurekaresearch/OpenDDE>（revision `eddd563ce96571f784012edd8f045181c8f8627d`）
4. 包：<https://pypi.org/project/opendde/1.0.3/>
5. 官方发布稿：<https://www.aurekabio.com/news/60>（2026-07-06）
6. Isomorphic Labs. *Accurate Predictions of Novel Biomolecular Interactions with IsoDDE.* Technical report, 2026-02-10（对标对象，闭源）
7. SAbDab2：<https://sabdab.opig.stats.ox.ac.uk/about>（VNAR 作为独立抗体类型）

### 9.2 本仓库配套

- [01 专题 §6 覆盖矩阵](../../专题/01_全原子表示与生成范式.md)：公开程度按 A–E 分级（本篇为 B 级）
- [02 专题 §5.1 公开透明度分级](../../专题/02_训练数据构成与切分.md)：A–E 分级口径
- [03 专题 §6 评测口径对照表](../../专题/03_评测体系与可复现性.md)：跨模型比较的切点／预算／分母校准
- [联合分析/全原子模型解题思路归类分析](../../联合分析/全原子模型解题思路归类分析.md)：路线 ③ 订正记录
- [对比表 CSV](../../联合分析/assets/全原子模型对比表_2026-11-04.csv)

### 9.3 仍待核实（交接清单）

- [ ] 2026ARK-AB 的收录日期窗口；
- [ ] baseline 的版本与采样预算（跨模型排名的第 2、3 项校准）；
- [ ] 训练集各子集条数与训练去重阈值；
- [ ] SAbDab 中 VHH／VNAR 实际占比；
- [ ] `opendde_abag.pt` 的训练配方；
- [ ] CLI 是否支持残基级 mask；
- [ ] "84 个 VNAR 高分辨率结构"的原始出处；
- [ ] 路线 ③ 若要独立成路线，缺一个"同一主干、只换采样空间"的对照实验。
