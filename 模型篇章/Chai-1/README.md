---
model: Chai-1
aliases: ["Chai-1", "chai-lab", "chaidiscovery/chai-lab"]
family: AF3 谱系（pair-bias trunk + 原子级扩散头），同一作者群后续有 Chai-2
granularity: 标准氨基酸/标准核苷酸每残基 1 token（蛋白按 atom37 编号，槽位上限 37；核苷酸按 36 原子编号）；配体与**修饰残基逐原子 1 token**（槽位索引恒 0）；全模型上限 2048 tokens（AVAILABLE_MODEL_SIZES 最大 2048）
backbone: 论文只写 "largely follows Abramson et al. [11], with a heavy reliance on pair-bias self-attention"，trunk block 数、pair/single 通道宽、MSA 模块 block 数、diffusion 层数/宽度**全部未披露**；代码以 6 个 TorchScript 组件发布（feature_embedding / bond_loss_input_proj / token_embedder / trunk / diffusion_module / confidence_head），无 PyTorch 源码
generative: EDM 式 x0 扩散；σ_data=16、s_max=80（不是 AF3 的 160）、s_min=4e-4、ρ=7，σ(t)=σ_data·power_interp ⇒ 有效 σ∈[0.0064, 1280]；Karras 式 churn 采样（S_churn=80、S_tmin=4e-4、S_tmax=80、S_noise=1.003、二阶）；默认 200 步、5 diffusion samples、1 trunk sample；论文评测 5 trunk ×5 diffusion = 25 个
conditioning: MSA（可选，Jackhmmer 3 迭代或 MMseqs2 server 或自带 .aligned.pqt）| 模板（PDB70，≤4 条）| constraints（CSV：covalent / contact / pocket）| ESM-2 3B 逐残基嵌入（默认恒开）| single-sequence 模式；无 antibody / nanobody 模式
data_scope: PDB + AlphaFoldDB；PDB 结构释放日 ≤2021-01-12（模板 PDB70 同切点）；不使用 AF3 的蒸馏集（AFDB 除外）；评测集 PDB 2022-05-01–2023-01-12、非 NMR、分辨率优于 4.5 Å、≤2048 tokens、40% 同源剔除
status: 代码与权重 Apache-2.0（LICENSE 与 README 均明确 "both code and model weights"）；但**论文摘要/正文写 non-commercial use**——两处许可表述并列保留、不择一，待法务口径确认；发布 2024-09-09（Chai Discovery），论文 bioRxiv v1 2024-10-11
verified_at: 2026-11-10
---

# Chai-1

> **三句话速判。** ① Chai-1 不是"AF3 的复现"，而是 AF3 主干 + 三处结构性改造：把 MSA 从"必需"降为"可选"并用 3B 蛋白语言模型嵌入补位、把配体与**修饰残基**改成逐原子 token、用"训练切点早于所有评测集"换掉 AF3 的"多模型分别评测"，因此可以单模型通吃并对外宣称 PoseBusters 77.05% vs AF3 76.34%。② 但"AF3 式"在它身上**落不到数字**：论文对 trunk block 数、pair/single 宽度、MSA 模块层数、diffusion 层数一字未写，只写"largely follows [11]"；能拿到的数字全在代码里（σ、步数、采样数、2048 token、MSA 16384 深度、模板 ≤4）。③ 最要命的一条：**训练切点论文写了，是 2021-01-12**（p.2、p.9、p.11 三处），二手来源流传的 2021-09-30 / 2022-05-01 / 2023-01 分别是 AF3 的切点与它自己的评测集起止日。

---

## 1. 定位与谱系

### 1.1 出处与版本（先把版本钉死）

| 项 | 值 | 核实方式 |
|---|---|---|
| 论文/技术报告 | bioRxiv `10.1101/2024.10.10.615955`，v1，**2024-10-11 posted**，18 页，标题页落款 `Date: September 9, 2024` | 【核实】bioRxiv v1 全文逐页核对（含每页水印） |
| 官方"技术报告 PDF" | `chaidiscovery.com/assets/chai1technicalreport.pdf` → **404**；README 指向的 `chaiassets.com/chai-1/paper/technical_report_v1.pdf` → **429（限流）** | 【核实】两次实测；因此全文一律引 bioRxiv v1，页码即该 PDF 页 |
| 仓库 | `github.com/chaidiscovery/chai-lab`，commit `c691da74d6f348b19bc9f931abe95d66d0b88e78`（2026-10-27） | 【代码／配置核实】GitHub API `/repos/...` + `/commits/<sha>` |
| PyPI | `chai-lab==0.6.1`，wheel 172 KB（**不含模型定义**） | 【代码／配置核实】下载 wheel 解包比对文件树 |
| 许可证 | 代码 + 权重 Apache-2.0；论文正文写 non-commercial | 【核实】LICENSE 全文 + README L189 + paper p.1/p.2 |

取证细节：`git clone` 与 `raw.githubusercontent.com` 不可达时改走 **GitHub contents API（base64）** + **PyPI wheel 0.6.1** 双路，二者文件集合一致。

### 1.2 相对 AF3：复现还是改了什么（逐项）

| 维度 | AF3 基线（本库 2026-11-04 笔记） | Chai-1 | 判定 |
|---|---|---|---|
| 表征 | 聚合物每残基 1 token；配体逐原子；`max_atoms_per_token=24` | 聚合物每残基 1 token（atom37 编号，上限 37 槽）；配体逐原子；**修饰残基也逐原子** | **改了**（修饰残基降级为逐原子）【代码／配置核实】+【推断】 |
| trunk | Pairformer 48 block，pair 128 / single 384，MSA 模块 4 block | 自述"largely follows"；**block 数与通道宽未披露** | **未披露**，无法判"改没改"【核实】 |
| MSA | 必需，最多 16,384 行 | **可选**；single-sequence 模式完全不用 MSA，改用 3B PLM 嵌入 | **改了**（结构性）【核实】 |
| diffusion | 24 个 token block，token 768 / atom 128；σ_data=16、σ_max=160、σ_min=4e-4、ρ=7；确定性 EDM | 层数/宽度未披露；σ_data=16、s_max=**80**（且再乘 σ_data）、s_min=4e-4、ρ=7；**Karras churn 随机采样器** | **改了采样器**，结构未披露【代码／配置核实】 |
| 条件 | MSA、模板、部分约束 | 加了一条 3B PLM 嵌入轨道；pocket/contact/docking 三类距离特征；共价键（跨链） | **改了**（新增 PLM 轨道 + 约束特征）【核实】 |
| 数据策略 | 主模型 + 多个"切到评测集之前"的重训模型分别打分 | **一个模型通吃**：训练切点 2021-01-12 早于所有评测集 | **改了**（这是全篇最重要的方法学差异）【核实】 |
| 训练目标 | 扩散损失 + confidence 损失（交叉熵） | **未披露**（论文没有 loss 段落，也没有蒸馏目标描述） | 未披露【未核实】 |

### 1.3 上下游

- **上游**：AF3（ref [11]）。数据预处理与聚类步骤、模板嵌入、pair-bias self-attention 均指向 AF3：`Templates are generated using PDB70`、`We largely follow the data preprocessing and clustering steps from Abramson et al. [11]` 。
- **并行**：Boltz-1 / Protenix 同代 AF3 复现品；Chai-1 的差异点在"PLM 轨道 + 单模型单切点 + 逐原子修饰残基"。
- **下游**：Chai-2（零样本抗体设计）。**Chai-1 没有任何 antibody/design 模式**，抗体能力是 Chai-2 才有的【核实：全仓库 grep `antibody|nanobody` 零命中；README 仅引用 Chai-2 报告】。
- **对外产品**：`lab.chaidiscovery.com` 免费含商用；开源包是"自建推理"路径 。

### 1.4 许可证与权重可得性（分开写，这里是坑）

| 对象 | 声明方 | 原文 | 结论 |
|---|---|---|---|
| 论文（文本） | bioRxiv | `Chai-1 model weights and inference code are available for non-commercial use¹`（脚注 1 指向 GitHub） | 非商用 |
| 仓库代码 | LICENSE 文件 | Apache 2.0 全文  | 可商用 |
| 仓库权重 | README L189 | `released under an Apache 2.0 License (both code and model weights)... both academic and commerical purposes, including for drug discovery`  | 可商用 |
| 权重文件本体 | 仓库树 | 300+ blob 中**没有第二份 LICENSE/NOTICE**，也没有 weights 专用条款 | 无独立条款，随 README 走 |

【推断】材料时间顺序：blog（2024-09-09，Apache 2.0）→ 论文（2024-10-11，non-commercial）→ 当前 README（2026，Apache 2.0 both）。论文那句极可能是发布前遗留、没跟着改；但这是真实的"论文 vs 代码"冲突——**两侧出处并列记录，本笔记不代为择一**，实际约束力须由法务按具体用途判定。另注意：论文 PDF 每页水印 `CC-BY-NC 4.0` 是 bioRxiv 对**论文文本**的许可，不是权重许可——二手转述里的"权重非商用"多半混了这两件事。

---

## 2. 表征粒度与表示层

### 2.1 token 划分（代码一手）

| 实体 | 每多少个原子 1 token | 原子槽位 | 中心原子 | 来源 |
|---|---|---|---|---|
| 标准氨基酸 | 1 残基 : 1 token | atom37 编号（`assert max_index <= 36`，即 ≤37 槽） | `CA` | 【代码／配置核实】tokenizer  |
| 标准核苷酸 | 1 残基 : 1 token | 36 原子编号（≤36 槽） | `C1'` | 同上 |
| 配体 / 非标准残基 / 共价修饰残基 | **1 原子 : 1 token** | `atom_within_token_index = 0`（单槽） | 原子自身 | 同上 |
| 糖链（`EntityType.MANUAL_GLYCAN`） | 1 原子 : 1 token | 1 槽 | 原子自身 | 同上 |

tokenizer 的原话是 `Tokenization is by residue if it is a standard amino acid or standard nucleotide; all ligands and all modified residues are tokenized per atom.` 。这一条与 AF3 的差别值得单独记：AF3 里 MSE、SEP 这类修饰残基仍算"聚合物链上的一个残基 token"，Chai-1 把它们**打散成逐原子 token**，主链连续性靠原子级 attention 去接。这也解释了它为什么对修饰残基敏感（论文自己承认的局限）。

### 2.2 原子槽位上限与可变原子数

- 蛋白 token 用 atom37 索引，所以**单 token 原子槽位上限 37**；核苷酸 36。**不是** AF3 的 24。这是"看似一样、其实换了索引体系"的典型【代码／配置核实】。
- 配体/修饰残基侧不存在"槽位"概念：每个原子独占一个 token，`atom_within_token_index` 恒 0，天然回避了"大配体超槽"问题【代码／配置核实】。
- 缺原子/多余原子怎么处理：tokenizer 侧用 atom37 的占位与 mask；**但模型内部怎么消费这些槽位无法核实**——`trunk.pt`、`diffusion_module.pt` 是 TorchScript 编译图，仓库没有 `.py` 源码 。所以"可变原子数用槽位+mask 还是动态原子列表"只能给到"输入侧按 atom37 槽位组织"，网络内部实现【未核实】。

### 2.3 residue frame 与扭转角

- 代码里存在 `backbone_frame_mask` / `backbone_frame_index` 这类字段（在 structure context 里随 token 一起产出），说明**输入侧准备了主链 frame 信息**【代码／配置核实】。
- 但"frame 是否真的进网络、是否用于 IPA 式更新"【未核实】：模型定义不在开源代码里。
- 侧链扭转角：仓库与论文**都没有** torsion 相关特征；论文自述"largely follows AF3"（AF3 明确不用 frame、不用扭转角）。**判定：无扭转角**；frame 只到"有字段、用途未核实"【核实 + 未核实】。

### 2.4 原子级 cross-attention

- `run_inference` 的 collate 调用传 `num_key_atoms=128, num_query_atoms=32` ，说明**存在原子级注意力且 kv 维 128、query 维 32**；层数、heads 与内部结构【未披露/未核实】（编译图）。

一句话：**表征层是 Chai-1 唯一"论文没写但代码能读全"的部分**，其余都在二进制里。

---

## 3. 主干与生成范式

### 3.1 论文自述：一句定性，零个数字

> `The Chai-1 neural architecture largely follows previous work Abramson et al. [11], with a heavy reliance on pair-bias self-attention. Key differences to previous work is summarized below.` （p.9 §5.1）

随后 §5.1.1 / §5.1.2 只讲了①语言模型嵌入轨道、②约束特征。**没有** block 数、通道宽、层数、heads、参数量。全篇 grep `block|hidden|channel|dimension|layers|attention head|number of parameters|triangle` 只有上面这一句命中【核实】。所以"AF3 式"在 Chai-1 上**落不到数字**，这是本笔记必须如实写出的第一结论。

### 3.2 能落到的数字（全部来自代码，逐项带路径行号）

| 项目 | 值 | 来源 | 与 AF3 比较 |
|---|---|---|---|
| trunk block 数 | **未披露** | 论文无；`trunk.pt` 为编译图 | AF3 48 |
| pair / single 通道宽 | **未披露** | 同上 | AF3 128 / 384 |
| MSA 模块 block 数 | **未披露**；但 MSA 整体**可跳过**（single-sequence 模式） | 论文 p.10  + 代码默认 `use_msa_server=False`  | AF3 4 block、必需 |
| diffusion 层数 / 通道 | **未披露** | `diffusion_module.pt` 编译图 | AF3 24 block / 768 / 128 |
| σ_data | 16.0 | `diffusion_schedules.py` L18 ；`chai1.py` L247  | 同 AF3 |
| σ 调度参数 | `s_max=160.0`、`s_min=4e-4`、`p=7.0`，但实际调用传 `s_max=DiffusionConfig.S_tmax=80.0` | L15-17 ；`chai1.py` L822  | AF3 直接用 160 |
| σ 的合成方式 | `sigmas = sigma_data * power_interpolation(times, val_0=s_max, val_1=s_min, p=p)` | L36-39  | **AF3 不做 σ_data 乘法** |
| 有效 σ 区间 | **[0.0064, 1280]**（=16×4e-4 … 16×80） | 由上式推得【推断，算式一手】 | AF3 [4e-4, 160] |
| 采样器 | Karras/churn：`S_churn=80`、`S_tmin=4e-4`、`S_tmax=80.0`、`S_noise=1.003`、`second_order=True`；γ=min(80/T, √2−1) | `chai1.py` L242-248、L831-832、L859、L877  | AF3 无 churn（确定性 Euler/EDM） |
| 默认步数 | 200 | `num_diffn_timesteps: int = 200`  | 同 200 |
| 默认 diffusion samples | 5 | `num_diffn_samples: int = 5` ；README L32  | AF3 5 |
| 默认 trunk samples | 1（论文评测用 5） | `num_trunk_samples: int = 1`  vs  | AF3 5 seeds |
| 论文评测总样本 | 5 trunk ×5 diffusion = **25** | 见本篇 §9 | 25 |
| recycling | 论文 **4**；代码 `num_trunk_recycles: int = 3` |  vs  | AF3 3（口径不同） |
| recycle 时 MSA 子采样 | `recycle_msa_subsample: int = 0`（默认不子采样） | 见本篇 §9 | — |
| 推理 MSA 深度 | 论文表：UniRef90 10000 / UniProt 50000 / Reduced BFD 5000 / MGnify 5000；代码截断 `MAX_MSA_DEPTH = 16_384` | （p.10 表）、 | AF3 16,384 |
| 模板数 | ≤4（`MAX_NUM_TEMPLATES = 4`） | 见本篇 §9 | AF3 4 |
| token 上限 | 2048 | 论文 p.10 ；`AVAILABLE_MODEL_SIZES` 最大 2048  | AF3 5,120 |
| 参数量 | **未披露**（论文未写，代码无权重清单） | — | AF3 ~214 M（trunk 口径另计） |

### 3.3 三个必须点出来的"数字疑点"

**(a) σ 的尺度被 σ_data 放大了一次。** AF3 的调度是直接在 σ 上做幂插值，区间 [4e-4, 160]；Chai-1 的 `get_schedule` 把插值结果再乘 `sigma_data=16` ，于是**名义 s_max=80 实际对应 σ=1280**。后果之一：`chai1.py` L831 用 `(sigmas >= S_tmin) & (sigmas <= S_tmax)` 判断"是否在 churn 窗口内"，而这里的 `sigmas` 已经是 ×16 之后的量、阈值却是没乘 16 的 4e-4/80，于是**前若干步（σ>80）被排除在 churn 之外**。**【推断，依据 L829-832 运算符与 L36-39 量纲】**是刻意还是量纲没对齐，作者未解释，属读代码该记的坑。

**(b) 采样器不是 AF3 的采样器。** churn + `S_noise=1.003` + 二阶更新  是 Karras 那一套随机采样。这意味着即使在同 σ 区间、同步数下，Chai-1 与 AF3 的样本分布也不同——做"同步数、同 seeds"的对齐实验时不能假设二者只差网络。

**(c) 论文 4 recycles vs 代码 3。** 论文明确写 `We run Chai-1 with 4 recycles, 5 trunk samples and 5 diffusion samples for a total of 25 predicted structures` ；代码默认 `num_trunk_recycles=3` ，循环体是 `for recycle_idx in range(num_trunk_recycles)` 。二手基准（J. Chem. Inf. Model. 纳米抗体表位预测）在服务器上记录的是"3 recycling steps / 5 diffusion samples"[S3]，与代码默认吻合。**最可能的解释**是"4 次 trunk 前向 = 1 次初始 + 3 次 recycle"的计数口径差；但论文没写清，代码也没写清，因此本笔记按"论文 4 / 代码 3"并列记录，不替作者抹平【推断】。

### 3.4 为什么拿不到架构数字（根因，不是懒）

模型以 **6 个 TorchScript 组件**分发：`feature_embedding.pt`、`bond_loss_input_proj.pt`、`token_embedder.pt`、`trunk.pt`、`diffusion_module.pt`、`confidence_head.pt`，从 `https://chaiassets.com/chai1-inference-depencencies/models_v2/{key}` 拉取（原文拼写 `depencencies`）。仓库 `chai_lab/model/` 下**只有** `diffusion_schedules.py` 与 `utils.py` 两个 `.py`——即"调度与工具开源，网络本体闭源"。这一点与 AF3（网络源码全开）、RFAA（全开）形成鲜明分层：**Chai-1 是"权重开放、架构不开放"**。任何声称"Chai-1 有 N 个 block"的二手说法都无一手依据。

---

## 4. 条件控制

### 4.1 接口总览（先给结论：**不是 YAML**）

Chai-1 的输入是 **FASTA（可带修饰残基/糖链语法）+ 可选 `.restraints` CSV + 可选 `*.aligned.pqt` MSA 目录 + 可选模板目录**，命令行 `chai-lab fold input.fasta output_folder` 。**没有 YAML 配置文件**——这与后来 Boltz 系列的 YAML 风格不同，别混。restraints 文件用 `pd.read_csv` 读，列名固定为：

| 列 | 是否生效 | 说明 |
|---|---|---|
| `restraint_id` | 标识 | 唯一 ID |
| `chainA` / `res_idxA` | 生效 | 约束的 A 侧；`res_idxA` 留空表示"整条链任一残基"（pocket 用法） |
| `chainB` / `res_idxB` | 生效 | B 侧 |
| `connection_type` | 生效 | `covalent` / `contact` / `pocket` |
| `max_distance_angstrom` | 生效 | 距离阈值 θ |
| `min_distance_angstrom` | **不生效** | README 原话：`currently not used by the model`  |
| `confidence` | **不生效** | 同上；不能用来给约束加权 |
| `comment` | 忽略 | 注释 |

【代码／配置核实】

### 4.2 逐项判定：接口 or 需重训

| 条件 | 能力 | 接口 | 需重训？ | 证据 |
|---|---|---|---|---|
| MSA（自动） | `--use-msa-server`：ColabFold MMseqs2 公共服务器，默认 URL `api.colabfold.com`，可 `--msa-server-url` 指向内网 | 命令行开关 | 否 | 【核实】README L82 ；代码默认 False  |
| MSA（自带） | `--msa-directory`，文件 `<name>.aligned.pqt`，列 `sequence/source_database/pairing_key/comment` | 文件 | 否 | 【核实】 |
| 模板 | `--use-templates-server`（PDB70 m8 服务）或本地模板目录；上限 4 条 | 命令行 + 目录 | 否 | 【核实】 |
| pocket 约束 | chainA + res_idxA + chainB，θ∈(6,20) Å | CSV `connection_type=pocket` | 否 | 【核实】 |
| contact 约束 | 残基对 + θ∈(6,30) Å | CSV `connection_type=contact` | 否 | 【核实】 |
| 跨链共价键 | 两残基间成键 | CSV `connection_type=covalent` | 否 | 【核实】examples/covalent_bonds |
| 链内共价（二硫键） | **不支持** | — | **需重训** | 【核实】`Chai-1 was not trained on intra-chain bonds (e.g., disulfides)`  |
| docking 特征（四箱 one-hot） | 存在生成器（bins 0-4/4-8/8-16/>16 Å） | **代码内特征，CLI 的 CSV 不暴露** | 否（但用不到） | 【代码／配置核实】 |
| ESM-2 3B 嵌入 | 默认恒开；`use_esm_embeddings` 可关；`esm_device` 可设 cpu | 参数 | 否 | 【代码／配置核实】 |
| single-sequence 模式 | 不给 MSA、不给模板，只给序列 + PLM 嵌入 | 默认行为 | 否 | 【核实】 |
| antibody / nanobody 模式 | **不存在** | — | 需外部方案（pocket 约束 + 配对链判断） | 【核实】全仓库零命中 |
| 表位定向（epitope conditioning） | 用 pocket 约束喂表位残基即为"表位条件" | CSV | 否 | 【核实】 |

几点必须说清：

1. **"single-sequence 模式"不等于"零外部依赖"。** 论文原话 `In single sequence mode, we provide only the input sequence and protein language model embeddings` ；代码默认 `use_esm_embeddings=True`，`esm.py` 会 `torch.jit.load` 一个 **esm2_t36_3B_UR50D** 并以 fp16 跑 。关掉 MSA 后仍要付 3B 语言模型的显存/时间，别把 `--use-msa-server false` 当"轻量模式"。
2. **约束特征的编码与论文不完全同构。** 论文只对 docking 说"one-hot 四箱"，pocket/contact 只说"距离阈值"；代码里 pocket/contact 走 `EncodingType.RBF`、`num_rbf_radii=6`、区间 6-20 / 6-30，`size=0.33` 对应论文的几何分布 p=1/3；docking 生成器确为 one-hot 四箱 。**结论：pocket/contact 是 RBF（论文未写），docking 与论文一致**。
3. **`confidence` 字段不生效**很关键：想让"高置信约束压过低置信约束"当前做不到，只能靠 `max_distance` 做软硬切换。
4. 论文写"训练时每类约束以 10% 概率独立加入"，开源推理代码里是 `include_probability=1.0`——这是推理默认值，不算冲突；训练侧未开源，10% 无法核实（见 §6 表）。

---

## 5. 训练数据构成与切分

### 5.1 切点问题：**论文写了，是 2021-01-12**

这是本次核查的核心任务，直接给三处直述原文 + 一处旁证：

1. **p.2 §2.1**：`we train a single model with a training data date cutoff of 2021-01-12 as opposed to training separate models for separate evaluations.` 
2. **p.9 §5.2**：`For PDB structures, we apply with a release date cutoff of 2021-01-12. ... Templates are generated using PDB70 with the same cutoff date.` 
3. **p.11 §5.6**：`Our evaluation set is constructed from a temporal split of PDB entries released between 2022-05-01 and 2023-01-12; these structures were all released after any data in our training set (most recent cutoff date of 2021-01-12).` 
4. **p.3 §2.3（旁证）**：`structures in this benchmark set are released on or after 2021-01-13 – after our training data cutoff.`  —— PoseBusters 的 428 个结构也全部晚于切点。

**结论：不是"未披露"。** 论文不仅写了训练集切点，还写了模板库同切点、评测集起止日，并把它当成"单模型可通吃"的合法性依据（`Unlike Abramson et al. [11], we use the same model for all evaluations since our training data cutoff does not overlap with the data used in any of our evaluation sets` ）。

### 5.2 四种说法的来源归因（结论表）

| 流传说法 | 有无一手依据 | 实际是什么 | 判断依据 |
|---|---|---|---|
| **2021-01-12** | **有，三处** | Chai-1 训练集（PDB）+ PDB70 模板库切点 | 见本篇 §9 |
| 2021-09-30 | **无**（对 Chai-1 而言） | **AF3 的 PDBe 训练切点**；被二手评测方借用为"统一保守切点"（如 LiPP 用 "October first 2021"） | 【核实】AF3 笔记 L184/L209；【二手】[S2] |
| 2022-05-01 | **无** | Chai-1 **评测集**的**起始日** |  原文 `released between 2022-05-01 and 2023-01-12` |
| 2023-01（2023-01-12） | **无** | Chai-1 **评测集**的**结束日** | 见本篇 §9 |
| "未披露" | **错** | 上一版 AF3 笔记（2026-11-04）第 289 行的表述，本次被推翻 | [S4] |

一句话记法：**2021-01-12 = 训练；2022-05-01 → 2023-01-12 = 评测；2021-09-30 = 别人家（AF3）的训练**。看到哪个数字就能反推说话人读的是哪一段。

### 5.3 代码侧有没有"泄露"日期？

- 仓库文件树（300+ blob）**没有**任何训练/数据准备脚本，没有 `docs/` 目录，没有 dataset 构建入口；`chai_lab/data/` 下全是为**推理**服务的解析、特征、collate 代码【代码／配置核实】。故"代码里 data preparation 脚本泄露切点"这条**不成立**——因为它压根没开源数据管线。
- 代码里能读到的只有**推理侧上限**：`MAX_MSA_DEPTH=16_384`、`MAX_NUM_TEMPLATES=4`、`AVAILABLE_MODEL_SIZES` 最大 2048 ；这些与论文的 2048 token 一致，与切点无关。
- 反向证据：`README` 写默认 `uses embeddings without MSAs or templates` ，也没有任何日期。

### 5.4 数据构成与规模

| 项目 | 值 | 来源 |
|---|---|---|
| 结构来源 | PDB + **AlphaFoldDB（AFDB）** | 见本篇 §9 |
| 切点 | PDB 释放日 ≤ **2021-01-12**；模板 PDB70 同切点 | 见本篇 §9 |
| 蒸馏结构 | **不使用** AF3 的蒸馏集（AFDB 除外） |  原文 `We do not train on any of the distillation datasets described in [11] other than AFDB` |
| 结构数 / 链数 / 原子数 | **未披露** | 全篇无 |
| 分辨率阈值、长度阈值 | **未披露**，自述沿用 AF3 的预处理与聚类 | 见本篇 §9 |
| 聚类阈值 | 训练侧未披露；**评测侧** 40% 序列一致性、≤9 残基与全部核酸用 100% | 见本篇 §9 |
| 训练算力 | 128×A100、batch 128、**30 天** | 见本篇 §9 |
| MSA 数据库 | UniRef90 / UniProt / Reduced BFD / MGnify（推理时 jackhmmer `-N 3`）；数据库版本与切点"沿用 [6]（AF2）" | 见本篇 §9 |
| 核酸 MSA | **不使用**（核酸一律 single-sequence） | 见本篇 §9 |
| PLM | 3B 参数模型、逐残基嵌入；修饰残基替换为母残基或 `X`；非蛋白链给 mask token | 见本篇 §9 |

值得单独记的一条：论文明确说"数据库日期切点与版本沿用 AF2 [6]"。所以 Chai-1 的 **MSA 数据库**切点其实比它的 PDB 切点更早、更保守（AF2 的库是 2018-2019 年的版本），这对"MSA 里是否含有测试集同源序列"是个隐含的额外保险——AF3 用的是更新版本的库。这一点论文没展开，属【推断，依据是  的 "same date cutoffs and versions described in [6]" + AF2 发布年份】。

---

## 6. 宣称与可验证性

### 6.1 宣称-证据对照

| 宣称 | 证据类型 | 核查状态 |
|---|---|---|
| PoseBusters 77%（AF3 76%），且**不用 MSA** | 表格（Table 1, p.12）+ 累积分布图 | 【核实】**但口径要打折**：Table 1 标题写的是 `Chai-1 77.05%`，正文 p.3 说"without MSA"是就"不给 MSA server"而言，而 single-sequence 仍喂 3B PLM 嵌入 ；另外 AF3 的 76.34% 是**引用 AF3 自己发布的预测**，不是作者复跑（脚注 3 原话 `we did not run AF3 ourselves`） |
| 蛋白-蛋白 0.751 vs AF2.3 0.677（p=6.24e-10） | 表格（Table 2, p.12）+ 箱线图 + Wilcoxon | 【核实】数值与 CI 均有一手；但基线只有 AF2.3，**没有 AF3/Boltz/Protenix** |
| 抗体-抗原 0.529 vs AF2.3 0.380 | 表格（Table 2）+ 图 4 | 【核实】但 AF3 缺位，作者理由是 `AF3 restricts commercial use`  |
| single-sequence 仍能打（PP 0.698 / PA 0.479 / 单体 0.852） | 表格（Table 2） | 【核实】；原文同时承认 0.698 vs 0.677 的差异 `not statistically meaningful`  |
| CASP15 单体 Cα-LDDT 0.849 vs AF2.3 0.843 | 表格 + 逐靶统计 | 【核实】口径是 **69/70**（剔除 T1169，因"无法在合理时间内完成推理"）；ESM3-98B 的 0.801 是 ESM3 自报，非作者复跑 |
| 少量 contact 约束把 DockQ 可接受率 35%→57% | 曲线（图 4） | 【核实】原文 `increases from the baseline of 35% to 57%`  |
| 4 个表位残基使成功率翻倍以上 | 曲线（图 4） | 【核实】原文 `more than doubles the DockQ success rate`；但作者同时承认"高质量仍只有 4-8%"  |
| "单模型通吃、无切点泄漏" | 仅文字 | 【核实】切点确实写了；但"所有评测集都在切点后"这句只对 PDB 成立，CASP15 靶（2023 年）与 PoseBusters（2024 年）当然也在其后；**AFDB 训练数据本身的日期未见披露** → 对 AFDB 部分【未核实】 |
| 核酸性能"与 RF2NA 相当" | 仅文字 + 附图 S3 | 【未核实】图 S3 在补充材料，本次未取 |
| 权重可商用 | README | 见 §6.2 冲突 |

### 6.2 论文 vs 代码：冲突清单（本篇最关键）

| # | 事实 | 论文怎么说 | 代码/仓库怎么说 | 影响 |
|---|---|---|---|---|
| C1 | **权重许可** | `available for non-commercial use`（p.1 摘要、p.2 §1） | `Apache 2.0 License (both code and model weights)... academic and commerical purposes`（README L189）；LICENSE 为 Apache 2.0 全文  | 两侧口径并列，不择一：Apache-2.0 见 LICENSE 全文与 README L189，non-commercial 见论文 p.1 摘要、p.2 §1；对外合规判断须法务确认 |
| C2 | **recycling 次数** | `4 recycles`（p.10） | `num_trunk_recycles: int = 3`（chai1.py） | 复现"论文数字"要显式设 4；默认跑出来与论文不同 |
| C3 | **采样预算** | 5 trunk samples × 5 diffusion = 25（p.10） | 默认 `num_trunk_samples=1`、`num_diffn_samples=5` ⇒ **5 个** ；README L32 亦写默认 5 个  | **默认只跑论文预算的 1/5**；只跑默认值就对不上论文任何数字 |
| C4 | **clash penalty** | `we did not have to use a clash penalty for sample ranking`（图 2 注） | `aggregate_score = 0.2*complex_pTM + 0.8*ipTM - 100*has_inter_chain_clashes`（rank.py L94-99） | 排序行为与论文描述不一致；有链间冲突的样本会被 −100 硬压 |
| C5 | **约束特征编码** | 只说 docking 是 one-hot 四箱；pocket/contact 未说编码 | pocket/contact 用 `EncodingType.RBF`、6 个径向基、min 6 / max 20 与 6 / 30；docking 才是 one-hot 四箱（`[0,4,8,16]` + 1 类） | 做"约束强度"实验时，RBF 的饱和行为与 one-hot 不同，论文没提醒 |
| C6 | **σ 上界** | 未写 | `s_max` 实参取 `S_tmax=80`，且整条调度再乘 `sigma_data=16` ⇒ σ 实际到 1280（AF3 是 160） | 想"对齐 AF3 噪声尺度"做消融，必须先换算 |
| C7 | **churn 窗口** | 未写 | γ 的生效条件是 `(sigmas>=4e-4)&(sigmas<=80)`，而 `sigmas` 已 ×16 ⇒ 高噪声段不 churn  | 【推断】疑似量纲不一致；会影响样本多样性与重现 |
| C8 | **训练时约束注入概率** | 每类特征独立以 10% 概率加入（p.9） | 推理代码 `include_probability=1.0`（这是推理默认，不算冲突）；训练代码未开源 | 10% 无法核实，标【未核实】 |
| C9 | **是否提供 docking 专用权重** | Table 1 列出 `Chai-1 - Docking 81.20%`  | 开源包只有一个模型目录（6 个组件），未见 docking 专用权重；CLI 也只接受 covalent/contact/pocket | 【未核实】该权重是否对外发布；复现 81.20% 目前无路径 |
| C10 | **"不使用 MSA"的措辞** | 摘要与 p.3 强调 without MSA  | single-sequence 仍跑 3B ESM（默认开） | 易被误读为"无任何进化信息"；§5.1.1 有交代，摘要没有 |

C2/C3/C4 三条是实操层面最容易踩的：**默认值跑出来既不是论文的采样预算，也不是论文的排序规则**。

---

## 7. 评测协议与成本

### 7.1 评测集与指标

| Benchmark | 规模 | 指标 | 预算 | 来源 |
|---|---|---|---|---|
| PoseBusters V1 | **427** 结构（原 428，剔除已 obsolete 的 7D6O；>2048 tokens 的 4/428 按"距配体最近的 2048 token"裁剪） | ligand RMSD ≤2 Å 成功率 | 25 样本（论文） | 【核实】 |
| 蛋白-蛋白低同源 | 2,362 界面 / 1,054 结构 / 929 聚类 | DockQ >0.23 成功率（按聚类平均） | 25 | 【核实】 |
| 蛋白-抗体低同源 | 268 界面 / 129 结构 / 122 聚类 | DockQ 成功率 | 25 | 【核实】；聚类数见 p.5 |
| 蛋白单体低同源 | 同一时间切分 | Cα-LDDT | 25 | 【核实】Table 2  |
| CASP15 单体 | 70 → **69**（剔 T1169） | LDDT | 未单列 | 【核实】 |
| CASP15 RNA | 9 靶 | C1′ LDDT | — | 【核实】 |
| 蛋白-核酸低同源 | 未给规模 | interface LDDT（iLDDT） | — | 【核实】p.10 §5.5  |

评测集构造（p.11）：PDB 2022-05-01–2023-01-12、非 NMR、分辨率优于 4.5 Å、≤2048 tokens、40% 同源剔除 。

### 7.2 基线数值（抄表，注明出处）

**Table 1（p.12）：PoseBusters V1，ligand RMSD ≤2 Å**

| 方法 | 成功率 |
|---|---|
| Chai-1 | **77.05 %** |
| AF3（引用其公开预测） | 76.34 % |
| RF2AA | 42 % |
| Chai-1 - Docking | **81.20 %** |

**Table 2（p.12）：mean (95% CI, 10,000 bootstrap)**

| 方法 | 蛋白-蛋白 | 蛋白-抗体 | 蛋白单体（Cα-LDDT） |
|---|---|---|---|
| Chai-1 (with MSA) | **0.751** (0.723, 0.778) | **0.529** (0.438, 0.620) | **0.915** (0.907, 0.922) |
| Chai-1 (single-sequence) | 0.698 (0.668, 0.728) | 0.479 (0.388, 0.570) | 0.852 (0.834, 0.867) |
| AlphaFold 2.3 multimer | 0.677 (0.646, 0.706) | 0.380 (0.298, 0.463) | 0.903 (0.895, 0.911) |

其它：CASP15 单体 0.849（AF2.3 0.843；AF2.3 弱项子集 n=14 上 0.643 vs 0.552，p=3.66e-4）；抗体 DockQ 随约束从 35%→57% 。

**基线口径三条警示**：① AF3 的 PoseBusters 数字是二手引用、非复跑 ；② PP/PA 完全没有 AF3/Boltz 对照，作者称因 AF3 商用限制 ；③ CASP15 剔除了一个推理超时靶（T1169），剔除方向对均值有利 。

### 7.3 成本

| 项 | 值 | 来源 |
|---|---|---|
| 训练 | 128×A100，batch 128，30 天 | 【核实】 |
| 推理显存 | 论文**未披露**；README 建议 A100 80GB / H100 80GB / L40S 48GB，小复合物可 A10/A30，社区反馈 RTX 4090 可用 | 【核实】 |
| 推理时长 | **未披露**（论文无 wall-clock 表；只提到 T1169 `could not complete inference in a reasonable timeframe`） | 【核实】 |
| 单次默认开销 | 200 步 × 5 samples + 3 次 trunk + 一次 3B ESM 前向 | 【代码／配置核实】 |
| 论文预算 | 200 步 × 25 samples + 4 次 trunk | 【核实】 |

一个容易忽略的成本项：**ESM-2 3B 是常驻开销**，即使 single-sequence 也要跑 ；做大批量 VNAR 扫描时它很可能比 diffusion 更早吃满显存。

---

## 8. 对我们的可复用点（VNAR 专列）

### 8.1 能不能当 VNAR / 纳米抗体预测器？

**能，但要限定用法。** 依据：

- VNAR/VHH 是单链单域（~120 aa）+ 抗原，正好落在"蛋白-蛋白/蛋白-抗体"评测形态内；Chai-1 的 PA 评测集是 268 个界面/129 个结构 ，其中含单链抗体类界面（论文按"是否判为 antibody"筛选，未区分 VHH/VNAR 与 Fab）。
- 无 MSA 也能跑，这对鲨鱼/驼类胚系在 UniRef 里覆盖稀疏的场景很关键：single-sequence 的 PA 从 0.529 掉到 0.479 ，掉幅 ~9.5%，比 PP 的掉幅（0.751→0.698，~7%）略大，属可接受但不可忽略。
- 更有价值的是**表位条件**：一条 15 Å 的抗体-抗原距离约束就把 DockQ 可接受率从 35% 提到 57%，4 个表位残基再翻倍 。VNAR 场景常常有丙肝/疟疾类表位知识（如已知 CDR3 接触残基、竞争表位），用 `connection_type=pocket` 喂进去即可，**不需要重训**。

### 8.2 MSA 依赖强度

| 场景 | 需要什么 | 备注 |
|---|---|---|
| 有 MSA | `--use-msa-server`（公共 MMseqs2，有限流）或自带 `.aligned.pqt` | 自带更稳：VNAR 胚系短，公共库常给出很浅的 MSA  |
| 无 MSA | 默认就是；但要付 ESM-2 3B 的算力 | `use_esm_embeddings` 默认 True  |
| 完全离线 | 模型 6 组件 + ESM 权重 + 参考构象 `conformers_v1.apkl` 预下载后可离线；MSA/模板服务需自建 | 【代码／配置核实】 |

### 8.3 能否微调

**不能，代码层面不成立。** 仓库无训练脚本、无 loss、无数据管线，模型是 TorchScript 编译图（`trunk.pt` / `diffusion_module.pt`）。要做 VNAR 微调只有三条路：① 向厂商要训练代码（非公开）；② 蒸馏到可训练骨架（Boltz-1/Protenix）；③ 用约束/模板做"提示工程"——**当前唯一可行的是 ③**。

### 8.4 抗体相关能力盘点

| 能力 | Chai-1 | 说明 |
|---|---|---|
| antibody 模式 | **无** | 全仓库 grep `antibody|nanobody` 零命中；对比 Chai-2 才有抗体设计 |
| 配对链（重/轻链）处理 | 无专门机制 | 通过 chain 分组 + `pairing_key` 走通用多链逻辑 |
| CDR 编号/掩码 | 无 | 依赖里出现 `anarci` 但包内未 import【代码／配置核实】 |
| 表位条件 | **有**（pocket 约束） | 最实用的一条  |
| 抗原多聚态 | 用户自己给 FASTA | 与 JCIM 纳米抗体基准的做法一致 [S3] |
| affinity / 排序 | 无亲和力头 | 只有 pTM/ipTM/pLDDT + clash 排序  |

### 8.5 硬边界（写清楚，别踩）

1. **许可**：论文说非商用、README/LICENSE 说 Apache-2.0（含权重）——引用时按 Apache-2.0 走，但保留一句"论文口径相反"。
2. **无训练代码**：不能微调，不能做 VNAR 专用 head。
3. **不支持链内二硫键**：`Chai-1 was not trained on intra-chain bonds (e.g., disulfides)` 。VNAR 的 canonical 半胱氨酸（以及 CDR3 里的额外二硫键）**无法作为约束输入**，只能靠模型自己学到的几何——这是 VNAR 场景最实际的精度风险点。
4. **2048 token 上限**：VNAR-抗原通常远低于此，但"VNAR + 大抗原多聚体"要裁剪 。
5. **默认采样只有 5 个**：论文数字是 25 个，做基准必须显式设 `--num-trunk-samples 5`（或等价）。
6. **架构不透明**：无法做层间分析、无法改 block、无法算参数量。

---

## 9. 疑问、心得与参考

### 9.1 未解问题

1. **trunk/diffusion 的真实规模**。理论上可从 `trunk.pt` 的 TorchScript `graph` 反查模块层次，但需先下数百 MB 权重并做图遍历；本次未做，这是后续唯一硬路径。
2. **AFDB 的日期切点**。论文只写 PDB 用 2021-01-12，AFDB 版本/生成日未写；AFDB 是 AF2 预测的全库，可能覆盖到 2022 年。若如此，训练集里就带进了"2022 年序列"的信息（预测结构而非实验结构，但仍有同源风险）。**最值得追问的一处**，标【未核实 + 推断】。
3. **docking 专用权重是否发布**（C9）。
4. **churn 窗口的量纲**（C7）到底是 bug 还是刻意。
5. **`confidence` 字段为何保留而不生效**：是不是留给未来 API 兼容。
6. 补充材料（图 S3、S4 等）本次未取；核酸 iLDDT 的具体数值因此缺失。

### 9.2 写作心得

- **"AF3 式"这个词在 Chai-1 上必须拆开用**：可以说"AF3 式主干 + 非 AF3 的采样器 + 非 AF3 的 MSA 策略"，但不能说"AF3 式 48 block"，因为一个数字都没有。
- **切点这类事实要用"三处交叉"法**：摘要（口径）、方法（参数）、评测集（用途）三段都读，才能把 2021-01-12 / 2022-05-01 / 2023-01-12 / 2021-09-30 四个日期区分开。只看摘要会以为"未披露"，只看评测集会以为"2022-05-01 是切点"。
- **许可证必须查 LICENSE 文件而不是摘要**：这篇是典型反例。
- 代码取证顺序建议：`README`（承诺）→ `chai1.py`（默认值）→ 具体 feature/tokenizer（语义）→ `ranking/`（隐藏的排序规则）。C3/C4 都是这样挖出来的。

### 9.3 参考（分组）

**论文（一手）**
- Chai Discovery Team. *Chai-1: Decoding the molecular interactions of life.* bioRxiv 2024.10.10.615955 (v1, 2024-10-11). DOI: [10.1101/2024.10.10.615955](https://doi.org/10.1101/2024.10.10.615955)
- 官方介绍页：https://www.chaidiscovery.com/news/introducing-chai-1 （2024-09-09；披露 PoseBusters 77%、CASP15 0.849、PP 69.8% 等口径，与论文一致）

**代码／配置核实点（路径 + 行）**
- `README.md` L21、L29-37、L82、L189（硬件、默认 5 样本、MMseqs2、许可证）
- `LICENSE`（Apache 2.0 全文）
- `chai_lab/chai1.py` L242-248（`DiffusionConfig`）、L480-500（`run_inference` 默认值，含 `num_trunk_recycles=3`、`num_diffn_samples=5`、`num_trunk_samples=1`、`use_esm_embeddings=True`）、L822-832/L859/L877（σ 实参、churn 窗口、二阶更新）、L679-894（6 个 TorchScript 组件与下载基址）
- `chai_lab/model/diffusion_schedules.py` L15-18、L21-39（σ 常数与 `sigma_data * power_interp`）
- `chai_lab/data/dataset/structure/all_atom_residue_tokenizer.py`（逐原子/atom37、`assert max_index <= 36`、中心原子 `CA`/`C1'`）
- `chai_lab/data/parsing/restraints.py`（CSV 列名、`connection_type` 三值）
- `chai_lab/data/features/generators/token_dist_restraint.py`、`token_pair_pocket_restraint.py`、`docking.py`（RBF vs one-hot 四箱）
- `chai_lab/data/dataset/embeddings/esm.py`（`esm2_t36_3B_UR50D`、fp16、CPU 默认）
- `chai_lab/ranking/rank.py` L53-55、L94-99（clash 阈值、aggregate score）
- `chai_lab/data/collate/utils.py`（`AVAILABLE_MODEL_SIZES` 到 2048）
- `examples/restraints/README.md`、`examples/msas/README.md`、`examples/covalent_bonds/README.md`

**二手转述（不作为一手证据）**
- 36kr《DeepMind 再迎挑战者…》（2024-10）：转述切点 2021-01-12，与论文一致。[S1]
- LiPP benchmark, *J. Chem. Inf. Model.* 2026：以 "October first 2021" 作为五种方法的统一保守切点——**评测方的假设，不是 Chai-1 的披露值**。[S2]
- *J. Chem. Inf. Model.* 病毒纳米抗体表位预测基准：Chai-1 服务器设置记录为 "3 recycling steps and 5 diffusion samples per seed, MMseqs2 for MSA, templates and restraint specification enabled"——与代码默认吻合、与论文 4 recycles 不同。[S3]
- 本库旧笔记 `2026-11-04\af3-model-note\outputs\AlphaFold3.md` L289："Chai-1 未披露"切点——本次核实推翻。[S4]
