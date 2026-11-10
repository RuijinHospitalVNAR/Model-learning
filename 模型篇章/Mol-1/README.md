```yaml
model: Mol-1
aliases:
  - Mol-1 Base
  - Mol-1-constraint（表位约束变体，仅网页端）
  - molence/Mol-1
family: 全原子结构预测 · AF3/Protenix 谱系（Protenix 派生，Molence Bio 2026 技术报告）
granularity: 残基级 token（≤24 原子/token）+ 配体每重原子 1 token；输出重原子全原子坐标
backbone: 48 层异构 pair trunk（8×PairMixer 448C → 40×Pairformer 320C）+ 4 层 MSA module；diffusion transformer 24 层（token 768C）+ atom encoder/decoder 各 3 层；577.4M 参数
generative: EDM 扩散；σ_data=16，σ_max=160 / σ_min=4e-4 / ρ=7；默认 200 步 × 5 样本 × 20 recycle（训练 K≤4，Beta 课程）
conditioning: 蛋白/DNA/RNA/配体（SMILES 或 CCD）/离子/修饰/共价键；MSA 必需；模板默认关（.a3m/.hhr 受 2021-09-30 截断，JSON 内联模板不过滤）；constraint 变体未开源
data_scope: PDB ≤2021-09-30；索引名标注 ≤4.5 Å、≤20 链；条目数/去重/聚类切分/蒸馏数据报告均未给
status: Apache-2.0，base 权重 577.4M（≈2.2 GB）可跑；constraint 变体、agent、VHH 设计流程均未开源
verified_at: 2026-11-10
```

# Mol-1（Molence Bio，2026）

> **三句话速判**
> 1. Mol-1 是 AF3/Protenix 架构线的一次“重新配平”：8 层宽 PairMixer（448C，去掉三角注意力）＋ 40 层窄 Pairformer（320C），共 48 层、577.4M 参数；报告称 FLOPs 只有同宽 48 层 Pairformer 的约 60%，拟合相当【论文自称，p.24】。
> 2. 它真正拉开差距的是**抗体–抗原**：FoldBench-AB 上 SR@DockQ>0.23 = 75.0%，Protenix-v1 52.3%、AF3 48.8%【论文自称，p.1/p.4】。但 0.23 是“可接受构象”门限，不等于高精度——同一套权重在蛋白–肽上 0.23 / 0.49 / 0.80 三段是 88.84% / 77.28% / 54.07%，掉得很快【论文自称，p.7 Figure 5】。
> 3. 对我们的直接价值：开源的 base 可以当 **VNAR–抗原复合物的预测器 + confidence 打分器**，但仓库里**没有设计接口、没有 constraint 接口、也没有任何鲨源 VNAR 验证**；84 条样本只够做 adapter/末几层式的局部微调，不够全权重微调【推断】。

---

## 1. 定位与谱系

### 1.1 是什么

Mol-1 是 Molence Bio 2026 年技术报告（39 页，`assets/report.pdf`）的主模型，仓库 `molence/Mol-1`，commit `e6e6486`（2026-10-30），Apache-2.0【核实：仓库元数据】。

模型家族：

| 成员 | 开源 | 说明 | 标记 |
|---|---|---|---|
| Mol-1 Base | ✅ 权重 2.2 GB | 通用全原子结构预测；本笔记主要对象 | 【核实】 |
| Mol-1-constraint | ❌ 仅网页 | 在 base 上冻结除 conditioner + 最后 8 层外的参数，48.7M 可训练（8.4%），专做表位/接触约束 | 【论文自称，p.32】 |
| 长时程 agent | ❌ | 用 Mol-1 反复预测–扩展–装配，单卡 H20 跑数十小时，最大 60k token | 【论文自称，p.9】 |
| de novo VHH 设计流程 | ❌ | “用早期 checkpoint”，不额外训扩散模型；方法细节未披露 | 【论文自称，p.11】 |

### 1.2 与 Protenix 的关系：不是纯复现

代码多个核心文件头写着 `Based on Protenix ... upstream revision bd54a05`，并逐文件列出 `# Changes:`。汇总下来至少改了六处【代码／配置核实】：

1. trunk 换成 PairMixer（无三角注意力）+ Pairformer 混排，pair 通道 448 → 320 两级；
2. 三角核换成 FlashPairformer 实现；
3. 移除 ESM 输入分支；
4. diffusion 侧加 conditioning cache（把 trunk 的单/对表征缓存后复用），并复用 pair bias；
5. 数据管线重写（聚类权重、crop 混合、按日期/分辨率筛选，见 §5）；
6. 单独训了 constraint conditioner 做表位条件化。

因此它**不是**“Protenix 换个权重”，但也不是全新范式：MSA module、diffusion module、confidence head 仍沿用 AF3/Protenix 的算法骨架【核实：`mol/config/base.py:230-270` 的层数与 AF3 Supplement 一致】。

### 1.3 与我们已有模型的相对位置

- 对比 **Protenix-v1**：同一评测口径下抗体 SR 高 22.7 个百分点（75.0 vs 52.3）【论文自称，p.4】；参数量 577.4M，比 Protenix-v1 大，但报告没有给 FLOPs 对齐表。
- 对比 **AF3**：抗体明显更强（75.0 vs 48.8），配体在低同源段更强（36.1% vs 24.6%），但配体总体仍落后 Boltz-2（报告的 Table 3 里 Boltz-2 更高，Mol-1 只赢 0–20% 相似度段）【论文自称，p.6】。
- 对比 **OpenDDE**：抗体四个基准里 Mol-1 全胜，但差距很薄，从 +5.0 pp（FoldBench-AB，75.0 vs 70.0）到 +7.3 pp（PXMeter-AB，58.3 vs 51.0），中间 AF3-AB +7.1、MolBench-AB-26 +5.5【论文自称，p.4 Figure 1】。

---

## 2. 表征粒度与表示层

### 2.1 token 化规则

| 实体 | token 规则 | 标记 |
|---|---|---|
| 标准氨基酸 | 1 残基 = 1 token，token 内最多 24 个重原子（`max_atoms_per_token=24`） | 【代码／配置核实】 |
| 配体 | **每个重原子 1 个 token** | 【代码／配置核实】 |
| 修饰残基/非标准残基 | 每个重原子 1 个 token | 【代码／配置核实】 |
| DNA/RNA | 1 核苷酸 = 1 token（≤23 原子） | 【代码／配置核实】 |
| 离子 | 按配体路径处理 | 【代码／配置核实】 |

这决定了 PAE 的索引语义：`pae` 是 `(n_token, n_token)`，蛋白区是残基索引，配体区是原子索引【代码／配置核实：`docs/output_format.md`】。原稿强调的“PAE 按 token 索引”仍然成立，且是踩坑点。

### 2.2 原子级特征

`c_atom=128`、`c_atompair=16`、`c_token=384`、`c_s_inputs=449`（=384+32+32+1）【代码／配置核实：`mol/config/base.py:115-121`】。原子特征含 `ref_pos / ref_mask / ref_element / ref_charge / ref_atom_name_chars / atom_to_token_idx`，在 diffusion 的 atom attention encoder 里作为残基内局部几何【代码／配置核实】。

### 2.3 输出与两套 confidence 量纲（原稿强项，保留）

同一个 JSON 里，**原子级和链级 pLDDT 的量纲不同**，混用会得到 100 倍误差【代码／配置核实：`docs/output_format.md`】：

| 字段 | 量纲 | 用途 |
|---|---|---|
| `atom_plddts` | 0–1 | 逐原子着色/过滤 |
| `plddt`（per-token） | **0–100** | residue-level 排序 |
| `chain_plddt` | 0–1 | 链级质量 |
| `pae` | Å，token×token | 域间/链间界面判断 |
| `ptm` / `iptm` | 0–1 | 全局/界面置信度，官方按 `iptm` 排序样本 |

另外 `chain_pair_pae_min`、`chain_ptm`、`chain_iptm`、`pde` 都在 `confidence-*.json` 里，不额外占显存；但 **2,456 token 时 confidence head 会 OOM**，与扩散主干的显存曲线不是同一条【代码／配置核实：`docs/inference_memory.md`】。

---

## 3. 主干与生成范式

### 3.1 结构主干（报告 p.23 + 代码对齐）

| 模块 | 配置 | 标记 |
|---|---|---|
| MSA module | 4 层 | 【代码／配置核实：`base.py:230-232`】 |
| trunk 前段 | 8 × PairMixer，`c_z_pairmixer=448`；PairMixer = 保留三角乘法/对转移/单表征更新，**去掉三角注意力** | 【代码／配置核实：`pairformer.py:237-275`】 |
| 投影 | 448 → 320 | 【代码／配置核实：`base.py:115-117`】 |
| trunk 后段 | 40 × Pairformer，`c_z_pairformer=320`，三角注意力用 FlashPairformer | 【论文自称，p.23；代码／配置核实】 |
| 总层数 | 48（`n_blocks=48`, `n_pairmixer_blocks=8`） | 【代码／配置核实：`base.py:122,134`】 |
| 单表征 | `c_s=384`，`c_s_inputs=449` | 【代码／配置核实】 |
| 参数量 | **577.4M** | 【论文自称，p.23/p.32】 |

报告给了一个有用的消融（Figure 16）：与“48 层全 Pairformer、448 通道”的 FLOPs 对齐基线相比，Mol-1 的 8+40 混排在**相同训练算力下拟合更好**，并且声称 FLOPs 约为基线的 60%【论文自称，p.24】。注意这是**训练算力轴**上的比较，不是同参数量比较。

### 3.2 扩散模块

沿用 AF3 Algorithm 20 的三段式【代码／配置核实：`mol/model/modules/diffusion.py`】：

1. **Atom attention encoder**：3 层 atom transformer，局部注意力窗口 `n_queries=32 / n_keys=128`，并通过 cross-attention 读取 token 级条件（trunk 的 `s`/`z` 投影 + σ embedding）——这就是常说的 “atom cross-attention”。
2. **Diffusion transformer**：24 层，`c_token=768`，在 token 级做全局注意力。
3. **Atom attention decoder**：3 层，回到原子级输出坐标更新。

训练时对每个 token 的原子做随机子采样（`max_atoms_per_token=24` 的上限内），推理时用全部原子【代码／配置核实】。

### 3.3 训练配方：recycle 与 crop 联合课程

报告明确写了训练期的 recycle 课程，这是它和 Protenix 默认配置的差异之一【论文自称，p.24】：

- 最大 recycle 深度 `K=4`；
- 采样分布从 `Beta(1,1)`（均匀 1–4）切到 `Beta(2,2)`，再到 `Beta(2,1)`（偏向浅层）；
- crop size 与 recycle 深度**联合调度**（Figure 16），先用小 crop 配浅 recycle，再放大 crop。

代码里训练 crop 默认 256，官方微调脚本用 384；crop 方式权重是 contiguous : spatial : interface = 0.2 : 0.4 : 0.4【代码／配置核实：`base.py:66`、`data.py:61-64`、`docs/finetune.md`】。

### 3.4 关键纠正：σ 调度与步数不是“未披露”，是“报告没写、代码写了”

这是重写本笔记最需要修正的一点。01 矩阵里 Mol-1 行原写“σ 调度未披露”“步数未披露”，实际是**报告确实没写，但开源配置写得很清楚**【代码／配置核实：`mol/config/base.py:184-207`】：

| 参数 | 值 | 位置 |
|---|---|---|
| 训练噪声 | `p_mean=-1.2`, `p_std=1.5`, `σ_data=16` | `base.py:184-188` |
| 推理 σ 上限 | `s_max=160`（即 σ_max） | `base.py:190` |
| 推理 σ 下限 | `s_min=4e-4`（即 σ_min） | `base.py:191` |
| 调度指数 | `ρ=7` | `base.py:192` |
| σ_data | 16（注意 EDM 原文是 1.0） | `base.py:193` |
| 默认采样步数 | **200** | `base.py:200` |
| 默认样本数 | **5** | `base.py:201` |
| 默认 recycle | **20** | `base.py:207` |
| 采样器 | γ0=0.8、γ_min=1.0、η=1.5 | `base.py:196-199` |

这组值与 AF3 的公开默认值一致（σ_data 16、σ_max 160、σ_min 4e-4、ρ=7、200 步），说明 Mol-1 没有改扩散调度，只改了主干与数据【推断，依据是数值完全重合】。报告本身对 σ/步数**一个字都没给**（p.23-24 只讲结构、参数、recycle 课程）。

---

## 4. 条件控制

### 4.1 输入面

`mol predict` 吃 JSON（或 `mol json` 从 CIF/PDB 转）。支持：蛋白链、DNA/RNA、配体（SMILES 或 CCD code）、离子、残基修饰、共价键、多链复合物；每条链可给 `unpairedMsaPath` / `pairedMsaPath` / `templatesPath`【代码／配置核实：`docs/infer_json_format.md`】。

**没有** symmetry 输入、**没有** constraint/表位输入、**没有**设计（序列生成）接口【代码／配置核实：`inference.py` 无相关字段，仓库无 design 模块】。

### 4.2 MSA 与模板：默认关、且有日期截断

- `use_msa: True`、`use_template: False` 是默认值【代码／配置核实：`inference.py`】。也就是说，**不显式开模板时，Mol-1 只靠 MSA**。
- 若用 `.a3m/.hhr` 模板，只取 **2021-09-30 之前**释放的模板（与训练截止一致），避免用测试集同期的结构【代码／配置核实：`docs/infer_json_format.md`】。
- 无 MSA 也能跑（`--use_msa false`），但官方明说会显著掉点，属于应急路径【代码／配置核实：`docs/infer_json_format.md`】。

### 4.3 JSON 内联模板：唯一绕过检索的通道

JSON 里可以直接塞 `templates` 数组，每项含 mmCIF 文本和 `queryIndices / templateIndices`。行为差异很关键【代码／配置核实：`docs/infer_json_format.md` §模板】：

| 行为 | `.a3m/.hhr` 模板 | JSON 内联模板 |
|---|---|---|
| 是否跑比对 | 是 | **否**（`No alignment is run`） |
| 2021-09-30 日期过滤 | 是 | **否** |
| 取模板哪条链 | 命中链 | **恒取第一条链** |
| 索引映射 | 自动 | 必须 1:1 手写 |

对我们的意义见 §8.2。

### 4.4 constraint 变体：只在网页端

报告里的 Mol-1-constraint 在四个抗体基准上把 SR 从 75.0/62.7/58.3/57.3 抬到 88.3/84.2/77.5/79.4（FoldBench-AB / AF3-AB / PXMeter-AB / MolBench-AB-26，DockQ>0.23，base 20 cycles、constraint 10 cycles）【论文自称，p.5 正文 + p.4 Figure 1】。做法是冻结除 constraint conditioner 与最后 8 层外的全部参数，只训 48.7M（8.4%），128 GPU、2,000 iters、384-token crop、每结构 32 个扩散样本、bf16、EMA 0.999【论文自称，p.32】。

但 conditioner 的**网络结构、输入编码（表位残基是 one-hot 还是 pair bias 注入）、损失权重都没写**；开源代码里 `constraint` 只出现在数据排除逻辑（不采离子），与设计约束无关【代码／配置核实：`mol/config/data.py:98`】。结论：**本地无法复现 constraint 变体**。

---

## 5. 训练数据构成与切分

### 5.1 报告只说了四句话

A.1 的数据段原文只有：PDB **2021-09-30 之前**释放的结构全部进训练；“在数据获取与清洗、系统质量过滤、合成数据生成上投入了大量工作”（substantial effort），然后直接进实现细节【论文自称，p.23】。

没有给：条目数、链数、界面数、分辨率分布、去重/同源阈值、聚类切分方式、是否做时间切分、是否用 AFDB/AF2/Protenix 预测结构蒸馏。全文检索 `distill / AFDB / AlphaFold DB / predicted structure` **零命中**【核实：全文关键词检索】。

### 5.2 代码能补出的部分（注意口径）

仓库随附的训练/微调数据配置能补出一部分**过滤与采样**，但它服务的是“用户微调/示例训练”，不能直接等同于 base 训练全集【代码／配置核实】：

| 项 | 值 | 位置 |
|---|---|---|
| 训练索引文件 | `indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz` | `data.py:90-93` |
| 由文件名可读出的过滤 | ≤20 链、≤4.5 Å、2021-09-30 截止、索引构建于 2026-01-07 | 同上【推断，依据文件名】 |
| 采样权重 | chain β=0.5 / interface β=1.0；prot α=3、nuc α=3、ligand α=1 | `dataset.py:978-1000`、`data.py:55-58` |
| crop 方式 | contiguous 0.2 / spatial 0.4 / interface 0.4 | `data.py:61-64` |
| 测试集 | `recentPDB_1536_sample384_0925`：低同源、≤1536 token、抽 384 | `data.py:83,105-116` |
| 发布版 MSA 包 | 覆盖 **7,074** 条 PDB 序列（供微调用） | `docs/data_sources.md` |

**重要口径**：`7,074` 是**发布包**里随附 MSA 的 PDB 序列数，不等于 base 训练用的结构数，也不等于界面数；把它写成“Mol-1 训练集规模”是错的【推断，依据 finetune.md 与 data_sources.md 的上下文】。

### 5.3 仍然无法回答的问题

1. **规模**：训练用了多少 PDB 条目/链/界面？报告未给。
2. **去重与切分**：聚类阈值、训练/验证如何隔离、测试集低同源是“按序列 identity 还是按 cluster”都没写；只给了测试集文件名暗示 `low_homology`。
3. **蒸馏数据**：Protenix 代码里保留 `Distillation` dataset 类，但 Mol-1 的 `data_configs` 只引用 `weightedPDB_before210930_v20260101`，**没有任何蒸馏数据配置**；因此“用没用 AFDB 蒸馏”答案是：**报告未给，配置里没有证据**【代码／配置核实】。
4. **合成数据**：报告说做了 “synthetic data generation”，但没说是结构蒸馏、还是 MSA/模板增强、还是负样本合成。

这是本笔记最大的空白，也是“不能把没写当成没做”的典型：数据部分只能标记“报告未给”，不能推断成“Mol-1 没用蒸馏”。

---

## 6. 宣称与可验证性

| 宣称 | 位置 | 可验证性 | 标记 |
|---|---|---|---|
| FoldBench-AB SR@0.23 = 75.0%（Protenix-v1 52.3） | p.1/p.4 | 基准公开、权重开源、协议写明（5 seeds × 5 samples × 20 cycles），**原则可复跑**；本次未复跑 | 【论文自称】 |
| MolBench-AB-26 = 57.3%（正文）/ 58.6–58.8%（Figure 17 加大 seeds/cycles） | p.3/p.26 | 同上；注意正文与图的采样预算不同 | 【论文自称】 |
| constraint 88.3/84.2/77.5/79.4% | p.5 | 变体未发布、conditioner 未描述 → **不可复现** | 【论文自称】 |
| 60k token、单卡 H20、每装配体“tens of hours”、8/10 TM≥0.90 | p.9-10 | 单模型可试，但 agent 未开源；9/10 优于 Claude Science 的对照流程也不可对等复核 | 【论文自称】 |
| VHH：13 靶点 30 候选、4 项 assay、round1 2/30 → round2 51/60、4-1BB KD 3.5/103 nM、BCMA 43 pM | p.11-14 | 无序列、无原始曲线、流程未描述、且用的是**早期 checkpoint** → **不可复现** | 【论文自称】 |
| “without training an additional diffusion model” | p.11 | 设计流程未给，无法判断是 inpainting、hallucination 还是别的 | 【论文自称】 |
| 数据“fair evaluation”（训练截止 2021-09-30） | p.23 | 截止日期可核；但同源阈值/切分细节未给 | 【论文自称】/部分【核实】 |

另外两点必须照实写：

- 报告自己在 A.2 承认 **各模型评测 cohort 不完全匹配**（不同论文的抗体集/过滤不同），所以跨模型的绝对差值要打折看【论文自称，p.25】。
- **MolBench-AB-26 / MolBench-Peptide-26 是 2026 年新基准，报告未说明维护方与数据许可**；把它当作中立第三方基准引用有风险【未核实】。

---

## 7. 评测协议与成本

### 7.1 协议（p.25）

- 检索/采样：每靶标 **5 seeds × 5 samples**，按 `iptm` 排序取 top-1；
- recycle：base **20 次**；配体/分子胶 **10 次**；constraint 变体 **10 次**；
- 成功判据：抗体/肽用 DockQ 阈值 0.23 / 0.49 / 0.80；配体用 Runs N' Poses；glue 用 MGBench；
- 基准规模：FoldBench-AB 160 个界面；MolBench-AB-26 242 个界面/101 个 PDB，取交集后 **172 簇/150 条目**【论文自称，p.25】。

### 7.2 结果（统一 SR@DockQ>0.23，%，越高越好）

| 基准 | Mol-1 | constraint | OpenDDE | Protenix-v1 | AF3 | ESMFold2 | OpenFold3 |
|---|---|---|---|---|---|---|---|
| FoldBench-AB | **75.0** | 88.3 | 70.0 | 52.3 | 48.8 | 56.2 | 34.9 |
| AF3-AB | **62.7** | 84.2 | 55.6 | 40.4 | 47.0 | 39.6 | 30.9 |
| PXMeter-AB | **58.3** | 77.5 | 51.0 | 40.2 | —（图中无 AF3） | 48.1 | 27.9 |
| MolBench-AB-26 | **57.3** | 79.4 | 51.8 | 41.4 | —（图中无 AF3） | 49.3 | 27.3 |

> 数值来源：report p.4 Figure 1a–d（逐柱读图核对，非文本抽取），Mol-1 base 20 cycles、constraint 10 cycles（A.2, p.25）【论文自称】。Boltz-1 只进了三张图：34.4（FoldBench-AB）/ 14.9（AF3-AB）/ 19.6（PXMeter-AB），MolBench-AB-26 未列。
>
> 口径警告：`58.3` 是 Mol-1 在 PXMeter-AB 的成绩，`77.5` 是 constraint 的成绩——两者相差 19.2 pp，不要把 77.5 当成 base 的数字；FoldBench-AB 的 75.0 才是 base 的最高分。另外 A.2 承认各模型的评测 cohort 不完全匹配（PXMeter 源 cohort：OpenDDE 494 条目/364 簇，Protenix-v2 报告的基线 516/376，Mol-1 自评 498/361），跨模型差值要打折看【论文自称，p.25】。

非抗体任务（报告口径）：

| 任务 | Mol-1 | 最强基线 | 标记 |
|---|---|---|---|
| 蛋白–配体 Runs N' Poses（0–20% 相似度段） | 36.1% | AF3 24.6% | 【论文自称，p.6】 |
| 蛋白–肽 MolBench-Peptide-26 @0.80 | 54.1% | OpenDDE 52.3% | 【论文自称，p.7】 |
| 分子胶 MGBench | 28.6% | Protenix-v1 19.1% | 【论文自称，p.7】 |

### 7.3 成本（原稿实测保留）

| 场景 | 数字 | 来源 |
|---|---|---|
| 权重 | 577.4M，`model.pt` ≈2.2 GB | 【核实】 |
| 644 token 推理 | 峰值 ≈14 GB；单次前向 43–62 s（chunk 5 / chunk 1）；含加载约 1–2 min | 【核实：README + `inference_memory.md`】 |
| 1,000 token / 1,499 token | ≈14.9 GB / ≈21.6 GB | 【核实】 |
| 2,456 token | confidence head OOM（95 GiB 卡） | 【核实】 |
| 60k token 装配 | 单卡 H20，数十小时 | 【论文自称，p.9】 |
| 微调默认 | 单卡 95 GiB，384-token crop，48 个扩散样本/结构，100k steps，lr 1e-3，warmup 2,000 | 【核实：`docs/finetune.md`】 |
| constraint 微调 | 128 GPU，2,000 iters，48.7M 可训练 | 【论文自称，p.32】 |

`mol doctor` 可以先查环境（CUDA、flash-attn、权重、显存），装完先跑它，别直接上大复合物【核实：`docs/installation.md`】。

---

## 8. 对我们的可复用点

### 8.1 能不能当 VNAR 预测器

**能，但只到“可用的复合物预测器”这一层**【推断，依据代码与示例】：

- tokenizer 对任意蛋白序列通用，VNAR 的 CDR1/2/3 与框架区不需要特殊处理；
- 仓库自带一个**骆驼源纳米抗体 + GPCR** 的 JSON 模板示例（`examples/example_with_json_template/demo_ab.json`），说明单域抗体 + 抗原是官方支持的输入形态【代码／配置核实】；
- 但报告里 **没有任何 shark / VNAR / IgNAR 数据**（全文 `shark`、`VNAR` 零命中），VHH 验证也集中在骆驼源；VNAR 的 CDR3 更长、框架更疏水，属于**分布外**，不能直接引用 VHH 的 75% 战绩【核实：全文检索】。

建议的最小验证：拿我们已有结构的 20–40 个 VNAR–抗原复合物，用“单链序列 + 抗原”跑 10 recycle × 20 seeds，按 ipTM 排序，算 DockQ>0.23 的通过率，再决定是否值得进生产【推断】。

### 8.2 模板驱动 vs 纯 MSA 的收益边界

Mol-1 的 JSON 模板通道适合我们，因为：

- **不跑比对、不受 2021-09-30 截断**，可以直接把某个已知 VNAR 框架结构（mmCIF 文本 + 1:1 索引）塞进去；
- 对 VNAR 这种 MSA 深度的序列，模板提供的框架先验比 MSA 更可靠【推断】；
- 代价：只取模板第一条链、索引必须手写对齐，且模板错了（比如 CDR 构象不同）会把误差直接注入。

**收益边界**：报告与文档都没有给出“开/关模板”的消融，也没有 VNAR/纳米抗体的模板增益数字；目前只能确认“通道存在且可用”，不能宣称“对 VNAR 一定涨点”。正确写法是：**机制上可行，收益未量化**【未核实】。

### 8.3 84 条样本能不能微调

结论：**能微调，但只能做局部/轻量微调，不能照默认配方全权重跑**【推断】。

理由（全部来自配置与报告）：

1. 官方默认 100k steps、每步 1 个 384-token crop、单卡；84 个复合物按 `prepare_training_data.py` 拆成链/界面条目大约 200–250 条，等于每条目被重复采样数千次 → 必然过拟合【推断】。
2. 报告的 constraint 微调给了一个可抄的“小数据”配方：**冻结除 conditioner + 最后 8 层外的所有参数**，只训 48.7M（8.4%），2,000 iters 就收敛【论文自称，p.32】。
3. 微调数据加载器**默认禁用模板**（报告说 base 训练时“有模板就用”），所以微调阶段会丢掉模板先验；如果 VNAR 依赖模板，需要在微调前确认这一点【代码／配置核实：`docs/finetune.md`】。
4. 自定义链在聚类文件里**全部同簇**（官方 `clusters.csv` 只覆盖 PDB），sequence-balanced 采样失效，需要手动复制/加权【代码／配置核实：`docs/finetune.md`】。
5. 官方微调**不划测试集**，必须自己留出 10–15 个复合物做 held-out【代码／配置核实：`docs/finetune.md`】。

建议路线：先零样本评测 → 若界面 PAE 系统性偏差，再按 constraint 配方冻结微调最后 8 层 + confidence head，lr 1e-4、2k–5k steps、早停；不要一上来全权重 100k steps【推断】。

### 8.4 工程注意事项（原稿强项，保留）

- **两套 pLDDT 量纲**：`atom_plddts`/`chain_plddt` 是 0–1，`plddt` 是 0–100，混用会差 100 倍。
- **PAE 按 token 索引**：抗体–抗原体系里配体/非标准残基是原子 token，取界面 PAE 前先建 token→chain 映射。
- **显存**：644 token ≈14 GB 是“能跑”的基线；上 1,500 token 要 ~22 GB；2,456 token 会先在 confidence head 炸，不是主干炸。
- **chunk**：`--chunk-size 5` 能把 644 token 的前向从 62 s 降到 43 s，但显存上升，按卡调。
- **recycle 与 seeds 的等价性**：Figure 17 里 30 cycles×5 seeds = 58.64%、20×10 = 58.83%、10×40 = 58.69%，说明**加 seeds 比加 cycles 更划算**； screening 阶段用 10 cycles × 20–40 seeds，比 20 cycles × 5 seeds 更稳【论文自称，p.26】。
- **微调 checkpoint 的处理**：优化器/调度器状态一起存；EMA 单独存；加载 base 时 strict=True，换 head 会报错；续训要显式 `--resume-from`，否则优化器状态是新的【代码／配置核实：`docs/finetune.md`】。

---

## 9. 疑问、心得与参考

### 9.1 疑问

1. **8+40 的配比是否对界面任务最优？** 报告只给了总体拟合曲线，没有按抗体/配体/多链拆分消融；“60% FLOPs”是训练算力口径，推理延迟是否也降 40% 没写。
2. **“synthetic data generation” 到底是什么？** 这是数据段唯一的实质性表述，却没有任何展开；若是结构蒸馏，会直接影响我们对“能否用小数据微调”的判断。需要向作者确认或等后续版本。
3. **抗体基准开没开模板？** 报告没写；代码默认关。若基准是“无模板”成绩，我们用模板时不能直接对比。
4. **constraint conditioner 怎么注入表位？** 只知冻结策略与参数量，不知是 pair bias、cross-attention 还是 token 拼接；这决定了我们能否用 84 条样本复刻一个 VNAR 表位约束版。
5. **正文 57.3% 与 Figure 17 的 58.6–58.8% 差 1.3 pp**，是 seeds 数不同（5 vs 10–40）还是 cycle 数不同？报告没有把两组预算对齐说明。
6. **7,074 条 PDB 序列的 MSA 包**是否就是 base 训练所用子集？文档只说“微调时默认读这个包”，不能反推训练集规模。

### 9.2 心得

- **读代码比读报告信息量大一个量级**：σ、步数、crop 比例、采样权重、测试集文件名全在 `mol/config/*.py`，报告只写“我们做了大量数据工作”。以后遇到“报告未披露”，先翻配置再下结论。
- **不要拿“报告没写”当“模型没做”**：Mol-1 的蒸馏数据就是典型——报告零提及，代码里 Protenix 的 Distillation 类还在，但 Mol-1 配置没引用。正确写法是“无证据”，不是“没有”。
- **AF3 的默认扩散参数被完整继承**（σ_data 16、s_max 160、s_min 4e-4、ρ 7、200 步），所以 Mol-1 的增益应归因于主干/数据，而不是采样器；这让我们复现时可以直接用 AF3 的 sampler 经验。
- **抗体强 ≠ VNAR 强**：报告全部是骆驼源 VHH 与 Fab，鲨源 VNAR 是分布外；把 75% 直接搬过来是过度外推。

### 9.3 参考

- 技术报告：`https://github.com/molence/Mol-1/blob/main/assets/report.pdf`（39 页；本笔记页码均指此 PDF）
- 仓库：`https://github.com/molence/Mol-1`，commit `e6e64866ee89640a91cef7ee777a85f1d8be03bb`（2026-10-30）
- 关键配置：`mol/config/base.py`（结构/扩散）、`mol/config/data.py`（数据/切分）、`mol/config/inference.py`（默认开关）
- 关键文档：`docs/infer_json_format.md`（JSON/模板）、`docs/finetune.md`（微调）、`docs/inference_memory.md`（显存）、`docs/output_format.md`（confidence 量纲）、`docs/data_sources.md`（MSA 包口径）
- 上游：Protenix（`upstream revision bd54a05`）；AF3 Supplement Algorithm 5/20（atom attention / diffusion）
- 取证日志：`d:\postdoc_work\文献学习\2026-11-10\mol1-note-rewrite\work\search_log.md`
