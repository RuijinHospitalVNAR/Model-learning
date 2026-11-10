---
model: Protenix
aliases: ["PX-v1", "PX-v2", "protenix_base_default_v1.0.0", "protenix_base_20250630_v1.0.0", "protenix-v2", "protenix_base_default_v0.5.0（2024 首版）"]
family: ByteDance Seed（原 AML AI4Science）"AF3 开源复现"主线。同族衍生：Protenix-Mini / Protenix-Mini+（arXiv 2507.11839 / 2510.12842，加速变体，非主报告）、PXMeter（配套评测套件）、Protenix-Dock（GPL-3.0，对接/共折叠）
granularity: 混合 token 化。聚合物每残基 1 token；配体与非标准残基"每个重原子 1 token"，超出 max_atoms_per_token=24 时按原子数折叠（代码常量 24，注释指明 DNA 鸟嘌呤最多 23 个原子）
backbone: Pairformer 48 blocks（c_s=384、c_z=128、16 heads）+ MSA module 4 blocks（c_m=64、msa_max_size=16384）+ template embedder（基础配置 0 块，v1/v2 为 2 块）。Protenix-v2 只做加宽：c_z=256、c_m=128、hidden_scale_up=True，未新增模块
generative: AF3 式 EDM 扩散。diffusion module：atom encoder 3 块/4 heads → transformer 24 块/16 heads/c_token=768 → atom decoder 3 块/4 heads；σ_data=16、σ_max=160、σ_min=4e-4、ρ=7；默认 N_step=200、N_sample=5；recycling=10（v1/v2 配置）；另含 TFG（training-free guidance）采样变体
conditioning: MSA（ColabFold 管线，UniRef 按 taxonomy 配对）| 模板（≤4，v1 起）| RNA MSA（MMseqs2 + nhmmer）| pocket / token-contact / atom-contact 约束（JSON 字段）| 配体共价键（bond token 对特征）| 任意多链；无 epitope 专用条件、无抗体框架/种系条件
data_scope: wwPDB ≤2021-09-30（约 15 万结构，v1/v2 基准模型）+ MGnify 2019-05 的 AF2 蒸馏单体约 1300 万 + 约 1.4 万 disorder 蒸馏 + 蒸馏集；另发布 2025-06-30 切点版本（用于生产，不适合公平基准）
status: 代码 Apache-2.0；权重 Apache-2.0（README 明示"code and model parameters"）；训练/微调脚本、模型配置、预处理训练数据（~1.5 TB）均公开；不依赖 AF3 权重；本篇代码锚定 tag v2.0.0 = 2475421477ab414b571149ad4a875c390ff8a35d
verified_at: 2026-11-10
---

# Protenix

**三句话判断。** Protenix 是目前"开放程度最彻底"的 AF3 复现线：不只给权重，还给训练代码、四阶段训练超参、预处理数据（~1.5 TB）与可微调的 368M 模型，这一点 Chai-1/Boltz-1 都不完全具备。它的 v1 报告用"相同数据切点、相同模型规模、相同推理预算"三同条件把 AF3 追平并在 FoldBench 聚合指标上小幅超过（88.57% vs 88.03% lDDT，Ab-Ag DockQ SR 52.31% vs 48.75%），v2 再把抗体-抗原从 52.3% 拉到 65.0%（FoldBench-AB）。但它的"评测口径"是自己造的 PXMeter，AF3 从未在 PXMeter 上被重跑（表里是"-"），所以"超过 AF3"这句话的可信区间应该理解为"在自己口径 + 引用他人自报数值"的合成，而不是同台对照。

---

## 1. 定位与谱系

### 1.1 三条时间线，一个仓库

Protenix 没有 v1/v2 两个仓库：GitHub 搜索 `q=protenix in:name` 只返回 `bytedance/Protenix`（Apache-2.0，2024-11-08 建仓）与 `bytedance/Protenix-Dock`（GPL-3.0，对接/共折叠）。v1 与 v2 是同一仓库内的**模型版本**，区分维度是 tag、权重名和 `docs/` 下的技术报告 。

- 2024-11-08 开源，2025-01-08 上 bioRxiv：`Protenix — Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction`（ByteDance AML AI4Science Team，19 页）。这篇的贡献不是刷榜，而是**把 AF3 补充材料里写错/写漏的地方逐条列出来**（见 §3.4），这在复现类工作里很少见。
- 2026-02-05 Protenix-v1：`Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction`（13 页，DOI 10.64898/2026.02.05.703733），宣称"第一个在严格同切点/同规模/同推理预算下超过 AF3 的完全开源模型"。
- 2026-04-08 Protenix-v2：`Protenix-v2: Broadening the Reach of Structure Prediction and Biomolecular Design`（15 页，无 DOI、不在 arXiv），主题从"复现"转向"抗体 + 设计"。

arXiv 上 `ti:"Protenix"` 只能搜到 Protenix-Mini（2507.11839）与 Protenix-Mini+（2510.12842），两篇都是**加速/剪枝**工作（Mini 用 few-step ODE 替换多步 AF3 采样器、剪掉不参与最终结构的块）。把 Mini 当主报告，会把"90% 加速、约 3% lDDT 代价"误当成 Protenix 的核心结论，这是本篇特意避开的坑。

### 1.2 v1 与 v2 的开放程度对照（本篇核查点 1）

| 维度 | Protenix-v1 | Protenix-v2 | 证据 |
|---|---|---|---|
| 代码仓库 | `bytedance/Protenix`（同一仓库，tag v1.0.0） | 同一仓库，tag v2.0.0 | 见本篇 §9 |
| 代码许可证 | Apache-2.0 | Apache-2.0 | 见本篇 §9 |
| 权重是否公开 | 是。`protenix_base_default_v1.0.0` / `protenix_base_20250630_v1.0.0`，URL 形如 `protenix.tos-cn-beijing.volces.com/checkpoint/{model_name}.pt` | 是。`protenix-v2`（464.44 M） | 见本篇 §9 |
| 权重许可证 | Apache-2.0（README 明示"code and model parameters"） | Apache-2.0（同段） | 见本篇 §9 |
| 权重/代码许可是否一致 | 一致 | 一致 | 见本篇 §9 |
| 训练代码 | 公开（`train_demo.sh` + `protenix/` 全量） | 公开（同一套） | 见本篇 §9 |
| 训练配置 | 公开（`configs/configs_base.py` 四阶段超参 + `docs/training_inference_instructions.md`） | 公开（v2 覆盖项写在 `configs_model_type.py`） | 见本篇 §9 |
| 训练数据 | 预处理数据可下载（~1.5 TB，含 MSA 索引；`weightedPDB_indices_before_2021-09-30_wo_posebusters`） | 同（代码共享同一套数据管线） | 见本篇 §9 |
| 发布模型清单 | base_default_v1.0.0（368.48 M，MSA/RNA MSA/Template）；base_20250630_v1.0.0（368.48 M，2025-06-30 切点）；旧版 base_default_v0.5.0（368.09 M）、base_constraint_v0.5.0（368.30 M）、mini_esm/ism（135.22 M）、mini_default（134.06 M）、tiny（109.50 M） | protenix-v2（464.44 M） | 见本篇 §9 |
| 参数量 | 368.48 M（README 写作 ~368 M） | 464.44 M（README 写作 ~464 M） | 见本篇 §9 |
| 特征开关 | ESM ✗ / MSA ✓ / Constraint ✗ / RNA MSA ✓ / Template ✓ | 同 v1（✗/✓/✗/✓/✓） | 见本篇 §9 |
| 技术报告 | 独立 PDF + bioRxiv DOI | 独立 PDF，无 DOI | 见本篇 §9 |
| 微调 | 有 `finetune_demo.sh`（可换 checkpoint、可改 `pairformer.nblocks`） | 同一脚本（权重换成 v2） | 见本篇 §9 |
| 配套评测 | PXMeter v1.0.0 | PXMeter v1.1.0 | 见本篇 §9 |
| HF 镜像 | **未核实**（本机 huggingface.co 不可达）；可核实的托管点是 ByteDance TOS | 同 | 见本篇 §9 |

结论（这是本篇最容易被写错的一格）：**v1 与 v2 在"开放程度"上没有实质差异**——同一个 Apache-2.0 代码库、同一套训练/微调管线、同一套数据下载脚本，权重许可也与代码一致。差异全在"能力与用途"：v2 用加宽（c_z 128→256）+ 抗体数据重训换来抗体-抗原 +13 个百分点，代价是参数量 +26%，且 v2 **没有** Constraint 与 ESM 特征（v0.5.0 的 constraint 权重是历史产物）。所以"v2 更闭源"或"v2 只放推理"这类说法都不成立；真正需要注意的是 v1 的 `20250630` 版本**训练切点晚于多数测试集**，不能拿来做公平比较（README 自己提示"请用 base_default_v1.0.0 做跨版本公平基准"）。

### 1.3 与 AF3 的"三同"承诺

v1 的核心方法论约束是"same training data cutoff, model size, and inference budget"。落到数字：切点 2021-09-30；规模 368.48 M vs AF3 约 386 M（2024 报告自己写的数）；预算 5 seeds × 5 samples × 200 步 × 10 recycles 。注意 368 M 对 386 M **不是严格相同**，v1 的措辞是"strictly adhering to the same … model size"，实际是"同量级"——这点在引用时值得加限定词。

---

## 2. 表征粒度与表示层

### 2.1 token 化：聚合物按残基、配体按原子

Protenix 沿用 AF3 的混合表征：蛋白/DNA/RNA 每残基一个 token；配体与非标准残基"每个重原子一个 token"，因此有"配体被 crop 成碎片"的风险，作者专门加了"配体与非标准残基必须整体保留"的裁剪约束 。代码里 `max_atoms_per_token=24`，注释写明 DNA 鸟嘌呤最多 23 个原子，即 24 是覆盖核酸最大残基的紧上界 。`protenix/data/tokenizer.py` 的 `get_token_array()` 负责逐原子到 token 的映射与聚合 。

这个设计决定了三件事：(1) 扩散模块的"序列长度"是**原子数**不是残基数，显存随配体原子数非线性上升；(2) 配体原子直接参与 atom attention，不需要单独的口袋编码分支；(3) 长链核酸是最贵的一类输入（同残基数原子更多），这也解释了官方成本表为什么用 N_token/N_atom 两列（见 §7.4）。

### 2.2 通道与头

- 输入嵌入：`c_s_inputs=449 = c_token(384) + 32 + 32 + 1`，即"token 特征 384 维 + MSA/模板剖面 32 + 32 + 一个占位标量"，代码里把恒等式写在注释里 。
- 主干：`c_s=384`、`c_z=128`、pairformer 48 blocks × 16 heads；MSA module 4 blocks、`c_m=64`、`msa_max_size=16384` 。
- 扩散：atom encoder 3 块/4 heads（`c_atom=128`、`c_atompair=16`）→ transformer 24 块/16 heads（`c_token=768`）→ atom decoder 3 块/4 heads 。
- 置信头：4 blocks，`stop_gradient=True`（不回传到主干）；输出头 pLDDT 50 bins、PDE 64 bins、PAE 64 bins、distogram 64 bins；排序键 `["plddt","gpde","ranking_score"]` 。
- frame：仓库里有 `protenix/model/modules/frames.py`，实现 AF3 Algorithm 29 的 frame 构造/对齐工具，用于结构对齐与手性判断，而不是"把残基离散成刚体 frame"的独立表征层 。

### 2.3 表示层的延伸：约束（restraint）与共价键如何进入特征

AF3 的"pocket 约束"在 Protenix 里是**三类训练特征**：`token_contact`、`atom_contact`、`pocket`（binder 链 + 口袋残基集合），实现在 `protenix/data/constraint/constraint_featurizer.py` 。推理侧也有入口：`examples/example_constraint_msa.json` 里出现 `"constraint": {"pocket": {…"contact_residues"…}}` 与独立的 `"contact"` 列表 。但 `docs/infer_json_format.md` 全文没有 constraint 章节 ——文档滞后于代码，这是我在读仓库时最明显的一处不一致（README 2025-07-17 条目宣称支持并给了 `#constraint` 锚点 ，锚点已失效）。

共价键不走约束通道，而走 bond token 对特征："bonds within ligands, bonds between ligands and polymers, and bonds within non-standard residues"。所以金属配位、共价抑制剂这类化学，是接口能力而非重训需求。

---

## 3. 主干与生成范式

### 3.1 架构总表（全部来自 `configs/configs_base.py` @ v2.0.0）

| 项 | 值 | 代码位置 | 备注 |
|---|---|---|---|
| `c_s` / `c_z` | 384 / 128 | L112–113 | v2 覆盖为 256 |
| `c_s_inputs` | 449 | L114 | = 384+32+32+1（注释自带） |
| `c_atom` / `c_atompair` | 128 / 16 | L115–116 | 原子级通道 |
| `c_token`（主干） | 384 | L117 | 扩散内部另用 768 |
| pairformer blocks | 48 | L118 | `n_blocks` 全局默认 |
| pairformer heads | 16 | L313 | |
| `max_atoms_per_token` | 24 | L119 | 注释：DNA G 最多 23 原子 |
| MSA module | 4 blocks，`c_m=64`，`msa_max_size=16384` | L271/L274/L280 | v2 的 `c_m=128` |
| template embedder | 0 blocks（基础）/ 2 blocks（v1.0.0、v2） | L265 /  | 唯一"模块级"差异 |
| diffusion transformer | 24 blocks，16 heads，`c_token=768` | L321–333 | |
| atom encoder / decoder | 3 blocks / 4 heads 各自 | L328–337 | |
| confidence head | 4 blocks，`stop_gradient=True` | L345/L353 | |
| σ_data / σ_max / σ_min / ρ | 16.0 / 160.0 / 4e-4 / 7 | L168–174 | 注释："in EDM, this is 1.0" |
| 默认 N_step / N_sample | 200 / 5 | L181–182 | |
| γ0 / γ_min / λ / η | 0.8 / 1.0 / 1.003 / 1.5 | L177–180 | AF3 Algorithm 18 的四个超参 |
| recycling（`N_cycle`） | 4（base 默认）/ 10（v1.0.0、v2 覆盖） | L249 /  | 论文一律写 10 |
| train_crop_size | 256（基础）/ 384→640→768（阶段） | L58 /  | |
| diffusion_batch_size | 48（base）/ 64（v2） | L122 /  | |
| 训练样本种子数 | `N_model_seed=1`（推理默认单 seed） | L248 | 评测时外部喂 5 seeds |

### 3.2 v2 到底改了什么（把"架构改进"翻译成参数）

v2 报告只写"architectural refinements and training optimizations"，不给逐项消融 。代码里的真实改动只有四行级别的覆盖项：`c_z: 128→256`（relative position encoding / pairformer / diffusion / confidence / distogram 同步）、`msa_module.c_m: 64→128`、`hidden_scale_up: True`（在若干 block 内把隐层临时加倍）、`template_embedder.n_blocks: 2`、`N_cycle: 10`、`diffusion_batch_size: 64` 。参数量 368.48 → 464.44 M（+26%）。

也就是说：**v2 没有新模块、没有新采样器、没有新表征**。它的增益（抗体 +13 点）主要来自容量 + 训练数据/配方，而不是结构创新。v2 报告选择不披露训练配方细节，这使"v2 的抗体提升可复现"这一条只能靠社区重训验证，目前无法从论文直接复算。这是我对 v2 最大的保留意见。

采样器方面，v2 唯一与"生成"相关的新东西是 **TFG（training-free guidance）**：在反向扩散的每一步施加手性、平面性、扭转角、成对距离的势函数梯度，灵感来自 Boltz-1x / projected diffusion / SHAKE，提供 `protenix-v2-TFG`、`protenix-v1-TFG` 两个推理变体 。代码里对应 `generator.py` 的 `guidance_configs` 参数与 `TFGEngine`、`parse_tfg_config()` 。TFG 的效果很直接：配体联合成功率 v1 48.37 → v1-TFG 61.11，v2 50.42 → v2-TFG 60.53（原始 PoseBusters 口径）。

### 3.3 采样器实现细节（与 AF3 的逐行对应）

`sample_diffusion()` 的签名就是 AF3 Algorithm 18 的超参集：`gamma0=0.8, gamma_min=1.0, noise_scale_lambda=1.003, step_scale_eta=1.5`，另加工程参数 `diffusion_chunk_size / inplace_safe / attn_chunk_size / enable_efficient_fusion` 。调度器用 EDM 形式（σ_data=16、s_max=160、s_min=4e-4、ρ=7），并把"EDM 里 σ_data 应为 1.0"的偏离写成注释 ——这是 AF3 的设定，不是 Protenix 的改动。

推理默认 `seeds=[101]`（单 seed），`enable_efficient_fusion=True`，`use_template=False`；当 token 数 >3840 时 confidence head 与 diffusion 都走 AMP，>2560 时只有 confidence head 走 AMP 。v0.7.0 加入"shared variable caching + kernel fusion + TF32"以获得推理加速 。

### 3.4 论文 vs 代码（不一致处单列）

| 项目 | 论文 | 代码/文档 | 判断 |
|---|---|---|---|
| recycling | 10（v1 p.12、v2 A.2 均写"10 recycles"） | base 默认 4；v1.0.0 与 v2 覆盖为 10  | 发布权重与论文一致；若有人用 base 配置自训，默认值会低 6 次循环，容易背锅 |
| 训练 crop | 2024 报告 Table 2：384 / 640 / **640** ；v1 报告：384 / 640 / **768**  | 文档：384 / 640 / **768 / 768**（4 阶段，第 4 阶段 confidence-only） | 三方不一致。最可能是 2024 报告表格排版错（640 重复），文档的 4 阶段是后续补的 |
| diffusion batch | 2024：48 / 32 / –  | 文档：48 / 32 / 32 / 32  | 同上，文档更细 |
| 模板块数 | v1 报告只说"top-4 templates"  | base 0 / v1、v2 为 2  | 配置层差异，论文未提"2 块" |
| 参数量 | README："~464M"/"~368M" ；2024 报告：AF3 "about 386 million"  | 精确 464.44 / 368.48 / 368.09 M  | 引用请用精确值 |
| c_z | v2 报告未给数字  | 128 → 256 ；`docs/supported_models.md` 亦写 256 | 只能以代码为准 |
| 步数/样本 | 论文只说"same inference budget"，不给 N_step  | 200 步、5 samples  | 论文缺失，代码补齐 |
| 约束功能 | README 宣称支持（2025-07-17） | 代码与示例 JSON 有，`docs/infer_json_format.md` 无章节  | 文档滞后 |

### 3.5 2024 报告修了 AF3 的哪些"错"（复现类工作的真正价值）

这张表是 Protenix 区别于"又一个 AF3 复现"的地方，逐条都是可验证的算法级修正 ：

1. **Alg. 1 L10**：`{z_ij} += MsaModule` → `{z_ij} = MsaModule`。MsaModule 内部已带残差，再 `+=` 等于重复累加。
2. **Alg. 3 L8**：`b_same_chain_ij` → `b_same_entity_ij`。`sym_id` 只在同一 entity 内有意义，用同链判断会把同源多聚体的对称关系算错。
3. **Alg. 18 L9**：`δ_l=(x_l−x̂_l)/t̂` → `δ_l=(x_noisy_l−x̂_l)/t̂`。修正后才与 L11 的 Euler 更新自洽；这是采样器正确性的关键。
4. **Alg. 23 L2**：`{b_i}=AttentionPairBias` → `{b_i}+=AttentionPairBias`。缺残差时 DiffusionTransformer 不收敛。
5. **Alg. 28 L11**：`x_align=R x+µ` → `+µ_GT`，判为笔误。
6. **Alg. 31 L3**：距离 bin 边界 `[3⅜, 5⅝, …, 21⅜]` → `[3⅜, 4⅝, …, 21⅜]`，修正后是 15 个等宽 1.25 Å 的 bin（原式不等宽）。
7. **SI 3.7.1 Eq.6**：`(t̂²+σ²)/(t̂+σ)²` → `(t̂²+σ²)/(t̂×σ)²`（EDM 的 per-sample 权重）。

另外 v1 报告承认对置信头动了刀：加 LayerNorm、加几个线性层、对 AdaptiveLayerNorm 的两个线性层与 Pairformer 的 N-cycle 连接块做零初始化，"按原文实现时置信损失收敛不佳"。这属于"论文没写清、复现者必须补"的部分，也意味着 Protenix 的置信分数与 AF3 的置信分数**不可直接互比**。

---

## 4. 条件控制

| 条件 | 支持形态 | 类别 | 代价/限制 | 证据 |
|---|---|---|---|---|
| MSA（蛋白） | JSON 内嵌或外部 a3m；ColabFold 管线，UniRef 按 taxonomy 配对；上限 16384 条 | 接口 | 与 AF3 的 MSA 源不同（AF3 用 UniProt 配对 + UniRef/BFD/MGnify 未配对；Protenix 只有 uniref30 + colabfold_envdb），因此同一条序列两侧 MSA 深度与多样性不同，跨模型比较必须固定 MSA 来源 | 见本篇 §9 |
| MSA 子采样 | 推理期 `k∈[1,16384]` 随机子采样 + pair embedding dropout 做集成 | 接口（推理技巧） | 属于 v1 的 inference-time 增强 | 见本篇 §9 |
| 模板 | ≤4 个；推理取 e-value 排序、切点前的模板；训练随机取 ≤4 个、且比目标早 60 天以上 | 接口（v1 起） | 默认 `use_template=False`；开启需先跑模板搜索 | 见本篇 §9 |
| RNA MSA | MMseqs2 聚类 + nhmmer 搜 Rfam/RNAcentral/Nucleotide | 接口（v1 起） | 核酸模型的两个特征列之一 | 见本篇 §9 |
| pocket 约束 | `constraint.pocket.contact_residues`（binder + 口袋残基） | 接口 | 需 constraint 特征；**v1/v2 权重均无 Constraint 特征**（只有 v0.5.0 的 constraint 权重有），因此当前旗舰权重能否吃 pocket 约束存疑 | 见本篇 §9 |
| token/atom contact | `"contact"` 列表（token 对 / 原子对） | 接口 | 同上 | 见本篇 §9 |
| 共价键（配体-蛋白） | bond token 对特征，训练与推理都在 | 接口 | 需要正确的 CCD/键表输入 | 见本篇 §9 |
| 多链 / 多聚体 | 任意链数与化学计量 | 接口 | 链数上限由 `filter.py` 与显存决定 | 见本篇 §9 |
| epitope 指定 | 无 | **需重训/无接口** | AF3 亦无；Protenix 未提供 epitope 条件通道 | 见本篇 §9 |
| 抗体框架/种系条件 | 无 | 需重训 | VHH-Fc 设计走的是外部设计管线 + Protenix 打分，不是条件输入 | 见本篇 §9 |
| 微调（自有数据） | `finetune_demo.sh`，可换 checkpoint、可改 `pairformer.nblocks`、可指定自定义 pdb_list | 接口 | 需按 `prepare_training_data.md` 备数据（完整预处理 ~1.5 TB，自建数据只需索引+特征） | 见本篇 §9 |

一句话：**序列、MSA、模板、约束、共价键、链组成都是接口；"结合位点语义"（epitope/paratope 指定）和"抗体专用条件"不存在，只能靠数据重训或外部管线**。

---

## 5. 训练数据构成与切分

### 5.1 规模与构成

v1 明确"training cutoff 2021-09-30，约 150k curated structures"。三类数据：

1. **WeightedPDB**（实验结构，权重最高 0.60）；
2. **Monomer Distillation**：从 MGnify（2019-05 版）聚类簇中取序列，丢弃少于 10 条序列的簇，用 AF2 的五套参数各预测一次取 pLDDT 最高者，得到**约 1300 万**蒸馏单体，"对序列长度与 pLDDT 不再设阈值"；
3. **PDB Disorder**：约 **1.4 万**条专门用于 disorder 蒸馏的结构 ，另加蒸馏集（权重 0.02）。

这套构成与 AF3 论文的"PDB + MGnify 蒸馏 + disorder"三件套同构，差别在规模与筛选细节（AF3 报的蒸馏规模更大，Protenix 未逐项对标）。

### 5.2 测试隔离

- **模板防泄漏**：训练期只允许"比目标样本早 ≥60 天发布"的模板 ；
- **训练集排除 PoseBusters**：索引文件名 `weightedPDB_indices_before_2021-09-30_wo_posebusters` 直接写明 ；
- **配体测试集隔离**：清洗阈值很硬——分辨率 ≤2.0 Å、occupancy=1.0、单残基、分子量 100–900 Da、元素限 {H,C,O,N,P,S,F,Cl}、≥3 重原子、与其它链无共价键、RSR≤0.2、RSCC≥0.95；并排除"训练集中存在序列同源 ≥0.4 且配体 Tanimoto ≥0.6 的复合物"的测试实体 ；
- **低同源切分**：PXMeter 统一用 MMseqs2 对切点前 PDB 做低同源检索，蛋白按 40% identity / 0.8 coverage 聚类，DNA/RNA 按 80%，配体按 CCD 归并，分辨率 <4.5 Å、token ≤2560、排除 NMR ；
- **诚实标注污染**：v1 在 Boltz-2 行标 `†` 并注明"该评测集含 Boltz-2 训练数据"；v2 沿用同一标注 。这种主动披露在同类论文里不常见。

---

## 6. 宣称与可验证性

标记：**实证**（论文给出数值且有第三方可复核口径）／**部分实证**（有数值但口径自建或基线为引用）／**仅宣称**（无数值或不可复核）／**未核实**。对应规范标记：实证→【核实】，部分实证→【论文自称】，仅宣称→【论文自称】，未核实→【未核实】；正文中的代码/配置类结论另以【代码／配置核实】标注出处路径。

| 宣称 | 证据类型 | 核查状态 |
|---|---|---|
| C1「第一个在相同切点/规模/预算下超过 AF3 的完全开源模型」 | 表格（FoldBench 0.8857 vs 0.8803 lDDT；Ab-Ag 52.31 vs 48.75 DockQ SR） | **部分实证**：AF3 数值取自 FoldBench 官方报告，**未重跑**；"same model size"实为 368 vs 386 M |
| C2「PXM-2024/PXM-2025 上全面领先 Chai-1/Boltz-1/Boltz-2」 | 表格（v1 0.8770/79.42/0.6003 vs Boltz-2 0.8568/71.29/0.5281） | **部分实证**：该表**无 AF3 行**（无权重可重跑） |
| C3「推理期扩展近似 log-linear」 | 表格（单 seed 36.01% → 10 seeds 42.92% → 80 seeds 47.68%，PXM-22to25-Ab） | **实证**：100 seeds 池做 bootstrap |
| C4「v2 抗体-抗原较 v1 提升 9–13 点」 | 表格（FoldBench-AB 52.3→65.0、PXMeter-AB 40.2→49.7、AF3-AB 40.4→53.5） | **部分实证**：单一来源（团队自评），5 seeds top-1 |
| C5「v2 在 5 seeds 就超过 v1 的 1000 seeds」 | 曲线（Fig.3） | **仅宣称**：只给曲线，无表格数值，不可复算 |
| C6「v2 配体"合理性"提升，TFG 后 v2 反超 v1」 | 表格（v2 50.42 vs v1 48.37；TFG 后 60.53 vs 61.11） | **实证（口径内）**：TFG 后 v1 仍略高于 v2（61.11 vs 60.53），与"v2 更强"的叙事有张力 |
| C7「VHH-Fc 设计 100% 靶点级成功、命中率最高 48%」 | 湿实验（16–30 测试预算/靶点） | **部分实证**：属设计管线（PXDesign 系）成果，不是结构预测精度；样本量与靶点数需读全文表格 |
| C8「完全开源可训练」 | 代码+权重+训练配置+预处理数据（仓库实证） | **实证**：HF 镜像未核实  |
| C9「约束（pocket）可控生成」 | 示例 JSON 字段  | **部分实证**：当前旗舰权重无 Constraint 特征 ，实际可用性待验 |

---

## 7. 评测协议与成本

### 7.1 PXMeter 是什么（本篇核查点 2）

PXMeter 是 ByteDance 自己出的开源评测套件（`github.com/bytedance/PXMeter`，Apache-2.0），论文 `From Dataset Curation to Unified Evaluation: Revisiting Structure Prediction Benchmarks with PXMeter`（bioRxiv 2025-07-22，Ma/Liu/Yang/Lu/Zhang/Xiao）。动机是"现有 benchmark 的数据集筛选与指标不统一"，因此同时提供**数据集**（近期 PDB 人工复核、剔除非生物接触）与**统一指标**。论文里的基准模型只有 **Chai-1、Boltz-1、Protenix**——AF3 不在列，因为权重不开放 。

指标清单：**LDDT、DockQ、pocket-aligned RMSD、PoseBusters validity** 。注意：**没有 CDR-H3 RMSD、没有 paratope 覆盖率、没有 epitope 恢复率**——这是判断"它能不能评纳米抗体"的关键。

测试集（`docs/available_datasets.md`）：

| 集合 | 内容 | 规模（entries/clusters） |
|---|---|---|
| PXM-2024 | 2024 年新沉积、低同源 | 单体 424/280、蛋白-蛋白 982/745 等（见表注） |
| PXM-2025 / PXM-2025-H2 | 2025 年（含下半年）新沉积 | 见 v1 Table 3 第二面板  |
| PXM-22to25-Ab-Ag | 2022–2025 抗体-抗原 | 283/70（v1 另报 498/359 的汇总版本） |
| PXM-22to25-Ligand | 2022–2025 蛋白-配体 | 526/209（v2 用 623/250） |
| PXM-Legacy | 历史基准合集 | — |
| antibody 子集 | antibody-protein 314/201、antibody_HL 201/103、antibody_H 107/104、scFv 6/6 | 见本篇 §9 |
| FoldBench（外部） | 单体 287、PL 441、PP 237、AB 104 | 见本篇 §9 |

与 AF3 原论文口径的差异，逐项列清：

| 维度 | AF3 原论文 | PXMeter / Protenix 报告 | 影响 |
|---|---|---|---|
| RecentPDB 定义 | 2022-05-01–2023-01-12，序列同源 <40%，<2560 tokens（Protenix 复刻版） | PXMeter 按年分层（2024/2025/H2/22to25），统一 MMseqs2 低同源 + 40%/0.8 聚类  | 年份越新越难，跨年不可直接比 |
| 基线是否重跑 | — | **不重跑**：v1 明说"从公开仓库收集结果，不重跑这些模型"；AF3 的 FoldBench 数字来自 FoldBench 官方报告，AF3-AB 来自 OpenFold3 团队  | "超过 AF3"是"自测 vs 他测"的拼接 |
| MSA 来源 | AF3 自有管线（UniProt 配对 + UniRef/BFD/MGnify） | 各模型用各自 MSA；OF3p2 用 MMseqs2 无模板  | v2 给出量化：同一 267 个界面上，OF3p2 在本管线 69.29% vs 官方 FoldBench 70.79%，差 1.5 点  |
| 采样预算 | 5 seeds × 5 samples（25 个结构） | 同（v1）；v2 主表用 5 seeds 的 ranked top-1 ；recycles 统一 10  | 预算可比，但 v2 的"top-1"是排序后取优，与"best-of-25"不同 |
| 排序 | AF3 用 pTM/ipTM 复合打分 | `complex_ranker_keys = [plddt, gpde, ranking_score]` ；v1 按链对 ipTM / 单链 pTM 排序  | 排序器不同会直接改 top-1 数值 |
| 配体指标 | PoseBusters + RMSD<2 Å | 两套并存：v2 Table 1 "RMSD<2 Å 且 PB-valid"；Table 3 "RMSD<2 & LDDT>0.8"  | 同一篇论文两个数，引用时必须带口径 |
| PoseBusters 版本 | v0.2.7  | v1.1.0 扩展了 sp2/酰胺平面性、sp3 非平面性检查  | 新版更严，跨版本数字不可比 |

### 7.2 关键数值（三张表，各带口径）

**(a) FoldBench（v1 报告 Table 2，面板 1，287/237/160/441/46/274/14/15）**

| 模型 | 单体 lDDT | PP DockQ SR | **Ab-Ag DockQ SR** | PL SR% | Prot-RNA | Prot-DNA | DNA lDDT | RNA lDDT |
|---|---|---|---|---|---|---|---|---|
| AlphaFold3 | 0.8803 | 71.73 | **48.75** | 62.59 | 65.22 | 75.91 | 0.5278 | 0.6140 |
| Boltz-1 | 0.8664 | 68.35 | 34.38 | 54.65 | 65.22 | 67.15 | 0.3388 | 0.4416 |
| Chai-1 | 0.8639 | 67.51 | 23.12 | 50.79 | 50.00 | 66.06 | 0.4574 | 0.4852 |
| Protenix-v0.5.0 | 0.8717 | 68.06 | 43.94 | 58.49 | 59.37 | 68.02 | 0.4960 | 0.6046 |
| **Protenix-v1** | **0.8857** | **72.70** | **52.31** | **62.54** | **68.46** | 69.13 | 0.5693 | 0.6547 |
| Protenix-v1（×4 seeds） | 0.8863 | 74.68 | 55.63 | 63.49 | 69.57 | 69.71 | 0.5682 | 0.6523 |

读法：v1 在 8 列里 7 列最好，唯一明显落后的是 Prot-DNA DockQ SR（69.13 vs 75.91，差 6.8 点），而 DNA/RNA 的 lDDT 反而领先，说明问题出在"蛋白-DNA 界面"而非核酸本身；报告没有专门讨论这一点。

**(b) PXM-2024 / PXM-2025（v1 Table 3）**

| 模型 | 单体 lDDT | PP DockQ SR | PP lDDT | **Ab DockQ SR** | Ab lDDT | 配体 RMSD SR | 配体 Lig SR | RNA DockQ SR | RNA lDDT | DNA DockQ SR | DNA lDDT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Protenix-v1 | 0.8770 | 79.42 | 0.6003 | **47.56** | 0.3988 | 77.13 | 54.95 | 69.53 | 0.4170 | 62.38 | 0.4214 |
| Protenix-v0.5.0 | 0.8622 | 72.75 | 0.5505 | 34.53 | 0.2788 | 71.31 | 52.55 | 53.85 | 0.3259 | 55.39 | 0.3878 |
| Boltz-2 | 0.8568 | 71.29 | 0.5281 | 23.44 | 0.1929 | 68.60 | 46.25 | 54.63 | 0.3592 | 54.51 | 0.3487 |

这张表最值得记的是 **Ab 从 34.53 → 47.56（+13.0）**，说明 v1 的抗体提升主要来自训练配方/数据而非规模（参数量没变）。PXM-2025 面板同向（35.22 / 0.3074）。

**(c) 抗体专表（v2 Fig.2，5 seeds ranked top-1）**

| 集合（entries/clusters） | AF3 | Boltz-1 | OF3p2 | PX-v1 | **PX-v2** |
|---|---|---|---|---|---|
| PXMeter-AB（516/376） | –（无权重） | 19.6 | 27.9 | 40.2 | **49.7** |
| FoldBench-AB（104/160） | 48.8 | 34.4 | 34.9 | 52.3 | **65.0** |
| AF3-AB（67/60） | 47.0 | 14.9 | 30.9 | 40.4 | **53.5** |

一个容易被忽略的细节：v1 报告 Table 4 里"Ab-Ag [AF3AB]"列，v1 是 36.58（DockQ SR）/0.3136（lDDT），而 AF3 在该列自报 44 ；到 v2 才通过同集合重测拿到 53.5。也就是说**"追上 AF3 的抗体水平"这件事在 v1 时期并未发生，是 v2 才发生的**。v1 摘要里"全面超过 AF3"的说法在抗体项上依赖于 FoldBench 而非 AF3 自建集。

**(d) 配体合理性（v2 Table 1，PXM-22to25-Ligand 623/250）**

| 模型 | RMSD<2 Å 且 PB-valid（原始） | 同（修订口径） |
|---|---|---|
| Boltz-1 | 35.56 | 34.38 |
| Boltz-1x（TFG） | 55.59 | 53.96 |
| Boltz-2† | 51.31 | 51.25 |
| Boltz-2x† | 63.51 | 62.86 |
| Protenix-v1 | 48.37 | 48.14 |
| Protenix-v1-TFG | **61.11** | **61.11** |
| Protenix-v2 | 50.42 | 49.22 |
| Protenix-v2-TFG | 60.53 | 60.46 |

结论要说白话：**在不加 TFG 的情况下，v2 的配体合理性只比 v1 高 2 点，且显著低于 Boltz-2x；加上 TFG 后 v1/v2 都在 60 附近，仍低于 Boltz-2x 的 63.5**。所以 v2 摘要里"ligand-related plausibility 的 additional update"是相对于 v1 自己，而不是相对于 Boltz 家族。

### 7.3 两套 benchmark 并排（v2 Table 3，统一指标）****

| 模型 | FoldBench 单体 lDDT | FoldBench PL（RMSD<2 & LDDT>0.8） | FoldBench PP（DockQ>0.23 SR） | PXMeter 单体 | PXMeter PL | PXMeter PP |
|---|---|---|---|---|---|---|
| AlphaFold3 | 88.0 | 62.6 | 71.7 | – | – | – |
| OpenFold3-preview2 | 86.8 | 56.6 | 66.4 | 85.4 | 47.9 | 63.4 |
| Protenix-v1 | 88.6 | 62.5 | 72.7 | 87.2 | 48.1 | 72.9 |
| **Protenix-v2** | **89.1** | **64.0** | **73.0** | **88.0** | **49.2** | **73.9** |

这表的真正信息量在两组数的**落差**：同一模型在 FoldBench 与 PXMeter 上差 1–14 点（PL 62.6 vs 不可得；PP 71.7 vs 63.4 对 OF3p2）。所以"Protenix 在 PXMeter 上 PP 73.9"与"AF3 在 FoldBench 上 71.7"**不是同一把尺子**，不能相减。

### 7.4 成本（官方唯一一组数字）

推理（`docs/training_inference_instructions.md`）：

| N_token | N_atom | 峰值显存 | 时延 |
|---|---|---|---|
| 500 | 5,000 | 6.1 GB | 17 s |
| 1,000 | 10,000 | 18.2 GB | 59 s |
| 2,000 | 20,000 | 66.6 GB | 226 s |
| 3,000 | 30,000 | 60.8 GB | 935 s |
| 4,000 | 40,000 | 78.1 GB | 1,424 s |

非单调（2000→3000 显存反降）说明中间触发了分块/AMP 切换；时延在 2000 token 后从 226 s 跳到 935 s，是长序列 attention 的典型表现。**这张表是 VNAR 可行性的硬边界**：一个 VHH-抗原复合物通常 400–900 token，落在 6–18 GB 档，单张 24 GB 卡可跑；但 5 seeds × 5 samples 的 25 个样本约需 25 分钟/复合物。

训练（同文档）：

| 阶段 | crop | diffusion batch | α_pae | 吞吐（A100-80G） | 峰值显存 |
|---|---|---|---|---|---|
| Initial | 384 | 48 | 0 | ~12 s/step | ~34 GB |
| FT1 | 640 | 32 | 0 | ~30 s/step | ~35 GB |
| FT2 | 768 | 32 | 0 | ~44 s/step | ~48 GB |
| FT3（confidence only） | 768 | 32 | 1.0 | ~13 s/step | ~24 GB |

2024 报告给的总算力是"192 GPUs × ~2 周，三阶段 75K/15K/4K steps"，v1 是"256 GPUs 多阶段"。两者的阶段数不一致（3 vs 4），见 §3.4。

---

## 8. 对我们的可复用点（VNAR 专列）

**能不能当 VNAR 预测器？** 结构层面可以，且没有任何"抗体专用开关"挡路：VNAR 就是一条 ~120 残基的单域蛋白，按普通链输入即可，MSA 用常规单链 MSA 管线（Protenix 的 ColabFold 管线不需要重链/轻链配对信息）。真正的问题是**没有 VNAR 的评测证据**：

1. **无 VNAR/纳米抗体子集**。PXMeter 的抗体子集只分 antibody-protein / antibody_HL / antibody_H / scFv 四类 ；v1 的抗体测试集用 SAbDab 元数据把链分成 scFv / HL / 单 H / 单 L 。单域重链（VHH/VNAR）会落进 "H" 类（107 个复合物），但报告从未声明这一类里含 VHH，更不含鲨鱼 VNAR。**"Protenix 在 VNAR 上如何"目前是空白，不是"表现良好"。**
2. **无 CDR-H3 指标**。PXMeter 只有 LDDT / DockQ / pocket-aligned RMSD / PoseBusters ，全部是整体或界面级；CDR-H3 局部精度（尤其 VNAR 的 CDR3 长度与构象多样性）必须自建指标。v1/v2 的"抗体提升"是 DockQ SR 意义下的界面提升，不等于 loop 精度提升。
3. **v2 的抗体增益是真的，但来源是数据不是结构**。Ab DockQ SR 34.53 → 47.56（v1 同规模，纯配方）→ 49.7/65.0（v2 加宽+重训）；v2 摘要明说"shows clear gains on antibody-antigen"，但没给 CDR 级分析，也没给"单域抗体"专表。

**相对 Chai-1 的真实优势是微调**：Chai-1/Boltz-1 只放权重，Protenix 放训练代码、四阶段超参、预处理数据和 `finetune_demo.sh`（可换 checkpoint、改 `pairformer.nblocks`、指定自定义 pdb_list）。VNAR 场景数据量小（公开 VNAR-抗原复合物是百量级），正确的打法是"v1 base（2021 切点）微调 + 自建 VNAR 留出集"，而不是零样本刷 PXMeter。

**硬边界，按优先级：**
- **显存/时长**：4000 token 就要 78 GB、1424 s ；VNAR-抗原（~600–900 token）单样本约 20–60 s，25 样本预算下可接受；但若要做 CDR3 系综采样（几十 seeds），成本线性上升。
- **序列长度上限**：PXMeter 的准入是 ≤2560 token ，官方成本表外推到 4000 token 已吃满 80 GB，长抗原（>1000 残基）必须裁剪或分域。
- **权重版本**：做基准必须用 `protenix_base_default_v1.0.0`（2021-09-30 切点）；`protenix_base_20250630_v1.0.0` 切点晚于多数公开 VNAR 结构，会泄漏 。
- **MSA 语义**：纳米抗体缺轻链，配对 MSA 退化为普通单链 MSA（与 AF3 处理 VHH 相同，但官方未说明）。
- **约束不可用（大概率）**：pocket 约束虽在 JSON 里有字段 ，但 v1/v2 旗舰权重没有 Constraint 特征 ，做"表位定向"的 VNAR 设计不能指望它，得自己微调。

**如果要在这条线上做一个 VNAR 课题，最小可行实验**：(1) 用 SAbDab/PDB 抓 VNAR-抗原复合物（含 CDR3 长度分层），按 2021-09-30 切点分训练/留出；(2) 零样本跑 `protenix_base_default_v1.0.0` 与 v2，各 5 seeds × 5 samples × 10 recycles，报 DockQ SR + CDR3 RMSD + epitope 恢复率；(3) 用 `finetune_demo.sh` 在 v1 base 上微调（crop 384 起步，FT3 confidence-only 收尾），比较"微调 vs 零样本 vs AF3 引用值"；(4) 用 v2 的 VHH-Fc 设计链路做一次小规模湿实验验证（v2 报告给的是 100% 靶点成功、48% 命中率 ，但那是设计与实验的联合结果，不能直接外推到 VNAR）。

---

## 9. 疑问、心得与参考

**我仍然不确定的五件事：**

1. **v2 的抗体增益到底来自哪一步**。论文只有"architectural refinements and training optimizations"一句 ，代码显示只是加宽 + 模板 2 块 + N_cycle 10 。若只是加宽，368→464 M 换 13 点 Ab 提升，性价比极高；若有未披露的数据增补（如 2025 年抗体结构），则 v2 的"公平性"要打折——因为它的切点仍写 2021-09-30 ，但抗体领域的结构在 2022–2025 增长最快。这是我读 v2 最想追问作者的一点。
2. **pocket 约束到底能不能用**。README 说支持、示例 JSON 有字段、`constraint_featurizer` 有实现 ，但旗舰权重无 Constraint 特征 ，文档也没章节。三种可能：权重里其实带该能力（表格标注过时）、需要专用 v0.5.0 constraint 权重、或者只是训练代码残留。**未核实**。
3. **PXMeter 的 5-seed top-1 与 AF3 的 best-of-25 是否等价**。v2 用"ranked top-1 at 5 seeds"，AF3 的 25 样本是 5 seeds × 5 samples 后按 pTM/ipTM 取优 。前者每 seed 只出 1 个结构，后者出 5 个再选，**采样预算差 5 倍**。v2 的 "+13 点"里有多少来自这个口径差，论文没做消融。
4. **v1 的 Prot-DNA 短板原因**。FoldBench 上 69.13 vs AF3 75.91 ，差了近 7 点，而 Prot-RNA 反而领先（68.46 vs 65.22）。报告未解释；我怀疑是 DNA 的 MSA/模板处理与 AF3 不同（Protenix 只对 RNA 加 MSA，DNA 无 MSA ），但这只是推测，**未核实**。
5. **HF 上的权重镜像是否存在**。本机 huggingface.co 不可达 ；若不存在，意味着国内以外用户需从 ByteDance TOS 拉取，这对"完全开源"是可访问性上的一处折扣（不是许可证问题）。

**判断。** 如果我要在一个课题里选"AF3 系里最适合做微调的基座"，Protenix 是当前首选：Apache-2.0 覆盖代码与权重、训练/微调/数据三件套齐全、368 M 单卡可训（crop 384 时峰值 34 GB）、并且有 PXMeter 这套可复算的评测脚本。如果我要"证明我的方法比 AF3 强"，Protenix 的报告只能作为**方法论模板**（三同条件 + 修错清单 + 主动标注污染），不能作为**对照数据**——因为它对 AF3 的所有比较都是"引用他人自报值"，而 AF3 从未在 PXMeter 上被重跑 。真正干净的对照只有一条：在同一个 PXMeter 版本、同一 MSA 来源、同一 5 seeds × 5 samples × 10 recycles 下，把 Protenix-v1/v2、Boltz-2、Chai-1、OpenFold3 全跑一遍——这正是 v2 附录 A.2 已经开了头的事（OF3p2 的 69.29% vs 70.79% 就是这种对照的副产品 ），但它没做完。

**给后来者的三条操作建议：**
1. 引 Protenix 的抗体数字时，务必写清是 FoldBench-AB、PXMeter-AB 还是 AF3-AB，三者差 10 点以上 ；
2. 复现时固定 `N_cycle=10`（base 默认 4 会显著掉点）；
3. 用 v1 的 `20250630` 权重做任何"新方法更强"的声明都是无效的，切点晚于公开测试集 。

---

### 9.1 参考（一手来源）

- 报告：`docs/Protenix_Technical_Report_2024.pdf`（bioRxiv 10.1101/2025.01.08.631967）；`docs/PTX_V1_Technical_Report_202602042356.pdf`（DOI 10.64898/2026.02.05.703733）；`docs/PX2.pdf` 
- 仓库：`github.com/bytedance/Protenix` @ v2.0.0 `2475421477ab414b571149ad4a875c390ff8a35d`；`github.com/bytedance/PXMeter` 
- 评测论文：PXMeter，bioRxiv 10.1101/2025.07.17.664878 
- 衍生（非本篇主体）：Protenix-Mini，arXiv 2507.11839；Protenix-Mini+，arXiv 2510.12842 
- 代码关键文件：`configs/configs_base.py`、`configs/configs_model_type.py`、`configs/configs_inference.py`、`protenix/model/generator.py`、`protenix/data/constraint/constraint_featurizer.py`、`docs/training_inference_instructions.md`、`docs/supported_models.md`
- 本篇所有事实的取证形态已写在正文标注里：论文页码（复现报告 19 页 / v1 报告 13 页 / v2 报告 15 页）、代码 `tag` 与 `file:line`、配置文件与 README 路径；不依赖任何本仓库之外的文件。
