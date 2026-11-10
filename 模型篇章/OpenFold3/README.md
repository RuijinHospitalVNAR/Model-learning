---
model: OpenFold3
aliases: ["OpenFold3-preview", "OF3p", "OF3", "OpenFold3-preview2（0.4.0，2026-03-13，另一套权重）", "OpenBind-0（0.5.0 起的默认权重，2026-08-21，非 preview）"]
family: AF3 谱系 · 开源"逐位复现"（AlQuraishi Lab + OpenFold consortium）
granularity: 逐原子；聚合物规范残基每残基 1 token，配体/非规范残基/共价修饰残基每原子 1 token（按 AF3 SI §2.6）；token 内原子槽位 23（AF3 官方实现为 24）
backbone: AF3 trunk 原样：pairformer 48 block（c_s=384, c_z=128）、MSA module 4 block（c_m=64, c_hidden_msa_att=8）、template 2 block（c_t=64, c_dgram=39）、diffusion transformer 24 block（c_token=768, c_hidden=48, 16 heads）、atom transformer 3+3 block（c_atom=128, c_atom_pair=16）、confidence head 4 block
generative: EDM 扩散：σ_data=16、σ_min=4e-4、σ_max=160、ρ=7、γ0=0.8、step_scale=1.5；默认推理 200 步 / 5 diffusion samples / 1 model seed / 3 次 recycle（评测时用 10 次）
conditioning: MSA（main+paired，推理子采样至 1024 条）| 模板（≤4 个模板链，仅蛋白）| 共价键（query JSON 有 covalent_bonds，但文档标 "coming soon"）；无 pocket / restraint / epitope / 抗体模式 / 无 MSA 自由模式
data_scope: 复现 AF3 SI 配方：weighted-PDB 0.5 + 蛋白单体蒸馏 0.495 + 另一单体集 0.005 + disordered-PDB 0.02；结构集切点同 AF3（PDB 2021-09-30）；验证集 2021-10-01–2023-01-13、≤4.5 Å、≤2048 token；无核酸自蒸馏；preview 白皮书称训练数据随最终版发布，0.5.1 README 称经 portal 提供复现 preview 所需的完整数据（含 13M MGnify 蒸馏集）
status: preview（0.3.x，研究与社区反馈用途）；本篇代码锚定 tag 0.3.1（2025-10-28）；0.5.0 起默认权重换为 OpenBind-0（June 2025 切点）；训练代码在仓库中但官方标注 "unsupported"，训练数据不随仓库发布（0.5.1 README 称经 portal 提供）
verified_at: 2026-11-10
---

# OpenFold3

> **三句话速判**：
> 1. OpenFold3-preview 是 AF3 的**结构级复现**，不是新方法——pairformer 48 block、MSA 4 block、diffusion 24 block、σ_data/σ_min/σ_max/ρ = 16/4e-4/160/7、200 步、5 samples，这些数字与 AF3 逐个对齐，连数据配比 0.5/0.495/0.005/0.02 都照抄；改动只有作者显式列出的 4 处（distogram 分箱、confidence head 前置 LayerNorm、去掉 diffusion 的 LayerNorm bias、最后一个 MSA block 跳过 MSA stack）加上 Protenix 的稳定性修改【核实：白皮书 p.1】。
> 2. "preview"的实质是：**权重公开（单一 2.29 GB checkpoint）+ 推理代码完整 + 训练代码在但不被支持 + 训练数据不发布**——所以它可复现"推理"，不可复现"训练"；论文自己写了"does not include the training dataset, which will be released with the final version"【核实：白皮书 p.5】。
> 3. 对 VNAR：可以当**单域抗体结构预测器**用（蛋白单体路径），但没有任何抗体专用条件、配对 MSA 对单链退化、无 epitope/affinity，且想微调就得自己重建整套 PDB+序列库数据管线——preview 阶段这是硬边界。

---

## 1. 定位与谱系

### 1.1 出处、版本与"论文"到底是什么

OpenFold3 **没有 DOI、没有 arXiv/bioRxiv 预印本**。它的"论文"是一份 8 页技术白皮书 PDF（`of3p1_technical_report.pdf`，随仓库 `assets/` 发布，0.5.1 tag 仍在），下文页码即该 PDF 的物理页。

- arXiv 检索 `ti:"OpenFold3"` 只命中一篇**衍生**工作 ConforNets（arXiv 2604.18559，2026-04-20，"Latents-Based Conformational Control in OpenFold3"），不是 OF3 本体【核实：arXiv API】。
- 仓库 `github.com/aqlaboratory/openfold-3`，许可证 **Apache License 2.0**，README 明说"for both academic and commercial use"。
- 本篇固定到 **tag `0.3.1`（2025-10-28）**：这是首次公开的"OpenFold3 Preview"；同日还有 `0.3.0`（09:06）与 `0.3.1`（11:19）两个 tag，另有 `0.3.1-zenodo`（2025-10-30）供引用【核实：GitHub releases API】。

版本时间线（**必须区分，否则会把 preview2/OpenBind 的事实错记到 preview 头上**）：

| tag | 日期 | 名称 | 与本篇关系 |
|---|---|---|---|
| 0.3.0 / 0.3.1 | 2025-10-28 | OpenFold3-preview（OF3p） | **本篇主体** |
| 0.3.1-zenodo | 2025-10-30 | 同上，Zenodo 可引用快照 | DOI 未核实（release 正文未给） |
| 0.4.0 | 2026-03-13 | OpenFold3 Preview2（OF3p2） | 另一套权重与训练（155,000 步），评测跑 10 次 recycle |
| 0.5.0 | 2026-08-21 | OpenBind Model Release | 默认权重换为 **OpenBind-0**，June 2025 训练切点 |
| 0.5.1 | 2026-10-28 | 维护版 | 本篇用于取权重/文档的对照版本 |

（整表来源：GitHub releases API；权重对象的时间戳另见。）

### 1.2 preview 状态意味着什么（本篇核心）：五列结论

| 维度 | preview（0.3.1）到底给了什么 | 怎么核的 | 与"完全开源可复现"的差距 |
|---|---|---|---|
| **权重** | **公开**。单一 checkpoint `of3_ft3_v1.pt`，S3 公开桶 `s3://openfold/openfold3_params/`，脚本用 `aws s3 cp ... --no-sign-request`（免凭证）；对象 **2,288,027,095 B（2.13 GiB）**，S3 `Last-Modified: 2025-10-26` | 【核实：`openfold3/scripts/download_openfold3_params.sh` L35-40】【核实：S3 HEAD 200 实测】 | 只有**一个** checkpoint，没有 monomer/multimer 分版；无参数量披露；无独立权重许可文件 |
| **代码** | **推理完整**：CLI `run_openfold predict`、ColabFold/自带 MSA 管线、模板、cif/pdb 输出、confidence JSON、多 GPU（Lightning）；`cuequivariance` 与 DeepSpeed4Science kernel 开关齐全 | 【代码／配置核实：`run_openfold.py`、`docs/source/Inference.md` L80-296】 | 推理侧功能矩阵仍有缺口，见 §4 |
| **训练配置** | **训练代码在，但"不支持"**：`run_openfold train` 入口、4 个 yaml（initial + finetune 1/2/3）、DeepSpeed 配置、预处理/缓存脚本、`train` preset（`blocks_per_ckpt: 1` 逐 block 梯度检查点）全在仓库里；白皮书原话 "includes model weights, inference code, and (unsupported) training code" | 【代码／配置核实：`examples/training_yamls/*.yml`、`model_setting_presets.yml` L1-15】 | 官方不承诺可跑通；README 把 "Training documentation & dataset release" 列为 **Upcoming**；yaml 里 `debug_mode: true`、路径为占位符 |
| **数据** | **未发布**。白皮书："This preview release lacks finalized documentation for training and does not include the training dataset, which will be released with the final version of OF3"；仓库只有**预处理脚本 + 缓存格式**（需自备 PDB 与 UniProt/UniRef90/MGnify/PDB-seqres 库） | 【核实：白皮书 p.5 L153-155】【代码／配置核实：`scripts/snakemake_msa/download_of3_databases.py`】 | 没有结构清单、没有 MSA 缓存、没有蒸馏集 → 训练复现的实际障碍在这里，不在代码 |
| **许可** | 代码 **Apache-2.0**（仓库 LICENSE + README "academic and commercial use"） | 【核实：README L13】 | **权重许可未单列**（仓库内无第二份 LICENSE/权重条款）；HF 端本机不可达 → 权重再分发/商用条款 **未核实** |

补一条容易踩的坑：任务线索里"权重在 HuggingFace"**不准确**。HF 的 `OpenFold/OpenFold3` 仓库在文档里只用于放 **示例输出**（`examples/` 目录）【核实 引文上下文】，真正的权重走 **公共 S3**。HF API 在本机被墙，故 HF 侧的模型卡/许可**未核实**。

### 1.3 相对 AF3：复现还是改了什么（逐项）

| 面 | 判定 | 依据 |
|---|---|---|
| 表征 | **复现，且有 1 处未声明的差异**：token 化按 AF3 SI §2.6（聚合物规范残基逐残基、其余逐原子）；但原子槽位 **23**，AF3 官方实现是 **24** | 【代码／配置核实：`tokenization.py` L140-158；`model_config.py` L48】＋AF3 基线笔记 `pipeline.py` L71 |
| trunk | **复现**：48 block / c_s 384 / c_z 128 / 4 heads / SwiGLU；MSA 4 block | 【代码／配置核实：`model_config.py` L276】 |
| diffusion | **复现**：24 block / 768 / 48 / 16 heads；σ_data 16、σ_max 160、σ_min 4e-4、ρ 7、200 步、5 samples | 【代码／配置核实：`model_config.py` L337、L366-374】 |
| 条件 | **缩水**：MSA/模板/共价键有接口；pocket、restraint、epitope、抗体模式、RNA 模板、共价配体均缺 | 【代码／配置核实：`inference_query_format.py`；`Inference.md` L29-71】 |
| 数据 | **复现**：AF3 SI 配方，结构集切点同 AF3；核酸自蒸馏**没有**（因无预训练 OF3 模型） | 【核实：白皮书 p.1 L40-43】 |
| 训练目标 | **复现**：diffusion loss = bond + smooth_lddt + mse；置信度/ distogram 头齐全 | 【代码／配置核实：`model_config.py` L448、L465】 |

白皮书自己点名的 4 处偏离（这是全篇最硬的"改了什么"清单）：

1. AF3 SI Algorithm 31 Line 3 的距离分箱改为 **38 个等宽箱，3.25–50.75 Å，另加一个"更远"箱**（故 `c_dgram = 39`）；
2. 每个 confidence head 的最终线性投影前加 **LayerNorm**；
3. diffusion module 里**去掉 LayerNorm 的 bias**；
4. **最后一个 MSA module block 跳过 MSA stack**（不影响输出的 pair 表示）。
   此外整体吸收了 Chen et al.（Protenix）Table 2 的稳定性修改。

### 1.4 谱系：上游 / 下游

- **上游**：AF3（Abramson et al., *Nature* 2024）→ Protenix（Chen et al.）提供"AF3 配方必须哪些稳定性补丁"的经验 → OF3p。
- **同代**：Boltz-1、Chai-1、Protenix 都是同一批判 AF3 复现；OF3p 是其中**唯一把"bitwise reproduction"写进目标函数**的（"aiming to be a bitwise reproduction"）。
- **下游**：OF3p → OF3p2（0.4.0）→ OpenBind-0（0.5.0）；另有 ConforNets 直接拿 OpenFold3 的 latent 做构象控制（arXiv 2604.18559），说明 OF3 的 `write_latent_outputs` 接口被外部当底座用（配置项见 `full_config.yml`，`write_latent_outputs: false`）。

---

## 2. 表征粒度与表示层

### 2.1 token 化：三条规则

`tokenize_atom_array()` 的 docstring 直接写明"according to section 2.6. in the AF3 SI"，并列出 4 步：① 取聚合物中规范残基的 token 起点；② 取小分子配体、聚合物中的非规范残基、氨基酸/核苷酸类小分子配体的**全部原子**；③ 取与其它化学物种成键的"被修饰"规范残基的原子；④ **集合 2、3 逐原子成 token，集合 1 减 3 逐残基成 token**【核实：`openfold3/core/data/primitives/structure/tokenization.py` L140-158】。

因此：

- 蛋白/RNA/DNA 的规范残基：**1 残基 = 1 token**；
- 配体、非规范残基、共价修饰残基：**1 原子 = 1 token**（修饰残基会被"原子化"，同时 `token_center_atom=True`）；
- token 中心原子由 `TOKEN_CENTER_ATOMS` 决定，修饰残基的中心原子被强制置真（L214-221）。

### 2.2 原子槽位：23 而不是 24

`max_atoms_per_token: 23`【核实：`model_config.py` L48】。AF3 官方实现是 24（AF3 基线笔记：`pipeline.py` L71）。白皮书**没有**提这条差异，属"代码里悄悄改了"。影响面【推断】：token 内 atom attention 的 KV 长度少 1 槽，对显存/带宽是微小利好，但也意味着**任何依赖 24 槽语义的权重/实现不能直接互换**；超出 23 个原子的残基如何截断/告警，本篇未逐行核实（`未核实`）。

### 2.3 没有 frame / 扭转角

`model_config.py` 全文 544 行**不含** `torsion`/`frame`/`chi` 键（唯一命中 `bond` 是 diffusion loss 名，`angle_threshold: 25.0`）【代码／配置核实】。即：与 AF3 一样**去掉了 residue frame 与扭转角参数化**，diffusion 直接在 `(n_token, n_atom_slot, 3)` 上回归坐标；这也解释了为什么"共价键条件"只能以 query 级 `covalent_bonds` 形式注入，而不是靠内部化学图。

### 2.4 原子级 cross-attention 仍在

atom transformer 分编码/解码两段，各 **3 block、4 heads、c_hidden 32**，`c_atom=128`、`c_atom_pair=16`，`n_query=32 / n_key=128`【核实：`model_config.py` L320、L353；L13-33】。这正是 AF3 "token 内局部注意力 + token 间全局注意力"的两级结构，OF3 原样保留。

---

## 3. 主干与生成范式

### 3.1 架构逐项（每个数字都带出处）

| 组件 | 数值 | 出处（`openfold3/projects/of3_all_atom/config/model_config.py`，tag 0.3.1） |
|---|---|---|
| single 通道 `c_s` | 384 | L13-33（dim 区块） |
| pair 通道 `c_z` | 128 | 同上 |
| MSA 通道 `c_m` / `c_t` | 64 / 64 | 同上 |
| token embedder 宽度 | 384 | 同上 |
| diffusion token 宽度 | 768 | 同上 |
| atom 通道 / atom-pair | 128 / 16 | 同上 |
| `max_atoms_per_token` | **23** | L48 |
| MSA module block 数 | 4 | L252 |
| MSA attn 中间宽 `c_hidden_msa_att` | 8（作者注释："8 or 32, possible typo in SI"） | L246 |
| MSA heads（MSA / pair） | 8 / 4 | L246-259 |
| MSA dropout（MSA / pair） | 0.15 / 0.25 | L246-259 |
| pairformer block 数 | **48** | L276 |
| pairformer heads / `c_hidden_mul` / `c_hidden_pair_att` | 4 / 128 / 32 | L276-290 |
| template block 数 | 2 | L218 |
| template `c_dgram` | 39 | L207 |
| template `c_hidden_tri_att` | **16**（作者注释：SI 写 64，代码是 16） | L214-216 |
| diffusion transformer block 数 | **24** | L337 |
| diffusion heads / `c_hidden` | 16 / 48 | L331-337 |
| atom transformer（enc / dec） | 3 / 3 block，4 heads | L320 / L353 |
| confidence head block 数 | 4 | L388 |
| `num_recycles` | **3** | L158 |
| 训练 rollout：samples / mini / full | 48 / (20 步, 1 sample) / (200 步, 5 samples) | L162-166 |
| 噪声表：σ_data / σ_max / σ_min / ρ | 16 / 160 / 4e-4 / 7 | L33、L366-368 |
| 采样器：γ0 / γ_min / noise_scale / step_scale | 0.8 / 1.0 / 1.003 / 1.5 | L371-374 |

### 3.2 生成范式：EDM + AF3 的"两步 rollout"

训练侧是 AF3 的 rollout 方案而非"一次扩散"：每个 crop **48 个样本**，其中 **1 个走 20 步 mini rollout、5 个走 200 步 full rollout**，其余用于中间监督【核实 L162-166】。这解释了为什么 `no_full_rollout_samples=5` 恰好等于推理默认的 5 个 diffusion samples——**默认采样预算就是从训练配方里继承的**。

推理侧默认：**200 步、5 samples、1 seed、3 次 recycle**【核实：`Inference.md` L144-148、L224-226；E16 L158】。也就是说，一条命令出 **5 个结构**，而不是 AF3 论文协议的 25 个。

### 3.3 显存工程：有开关，没有数字

仓库给的是**旋钮**，不是**实测**：

- `predict` preset：`chunk_size: 4`（自动调优的下限）、`offload_inference.confidence_heads: true` + `token_cutoff: 2800`（超过 2800 token 才卸载置信度头）【核实：`model_setting_presets.yml` L17-30】；
- `low_mem` preset：`clear_cache_between_steps: True`、`per_sample_token_cutoff: 0`、`per_sample_atom_cutoff: 0`（即强制逐样本串行）、把 `msa_module` 与 `confidence_heads` 都 offload【核实: L38-48】。文档明确警告"significant slowdowns may occur"【核实 L257】；
- `train` preset：`blocks_per_ckpt: 1`（逐 block 梯度检查点）、`swiglu_seq_chunk_size: 4000`、diffusion loss `chunk_size: 4`【核实: L1-15】；
- kernel：`use_cueq_triangle_kernels`、DeepSpeed4Science 选项在配置里。

**README、白皮书、docs 都没有给出任何显存 GB 数或单结构耗时**（`docs/source/kernels.md` 也无数字）→ 所以"OpenFold3 更省显存/能跑更长序列"这个常见说法，在 preview 阶段**没有一手数字支撑**【未核实】【核实：负证据留档】；能核实的只有"提供了 low_mem 串行 + offload + chunk 的机制"。

### 3.4 推理 MSA 深度

`predict` preset 同时设 `subsample_main_msa: False`、`subsample_all_msa: True`【核实 L27-30】，配合 1024 上限 → **推理时所有链的 MSA 都截到 1024 条**，与白皮书评测口径一致（"subsample the input MSA to 1,024 sequences"）。抗体例外：评测保留 paired MSA、只子采样 unpaired。

---

## 4. 条件控制

接口文件是 **query JSON**（`inference_query_format.py`，Pydantic 模型），字段与能力逐项核对如下：

| 条件 | 是否支持（preview） | 接口/字段 | 性质 |
|---|---|---|---|
| MSA（unpaired/main） | ✅ | `use_msas`、`use_main_msas`、`main_msa_file_paths` | 推理接口 |
| MSA（paired） | ✅（仅异源多聚蛋白自动生成） | `use_paired_msas`、`paired_msa_file_paths` | 推理接口 |
| 模板 | ✅ 但**仅蛋白**、最多 4 个模板链 | `template_alignment_file_path`、`template_entry_chain_ids` | 推理接口 |
| 共价键 | ⚠️ schema 有、文档标 "coming soon" | `covalent_bonds: list[Bond]`，`Bond(atom1: Atom, atom2: Atom)`，`Atom(chain_id, residue_id, atom_id)` | 接口在，功能未完成 |
| 配体 | ✅ 仅**非共价**配体（`smiles` / `ccd_codes` / `sdf_file_path`） | `Chain.smiles`、`Chain.ccd_codes`、`Chain.sdf_file_path` | 推理接口 |
| 非规范残基 | ✅ | `Chain.non_canonical_residues: dict[int, str]` | 推理接口 |
| pocket / 口袋约束 | ❌ | 无字段 | 缺失 |
| restraint（contact / epitope / 距离） | ❌ | 无字段 | 缺失 |
| 抗体特定模式 | ❌ | 无字段（不像 Chai-1 有 antibody mode） | 缺失 |
| 多链模板（跨链配对模板） | ⚠️ 部分：模板按 entry chain 指定，白皮书评测确有给模板；但**存在已知 bug**：inter-chain mask 用"乘 chain ID"而非"判断相等"，AF3 SI Algorithm 16 Line 5 写错，OF3p 复刻了这个错 |  | 已承认，preview2 修复 |
| RNA 模板 | ❌ | `Inference.md` L55-59 "Coming soon" | 缺失 |
| seed 控制 | ✅ | `--num_model_seeds` / `experiment_settings.seeds` | 推理接口 |
| MSA-free | ⚠️ 名义支持、实际劝退 | 需喂"只含 query 序列的 dummy MSA"；文档："a completely MSA-free inference mode is currently discouraged" | 半支持 |

---

## 5. 训练数据构成与切分

### 5.1 配方：照抄 AF3 的四路混合

`examples/training_yamls/initial_training.yml` 的 `dataset_configs.train`：

| 数据集 | 类 | 权重 | 用途 |
|---|---|---|---|
| weighted-pdb | `WeightedPDBDataset` | **0.5** | PDB 结构，按分辨率/聚类加权 |
| long-monomer-distillation | `ProteinMonomerDataset` | **0.495** | 蛋白单体蒸馏（AF2 式） |
| （第三个单体集） | `ProteinMonomerDataset` | **0.005** | 少量补充单体 |
| disordered-pdb | （disordered） | **0.02** | 无序区训练 |

这 4 个数字与 AF3 SI 的 0.5 / 0.495 / 0.005 / 0.02 完全一致——**这是"bitwise 复现"最有力的间接证据**：连数据混合比例都没有重新调（作者也自述"following all details of the AF3 paper and SI, including ... training datasets"）。

其他训练超参：`devices: 8`、`num_nodes: 32`（= **256 卡**）、`precision: bf16-mixed`、`batch_size: 1`、`epoch_len: 128000`、`seed: 1272025`、梯度检查点 `blocks_per_ckpt: 1`、`token_budget: 384`、crop 权重 contiguous/spatial/spatial_interface = 0.2/0.4/0.4、`n_templates: 4` 且 `take_top_k: false`。

### 5.2 预处理与阈值

- PDB 训练缓存：分辨率阈值 **≤ 9.0 Å**（脚本 docstring），按链长/聚类加权【代码／配置核实：`scripts/data_preprocessing/.../pdb_weighted.py`】——与 AF3 的 9 Å 口径一致。
- 验证集：**2021-10-01 至 2023-01-13**、分辨率 **≤ 4.5 Å**、长度上限 **2048 token**（早期版本 2560）【核实：`create_pdb_validation_dataset_cache.py`】。注意这与 AF3 论文的"2022-05-01 至 2023-01-23"不同窗口，OF3 用的是自己划的"PDB 新近窗口"。
- MSA 序列库（自带管线）：`uniprot`、`uniref90`、`mgnify`、`pdb_seqres`（JackHMMER/hhblits 路线另需 `uniref30`，可选 `bfd`），从公共桶下载【代码／配置核实：`scripts/snakemake_msa/download_of3_databases.py`】。

### 5.3 切点与蒸馏

- **结构集切点：与 AF3 相同**（即 PDB 2021-09-30），白皮书原话："Date cutoffs for structural training sets were the same as AF3"。
- **核酸自蒸馏：没有**。"As we did not have a pretrained OF3 model, self-distillation sets for nucleic acids were not used."——这直接解释了为什么 OF3p 的 RNA 只是"打平 AF3"而不是超越。
- 蛋白序列库比 AF3 新（白皮书："using newer protein sequence databases for MSA construction"）。

### 5.4 测试集与隔离

CASP16（蛋白单体、蛋白复合物、RNA 单体；发生在训练切点之后，故可作盲测）、FoldBench（蛋白复合物、DNA/RNA/肽）、AF3 论文的 antibody-antigen 测试集、Ludaic & Elofsson RNA 集、Runs N' Poses（蛋白-配体，按口袋相似度分箱）。隔离方式=时间切点，未见额外去冗余披露（`未核实`）。

### 5.5 preview 是否披露

**没有**。白皮书："does not include the training dataset, which will be released with the final version of OF3"。仓库里只有预处理脚本与缓存类，没有清单/缓存成品。作为对照：**0.4.0（preview2）才发布蒸馏集**（13M MGnify 2025_03 单体 + RNA 蒸馏），**0.5.x 才把"完整训练数据"放到 portal**（"We provide the full training data for reproducing OpenFold3-preview, including our reproduction of the MGnify-based 13M-sequence distillation dataset"）。

---

## 6. 宣称与可验证性

### 6.1 宣称核查表

| 宣称 | 证据类型 | 核查状态 |
|---|---|---|
| "aiming to be a bitwise reproduction of AF3" | 仅文字（README 首段） | **部分证实**：架构数字逐个对上（48/4/24、384/128、σ 四项、200 步、数据配比），但槽位 23≠24、默认 recycle 3≠10、ranking 明显落后 → "bitwise"目前指**结构级**而非**权重级**，白皮书自己也退一步写成"modulo literal model weights parity" |
| "close to the best current AF3 reproduction for any given modality" | 柱状图（Fig. 1–4），**无表格、无源数据** | 【论文自称】+ 无法独立复算（图读数不可作一手数值） |
| "at parity with AF3 on RNA" | 柱状图 + 案例（R1241、8TJU） | 【论文自称】；机制上可信（有 RNA MSA，其他复现没有；8TJU 提 ~23 Å RMSD） |
| "Apache 2.0, academic and commercial" | 许可证文件 + README | **证实**（代码层面）；权重许可未单列 |
| "prerelease includes ... training code" | 仓库实文件 | **证实存在**，但官方定语是 "(unsupported)"，且无数据 → 只算"可读"，不算"可跑" |
| "Templates for protein monomers" | README + query schema | **证实**：`template_alignment_file_path` + `template_entry_chain_ids`，最多 4 链；RNA 模板 "coming soon" |
| "Optimized kernels (cuEquivariance / DeepSpeed4Science)" | 配置开关 | 开关存在，**加速比未披露** |
| "Multi-query processing" | README + CLI | 证实（一个 JSON 多 query、MSA 去重复用） |
| "Improved performance on par with AlphaFold3"（Upcoming 列表） | — | 当时**未实现**，属路线图 |
| "更低显存 / 更长序列" | 无 | **未见任何一手数字**；只有 low_mem 机制【未核实】 |

### 6.2 论文 vs 代码（本篇最重要的一张表）

| # | 论文/文档说法 | 代码/配置实际 | 性质 |
|---|---|---|---|
| 1 | 评测"run 10 trunk recycles"（OF3p2 白皮书 p.4） | `num_recycles: 3`（0.3.1 **与** 0.5.1 都是 3）【核实 L158】 | **口径冲突**：10 次是靠 runner 覆盖传入，仓库默认值是 3 → 用默认命令跑不出论文的采样预算 |
| 2 | 评测"5 MSA seeds × 5 diffusion samples = 25" | CLI 默认 `--num_model_seeds 1`、`--num_diffusion_samples 5` → **5 个结构**；文档还专门写"By default, only 1 random model seed is used with 5 diffusion samples"【核实 L224】 | **默认值 ≠ 论文协议**，复现者必须显式加参数 |
| 3 | 白皮书未提原子槽位 | 代码 `max_atoms_per_token: 23`；AF3 官方实现 24 | **未声明的实现差异** |
| 4 | AF3 SI 写 `c_hidden_tri_att = 64` | 代码 16，作者自己加了注释 "DISCREPANCY: ... In the code, it's 16." | 作者主动标注的 SI/代码冲突（AF3 侧就有） |
| 5 | AF3 SI 对 `c_hidden_msa_att` 写法有歧义 | 代码 8，注释 "8 or 32, possible typo in SI" | 同上 |
| 6 | README 功能矩阵说蛋白"Prediction without MSA"是 Supported（L22） | 同一文档 L175 说 MSA-free "currently discouraged"，需要 dummy MSA | **同一文档内部不一致** |
| 7 | `Inference.md` L31 把"共价键"列为 Coming soon | query schema 已有 `covalent_bonds` 字段 | schema 先行、功能未完成 |
| 8 | 白皮书称给"结构模板作为输入"（全模态评测） | README 只承诺"protein monomers"；RNA 模板 Coming soon | 措辞范围不一致 |
| 9 | 白皮书称模板已实现 | 同时承认 inter-chain mask bug（乘 chain ID 而非判等），preview2 才修 | 已知缺陷，影响多链模板 |

---

## 7. 评测协议与成本

### 7.1 benchmark 与指标（数值层面：未披露）

| 模态 | 测试集 | 指标 | 基线 | 结果出处 |
|---|---|---|---|---|
| 蛋白单体 | CASP16 全部公开单体 | lDDT | AF3 / Protenix / Chai / Boltz-1 | Fig. 1A |
| 蛋白复合物 | CASP16 + FoldBench | DockQ（FoldBench 界面级；CASP16 全界面聚合） | 同上 | Fig. 1B |
| 抗体-抗原 | AF3 论文 AbAg 集 | DockQ（界面级）、oracle + ranking | 同上 | Fig. 1C |
| RNA 单体 | CASP16 RNA + Ludaic & Elofsson | lDDT | 同上 | Fig. 2A |
| 蛋白-配体 | Runs N' Poses（按口袋相似度分箱） | success rate（RMSD<2 Å 且 lDDT-PLI>0.8） | 同上（此图 AF3 **带模板**） | Fig. 4 |

**关键限制：白皮书只给柱状图，没有任何数值表、也没有源数据**（8 页里 Fig. 1–4 全是条形图/散点图）。因此"OF3p 比 Protenix 高多少"这类数字**无法从一手材料取得**，本篇一律写"未披露"，不做图读数估算。唯一给出的定量对比是 RNA MSA 消融：8TJU 上"incorporating RNA MSAs improves the predicted model by ~23Å RMSD"。

定性结论（白皮书原文）：OF3p 在多数模态"close to the best current AF3 reproduction"，RNA 与 AF3 打平；**AF3 在抗体-抗原与蛋白-配体（尤其新口袋）仍明显领先**；OF3p 的 **oracle 表现好于 ranked 表现**，ranking 是当前最大短板。

### 7.2 采样预算与成本

| 项 | 值 | 出处 |
|---|---|---|
| MSA seeds × diffusion samples | 5 × 5 = 25（抗体 50 seeds，再随机抽 5 个模拟 5-seed） |  |
| MSA 深度 | 1024（抗体保留 paired MSA，只子采样 unpaired） |  |
| 模板 | 除抗体外均给模板；AF3 对照除注明外**不给模板** |  |
| 扩散步数 | 200（配置默认，未在论文另述） | 【核实 L166】 |
| recycle | 10（评测）/ 3（代码默认） |  |
| 训练规模 | 256 卡（8×32，bf16-mixed，batch 1），74,250 + 1,750 + 250 + 1,750 步 |  |
| **推理显存 / 单结构耗时** | **未披露**（无 GB、无秒/结构、无 GPU 型号实测） | 【未核实】 |

### 7.3 复现者必须注意的两件事

1. **默认不是论文协议**：`run_openfold predict` 只跑 1 seed × 5 samples（§6.2 第 2 行）。若要与白皮书数字对齐，须显式 `--num_model_seeds 5`，并把 runner 的 `num_recycles` 覆盖成 10。
2. **ranking 短板会被默认流程放大**：论文说 ranking 明显落后 AF3，而默认只出 5 个样本、按 `sample_ranking_score` 选一个（`ptm/iptm/disorder/has_clash` 加权，见 `Inference.md` L360-382）。样本越少，ranking 越吃亏 → 想复现"ranked"指标必须加预算。

---

## 8. 对我们的可复用点（VNAR 专列）

### 8.1 能否直接当 VNAR / 纳米抗体预测器

**能，但只是"当普通单域蛋白折叠"用**【推断，依据如下】：

- VNAR（约 120 aa 的单域抗体）在 OF3 里就是一条 protein chain，走单体/多聚体路径；OF3 的 CASP16 单体集与复合物集覆盖这一类长度区间；
- 不需要任何"抗体模式"即可提交（query JSON 只写 `molecule_type: protein` + 序列 + 可选 MSA/模板）；
- 但 OF3 **没有** Chai-1 那种 antibody 专用输入、也**没有** Boltz-2 的 affinity 头 → 只能出结构，不出结合强度。

### 8.2 抗体能力：有 benchmark，没有特权

- 白皮书确实在 AF3 的 antibody-antigen 集上评了 OF3p，且是**唯一明确说"AF3 仍大幅领先"的模态**（Fig. 1C）；
- 抗体的评测口径是"保留 paired MSA、只子采样 unpaired"——对 VNAR 这类**没有轻链**的单域，paired MSA 天然退化为主链 MSA，等于失去了这条为抗体设计的加成；
- 50 seeds/结构的评测预算说明作者知道抗体任务方差大；实际使用若只跑默认 5 样本，可靠性会明显低于论文图。

### 8.3 MSA 依赖强度：强

- 蛋白默认走 ColabFold server（只提交蛋白序列），或用预计算 MSA；**完全没有 MSA 会被劝退**（需 dummy MSA，且"prediction quality will be significantly reduced"）；
- 纳米抗体序列在公共库里深度浅，MSA 有效深度低 → 这正是 VNAR 场景的先天弱点，OF3 没有任何"低 MSA 深度"专门处理；
- 反过来，模板是 VNAR 的可利用点：OF3 支持**蛋白模板**（≤4 个模板链、`template_entry_chain_ids` 可指定），同源 VNAR/IG 结构可作模板注入。但注意 §6.2 第 9 行的多链模板 bug——**单链模板不受影响**，多链（VNAR+抗原）模板在 preview 里不可信。

### 8.4 能否微调：代码在，路径不通

| 问题 | 结论 | 依据 |
|---|---|---|
| 有 finetune 入口吗 | **有**：`run_openfold train`；`examples/training_yamls/finetune_1.yml`（640 token）、`finetune_2.yml` / `finetune_3.yml`（768 token）、DeepSpeed 配置、`train` preset | 【代码／配置核实】 |
| 能直接拿 preview 权重续训吗 | 技术上可以（`restart_checkpoint_path`），但官方标注 training code **unsupported**，且 yaml 里 `debug_mode: true`、路径占位 |  |
| 有"自定义非 PDB 数据"工作流吗 | **没有**，README 把它列在 Upcoming |  |
| 有训练数据/缓存吗 | 仓库内**没有**；但 0.5.1 README 称"We provide the full training data for reproducing OpenFold3-preview, including our reproduction of the MGnify-based 13M-sequence distillation dataset"，经 portal 申请可得（与白皮书"随最终版发布"的口径需对齐）；自建管线脚本见 `scripts/data_preprocessing`、`scripts/snakemake_msa` | +【代码／配置核实：0.5.1 README】 |
| 显存门槛 | 官方训练配置是 256 卡；单卡微调需要自己改 batch/crop/offload，可行性与显存**未披露** |  |

### 8.5 硬边界（VNAR 场景）

1. **无 epitope / pocket 条件** → 不能做"指定表位"的 VNAR 对接式预测，只能无条件折叠或靠模板偏置；
2. **无 affinity / 结合排序** → 不能做 VNAR 亲和力筛选（与 Boltz-2 的分工点）；
3. **无抗体专用 MSA 配对策略**（只有"异源多聚体自动 paired"这一通用机制），单域 VNAR 拿不到抗体加成；
4. **preview 不可复现训练** → 想在自己的 VNAR 数据上做领域适配，成本 ≈ 重建整套 AF3 数据管线，不适合作为短期方案；
5. **权重许可未单列** → 商用/再分发前必须确认权重条款（代码 Apache-2.0 不代表权重同许可）【未核实】；
6. **显存/时长无一手数字** → 不能据此判断"能否在单张 24 GB 卡上跑 200 aa 的 VNAR+抗原复合物"，需自测。

---

## 9. 疑问、心得与参考

### 9.1 未解问题

1. **23 vs 24**：为什么槽位减 1？是显存优化、还是 AF3 的 24 槽里有 1 槽其实是"中心原子占位"？白皮书零解释（`未核实`）。
2. **recycle 3 vs 10**：默认 3 是训练时的值（训练在 0..3 均匀采样），评测却用 10。把评测值设成远高于训练分布，是有意（AF3 SI 也这么做）还是配置漂移？白皮书没说（`未核实`）。
3. **参数量**：2.29 GB checkpoint 对应多少参数、是否含 EMA/优化器状态，官方未披露（`未披露`）。用 4 字节/参数粗算 ~5.7 亿，但这是**推测，不写进正文结论**。
4. **ranking 为什么落后**：白皮书只承认现象，未给消融。是 confidence head 训练不足（250 步 ft2）还是样本预算不足？（`未核实`）
5. **HF 侧许可/模型卡**：本机 HF API 不可达，`OpenFold/OpenFold3` 的许可与文件清单未核实（`未核实`）。

### 9.2 心得

- **判断"复现"要看配置，不看口号**。OF3 的"bitwise"最有说服力的证据不是 README，而是 `model_config.py` 里 48/4/24、384/128、16/4e-4/160/7 与 AF3 逐个对齐，以及训练 yaml 里 0.5/0.495/0.005/0.02 的数据配比。反过来，槽位 23、recycle 3 这两处"没写进论文的差异"也提醒：**复现品的默认配置不等于论文协议**。
- **preview 的"开源"要拆成四件事**：权重（给）、推理代码（给全）、训练代码（给但不支持）、训练数据（不给）。任何"OpenFold3 完全开源可复现"的说法，在 0.3.1 这个时点上是**不成立**的。
- **评测章节只有图没有表**，是这类"白皮书式发布"的通病：想要数字只能自己跑，或者等 preview2/OpenBind 的后续报告（0.4.0 起才有更完整的蒸馏集与 155k 步训练叙述）。
- 对 VNAR，OF3 的价值不在"抗体能力"（它明确弱于 AF3），而在**可读的训练栈**：`train` preset、四段 finetune yaml、预处理脚本都给出来了，是少数能看清"AF3 式训练怎么组织"的开源实现——即便 preview 阶段跑不通。

### 9.3 参考

**一手（论文/白皮书）**
- OpenFold3-preview Technical Report（white paper），8 页，`assets/of3p1_technical_report.pdf`（仓库 0.5.1 tag）。**无 DOI、无 arXiv/bioRxiv 版本**。
- OpenFold3-preview2 Technical Report（8 页），`https://portal.openfold.omsf.io/reports/of3p2_technical_report.pdf`。
- AF3：Abramson et al., *Nature* 2024（本系列基线笔记 `2026-11-04\af3-model-note\outputs\AlphaFold3.md`）。

**代码/配置核实点（tag 0.3.1，路径 + 行号）**
- `openfold3/projects/of3_all_atom/config/model_config.py` L13-33（通道）、L33（σ_data）、L48（23 槽）、L158（recycle 3）、L162-166（rollout）、L207-218（template）、L246-259（MSA）、L276-290（pairformer）、L320/L337/L353/L388（atom enc/diff/dec/heads）、L366-374（噪声表与采样器）、L448/L465（diffusion loss）。
- `openfold3/projects/of3_all_atom/config/model_setting_presets.yml` L1-48（train / predict / low_mem / pae_enabled）。
- `openfold3/projects/of3_all_atom/config/inference_query_format.py` L30-70（`Atom`/`Bond`/`Chain`/`Query` 字段）。
- `openfold3/core/data/primitives/structure/tokenization.py` L140-158（token 化四步）。
- `openfold3/scripts/download_openfold3_params.sh` L35-40（S3 桶 + `--no-sign-request`）。
- `docs/source/Inference.md` L10-71（功能矩阵）、L144-148 + L224-226（默认 5 samples / 1 seed）、L175（MSA-free 劝退）、L255-264（low_mem）。
- `examples/training_yamls/initial_training.yml`（256 卡、bf16、384 token、数据配比）、`finetune_1/2/3.yml`（640/768 token）。
- `scripts/data_preprocessing/create_pdb_validation_dataset_cache.py`（2021-10-01–2023-01-13、4.5 Å、2048 token）。
- `scripts/snakemake_msa/download_of3_databases.py`（uniprot/uniref90/mgnify/pdb_seqres）。

**二手（不作为一手证据）**
- `https://bio.rodeo/models/openfold3`（第三方模型目录，仅用于交叉确认"开源复现 AF3"的定位）。
- OpenBind 博客 `openbind.uk/news/blog-openbind-0-...`（0.5.0 的 OpenBind-0 公告；本篇未读全文，仅引 README 转述）。
- arXiv 2604.18559 *ConforNets: Latents-Based Conformational Control in OpenFold3*（衍生工作，2026-04-20；用于说明 OF3 被当底座，不用于 OF3 自身事实）。
