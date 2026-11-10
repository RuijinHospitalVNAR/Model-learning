---
model: Boltz-2
aliases: ["Boltz2", "boltz2_conf.ckpt + boltz2_aff.ckpt", "jwohlwend/boltz（main 即 Boltz-2）"]
family: "Boltz 线第二代（Passaro / Corso / Wohlwend / Reveiz / Thaler …，与 Recursion 合作）；上游 Boltz-1（AF3 骨架），下游 Boltz-2 affinity 任务"
granularity: "蛋白/RNA/DNA 每残基 1 token（非标准残基也压成 1 token）；配体逐原子 token；原子侧不用 AF3 的 24 槽位，改用 atom window（queries 32 / keys 128）"
backbone: "PairFormer 64 blocks / 16 heads，single 384、pair 128、num_bins 64 + MSA 模块 4 blocks（msa_s 64、pairwise_head_width 32）；affinity 分支另挂 8 层（成员 1）/ 4 层（成员 2）的 pair-only PairFormer"
generative: "原子级 EDM 扩散：σ_data 16、σ_min 1e-4、σ_max 160、ρ 7、step 1.5、noise 1.003、γ0 0.8、γmin 1.0；默认 200 步 / 1 sample / 3 recycle，affinity 通道 200 步 × 5 samples / 5 recycle"
conditioning: "MSA（a3m / 多链 CSV / empty / 服务器）| 多链模板（force / threshold）| pocket 与 contact（4–20 Å）| covalent bond（仅 CCD 配体与标准残基）| method 标签 | affinity（binder 必须是 ligand）"
data_scope: "PDB 截止 2023-06-01（Biological Assembly 1，≤7 MB / ≤5000 残基）+ MISATO 11,235 + ATLAS 1,284 + mdCATH 5,270 + AFDB 约 500 万单体 + Boltz-1 多模态蒸馏；亲和力 = ChEMBL v34 + BindingDB + PubChem 1.8.1 + CeMM + MIDAS（1.2M 值点 / 3.4M 二分类点，iptm≥0.75 过滤后）"
status: "代码 MIT（LICENSE 核实）与权重 MIT（仅 README 声明，HF 模型卡本次不可达、未核实）——两处出处分别记录；训练代码仍是 Boltz-1 版，Boltz-2 训练/评测 README 标 coming soon；亲和力只有推理、特征与裁剪代码"
verified_at: 2026-11-10
---

# Boltz-2

> 三句话速判：**Boltz-2 = AF3 骨架 + 一次结构侧升级（48→64 层、AF3 式模板、crop 512→768、σ_min 1e-4、ρ 7）+ 一个新任务头**。新东西是 **affinity：不是新骨干，而是挂在共享 trunk 表示之上、读 ipTM 选优后预测坐标、训练时梯度被显式截断的"下游读取器"**（Algorithm 1 与 `affinity.py` 双证），把蛋白–小分子亲和力做到接近 FEP（4-target Pearson 0.66 vs FEP+ 0.78），代价是必须跑第二次完整前向。**蛋白–蛋白／纳米抗体–抗原亲和力在代码层被直接拒绝（binder 必须是 ligand），所以它不能当 VNAR 亲和力预测器。**

---

## 1. 定位与谱系

### 1.1 出处

| 项 | 内容 | 标记 |
|---|---|---|
| 论文 | 《Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction》，bioRxiv `10.1101/2025.06.14.659707` v1，**posted 2025-06-18**，55 页，CC-BY 4.0；Passaro, Corso, Wohlwend, Reveiz, Thaler, Somnath, Getz, Portnoi, Roy, Stark, Kwabi-Addo, Beaini, Jaakkola, Barzilay | 【核实】 |
| 版本 | arXiv API `all:"Boltz-2"` 命中全是引用它的第三方论文，**无 arXiv 主论文**；bioRxiv API 只返回 `version: 1`、`published: NA`，**无 v2**；未检索到期刊版 | 【核实】 |
| 仓库 | `github.com/jwohlwend/boltz`，public，`main`，MIT，created 2024-11-17，约 4.3k star；取文件时 commit **`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`**（2026-05-29） | 【核实】 |
| 权重 | HF `boltz-community/boltz-2`：`boltz2_conf.ckpt`、`boltz2_aff.ckpt`、`mols.tar`；CCD 仍用 `boltz-community/boltz-1/ccd.pkl` | 【代码核实：`main.py` L36-51】 |

### 1.2 相对 AF3：复现还是改了什么

论文不谈"复现 AF3"，但架构同构。下表 AF3 列取自同项目 AF3 笔记（代码默认值），Boltz-1 列取自 `main.py` dataclass 默认值，Boltz-2 列取自 `PairformerArgsV2` / `Boltz2DiffusionParams` / 论文 Table 6-7：

| 维度 | AF3 | Boltz-1 | **Boltz-2** |
|---|---|---|
| 表征 | 残基 token + 配体逐原子；`max_atoms_per_token=24` 槽位 | 同思路 | 非标准残基压成 1 token；原子侧用 window（32/128）【P23】 |
| trunk | 48 blocks，single 384 / pair 128 | 同 AF3 | **64 blocks**，宽度不变（加深）【P23】 |
| MSA | 4 blocks | 4 blocks，msa_s 64 | 同参【`main.py` L91-105】 |
| 模板 | 有 | **无** | **AF3 式模板**（2 blocks、pairwise dim 64、多链）（新增）【P23-24】 |
| 扩散 / 置信 | 24 token blocks、σ_min 4e-4、ρ 7、γ0 0.8；4 层置信 PairFormer | σ_min 4e-4、**ρ 8**、step 1.638、γ0 0.605；**48 层**置信头 | **σ_min 1e-4、ρ 7、step 1.5、γ0 0.8**；**8 层**置信头，PDE/PAE 拆链内/链间（详见 §3.3–3.4）【Table 7；`main.py` L128-144；P25】 |
| 条件 | MSA/模板/pocket | MSA/pocket（single 通道） | pocket/contact 走 **pair 通道**，模板可 force，加 method/cyclic/修饰【P24-25】 |
| 目标 | 结构 + 置信 | 结构 + 置信 | **加集合监督 + B-factor + distogram + affinity 双头**（新头）【P3 ；P23】 |

**判断**：Boltz-2 = AF3 架构的第三方复现 + 结构侧升级（64 层、模板、crop 768）+ 一个新任务头；结构侧改动服务于亲和力表示质量（"affinity prediction builds on the latent representation driving the cofolding process"，）。

### 1.3 谱系位置

```
AF3 (Nature 2024, 不开源)
 ├─ Boltz-1 (2024-11, MIT, 首个逼近 AF3 的开源复现：48 层、无模板、无亲和力)
 │    └─ Boltz-2 (2025-06, MIT, 64 层 + 模板 + affinity 头 + 集合监督)
 ├─ Chai-1 (2024-10, Apache-2.0 代码 + 权重；论文正文另称 non-commercial)
 ├─ Protenix (字节, Apache-2.0)
 └─ OpenFold3
```

直接上游是 **Boltz-1**，论文明列三处结构差异：模板模块、48→64 层、bf16 + trifast 核把 crop 从 512 抬到 768【P23】。下游意义：affinity 头是"共折叠表示 + 轻量下游头"在开源界的首个完整实现，其"梯度截断 + 界面 mask + 成对差损失"可移植到其他共折叠模型。

### 1.4 许可证与可得性（代码与权重分开写）

| 对象 | 许可 | 原文/依据 | 标记 |
|---|---|---|---|
| **代码** | **MIT** | `LICENSE` 为 MIT 全文，`Copyright (c) 2024 Jeremy Wohlwend` | 【核实】 |
| **权重** | **MIT（作者声明）** | README L19："All the code and weights are provided under MIT license, making them freely available for both academic and commercial uses."；L83 复述 | 【核实（README 原文）；HF 卡片未核实】 |
| 论文 / 训练 / 评测 | 论文 CC-BY 4.0；训练代码只有 Boltz-1 版；评测代码未发布 | README L70、L60 均为 "⚠️ Coming soon" 句式 | 【核实】 |

**代码与权重同为 MIT，是开源共折叠模型里最宽松的一档**（Chai-1 权重非商用；AF3 两者均受限）。但 MIT 只覆盖许可、不覆盖"能训"：亲和力训练管线未开源，`affinity` 只出现在推理/特征文件，`train/configs/*.yaml` grep 不到【代码核实】。

---

## 2. 表征粒度与表示层

### 2.1 token 化

- 蛋白按**氨基酸**、RNA/DNA 按**核苷酸**、其他分子按**原子**切 token；与 AF3/Chai-1/Boltz-1 唯一差别是**非标准氨基酸/核苷酸不再拆成逐原子 token，而保留 1 个 token**【核实：P23】。
- 配体（SMILES 或 CCD）逐原子成 token；`mols.tar` 给每配体最多 10 个 3D 构象、训练随机抽 1【核实：P16】。
- 多链同构实体用 `id: [A, B]` 声明，tokenizer 按 `asym_id` 区分【`docs/prediction.md` L71；`tokenize/boltz2.py` L172-174】。

### 2.2 原子槽位：不是 AF3 的 24，而是 atom window

AF3 用 `max_atoms_per_token=24` 槽位 + mask；Boltz-2 **无固定槽位**，按 `atoms_per_window_queries=32` / `keys=128` 切窗 + `atom_pad_mask`，窗口数随原子数变化（`models/boltz2.py` L77-78；`featurizerv2.py` L1118/L1504-1509）。好处是原子多的残基不被截断，代价是需预先知道总原子数；对大配体更安全【推断】。

### 2.3 无 frame、无扭转角；原子级 cross-attention 存在

与 AF3 一致：**无 residue frame、无扭转角**，坐标直接在原子层扩散，token 只作索引与单位；`token_to_rep_atom` 把原子坐标池化回 token 表示（`affinity.py` L104-105 即用它把 `x_pred` 投影为 token 坐标）。原子↔原子注意力在窗口内由 `AtomAttentionEncoder/Decoder` 完成，token 间由 `DiffusionTransformer` 做全局注意力；该去噪模块"与 Boltz-1 相同"，无变化【核实：P23】。新增输入特征（P23 ）：single 层 `cyclic` / `modified` / `mol_type` 与**实验方法 one-hot**，pair 层**键级 one-hot**、环状聚合物位置编码、对称链 relative chain 编码，MSA 层**配对序列标志**。

---

## 3. 主干与生成范式

### 3.1 Trunk

| 参数 | 值 | 来源 |
|---|---|---|
| PairFormer blocks | **64**（16 heads，dropout 0） | P23："increased from 48 to 64" + `PairformerArgsV2.num_blocks=64` |
| single `token_s` / pair `token_z` | **384 / 128** | `scripts/train/configs/structure.yaml` L81-82（Boltz-1 版；Boltz-2 未改宽度） |
| `num_bins` / `atom_s` / `atom_z` | **64 / 128 / 16** | 同上 L73、L79-80；`DistogramModule(token_z, num_bins)` |
| recycle | 训练 5（affinity 通道固定 5）；推理默认 **3** | P26；`--recycling_steps` 默认 3 |
| 训练 crop / 精度 | 384 → 512 → 640 → 768（4 阶段）；trunk 大部分 bf16 + trifast 三角核、去噪模块 float32 | Table 5 ；P23 |

**"AF3 式"落到数字**：64 blocks × 16 heads、pair 128 / single 384 —— 相对 AF3 是**同形状加深 1/3，宽度完全没动**：只加深，不加宽。

### 3.2 MSA 模块与采样策略

- 模块：4 blocks、`msa_s=64`、`pairwise_head_width=32`、`pairwise_num_heads=4`、`use_paired_feature=True`、可选 `subsample_msa`（1024 行）【代码：`main.py` L91-105】；训练 MSA 深度最多 **8192 条**（Table 6 ）。
- 相对 Boltz-1 的改动：**不再贪婪取 top-k，而在 top 16k 命中里随机采样**；且**5% 的训练迭代把整复合物的 MSA 丢掉**（逼模型学单序列）【核实：P28】。
- 推理侧：`msa` 必填，或 `--use_msa_server` 自动搜，或 `msa: empty` 走单序列（不推荐）【`docs/prediction.md` L74-77】。

### 3.3 扩散模块（EDM）

去噪网络与 Boltz-1 相同，只改扩散超参。Table 7 原注："except for sigma min, we opted for AlphaFold3's default hyperparameters"【核实】：

| 参数 | Boltz-1 | **Boltz-2** | AF3 | 来源 |
|---|---|---|---|---|
| σ_data / σ_max | 16.0 / 160 | **16.0 / 160** | 16.0 / 160 | `main.py` L119/L137-139 |
| σ_min | 4e-4 | **1e-4** | 4e-4 | Table 7；`main.py` L137 |
| ρ | 8 | **7** | 7 | Table 7；`main.py` L135 |
| γ0 / γ_min | 0.605 / 1.107 | **0.8 / 1.0** | 0.8 / 1.0 | 同上 |
| noise_scale / step_scale | 0.901 / 1.638 | **1.003 / 1.5** | 1.003 / 1.5 | 同上 |
| P_mean / P_std | -1.2 / 1.5 | -1.2 / 1.5 | -1.2 / 1.5 | `main.py` L140-141 |
| 步数 / 样本数 | 200 / 1 | **200 / 1**（CLI）、**200 × 5**（affinity） | 200 / 1 | `--sampling_steps` / `--diffusion_samples_affinity` |

c_skip / c_out / c_in / λ(σ) 与 AF3 同形式（`diffusionv2.py` L240-249：`σ_data²/(σ²+σ_data²)`、`σ·σ_data/√(σ²+σ_data²)`、`log(σ/σ_data)·0.25`）【代码核实】。

**扩散模块的层数/宽度：未披露**。`num_blocks`、`token_s`、`atom_s` 由 checkpoint 超参 `score_model_args` 注入（`boltz2.py` L277-286），代码无默认值、论文也没给；AF3 的"24 token blocks"不能平移【未核实】。

### 3.4 置信度模块（一次重设计）

AF3 用 4 层 PairFormer；**Boltz-1 的坑**是置信模型与 trunk 同规模（48 层）且用 trunk 权重初始化，等于把 trunk 再跑一遍。**Boltz-2 用 8 层**，只架在 trunk 末端 pair 表示 + 预测坐标编码上，并把 PDE/PAE logits 的最后一层拆成"链内/链间"【核实：P25】。训练侧：单阶段、crop 512、只用 PDB；每次迭代随机抽 **20/50/200** 步与 **1.0–1.5** step scale，让置信头对推理预算不敏感【P29】。

### 3.5 ★affinity：新任务头，不是新骨干

结论：**新增任务头（a new task head attached to the shared trunk representation），不是新骨干，也不是独立网络。**

#### 3.5.1 论文侧证据

1. §3 把架构拆成四块，affinity 是第四块，与 trunk / denoising / confidence 并列【核实：P4】。
2. Algorithm 1 的输入只有三样：**trunk 5 次 recycle 后的 `z_trunk`**、**输入特征 `s_inputs`**、**预测坐标的 distogram `D_ij`**——没有新的序列编码器、MSA 通路或原子编码器【核实：P26】。
3. 模块本体 = `PairwiseConditioner` + `PairFormerModule`（mask 只留蛋白–配体与配体内部）+ 均值池化 + 两个 MLP 头【同上】。
4. 训练时**梯度从 trunk 截断**："Affinity training is performed after structure and confidence training, with gradients detached from the trunk."【核实：P5】→ trunk 权重不被亲和力目标更新。
5. 两个集成成员只差 3 个标量：**层数 8 vs 4、λ_focal 0.8 vs 0.6、样本 55M vs 12.5M**（Table 3）——一个头的自由度就这么多，不是"新骨干"量级【P27】。
6. 图 2 中 affinity 的输入边穿过 **"Stop Gradient"**，自带独立 "PairFormer Module" 方块，输出 "Binding Likelihood" 与 "Affinity Value"【P4 图注】。

#### 3.5.2 代码侧证据（与论文逐条对上）

| 论文表述 | 代码实现 | 位置 |
|---|---|---|
| 输入 `z_trunk`（5 recycle 后） | `z = z_affinity.detach()`，且进模块前先 `z * cross_pair_mask` 硬置零 | `boltz2.py` L615-620、L633 |
| 输入 `s_inputs` | `self.input_embedder(feats, affinity=True)` 后 `.detach()` | L627、L632 |
| 输入 `D_ij` | ipTM 最高 sample 的原子坐标 → `torch.cdist` → 64 分箱（2–22 Å）→ `nn.Embedding` | `affinity.py` L95-108 |
| `PairwiseConditioner` | `PairwiseConditioning(token_z, token_z, num_transitions=2)` | L60-64 |
| `PairFormerModule`（mask = 蛋白–配体 + 配体内部） | `PairformerNoSeqModule`（**纯 pair、无 MSA/序列更新**），mask = lig×rec + rec×lig + lig×lig | L66、L121-130 |
| 均值池化 + 两个 MLP 头 | `sum(z*mask)/sum(mask)`，头内再乘 mask 并**减单位对角**；回归 3×Linear+ReLU→1，分类 2×Linear+ReLU→1 再 `Linear(1,1)` | L154-218 |
| 类注释 | `class AffinityModule(nn.Module): """Algorithm 31"""` | L34-35 |

**"Algorithm 31" 是关键线索**：AF3 算法编号到 30 为止，作者在 docstring 里给新模块编号 31，等于承认"AF3 算法清单之外新增的一块"。

#### 3.5.3 并联还是串联：两个答案

- **图上并联**：affinity 与 confidence 同层，都从 trunk 末端读表示（图 2）。
- **计算上串联**：`affinity_prediction` 在 `confidence_module` 之后执行，硬依赖 `sample_atom_coords` 与 `iptm`，**必须先完成扩散采样**（`boltz2.py` L586-624）。
- **部署上两次完整前向**：affinity 不复用结构结果，而是**另起 `Boltz2.load_from_checkpoint(boltz2_aff.ckpt)`**，5 recycle / 200 步 / 5 samples、steering 全关，从 `predictions/` 读回结构再算界面【`main.py` L1372-1409】。

#### 3.5.4 输出是两个语义不同的标量

README 明确"trained on largely different datasets … should be used in different contexts"：

| 输出 | 语义 | 值域 | 用法 |
|---|---|---|---|
| `affinity_pred_value` | `log10(IC50)`，IC50 以 **µM** 为基准 | 无界（−3 = 1 nM；0 = 1 µM；2 = 100 µM） | hit-to-lead / lead-opt 的**排序与小幅改构效应** |
| `affinity_probability_binary` | 结合概率 = `sigmoid(logit)` | 0–1 | hit-discovery 的 **binder vs decoy 判别** |

集成后加**分子量校正**：论文写 `ŷ = C0·(y1+y2) + C1·MW + C2`；代码是 `1.03526·mean(y1,y2) − 0.59993·MW^0.3 + 2.83288`（MW 先取 0.3 次幂）【代码核实】。

### 3.6 训练目标（四件套）

1. 结构：one-hot distogram 的加权多类交叉熵（集合监督，K 个构象的 distogram 先聚合）+ 坐标去噪损失（每次迭代随机抽一个构象）【核实：P28】。
2. B-factor 头：用 trunk 末端 single 表示预测代表原子的 B-factor；MD 结构由 RMSF 换算 `B = (8π²/3)·RMSF²`【核实：P28】。
3. 置信：pLDDT/PAE/PDE/pTM/ipTM（与 AF3 同名）。
4. 亲和力：Huber（δ=0.5）绝对值项 + **更强权重的 intra-assay 成对差项** + focal loss（二分类）+ 删失感知（`'>'` 视为下界，只有预测低于该界才计入损失）【核实：P6、P33】。

### 3.7 论文 vs 代码

| # | 事实 | 论文 | 代码 | 影响 |
|---|---|---|---|---|
| 1 | 界面 mask 位置 | Algorithm 1 在 PairFormer 前用 `pair mask` 屏蔽 | 进模块前先 `z_affinity = z * cross_pair_mask` **硬乘**，模块内再用 `pair_mask` 一次 | 数值等价；表示模长略变 |
| 2 | MW 项 | `C1·MW` | `C1·MW^0.3` | **非线性隐含在代码里**，论文公式未写幂次 |
| 3 | 集成系数 | `C0·(y1+y2)+C1·MW+C2` | `1.03526·mean(y1,y2)+…` | 等价换算 C0' = 0.5176，字面值不等 |
| 4 | 二分类头形式 | 4 个 Linear+ReLU 后 SoftMax | 2 个 Linear+ReLU→1→`Linear(1,1)` 出 logit，外部 sigmoid | 伪代码把两头写成对称，实际回归 3 层、分类 2 层 |
| 5 | affinity 的 recycle 次数 | `z_trunk after 5 recycling` | 结构阶段 CLI 默认 3，affinity 阶段**强制 5** | 两阶段 recycle 预算不同 |
| 6 | 训练配置 | 64 层 / σ_min 1e-4 | `scripts/train/configs/*.yaml` 仍是 48 层 / σ_min 4e-4 | **仓库配置不是 Boltz-2 的配置**，不能当复现依据 |

---

## 4. 条件控制

### 4.1 接口总表

| 条件 | yaml 字段 | 接口位置 | 需重训 | 标记 |
|---|---|---|---|---|
| MSA | `sequences[].protein.msa: PATH`（.a3m 或两列 CSV）/ `empty` / `--use_msa_server` | `docs/prediction.md` L74-77；`featurizerv2.py` | 否 | 【代码核实】 |
| 模板 | `templates: [{cif\|pdb, force, threshold, chain_id, template_id}]`；多链同构用 `id: [A, B]` | 同上 L71、L97-101；`TemplateModule`（2 blocks、pairwise dim 64） | 否（模板能力训练时就有，非 zero-shot 外挂） | 【核实：P24；代码 L2305 附近】 |
| pocket | `constraints: [{pocket: {binder, contacts: [[chain, res_idx/atom_name]], max_distance, force}}]` | L37-41 | 否 | 【代码核实】 |
| contact / restraint | `constraints: [{contact: {token1, token2, max_distance, force}}]`，`max_distance` **4–20 Å，默认 6 Å** | L42-46、L94 | 否 | 【代码核实】 |
| 共价键 | `constraints: [{bond: {atom1: [chain, res_idx, atom], atom2: [...]}}]`，**仅 CCD 配体与标准残基** | L90 | 否 | 【代码核实】 |
| 环肽 / 修饰残基 | `cyclic: true`、`modifications: [{position, ccd}]`（ccd 仅 CCD 配体） | L26-29、L79-83 | 否（训练期特征） | 【核实：P23 + 文档】 |
| 实验 method | 无 yaml 字段；推理由 `override_method` 注入（affinity 通道固定 `"other"`） | `main.py` L1368 | 否 | 【代码核实】 |
| **亲和力** | `properties: [{affinity: {binder: CHAIN_ID}}]`，**binder 必须是 ligand、只能一条链、不能多拷贝** | `schema.py` L1045-1077 | 否（推理开关） | 【代码核实】 |
| 抗体特定条件 | 无专门字段，靠 MSA + 模板 + 多链 | — | — | 【文档无此字段】 |

### 4.2 三个值得单独说的点

1. **pocket/contact 从 single 通道搬到 pair 通道**。Boltz-1 把口袋塞进 single token 且只能给一个 restraint；Boltz-2 改成 pair 特征（contact 类型 one-hot + 距离 Fourier 嵌入 `(d−4)/16`），可给多个【P25】。接口层变化、不需重训，但 Boltz-1 的用法不能直接迁移。
2. **模板可以 force**：`force: true` + `threshold` 用势函数把主链拉在模板附近（`E_planar = Σ max(‖x_i − x_i^ref‖ − α_cutoff, 0)`），且**目前只在蛋白链内部生效**（"we only allow for potentials within protein chains"）【核实：P24】。对"框架保守、CDR3 高变"的纳米抗体，这是天然的框架约束工具。
3. **affinity 通道不用模板**：`if data.templates and not compute_affinity:`（`featurizerv2.py` L2312）——亲和力分支显式跳过模板特征【代码核实】。

---

## 5. 训练数据构成与切分

### 5.1 结构数据

| 来源 | 规模 / 处理 | 标记 |
|---|---|---|
| PDB | 截止 **2023-06-01**；**Biological Assembly 1**；>7 MB 或 >5000 残基剔除；沿用 AF3 清洗（去结晶助剂、去冲突链、<4 个已解析残基的链剔除）；NMR 多模型逐帧；保留 B-factor；每配体最多 10 个 3D 构象、训练随机抽 1 | 【核实：P16】 |
| MSA | ColabFold search，**仅蛋白链**；模板命中只取查询沉积日 **≥60 天前**的 PDB 序列（防泄漏） | 【核实：P16】 |
| MISATO（MD） | 300 K、8 ns、去掉配体漂离 >12 Å 的轨迹；100 帧；过滤后 **11,235 系统** | 【核实：P16-17】 |
| ATLAS（MD） | 100 ns NPT；取最后 10 ns 的 100 帧；**1,284 蛋白** | 【核实】 |
| mdCATH（MD） | 320 K、最长 500 ns；只取轨迹最后 10%；**5,270 系统** | 【核实】 |
| AFDB 蒸馏 | AF2 高置信单体，global lDDT ≥ 0.5，**约 500 万蛋白** | 【核实：P17-18】 |
| Boltz-1 蒸馏 | RNA（Rfam 14.9）、protein-DNA（JASPAR 2024）、RNA-ligand（R-SIM ~2500）、protein-ligand（被亲和力主训练集排除的稀疏数据）、TCR-pMHC（VDJdb）、MHC（IEDB）；质量阈值 PDE/iPDE ≤ 2.0、ipTM ≥ 0.7–0.9；每条 3 recycle × 3 diffusion 样本 | 【核实：P17-18】 |

**采样权重**（Table 2）两点：**MISATO 的 MD 数据权重 0.230，仅次于 PDB（0.455）与 AFDB（0.380）**，比所有 Boltz-1 蒸馏模态加起来还多；ATLAS 仅 0.003、mdCATH 0.010。15 项相加约 **1.24**，故是**相对权重**（未归一化）而非概率分布【P18；"相加约 1.24"为据表逐项相加所得，属推断】。

### 5.2 四阶段训练

LR 1e-3 → 5e-4，crop 384（88k steps）→ 512（4k）→ 640（4k）→ 768（1k）；阶段 1、4 不用 MD 数据，**阶段 4 同时去掉 MD 与蒸馏数据**，只留最高质量实验结构【核实：P29】——是 Boltz-2 相对 Boltz-1 训练策略的实质差别之一。

### 5.3 亲和力数据（重点，逐项给来源；论文没写的写"未披露"）

#### 5.3.1 总量（Table 1，P4）

| 来源 | 类型 | 监督 | binders | decoys | targets(90% id 簇) | compounds |
|---|---|---|---|---|---|---|
| ChEMBL + BindingDB | optimization **values** | values | **1.2M**（1.45M） | 0 | 2k（2.5k） | 600k（700k） |
| PubChem small assays | hit-discovery | **both** | 10k（13k） | 50k（70k） | 250（300） | 20k（25k） |
| PubChem HTS | hit-discovery | binary | 200k（400k） | 1.8M（3.5M） | 300（500） | 400k（450k） |
| CeMM Fragments | hit-discovery | binary | 25k（45k） | 115k（200k） | 1.3k（2.5k） | 400（400） |
| MIDAS Metabolites | hit-discovery | binary | 2k（3.5k） | 20k（35k） | 60（100） | 400（400） |
| ChEMBL + BindingDB 合成 decoy | — | binary | 0 | 1.2M（1.45M） | 2k（2.5k） | 600k（700k） |

括号内是**未加 iptm≥0.75 过滤**前的数字，括号外是过滤后用量【P4】。表注"Targets = 90% 序列一致性聚类数"——**靶点级**去重，非复合物级。合并口径：值监督约 **1.21M 点**，二分类约 **3.4M 点**【推断，Table 1 相加】。

#### 5.3.2 连续亲和力（values）

- **数据源与活性类型**：PubChem、ChEMBL（v34）、BindingDB（无版本号）【P3 ；P19-20】；`Ki / Kd / IC50 / XC50 / EC50 / AC50` 六个活性类型进同一个回归头【P3、P6】。
- **单位与变换**：统一 **log10、以 1 µM 为基准**，`IC50=10⁻⁹ M → 输出 −3`【核实：P3 + 文档 L245-247】。
- **截断/删失**：`'>'` 视为下界，做 censoring-aware Huber；`'<'` 在 hit-to-lead **直接丢弃**；`'>'` 且活性 < 10 µM 的也丢弃【核实：P21、P33】。
- **assay 级过滤（最严一档）**：ChEMBL 只留 `confidence=9`、`target_type=SINGLE PROTEIN`、`assay_type ∈ {biochemical, functional}`、排除 unreliable source；两库都去 PAINS、去 >50 重原子配体；去"平均成对 Tanimoto < 0.25"、数据点 < 10、活性标准差 < 0.25、唯一活性值 < 10 或"唯一/总数 < 0.2"、含 `< 10⁻⁶ µM` 值的 assay【核实：P20-21】。
- **BindingDB 去重**：只留 ChEMBL 未覆盖的记录，用 DOI 当 assay id，排除 >1 条链的蛋白【核实：P20】。

#### 5.3.3 二分类（binary）

- **数据源**：PubChem HTS、**CeMM fragment screening**、**MIDAS**（University of Utah 蛋白–代谢物互作组）【核实：P3-4】。
- **过滤**：PubChem HTS 只留 ≥100 化合物且命中率 <10% 的 assay；对每个 (蛋白序列, SMILES) 反查是否有独立 assay 的定量值且标 Active；论文自评"高通量初筛约 **40%** 的 active 可能是假阳性"【核实：P21】。
- **合成 decoy**：把 hit-to-lead 的 binder 跨靶点打乱生成 decoy，要求与同源蛋白所有已知 binder **Tanimoto < 0.3**（压低假阴性）【核实：P3-4】。
- **hit-discovery 打标**：`'>'` 一律 inactive；`'='` 且活性 < 2.0 µM 判 active；其余丢弃【核实：P21】。

#### 5.3.4 采样与 batch 构造

| 来源 | 监督 | 权重 |
|---|---|---|
| ChEMBL + BindingDB | values | **0.25** |
| PubChem small assays | values | 0.005 |
| PubChem HTS | binary | **0.44** |
| PubChem small assays | binary | 0.02 |
| CeMM Fragments | binary | 0.03 |
| MIDAS Metabolites | binary | 0.005 |
| ChEMBL + BindingDB 合成 decoy | binary | **0.25** |

- 每 batch **固定 5 个样本且同属一个 assay**（"学 intra-assay 差异"的前提）【核实：P32】。
- **activity-cliff 采样**：以 assay 内亲和力值的 **IQR** 为悬崖分数，assay 采样概率 ∝ IQR（Algorithm 4）；二分类侧是"1 binder + 同 assay 4 个 decoy"【核实：P32】。
- **训练成本**：两个成员均 **128 张 A100**、AdamW、weight decay 0.001、LR 1e-4【核实：P27】。

#### 5.3.5 论文**没有**披露的项

**未披露**：assay 总数（只给 targets/compounds）、Ki/Kd/IC50 各自条目占比、时间切分、训练/验证/测试比例（只说验证集来自 hit-to-lead）、复合物级去重、亲和力与结构数据的显式配比（仅由 5.3.4 采样权重间接决定）、是否用 AlphaFold 预测结构训练（见 C.2.3）。

#### 5.3.6 隔离与泄漏控制

- **结构测试集**：PDB 2024-01-01 之后提交、"significantly different from any structure that any of the models had seen"的复合物【P6】；验证集 **2023-06-01（不含）–2024-01-01（含）**、分辨率 <4.5 Å、序列不在训练簇内、配体 Tanimoto ≤0.85，共 **398 个结构**【P18-19】。
- **亲和力泄漏控制**：所有亲和力数据集先按 `mmseqs easy-cluster --min-seq-id 0.9 --cov-mode 0 -c 0.01` 聚类，训练集里与验证/测试同簇的蛋白全删；**CASP16 与 Recursion 内部集例外**（前者晚于训练截止，后者是拿不到的私有靶点）【核实：P37】。
- **化合物泄漏**：FEP+ 上算每个测试化合物对训练集的最大 Tanimoto（assay 内均值池化），"no significant dependence"；CASP16 上最大 Tanimoto 分别 0.41（L1000）/ 0.59（L3000）【核实：P37】。**TYK2 特别声明**：TYK2 在**测试集**里，前瞻筛选无数据泄漏【核实：P10】。

---

## 6. 宣称与可验证性

四级：**A=一手可核**（表格/代码，已核）、**B=一手但口径不完整**、**C=仅文字**、**D=未核实**。对应规范标记：A→【核实】，B/C→【论文自称】，D→【未核实】。

| 宣称 | 证据类型 | 核查状态 |
|---|---|---|
| "first AI model to approach the performance of FEP"（P1） | 表格（Table 11/12，FEP+/OpenFE 数值） | **B**：0.66 vs 0.78（4 targets）、0.62 vs 0.63（OpenFE）；但"first"是时间性主张，无法证伪  |
| "**at least 1000× more computationally efficient than FEP**"（P1） | 表格（Table 12 耗时列） | **B（算术可支持，对象不是 FEP+）**：Boltz-2 20 GPU sec、OpenFE 6–12 GPU h、ABFE >20 GPU h、**FEP+ 为 "—"**；对 OpenFE → 1080×–2160×，对 ABFE → ≥3600×，对 FEP+ 不可复算  |
| "over 1000 times faster (see Figure 1)"（P2） | 曲线（Fig.1 accuracy-vs-time 散点） | **B**：只能读量级；PDF 文本层无坐标轴数值  |
| "approaches SOTA free energy simulations"（P9） | 表格（Table 12） | **A**：0.66 vs FEP+ 0.78 / ABFE 0.75 / OpenFE 0.66；**PMAE 0.85 优于 OpenFE 1.09、ABFE 0.95**  |
| "outperforms all top-ranking participants by a clear margin"（CASP16，P9） | 表格（Table 14） | **A**：Boltz-2 0.65 vs 最佳参赛 LG016 0.54；Kendall 0.45 vs 0.42  |
| "nearly doubling the average precision"（MF-PCBA，P10） | 表格（Table 13） | **A**：AP 0.0248 vs GAT 0.0133（1.86×）、BACPI 0.0131；EF@0.5% = **18.39** vs 11.12；AUROC 0.8122 vs 0.7928  |
| "Pearson R = 0.95, MAE 0.42 kcal/mol"（TYK2 前瞻，P10） | 表格（预测–预测一致性） | **B**：只对标 Boltz-ABFE 的一致性（centered MAE 0.42、offset 0.92 kcal/mol），**无湿实验验证**  |
| "8/10 + 10/10 命中"（Enamine 虚拟筛选，P11） | 仅文字（计数） | **C**：只给预测为 binder 的计数，**无实验验证**；化合物来源与去重方式正文未说明  |
| "matches or moderately improves over Boltz-1"（结构，P6） | 柱状图（Fig.4，无表） | **A（方向）**：正文有结论、Fig.4 有柱；但文本层无法对齐图例与数值，具体 DockQ 值 **D**  |

**"1000×"怎么算**：Boltz-2 亲和力流程 = 结构再跑一遍（5 recycle + 200 步 × 5 samples）+ affinity 头 = 20 GPU sec；OpenFE 6–12 GPU h = 21,600–43,200 s → 慢 **1,080×–2,160×**；ABFE >20 GPU h → **>3,600×**；FEP+ 的 time 列是 "—"，无法相除。**结论：该宣称在"对 OpenFE/ABFE"口径下成立且偏保守，在"对 FEP+"口径下无法复算**——"数值成立但对象漂移"的典型例【推断，Table 12】。

---

## 7. 评测协议与成本

### 7.1 结构 benchmark

测试集为 2024/2025 年提交、与所有模型训练集显著不同的 PDB 复合物，覆盖 protein / protein-ligand / protein-DNA / RNA；指标为 LDDT-PLI / DockQ / 模态成功率（不逐项抄数值）。结论：与 Boltz-1 持平或小幅提升（RNA 链、DNA-蛋白最明显），优于 Chai-1 与 ProteinX，略低于 AF3【P6】。抗体子集 Mean DockQ 约 AF3 0.49 / Boltz-1 0.37 / Boltz-2 0.38，但**文本层数值与图例无法可靠对齐**【未核实】。

### 7.2 亲和力 benchmark（四个值测试集 + 一个二分类 + 一个私有集）

| benchmark | 规模 | 指标 | Boltz-2 | 最强基线 | 出处 |
|---|---|---|---|---|---|
| **FEP+ 4-target 子集**（CDK2/TYK2/JNK1/P38，87 个中性化合物） | 4 靶点 | Pearson / Kendall / PMAE / MAE(centered) | **0.66** / 0.48 / **0.85** / **0.59** | FEP+ 0.78 / 0.63 / 0.77 / 0.53；ABFE 0.75 / — / — / 0.65；OpenFE 0.66 / — / 1.09 / 0.75 | Table 12  |
| **FEP+ / OpenFE 全量** | **876** 条 hit-to-lead 测量 | Pearson / MAE(centered) | **0.62** / **0.64** | OpenFE 0.63 / 0.94；GAT 0.28；BACPI 0.29 | Table 11  |
| **CASP16** | **140** 个蛋白–配体对、2 靶点（L1000/L3000） | Pearson / MAE(centered) | **0.65** / **0.95** | 最佳参赛 LG016 0.54 / 1.03；GAT 0.50 | Table 14  |
| **内部 hit-to-lead 验证集** | 未给条数 | Pearson | **0.4246** | GAT 0.2512、BACPI 0.1997 | Table 15  |
| **Recursion 8 个盲测** | 8 assay | Pearson（均值）/ MAE(centered) | **0.39** / **1.36 kcal/mol**（验证集 0.86） | 无对照 | P44  |
| **MF-PCBA**（二分类） | 10 个生化 HTS assay、每 assay 下采样至 50,000 对 | AP / EF@0.5% / AUROC | **0.0248 / 18.39 / 0.8122** | GAT 0.0133 / 11.12 / 0.7928 | Table 13  |

指标口径：Pearson / Kendall 在 **assay 内**算再平均（"Pearson correlation averaged over each assay"）；PMAE 是"pIC50 对多构象平均后的 MAE"；MAE(centered) 去掉 assay 系统偏移，衡量**排序**而非绝对标定【Fig.6 图注 + Table 15】。

**没有 CASF-2016**：全文 grep `CASF` **0 命中**；实际评测是 FEP+（两子集）、CASP16、内部验证集、Recursion 私有集、MF-PCBA 五块【-】。

### 7.3 采样预算与成本

结构默认 3 recycle × 200 步 × 1 sample；亲和力 **5 recycle × 200 步 × 5 samples**（取 ipTM 最优）、float32（`main.py` L1372-1380）；数据预处理还需每靶点抽 10 个 binder × (10 recycle, 200 步, 5 samples) 算共识口袋（P30 ）。**单次亲和力预测 20 GPU sec**（A100 级，Table 12 ）；训练侧 4 阶段共 97k steps + 亲和力两成员 55M/12.5M 样本、128×A100，**显存未披露**（Table 5、Table 3）。

---

## 8. 对我们的可复用点（VNAR 专列）

### 8.1 能不能直接当 VNAR–抗原亲和力预测器？**不能**

三层硬边界，从外到内：

1. **接口层**：`binder` 必须是 `ligand` 链，否则抛 `ValueError("Chain {binder} is not a ligand! …")`，且只能 1 条链、不能多拷贝【`schema.py` L1065-1077，】。VNAR 是蛋白链 → **连输入都过不去**。
2. **表示层**：界面 mask 由 `rec_mask = (mol_type == 0)` 与 `lig_mask = affinity_token_mask` 双向组合；若强行把蛋白标成 ligand，重叠会令 mask 语义崩塌【推断，`affinity.py` L112-125】。
3. **数据层**：亲和力训练集全是**小分子**（ChEMBL/BindingDB/PubChem/CeMM/MIDAS），没有任何蛋白–蛋白 Kd 标注；即便绕过接口，输出也没被训练过【核实：Table 1】。

补充：`affinity_mw` 用配体分子量，MW 校正项 `−0.59993·MW^0.3` 对 15 kDa 的 VNAR 会给出荒谬偏移【代码核实】。

### 8.2 能做什么：结构侧才是 VNAR 的可用面

| 能力 | 是否支持 | 依据 |
|---|---|---|
| 单链纳米抗体 + 抗原复合物结构 | **支持**（两条蛋白链 + MSA + 可选模板） | yaml 通用能力 |
| 抗体/纳米抗体专项模式 | **无**（不像 Chai-1 有 antibody mode）；只有 PDB 里抗体界面被 upsampling（权重 0.025） | Table 2  |
| 抗体 benchmark | 略优于 Boltz-1、低于 AF3（详见 §7.1） | §7.1 评测表；口径为论文自报 |
| 多链模板（用已知 VNAR 框架） | **支持**，可 `force: true` + `threshold` 拉主链；但**无 CDR 专用接口**，只能用 contact/pocket 约束 | P24、文档 |
| 抗原–纳米抗体亲和力 / 微调 | **均不支持**：见 8.1；训练代码只有 Boltz-1 版 | 代码核实：`schema.py` L1065-1077、`scripts/train/` 配置清单 |

### 8.3 MSA 依赖强度

- 蛋白链**默认必须有 MSA**；`msa: empty` 显著掉点（README："not recommended, as it reduces accuracy"）。
- 训练时 **5% 迭代整复合物丢 MSA** + top-16k 随机采样，说明作者在刻意降低对 MSA 精度的依赖，但**没降到可以不用**。
- 对 VNAR 的实际影响：CDR3 短、体细胞突变多，自动搜到的 MSA 往往很浅（germline 序列占多数），会压低 CDR3 置信度；建议用**自定义 CSV 多链 MSA**（VNAR 对齐同家族序列、抗原单独一列）【推断，`docs/prediction.md` L76】。

### 8.4 微调可行性与替代路线

- **代码层**：`scripts/train/` 有 Boltz-1 的 Hydra 配置（`structure.yaml` / `confidence.yaml` / `full.yaml`），**没有 affinity 训练配置**；`src/boltz/data/module/` 有 `training.py` 但无 `training_affinity.py`【代码核实】。
- **可行路径（若要做 VNAR–抗原亲和力）**：复用 `AffinityModule` → 把 `rec_mask`/`lig_mask` 改成 "paratope 链 vs epitope 链" → 用 SKEMPI/SAbDab 的 Kd 重训 → 重写采样（activity-cliff 与成对差损失对蛋白–蛋白未必适用）。**无官方支持，属二次开发**【推断】。
- **更现实的做法**：把 Boltz-2 当**结构 + 界面置信度**工具（ipTM/PAE/界面 pLDDT），亲和力排序另用专门的蛋白–蛋白亲和力模型或 FEP。

### 8.5 硬边界汇总

三条：`binder` 只能是 ligand（硬拒绝，单链纳米抗体只可做结构）；多链模板可用，但模板势"only within protein chains"；代码与权重均 MIT、可商用，而训练/评测代码未发布、显存未披露，需 A100 级【；未核实】。

---

## 9. 疑问、心得与参考

### Q1：affinity 到底是"新头"还是"新骨干"？

**新任务头**。六条判据见 §3.5.1，最关键两条：梯度显式截断、输入里没有新的序列/MSA/原子编码器。**反面判据**（什么才算新骨干）：自带 tokenizer/atom encoder/MSA 通路、与结构头共享扩散网络联合训练、trunk 被亲和力目标更新 —— 三条均不成立。**易误判点**：Fig.2 里 affinity 自带 "PairFormer Module" 方块且有 8 层（trunk 的 1/8），像"小骨干"；但它只读 trunk 表示 + 预测坐标 distogram，故是"头"。

### Q2：affinity 训练时用的是真结构还是模型预测？

**模型预测**。C.2.3 流程："用 Boltz-2 跑 5 recycle / 200 步 / 5 samples，取 ipTM 最优结构，存坐标 + trunk pair 表示 + 裁剪索引"，再用这份表示训 affinity 头。它学的是"如何读自己 trunk 的表示"而非"如何读实验结构"——这是它与"结构模型 + 外挂亲和力模型"的本质区别【P31】。

### Q3：三个容易引错的口径

1. **Table 1 的括号**：括号内是未过 iptm 0.75 过滤的数字（PubChem HTS 400k binders），括号外是过滤后用量（200k）；正文两种都引，引用必须带口径。
2. **"1000×"要有对象**：写"相对 OpenFE/ABFE 成立（1080×–3600×），相对 FEP+ 无法从论文复算（Table 12 的 time 列为 '—'）"，不要裸写"比 FEP+ 快 1000 倍"。
3. **仓库配置不是 Boltz-2 的配置**：`scripts/train/configs/*.yaml` 是 48 层、σ_min 4e-4 的 **Boltz-1 配方**（README："currently for Boltz-1 but soon for Boltz-2"），拿去复现 Boltz-2 会得到不同模型。

### 写作心得

1. **"Algorithm 31" 是决定性线索**：AF3 只到 Algorithm 30，作者给新模块编号 31，等于承认"AF3 之外的新算法块"，比读图猜结构可靠。
2. **论文 vs 代码差异集中在实现细节**（mask 位置、MW 幂次、头层数、recycle 预算），不影响结论但影响数值复现。
3. **数据章节"规则细、数字缺"**：过滤细到"活性标准差 < 0.25 丢弃"，但 assay 数、类型占比、时间切分全缺；综述必须区分"给了规则"与"给了数字"。

### 参考资料

- 论文：Passaro, Corso, Wohlwend, Reveiz, Thaler, Somnath, Getz, Portnoi, Roy, Stark, Kwabi-Addo, Beaini, Jaakkola, Barzilay. *Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction.* bioRxiv `10.1101/2025.06.14.659707` v1（2025-06-18），CC-BY 4.0。本篇页码即 bioRxiv v1 PDF 的物理页。
- 代码核实点（commit `b1ebfc4`，逐条已在正文标注 file:line）：`affinity.py` L34-218、`boltz2.py` L547-698、`main.py` L36-144 / L1229-1409、`schema.py` L1045-1077、`tokenize/boltz2.py` L172-174、`featurizerv2.py` L658/1107/1647/2305-2312、`trunkv2.py` L765-827、`crop/affinity.py` L16-17、`structure.yaml` L73-83（`token_s 384 / token_z 128 / num_bins 64`，**Boltz-1 配方**）。
- 二手转述（不作一手证据）：384/128 两个宽度**论文没写数字**，只说"trunk 表示与 Boltz-1 相同"，引用须注明来自仓库配置；结构对比只采信 "edging Chai-1 and ProteinX, but lagging a bit behind AlphaFold3"。
