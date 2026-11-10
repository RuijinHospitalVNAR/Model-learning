# 结构预测 SOTA 横向对比：训练数据 · 方案 · 架构

> **涉及的模型**：[AlphaFold3](../模型篇章/AlphaFold3/README.md) · [OpenDDE](../模型篇章/OpenDDE/README.md) · [Mol-1](../模型篇章/Mol-1/README.md) · [Boltz-1](../模型篇章/Boltz-1/README.md) · [Boltz-2](../模型篇章/Boltz-2/README.md) · [Chai-1](../模型篇章/Chai-1/README.md) · [Protenix](../模型篇章/Protenix/README.md) · [OpenFold3](../模型篇章/OpenFold3/README.md)，参照系 [RFAA](../模型篇章/RFAA/README.md)（非生成式谱系起点）
> **记录人**：（待填）
> **最后更新**：2026-11-10
> **材料来源**：全部取自本仓库上述模型篇章；未引入篇章之外的二手数字。

---

## 0. 阅读约定：为什么这篇要标证据等级

这一代模型（AF3 谱系）**架构高度同构**，真正拉开差距的是训练数据与评测口径。因此"看起来可比的表"极易造假：把 A 论文的自测数、B 论文的他报数、C 代码里的默认值拼进一行，就会得出"X 超过 Y"的伪结论。

本文每个关键数字后附证据等级，含义如下：

| 标签 | 含义 |
|---|---|
| 【核实】 | 已在一手论文/白皮书正文（含页码）核对到原句 |
| 【论文自称】 | 论文写了，但没有第三方复现或代码佐证 |
| 【代码／配置核实】 | 从仓库代码、yaml、配置常量或权重文件名读到 |
| 【推断】 | 由已核实的事实推导，非原文表述 |
| 【未核实】 | 篇章里明确写了"未披露/无法核实"，本文不补数 |

**三条硬规则**（违反其中任一条，该格作废）：

1. 一格只放一个来源的数；来源不同则拆格或加注（例：AF3 的 PoseBusters 有两个数，见 §7.4）。
2. 默认配置 ≠ 论文协议。凡是"论文 5 seeds、代码 1 seed"这类，两列并写并标冲突。
3. 「某模型超过 AF3」这类结论，必须同时给出 AF3 在**同一基准、同一打分脚本**下是否被重跑；栏里是「–」就不构成同台比较。

---

## 1. 九个模型的一句话定位

| 模型 | 谱系位置 | 一句话定位 | 关键差异点 |
|---|---|---|---|
| **AlphaFold3** | 谱系原点 | Pairformer + 原子级 EDM 扩散的原始配方，其余八篇都在对齐它 | 数据与权重均未开放；多模型分切点评测 |
| **OpenDDE** | AF3 复现 + 加宽 | 把 c_z 从 128 抬到 384（参数 ×3），并在抗体/复合物上做同口径基线重跑 | 唯一把基线全部重训重跑的一篇 |
| **Mol-1** | AF3 复现 + 异构 trunk | 8 层 PairMixer（去三角注意力）+ 40 层 Pairformer 的混合主干 | 无蒸馏集；索引侧写明 ≤20 链 / ≤4.5 Å |
| **Boltz-1** | AF3 复现（首个逼近） | 删模板、MSA 可跳过、置信度改为"第二个 48 层 trunk" | 置信头 = 整套 trunk 复刻，开销最大 |
| **Boltz-2** | Boltz-1 升级 | 48→64 层 + 模板回归 + **亲和力头** | 唯一有 affinity 头；切点最晚（2023-06-01） |
| **Chai-1** | AF3 复现 + PLM 轨道 | MSA 从"必需"降为"可选"，用 ESM-2 3B 补位；单模型单切点 | 切点最保守（2021-01-12）；架构数字全未披露 |
| **Protenix** | AF3 复现（开放最彻底） | 权重 + 训练代码 + 四阶段超参 + 预处理数据（~1.5 TB）全给 | 「超 AF3」的 PXMeter 行 AF3 是「–」 |
| **OpenFold3** | AF3 逐位复现 | 连数据配比 0.5/0.495/0.005/0.02 都照抄；槽位 23 与 recycle 3 是未声明差异 | 训练代码"unsupported"、数据 0.3.1 不发布 |
| **RFAA**（参照） | 非生成式前身 | 单次前向 + 4 recycle，SE3 等变网络 + 全原子 FAPE，无扩散 | 无随机采样、无 MSA 条件、无蒸馏 |

---

## 2. 维度一：训练数据

### 2.1 切点 · 过滤 · 规模

| 模型 | PDB 释放切点 | 分辨率过滤 | 链数 / 规模限制 |
|---|---|---|---|
| AlphaFold3 | **2021-09-30**【核实·论文】 | 训练阈值未披露（SI 截断）【未核实】 | 训练集条目数未披露【未核实】；crop 384→640→768、batch 256、150k 步【核实·SI】 |
| OpenDDE | **2021-09-30**【核实·论文】 | 未披露【未核实】 | 各子集条数未披露【未核实】；算力 ≈414K GPU-hours【论文自称】 |
| Mol-1 | **2021-09-30**【代码／配置核实·索引名 `indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz`】 | **≤4.5 Å**【代码／配置核实·同一文件名】 | **≤20 链**【代码／配置核实·同一文件名】；条目数未披露【未核实】 |
| Boltz-1 | **2021-09-30**【核实·论文】 | 论文写 "at least 9 Å"；代码 `structure.yaml` 为 9.0、`confidence.yaml`/`full.yaml` 为 4.0【冲突：论文 vs 代码，未裁决】 | 未披露链数；crop 384 token/3456 atom → 512/4608【核实·代码】；68k 步 × batch 128 |
| Boltz-2 | **2023-06-01**【核实·论文】 | 未披露【未核实】 | Biological Assembly 1、≤7 MB、≤5000 残基【核实·论文】 |
| Chai-1 | **2021-01-12**【核实·论文 p.2 §2.1 / p.9 §5.2 / p.11 §5.6 三处】 | 未披露【未核实】 | 结构数 / 链数阈值未披露【未核实】；128×A100、batch 128、30 天【核实·论文】 |
| Protenix | **2021-09-30**【核实·论文】 | 论文未给单一阈值（WeightedPDB 按分辨率加权）【未核实】 | **约 150k 结构**【论文自称】；另发布 2025-06-30 切点版本（不适合作公平基准）【核实】 |
| OpenFold3 | **2021-09-30**（白皮书："same as AF3"）【核实·白皮书】 | **≤9.0 Å**（预处理脚本 docstring）【代码／配置核实】 | 未披露；训练 256 卡、74,250 + 1,750 + 250 + 1,750 步【代码／配置核实】 |
| RFAA（参照） | PDB 快照日期**未核实**；模板库 `pdb100_2021Mar03`【代码／配置核实】 | 未披露【未核实】 | crop 256→375【核实·论文】 |

> **切点这一列的两个高频错误**（必读）：
> 1. **Chai-1 是 2021-01-12**，不是 2021-09-30。后者是 AF3 的切点，两者常被二手来源混用。
> 2. Chai-1 论文里的 **2022-05-01–2023-01-12 是评测集窗口**，不是训练切点。看到"Chai-1 cutoff 2022-05"即知材料串行了。
> 附带：Boltz-2 的 2023-06-01 比同代晚 21 个月，这不是笔误，而是它敢碰亲和力任务的**前提**（见 §6.1）。

### 2.2 去重 / 聚类 · 蒸馏集 · 专项数据

| 模型 | 去重 / 聚类阈值 | 蒸馏集（AFDB / MGnify 等） | 亲和力 / 复合物专项 |
|---|---|---|---|
| AlphaFold3 | 训练集聚类阈值未披露【未核实】；测试集 40% 同源剔除【核实】 | AF2-Multimer v2.3 蒸馏【论文自称】；配比 Weighted-PDB 0.5 / disordered 0.02 / 单体蒸馏 0.495【核实·SI】 | 无亲和力专项【核实】；复合物靠 PDB 原生 |
| OpenDDE | 训练去重阈值未披露【未核实】 | AFDB-multimer + Teddymer + MGnify（长/短单体）+ Swiss-Prot + Disordered motifs【论文自称】 | SAbDab 抗体占比从 0% 递增到 13%【论文自称】；复合物靠 AFDB-multimer |
| Mol-1 | 未披露【未核实】 | **未见蒸馏集**：全文检索 distill / AFDB 零命中【负证据·核实】 | 无【核实】；随附 MSA 包覆盖 **7,074** 条 PDB 序列（**不是**训练集规模）【代码／配置核实】 |
| Boltz-1 | 未披露；测试集按 40% identity 剔除【核实】 | **OpenFold 蒸馏约 270K 结构**，明确不用 AF3 蒸馏【论文自称】 | 无亲和力；pocket / 表位作条件（30% 迭代、6 Å 部分揭示）【核实·代码】 |
| Boltz-2 | 亲和力侧靶点 90% identity 聚类【论文自称】 | AFDB 约 **500 万**单体 + Boltz-1 多模态蒸馏【论文自称】 | **MISATO 11,235 + ATLAS 1,284 + mdCATH 5,270**；亲和力过滤后 1.2M 值点 / 3.4M 二分类点（iptm≥0.75）【论文自称】 |
| Chai-1 | 未披露；评测集按 40% 同源剔除【核实】 | PDB + **AlphaFoldDB**；明确不使用 AF3 的蒸馏集（AFDB 除外）【核实·论文】 | 无【核实】 |
| Protenix | 未披露【未核实】 | MGnify 2019-05 的 AF2 蒸馏单体**约 1300 万** + disorder 蒸馏约 1.4 万【论文自称】；WeightedPDB 权重 0.60【核实·代码】 | 训练集**排除 PoseBusters**【核实·代码】；无亲和力 |
| OpenFold3 | 未披露【未核实】 | 照抄 AF3 四路配比；**核酸自蒸馏没有**（无预训练 OF3 模型）【核实·白皮书】 | 无【核实】 |
| RFAA（参照） | **30% 序列同一性聚类**（验证取 10% 簇）【论文自称】 | 无（AF2 时代产物）【核实】 | 蛋白-小分子 **121,800 结构 / 5,662 簇**；蛋白-金属 **112,546 / 5,324**；共价修饰 **12,689 / 1,099**；CSD 补充小分子【论文自称】 |

### 2.3 预处理管线与数据本体是否公开

| 模型 | 代码 | 权重 | 训练代码 | 预处理数据本体 |
|---|---|---|---|---|
| AlphaFold3 | 未开源【核实】 | 未公开发布【核实】 | 否 | 否 |
| OpenDDE | Apache-2.0【核实】 | Apache-2.0，PyPI `opendde`【核实】 | 公开【核实】 | 数据本体是否随仓库发布【未核实】 |
| Mol-1 | Apache-2.0，commit `e6e6486`【核实】 | Apache-2.0，base 577.4M【核实】 | 公开【核实】 | 训练索引与 MSA 包公开；训练数据本体【未核实】 |
| Boltz-1 | MIT【核实】 | MIT【核实】 | 公开【核实】 | 蒸馏集是否随包发布【未核实】 |
| Boltz-2 | MIT【核实】 | MIT（README 原文）【核实】 | **Boltz-1 版**；Boltz-2 训练/评测标 "coming soon"【核实】 | 亲和力训练管线未开源【核实】 |
| Chai-1 | Apache-2.0，但只发 **6 个 TorchScript 组件**，无 PyTorch 源码【核实】 | Apache-2.0（论文正文写 non-commercial，两处冲突）【核实】 | 否 | 否 |
| Protenix | Apache-2.0【核实】 | Apache-2.0【核实】 | 公开（`train_demo.sh` + 四阶段超参）【核实】 | **公开，约 1.5 TB**（含 MSA 索引）【核实】 |
| OpenFold3 | Apache-2.0【核实】 | 公开（S3 免凭证，单一 2.29 GB ckpt）；权重许可未单列【未核实】 | 在仓库但官方标注 **unsupported**【核实】 | 0.3.1 **不发布**；0.5.1 经 portal 提供【核实】 |
| RFAA（参照） | BSD-3-Clause【核实】 | 1.34 GB 直接下载【核实】 | 【未核实】 | 【未核实】 |

---

## 3. 维度二：方案

### 3.1 训练目标与损失构成

| 模型 | 扩散损失 | 其他损失项 | 损失细节公开程度 |
|---|---|---|---|
| AlphaFold3 | EDM x0 参数化：MSE + smooth LDDT + bond【核实·SI】 | 置信度交叉熵（pLDDT / PAE / PDE）【核实】 | 高（SI 给全） |
| OpenDDE | 沿用 AF3 三项【推断】 | confidence + structural-token refiner 的辅助目标【未核实·权重未披露】 | 中（架构给全，损失权重未给） |
| Mol-1 | AF3 式扩散损失【推断】 | 异构 trunk 端到端训练，未给逐项权重【未核实】 | 低 |
| Boltz-1 | 扩散损失权重改回 EDM 原式（非 AF3 的改法）【核实·代码】 | confidence（= 第二套 48 层 trunk）【核实·代码】 | 高（五处算法差异逐条列出） |
| Boltz-2 | 结构扩散 + **集合监督 + B-factor + distogram** | affinity 双头（`affinity_pred_value` 回归 log10 IC50 µM + `affinity_probability_binary`）；`z = z_affinity.detach()` 梯度截断【核实·代码】 | 中高（结构侧细节未披露，affinity 侧算子级可见） |
| Chai-1 | **论文无 loss 段落**，`trunk.pt` 为编译图【未核实】 | 【未核实】 | 极低 |
| Protenix | AF3 式 diffusion + confidence【核实·代码】 | 四阶段配置公开，第 4 阶段为 confidence-only【核实·代码】 | 高 |
| OpenFold3 | `bond + smooth_lddt + mse`【代码／配置核实】 | confidence head + distogram（39 箱）【代码／配置核实】 | 高（配置可读） |
| RFAA（参照） | **无扩散** | 全原子 FAPE + 结构监督，单次前向【核实·论文】 | 中 |

### 3.2 recycle 课程与采样配置

| 模型 | 训练 recycle | 推理默认（代码） | 论文评测协议 |
|---|---|---|---|
| AlphaFold3 | 未披露【未核实】 | 200 步 × 5 samples【核实】 | **5 seeds × 5 samples = 25**【核实】 |
| OpenDDE | 未披露【未核实】 | 200 步 / 10 recycle / 5 sample【代码／配置核实】 | **20 步 / 4 cycle / 5 sample**【论文自称 Table 1b】→ **与代码冲突** |
| Mol-1 | K≤4，Beta(1,1)→Beta(2,2)→Beta(2,1)【论文自称】 | 200 步 × 5 样本 × 20 recycle【核实】 | base 20 cycle、constraint 10 cycle；5 seeds × 5 samples【核实】 |
| Boltz-1 | 未披露【未核实】 | 200 步 / 1 sample / 3 recycle【代码／配置核实】 | 200 步 / 10 recycle / **5 输出**【核实】→ 默认 ≠ 协议 |
| Boltz-2 | 未披露【未核实】 | 200 步 / 1 sample / 3 recycle | affinity 通道 200 步 × 5 samples / 5 recycle【代码／配置核实】 |
| Chai-1 | 未披露【未核实】 | 200 步 / 5 diffusion / **1 trunk**；3 recycle【代码／配置核实】 | 5 trunk × 5 diffusion = 25；**4 recycle**【核实·论文】→ 4 vs 3 为计数口径差【推断】 |
| Protenix | 未披露【未核实】 | 200 步 / 5 samples / recycle 10（base 默认 4）【代码／配置核实】 | 5 seeds × 5 samples × 200 步 × 10 recycle【核实】 |
| OpenFold3 | 训练在 0..3 均匀采样【代码／配置核实】 | 200 步 / 5 samples / **1 seed** / 3 recycle【代码／配置核实】 | **10 recycle + 5 MSA seeds × 5 = 25**（抗体 50 seeds）【核实】→ 默认 ≠ 协议 |
| RFAA（参照） | **固定 4 次**（MAXCYCLE 4）【代码／配置核实】 | 单次确定性前向，无随机采样【核实】 | 无 5×5 预算概念 |

### 3.3 输出头：置信度 / 亲和力 / 约束

| 模型 | 置信度头 | 亲和力头 | 约束头 |
|---|---|---|---|
| AlphaFold3 | 4 层 mini-trunk；pLDDT 50 bin、PAE/PDE 64 bin；pTM/ipTM 由 PAE 导出【核实】 | 无【核实】 | 无专用头（约束走输入特征） |
| OpenDDE | 沿用 AF3【推断】 | 无【核实】 | contact / epitope / movable_chains / covalent_bonds 走输入通道【核实】 |
| Mol-1 | 沿用（细节未披露）【未核实】 | 无【核实】 | **无 constraint 接口**（constraint 变体仅网页端）【核实】 |
| Boltz-1 | **整个 48 层 trunk 复刻 + 扩散轨迹聚合**【核实·代码】 | 无 | pocket / 表位 / 共价键走输入 |
| Boltz-2 | 8 层；PDE/PAE 拆链内与链间【核实·代码】 | **有**：8 层（成员 1）/ 4 层（成员 2）集成；读 ipTM 选优后坐标 + `s_inputs` + distogram【核实·代码】 | pocket/contact 走 **pair 通道**（4–20 Å，默认 6 Å）【核实·代码】 |
| Chai-1 | 有 `confidence_head` 组件，层数未披露【核实·组件存在／未披露·结构】 | 无【核实】 | CSV：covalent / contact / pocket（pocket 6–20 Å、contact 6–30 Å）【核实】 |
| Protenix | 4 blocks，`stop_gradient=True`；pLDDT 50 / PDE 64 / PAE 64；排序键 `[plddt, gpde, ranking_score]`【核实·代码】 | 无 | token_contact / atom_contact / pocket（JSON）【核实·代码】 |
| OpenFold3 | 4 blocks，最终线性投影前加 LayerNorm【代码／配置核实】 | 无【核实】 | `covalent_bonds` schema 有、文档标 coming soon【核实】 |
| RFAA（参照） | 未见置信头描述【未核实】 | 无 | 无 |

### 3.4 评测协议：用谁的测试集、基线是否重跑

| 模型 | 测试集 | 基线是否重跑 | 备注 |
|---|---|---|---|
| AlphaFold3 | 自建 Recent PDB **8,856**（2022-05-01–2023-01-23、<5,120 tokens、40% 同源剔除）；PoseBusters **428**；CASP15 **10 个 RNA 靶** | 不适用（它是被比的对象） | PoseBusters 用 **2019-09-30 重训模型**，非主模型【核实】 |
| OpenDDE | PXMeter-AB / FoldBench-AB / 2026ARK-AB | **全部同口径重跑**：ESMFold2 51.0、Protenix-v1 50.0、OpenFold3 21.9、Boltz-1 25.0、Chai-1 16.7 | 九篇中唯一自己重跑全部基线的 |
| Mol-1 | FoldBench-AB / PXMeter-AB / MolBench-AB-26 | 未说明 | **PXMeter-AB 与 MolBench-AB-26 图上无 AF3 柱**【核实】 |
| Boltz-1 | 自建 val **553** / test **593**（2023-01-13 后、4.5 Å、100–2000 残基、40% identity、配体 Tanimoto ≤0.8），计入 **541** | AF3 用引用值【核实】 | 测试集自建，与 AF3 的 8,856 不同集合 |
| Boltz-2 | 结构：PDB **2024-01-01 后**提交；验证 398（2023-06-01–2024-01-01） | 亲和力侧 FEP+ / ABFE / OpenFE 自跑【论文自称】 | FEP+ 耗时为「—」，见 §7.5 |
| Chai-1 | PDB **2022-05-01–2023-01-12**、非 NMR、<4.5 Å、≤2048 tokens、40% 同源剔除 | AF3 用引用值 **76.34%**，未复跑【核实】 | 与 AF3 的 8,856 集合有重叠但窗口不同 |
| Protenix | FoldBench / PXMeter | v1 明说基线"从公开仓库收集结果，**不重跑**"【核实】 | PXMeter 上 AF3 行为「–」，见 §7.1 |
| OpenFold3 | CASP16 / FoldBench / AF3 论文 AbAg 集 / Ludaic-Elofsson RNA / Runs N' Poses | 未说明；**只给柱状图、无任何数值表**【核实】 | AF3 对照除注明外**不给模板**（对 AF3 不利）【核实】 |
| RFAA（参照） | 自建 10% 簇验证 + 外部基准 | 未说明【未核实】 | 非生成式，指标口径与扩散模型不同 |

---

## 4. 维度三：架构

### 4.1 trunk 与 diffusion module

| 模型 | trunk（层数 / 通道） | diffusion module | 参数量 |
|---|---|---|---|
| AlphaFold3 | Pairformer **48** 块（pair 128 / single 384）；MSA **4** 块、最深 16,384 行 | **24** 个 token block；token 768 / atom 128；atom transformer 3 块；16 heads | 官方未披露【未核实】 |
| OpenDDE | Pairformer **48** 块，**c_s = c_z = 384**（加宽）；structural-token refiner 4 块/8 heads、`n_roles=7`；MSA 4 块（c_m 128、深度 2,048） | **24** 块，内 c_token 768；atom encoder/decoder 各 3 块；distogram 96 bins [2.25, 25.75] Å | **655,791,538**（≈656 M）【代码／配置核实】 |
| Mol-1 | **48 层异构**：8×PairMixer 448C（**去掉三角注意力**）→ 40×Pairformer 320C + 4 层 MSA | 24 层 token 768C；atom encoder/decoder 各 3 层 | **577.4 M**（base）【核实】 |
| Boltz-1 | PairFormer **48**（single 384 / pair 128 / 16 heads）；MSA 4 块（msa_s 64，**可整模块跳过**） | **24** token blocks / **384** 通道 / 16 heads；atom transformer 3+3 / 128 / atom_z 16 / 4 heads | 未披露【未核实】 |
| Boltz-2 | PairFormer **64**（16 heads，384 / 128，num_bins 64）+ MSA 4 块 | **层数与宽度未披露**（由 checkpoint 注入）【未核实】 | 未披露【未核实】 |
| Chai-1 | **未披露**（论文仅 "largely follows Abramson et al. [11]"） | **未披露**（`diffusion_module.pt` 编译图） | **未披露** |
| Protenix | Pairformer **48**（384 / 128 / 16 heads）+ MSA 4 块（c_m 64、16,384）+ template embedder 0 块（base）/ 2 块（v1、v2） | **24** 块 / 16 heads / c_token 768；atom encoder 3 / decoder 3 | v1 **368.48 M**；v2 **464.44 M** |
| OpenFold3 | Pairformer **48**（384 / 128 / **4 heads**）+ MSA 4 块（c_m 64、`c_hidden_msa_att` 8）+ template 2 块（**c_dgram 39**） | **24** 块 / 768 / `c_hidden` 48 / 16 heads；atom transformer 3+3（128 / 16） | 未披露（2.29 GB ckpt；"~5.7 亿"为 4 B/参推测，**不进结论**） |
| RFAA（参照） | **32 main + 4 extra + 4 ref** 块；SE3Transformer 1 层 / 32 通道 / degree 2 | **无 diffusion module**（非生成式） | 未披露【未核实】 |

> 读法：除 OpenDDE（加宽到 384/384）、Mol-1（异构 + 去三角注意力）、Boltz-2（加深到 64）外，**trunk 都是 AF3 的 48 块 / 128-384**。这一列的方差远小于 §2 的数据列。

### 4.2 原子槽位与表征轨道

| 模型 | token 内原子槽位 | 表征轨道 | frame / 扭转角 |
|---|---|---|---|
| AlphaFold3 | **24**（`max_atoms_per_token`）【核实·代码】 | 双轨：token 级 + atom 级 | 无 frame、无扭转角【核实】 |
| OpenDDE | **24**【代码／配置核实】 | 双轨 + 独立 structural-token refiner 层 | 无【推断】 |
| Mol-1 | **24**【代码／配置核实】 | 双轨 | 未见 frame 描述【未核实】 |
| Boltz-1 | **无固定槽位**（变长 `atom_num`；原子注意力 32 query × 128 key 局部块稀疏）【核实·代码】 | 双轨 | 无 |
| Boltz-2 | 无固定槽位（atom window 32 / 128）；非标准残基压 1 token【核实·代码】 | 双轨 | 无 |
| Chai-1 | **atom37 = 37 槽**（核苷酸 36）；配体/修饰残基每原子 1 token、槽位索引恒 0；全模型 2,048 token 上限【代码／配置核实】 | 双轨，但修饰残基被**打散成逐原子 token** | 无扭转角；frame 字段存在、用途未核实【核实+未核实】 |
| Protenix | **24**（代码注释：DNA 鸟嘌呤最多 23 原子，24 为紧上界）【代码／配置核实】 | 双轨 | frame 工具仅用于对齐，非表征层【核实·代码】 |
| OpenFold3 | **23**【代码／配置核实，白皮书未提】 | 双轨；共价修饰残基原子化 | 无 frame / 扭转角（配置全文无 torsion/frame 键）【代码／配置核实】 |
| RFAA（参照） | 非槽位：非聚合物逐原子 **46 种元素 token、4 种键级、3D 轨手性符号**；重原子仅平移更新【核实·论文】 | 残基 token（Cα frame）+ 逐原子 token，统一进图，无独立 atom 表征层【推断】 | 有 Cα frame（RFAA 用 frame 做等变更新）【核实】 |

> 这一列最容易被误读成"更小更好"或"更大更好"。实际含义只有两个：**(a) 权重能否互换**（24 ≠ 23 ≠ 37 的 token 内 KV 长度不同）；**(b) 显存与长配体的处理方式**。Chai-1 的 37 与 AF3 的 24 不是"同一个槽位制换了个数"，而是两套索引体系。

### 4.3 条件接口矩阵

| 模型 | MSA | 模板 | pocket | 共价键 | 其他约束 |
|---|---|---|---|---|---|
| AlphaFold3 | **必需**，16,384 行【核实】 | ≤20 命中取前 4，不取同一 PDB【核实】 | 需单独微调【未核实】 | CCD 支持【核实】 | **无显式手性 / 二硫键约束**【核实】 |
| OpenDDE | 可关；RNA MSA 默认关【核实】 | 默认关【核实】 | 支持【核实】 | 支持【核实】 | contact / epitope / movable_chains【核实】 |
| Mol-1 | 必需【核实】 | 【未核实】 | 无【核实】 | 【未核实】 | 无（constraint 变体仅网页端）【核实】 |
| Boltz-1 | 可整模块跳过【核实】 | **删除模板**【核实】 | 支持（30% 迭代、6 Å 部分揭示）【核实】 | 支持 | 表位条件【核实】 |
| Boltz-2 | a3m / 多链 CSV / empty / server【核实】 | 支持（多链，force / threshold）【核实】 | 支持（4–20 Å）【核实】 | 仅 CCD 配体与标准残基【核实】 | method 标签 + affinity【核实】 |
| Chai-1 | **可选**；关则跑 ESM-2 3B（代价仍在）【核实】 | ≤4（PDB70）【核实】 | 支持（6–20 Å）【核实】 | 支持（跨链）【核实】 | contact 6–30 Å；**不支持链内二硫键**【核实】 |
| Protenix | 支持（ColabFold 管线）【核实】 | ≤4【核实】 | 支持（token/atom contact + pocket JSON）【核实】 | 支持（bond token 对）【核实】 | 无 epitope 专用【核实】 |
| OpenFold3 | 支持（推理子采样至 1,024）【核实】 | ≤4，**仅蛋白**【核实】 | **无**【核实】 | schema 有、标 coming soon【核实】 | 无 restraint / epitope / 抗体模式【核实】 |
| RFAA（参照） | **不用 MSA**（单序列 + 结构模板）【核实】 | 结构模板（`pdb100_2021Mar03`）【代码／配置核实】 | 无 | 共价修饰数据参与训练【论文自称】 | 无 |

---

## 5. 表格之外：哪些差异是数据、哪些是架构、哪些只是口径

### 5.1 数据决定的差异

1. **Boltz-2 是唯一能碰亲和力的，靠的是数据而不是新骨干。** 切点 2023-06-01（比同代晚 21 个月）+ AFDB 约 500 万单体 + 亲和力 1.2M 值点 / 3.4M 二分类点。它的 affinity 头本身只有 8 层 / 4 层，并且**梯度从 trunk 截断**（`z.detach()`），本质是"读共享表示的轻量下游"。把它当成"架构创新带来亲和力"会误判可迁移性。
2. **Protenix 的单体微弱优势更像同配方下的训练稳定性，不像容量红利。** FoldBench 单体 0.8857 vs AF3 0.8803（差 9e-4）。Protenix 公开的四阶段配方与稳定性补丁，被 OpenFold3 明确反向吸收（"整体吸收 Chen et al. Table 2 的稳定性修改"）——这条链条说明差距在训练工程，不在模型结构。
3. **OpenFold3 的 RNA 只能"打平"是缺数据造成的，不是构架弱。** 白皮书原话："As we did not have a pretrained OF3 model, self-distillation sets for nucleic acids were not used."
4. **Chai-1 切点最保守，是它所有跨代比较的前提。** 2021-01-12 早于全部同代评测集，单模型通吃。换句话说，它 PoseBusters 的数字里含有"没见过测试集"的红利，而 AF3 的 PoseBusters 数字来自一个专门重训到 2019-09-30 的模型——两者都不是主模型，但**不是同一个模型**。
5. **Mol-1 无蒸馏集**（全文 distill / AFDB 零命中），却把 FoldBench-AB 做到 75.0——它的抗体分数主要来自 constraint 变体的 88.3，裸预测 75.0 仍需与其自建 MolBench-AB-26 口径一起看。

### 5.2 架构决定的差异

1. **真正的架构分叉只有两条**：
   - **置信度用什么算**：Boltz-1 用整个 48 层 trunk 复刻 + 扩散轨迹聚合（最贵，换来 Boltz-1x 约 97% 物理有效性）；其余走 4–8 层轻头。这不是精度必需，而是"排序质量 vs 算力"的取舍。
   - **是否引入新任务头**：只有 Boltz-2 的 affinity 是真新头；pocket / contact / epitope 在多数模型里是输入特征，不是新模块。
2. **容量怎么加，两派做法**：OpenDDE 加宽（c_z 128→384，参数 ×3、计算 ×9）；Boltz-2 加深（48→64，+33%）。前者把增益放在复合物界面表征，后者把增益放在亲和力读取器。**同一代里"加参数"没有唯一答案，取决于你要解的任务。**
3. **槽位 24 / 23 / 37 / 无上限几乎不影响精度。** 它影响的是：权重能否互换（OF3 的 23 与 AF3 的 24 不可直接换）、长配体的显存、以及修饰残基的处理（Chai-1 与 OF3 都把修饰残基"原子化"，AF3 保留为聚合物残基 token）。把这一列当能力指标是常见误读。
4. **采样器只有 Chai-1 换了。** 它用 Karras churn（S_churn 80、`S_noise` 1.003、二阶），且 σ 被 `σ_data=16` 再乘一次，有效 σ∈[0.0064, 1280]。其余全是 AF3 的确定性 EDM。**这意味着"Chai-1 vs AF3 在同步数同 seeds 下的差"里，有一部分是采样器而不是网络。**
5. **RFAA 是唯一没有"采样预算"概念的。** 单次前向 + 4 recycle，无随机性；它的指标方差来源与扩散模型完全不同，5 seeds × 5 samples 那套预算对它无意义。

### 5.3 只是评测协议的差异（本篇最该记住的一节）

**这一节里的每一条，都足以推翻一张"谁超过谁"的表。**

1. **Protenix「超 AF3」在 PXMeter 上是拼接。** PXMeter 的对照只重跑了有开源权重的 Chai-1 / Boltz-1 / Protenix，**AF3 那行是「–」**（无权重、无法重跑）。Protenix 在 FoldBench 的 0.8857 vs 0.8803 里，AF3 一侧是引用原文数值。结论应写成「Protenix 追平 AF3」，不能写成「超过」。
2. **Mol-1 的 PXMeter-AB 58.3 与 MolBench-AB-26 57.3，图上都没有 AF3 柱。** 同一篇里 FoldBench-AB 75.0 vs AF3 62.7 的 12.3 点差距，AF3 是引用还是重跑未说明【未核实】。
3. **OpenDDE 的数字与其他篇相差一个量级——这是"同一基准"最危险的证据。** OpenDDE 报 Chai-1 **16.7**、Boltz-1 **25.0**、OpenFold3 **21.9**；而 Mol-1 表里 Protenix-v1 **52.3**、AF3 **48.8**。同一批模型、都叫"复合物 DockQ 成功率"，差 2–3 倍。差异来源只能是打分脚本、对齐方式、top-rank 选择规则。**跨论文的 DockQ SR 不能横读，只能在同一篇内比较。**
4. **AF3 自己在 PoseBusters 上有两个数：约 58% 与 76.34%。** 前者来自 Boltz-1 的比较（标准 AF3 权重 + 更晚的复合物子集），后者来自 Chai-1 引用的 AF3 原文（PoseBusters-428 + 2019-09-30 重训模型）。**这两个数绝不能放进同一列**——它们甚至连"哪个模型"都不同。
5. **Boltz-2 的「比 FEP 快 1000 倍」对 FEP+ 没有分母。** Table 12 里 FEP+ 的耗时是「—」。可复算的是：对 OpenFE 1080×–2160×、对 ABFE >3600×。这句话只能保留为「相对 OpenFE 约 10³ 量级」，不能作为对比表结论行。
6. **默认配置 ≠ 论文协议，几乎每一篇都踩了**：
   - Boltz-1：CLI 1 sample / 3 recycle vs 论文 5 输出 / 10 recycle
   - Chai-1：代码 1 trunk sample vs 论文 5；代码 3 recycle vs 论文 4
   - OpenFold3：CLI 1 seed / 3 recycle vs 论文 5 seeds / 10 recycle
   - Protenix：base 默认 4 recycle vs 发布权重与论文 10
   - OpenDDE：代码 200 步 / 10 recycle vs 论文 20 步 / 4 cycle

   复现失败时，先查这一层，再怀疑模型。

---

## 6. 我认为最实质的三条差异

1. **决定能力上限的是切点与蒸馏集，不是 trunk 图纸。** 九个模型的 trunk 几乎同构（48 层 / 128-384），而 Boltz-2 靠 2023-06 切点 + 500 万 AFDB 才够到亲和力；Chai-1 靠最保守的 2021-01 换来"单模型通吃"；OpenFold3 缺核酸自蒸馏因此 RNA 只能打平。**先比数据，再比架构。**
2. **"加容量"有两条互斥路线，且都没有被证明是唯一解。** OpenDDE 加宽（表征更厚）与 Boltz-2 加深（+ 新读取器）在各自目标上有效，但 Protenix v2 的抗体 +13 点来自"加宽 + 抗体数据重训"，报告选择不披露配方——意味着这条增益目前**不可从论文复算**。真正的架构共识只有一条：置信度用轻头（4–8 层）已经够，Boltz-1 的 48 层复刻是可选的重投入。
3. **跨论文的排名不可横读，误差大于模型差。** 同一 AF3 在 PoseBusters 上 58% vs 76.34%；同一 Chai-1 在 PXMeter-AB 上 16.7 vs 别处 50+ 量级。Protenix 的「–」与 Boltz-2 的「1000×」是这条规律的两个极端样本。任何"X 是 SOTA"的结论，先在**同一篇、同一脚本、同一 rank 规则**内验证。

---

## 7. 明确不可比 / 留空的格子

### 7.1 不可比结论（不要写进任何对比表）

| 命题 | 为什么不可比 |
|---|---|
| Protenix 在 PXMeter 上超过 AF3 | AF3 行为「–」，AF3 从未在该基准重跑 |
| Mol-1 在 PXMeter-AB / MolBench-AB-26 上超过 AF3 | 同一张图里无 AF3 柱 |
| Boltz-2 比 FEP 快 1000 倍 | Table 12 中 FEP+ 耗时为「—」，无可复算分母 |
| AF3 的 PoseBusters 通过率是 X% | 存在 58% 与 76.34% 两个来源不同的数，且对应不同权重（标准权重 vs 2019-09-30 重训） |
| Chai-1 的 DockQ SR 是 16.7 | 该数只出现在 OpenDDE 的 PXMeter-AB 重跑口径下，与其他篇的同名指标差 2–3 倍 |
| OpenDDE 的 FoldBench-AB 70.0 高于 Mol-1 的 75.0 或反之 | 两篇的 top-rank / oracle 定义与样本预算不同，需在同一脚本内重跑才能比 |

### 7.2 因证据不足留空的格子

| 模型 | 留空项 | 原因 | 补齐需要什么 |
|---|---|---|---|
| AlphaFold3 | 训练分辨率阈值、训练集条目数、聚类阈值 | SI 截断/未披露 | AF3 SI 完整版或官方数据清单 |
| OpenDDE | 各子集条数、训练去重阈值、损失权重 | 论文未给 | 训练配置或数据统计表 |
| Mol-1 | 训练去重阈值、损失逐项权重、模板支持状态 | 未披露 | 技术报告附录或训练代码 |
| Boltz-1 | ρ 与 γ0（本篇篇章记 7 / 0.8，Boltz-2 篇章转述论文表为 8 / 0.605） | **两处笔记冲突，未裁决** | 回 Boltz-1 论文 Table 或 `main.py` 常量比对 |
| Boltz-2 | diffusion module 层数与宽度 | 由 checkpoint 注入，论文未给 | 权重结构打印 |
| Chai-1 | trunk block 数、pair/single 宽度、MSA 层数、diffusion 层数、参数量、结构数 / 链数 / 分辨率阈值、loss | 只发 TorchScript，无论文数字 | 官方 PyTorch 源码或模型卡 |
| Protenix | v2 训练配方（数据配比、阶段超参） | v2 报告只写"architectural refinements" | v2 训练配置细节 |
| OpenFold3 | 参数量、推理显存、单结构耗时、全部 benchmark 数值 | 官方未披露；评测只给柱状图 | 自测或用 preview2/OpenBind 报告 |
| RFAA | PDB 快照日期、参数量、是否有置信头 | 篇章未覆盖 | RFAA 论文 SI 或代码 |
| 全部 | 训练集"去重后有效条目数" | 九篇中无一篇给全 | 只能逐篇自建统计 |

---

## 8. 参考

- 本仓库模型篇章（一手材料入口）：[AlphaFold3](../模型篇章/AlphaFold3/README.md)、[OpenDDE](../模型篇章/OpenDDE/README.md)、[Mol-1](../模型篇章/Mol-1/README.md)、[Boltz-1](../模型篇章/Boltz-1/README.md)、[Boltz-2](../模型篇章/Boltz-2/README.md)、[Chai-1](../模型篇章/Chai-1/README.md)、[Protenix](../模型篇章/Protenix/README.md)、[OpenFold3](../模型篇章/OpenFold3/README.md)、[RFAA](../模型篇章/RFAA/README.md)
- 相关联合分析：[全原子模型解题思路归类分析](全原子模型解题思路归类分析.md)
- 评级口径参考：PoseBusters（物理有效性）、FoldBench（界面 DockQ）、PXMeter（配套评测套件）、DockQ>0.23（可接受界面）
