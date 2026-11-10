---
model: Chroma
aliases: [Chroma, Generate:Chroma, generatebio/chroma]
family: 结构生成（无条件 / 可编程条件设计）
granularity: 主链逐原子（N、Cα、C、O 四个显式点做扩散）+ 侧链扭转（χ 角）+ 序列，三者由 design network 联合自回归解码；不是"残基级"也不是"全原子扩散"
backbone: 等变图神经网络（random graph，子二次复杂度）；扩散为 DDPM 式 x0 预测，"correlated diffusion" 指前向加噪用聚合物协方差（BackboneMVNGlobular，Brownian 协方差 + Rg ∝ N^(2/5)），不是条件化机制
generative: 生成式；主链 SE(3) 扩散（BLNL/布朗运动式前向过程）+ 条件采样（classifier / energy guidance + 低温退火）
conditioning: 无 MSA、无模板；条件接口为可组合 Conditioner：substructure（motif 坐标）/ symmetry / shape / ProClass 折叠类分类器 / ProCap 自然语言 caption 分类器；设计模式可给定主链出序列 + χ 角
data_scope: PDB（2022-03-20 查询）+ UniProt 2022_01 + Pfam 35；train/test/validation 的 PDB ID 清单公开于 Zenodo 10.5281/zenodo.8285077（519.2 kB CSV）；具体条目数本次未取到【未核实】
status: 学习中
verified_at: 2026-11-10
---

# Chroma

> **状态**：学习中（机制、发布形态、许可证已核实；参数量与训练集条目数未核实）
> **记录人**：团队共同学习
> **最后更新**：2026-11-10

---

## 1. 学习背景与目标

- **为什么学**：Chroma 是"**不做 MSA、直接对主链坐标做扩散、把条件写成能量项**"这条路线最完整的公开实现，也是本仓库中唯一自带**自然语言条件接口**的生成模型。对我们做 VNAR/纳米抗体设计，它的两个点最值得借：① 用 classifier/energy guidance 在不重训的情况下加约束；② 论文里明确演示过 **VHH CDR loop 重设计**。
- **本次目标**：
  1. 把"correlated diffusion"讲清楚——它是**前向噪声的协方差**，不是条件化，也不是"相关性建模"这种模糊说法；
  2. 明确表征粒度：主链是逐原子，侧链是 χ 角，不是"全原子扩散"；
  3. 核实发布形态（有没有 Chroma-large、权重怎么拿、许可证是什么）；
  4. 逐条核对三种条件接口的实现方式与代价。
- **一句话判断**：Chroma 的核心创新在**前向过程**（把独立高斯噪声换成聚合物统计的相关的噪声，使中间态始终像"塌缩的蛋白链"），条件化则统一成"在扩散采样里加能量项"的贝叶斯推断框架；后者的正确性取决于能量项本身是否可微、是否与真实约束一致——几何约束"按构造正确"，语义约束（折叠类、自然语言）只是分类器近似。【核实：论文正文 + 代码】

---

## 2. 核心概念

| 术语 | 含义 |
| ---- | ---- |
| correlated diffusion | 前向加噪不是逐原子 i.i.d. 高斯，而是从一个**聚合物统计的多元高斯**采样噪声，协方差由链连接与回旋半径约束给出；论文表述为"correlated noise process to match the distance statistics of natural proteins"，使噪声态呈现"随机塌缩的聚合物"而非散点云。【核实：论文正文】 |
| BackboneMVNGlobular | 上述协方差在代码里的实现类，`covariance_model="brownian"`，并显式写出标度律 `Rg = Rg0 · N^ν`，`ν = 2/5`（自回避链的 Flory 指数）。【代码／配置核实：`chroma/layers/structure/mvn.py`】 |
| BLNL backbone | 公开发布的主链权重所对应的训练任务名——"BLNL backbone model training with EMA, trained July 2023"（BLNL 在代码中作为任务标识出现，论文正文未展开该缩写）。【代码／配置核实：`chroma/constants/named_models.py`】 |
| design network | 给定主链，**自回归**联合预测序列与 χ 角（two-stage）；不是扩散，也不是一次性 softmax。【代码／配置核实：同名文件 `task` 字段】 |
| Conditioner | 可组合的条件对象，向采样过程提供能量/梯度；几何类条件"按构造正确"，语义类条件依赖训练好的分类器。【核实：论文正文】 |
| ProClass / ProCap | 折叠类分类器 / 结构 caption（语言）模型；ProCap 用 M5 ProClass GNN 嵌入做交叉熵训练。【代码／配置核实：`named_models.py`】 |
| 低温采样 | 论文提出的一般化低温采样算法，用退火把条件分布往高约束满足度推；温度是显式的精度-多样性旋钮。【核实：论文正文】 |
| random graph | 每层重采样的稀疏图（而非全连接），使复杂度对残基数**子二次**。【核实：论文正文 + README】 |

---

## 3. 原理与推导

### 3.1 "correlated diffusion" 到底是什么

- **它改的是前向过程**：标准 DDPM 里 $x_t = \alpha_t x_0 + \sigma_t \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$；Chroma 改成 $\epsilon \sim \mathcal{N}(0, \Sigma)$，$\Sigma$ 由蛋白链的几何统计给出（链相邻点强相关，随序列距离衰减，整体受回旋半径约束）。代码里就是 `BackboneMVNGlobular` 的 Brownian 协方差 + `Rg ∝ N^{2/5}` 标度。【核实：论文 + 代码】
- **它带来什么**：反向去噪的每一步中间态都"像一条塌缩的聚合物"，网络不必从无结构的散点里凭空长出链拓扑；同时因为噪声本身满足链约束，采样可以用更少的步数/更低的温度而不崩。**这属于"更好的先验"，不改变条件接口。**
- **它不是什么**：
  - 不是"对残基间相关性建模"的注意力机制（那是 backbone network 的事）；
  - 不是条件化（条件是另一套 Conditioner + 能量项）；
  - 不等于"等变"（等变由 equivariant layers 保证，是另一件事）。
- **一个可检验的推论**：如果把协方差换成各向同性（identity），模型应当仍能训练，但在长链/大复合物上采样稳定性下降、需要更多步数。论文没有给这个消融的完整曲线，只有定性论述。**该消融是否做过：【未核实】**（正文与 Methods 抓取范围内未见）。

### 3.2 表征粒度：到哪一层

| 组分 | 表示 | 产生方式 |
| ---- | ---- | ---- |
| 主链 N、Cα、C、O | 显式三维点 | 扩散生成（correlated diffusion + 等变图网络） |
| 侧链 | χ 角（扭转） | design network 自回归解码，按残基类型查内坐标重建原子 |
| 序列 | 20 字母表 | 与 χ 角联合自回归预测 |
| 氢原子 | 不显式 | 需后处理加氢 |
| 配体/核酸/金属 | **不支持** | 模型是纯蛋白模型 |

- 因此按专题 [01 §2.2](../../专题/01_全原子表示与生成范式.md) 的五档判据，Chroma 应落在"**主链逐原子 + 侧链扭转**"这一混合档，**不能**标成"逐原子（全原子）"——它的全原子性来自"主链原子 + χ 角重建"，不是"每个重原子都被扩散"。【核实：论文 + 代码 `sidechain.py` / `protein_graph_allatom.py`】
- 与 RFAA 的区别：RFAA 是"残基 frame + 重原子平移"，Chroma 是"主链点直接扩散 + χ 角解码"；两者都输出全原子，但**被生成/被预测的自由度完全不同**。

### 3.3 主干与条件接口

- **没有 MSA、没有模板**：输入只有残基数（和可选的条件），序列由模型自己出。这与 AF3/RFAA 的"MSA + 模板"接口是**互斥的两类信息源**——Chroma 把"进化信息"换成了"学到的结构先验"。【核实：论文；README 无条件生成示例】
- **条件化的统一框架**：把设计写成贝叶斯推断 $p(x \mid c) \propto p(x)\,p(c \mid x)$，$p(c\mid x)$ 用可微能量 $E_c(x)$ 表示，在反向扩散的每一步加 $\nabla_x E_c$（classifier / energy guidance），配合低温退火采样。【核实：论文正文 "Bayesian inference under external constraints"】

| 条件类型 | 实现方式 | 代价 / 局限 |
| ---- | ---- | ---- |
| substructure（motif 固定） | 直接把给定坐标写进采样（按构造满足），并对未固定部分做条件去噪 | 无近似误差；但 motif 与生成部分的接口几何需自洽，冲突时只能靠采样"绕开" |
| symmetry | 对坐标做对称化投影 / 在能量项里加对称约束 | 严格对称可实现；非严格（C2 二聚体界面）需调温度 |
| shape | 用可微的包络/占据体积能量约束整体形状 | 形状是粗粒度约束，无法指定二级结构组成 |
| ProClass 折叠类 | 训练一个**结构分类器**，用其对数概率作能量（classifier guidance） | 分类器 ≠ 真值：论文自述 Ig fold、β-barrel 只学到部分特征（β 桶不闭合）【论文自称】 |
| 自然语言（ProCap） | 训练 caption 模型 $\to$ 用其似然作能量 | 论文自述"偶尔能对齐，尤其在训练中高频的蛋白类别"；本质是"把功能条件化降级为分类器准确率问题"【论文自称】 |

- **算力代价**：每加一个 Conditioner 就要在**每个采样步**多算一次前向 + 反传（或缓存的分类器梯度）；几何条件便宜（解析），语义条件贵（一个 GNN 分类器）。README 称"在消费级 GPU 上几分钟生成大复合物"，**未给具体卡型/步数/复合物规模**【作者自称；README】。

### 3.4 与同期路线的位置

- 与 RFdiffusion：RFdiffusion 微调 RF2 的结构模块做扩散、需要自条件化与对称化技巧、依赖 AF2/RF2 的 frame 表示；Chroma 从零训练、无预训练预测器、直接对主链点扩散。【推断，依据两篇方法】
- 与 Proteina / La-Proteina：那两篇用**流匹配 + 潜变量**替代扩散，并强调"全原子"（含侧链/配体）；Chroma 仍是 DDPM + 显式主链点，且**不支持非蛋白组分**。
- 与本仓库主线的关系：Chroma 提供的是"**无 MSA 的纯结构先验 + 可插拔能量条件**"范式；做纳米抗体 CDR 重设计时，它的 substructure 条件 + CDR 区间掩码是现成可用的最小方案（论文 Fig. 3 有 VHH 例子）。【核实：论文图 3】

---

## 4. 训练数据构成与切分

- **数据来源**（【核实：论文 Data availability 原文】）：
  - PDB，**查询日期 2022-03-20**；
  - UniProt **2022_01**（用于序列/注释侧信息）；
  - Pfam **35**（用于家族/语义标签）。
- **切分**：作者公开了 train / test / validation 的**逐条 PDB ID 清单**（Zenodo `10.5281/zenodo.8285077`，CSV 519.2 kB，CC-BY 4.0）。因此切分方式**可复核**，但：
  - 具体条目数、按序列同一性聚类的阈值：**本次未取到**。我查过：论文正文 Data availability（只给来源与 Zenodo 链接）、Zenodo 记录页描述（只说明 `pdb` 与 `split` 两列）、`named_models.py`（只给 "Generate Structure ETL: July 25 2022"）、README（无数据节）。shell 侧 DNS 受限无法下载 CSV 统计，故标【未核实】而非"作者没写"。
- **时间切分**：有快照日期（2022-03-20）但**未见按时间前后切分**的表述；test/validation 的具体构造规则论文正文未展开【未核实：是否按时间/聚类切分】。
- **偏置**：
  - PDB 天然偏置（可溶、可结晶、药物相关家族过表征）——这也解释了为什么自然语言条件"在高频类别上更好"；
  - 训练后 ETL 又更新到 2022-07-25（`named_models.py` 的 "Generate Structure ETL" 字段与 PDB 查询日不一致），**说明公开权重用的数据快照晚于论文声明的 2022-03-20**，两者不是同一份数据。【代码／配置核实 + 推断】

---

## 5. 宣称与可验证性

| 宣称 | 证据类型 | 核查状态 | 备注 |
| ---- | ---- | ---- | ---- |
| 参数量（Chroma / Chroma-large） | — | **未核实** | 公开权重只有单一规模 `chroma_backbone_v1.0.pt`；仓库、权重页、Zenodo 描述、README 均无 "Chroma-large" 这一名称，也均未标参数量。**"Chroma-large" 在公开材料中查无此模型**，引用时应避免 |
| 公开发布形态 | 文件 + 代码常量 | 已核实 | 四个权重：`chroma_backbone_v1.0.pt`（主链扩散）、`chroma_design_v1.0.pt`（序列+χ 角）、`chroma_proclass_v1.0.pt`（折叠类）、`chroma_procap_v1.0.pt`（caption） |
| 代码许可 Apache-2.0 | LICENSE | 已核实 | 论文 Code availability 原文与仓库 LICENSE 一致 |
| 权重许可 | 许可文本 | 已核实 | 学术/非营利免费，**需 API key 并接受"不得转授、不得商用、输出亦不得商用"条款**；商用需另谈（Delaware 法） |
| 子二次复杂度、可生成大复合物 | 曲线 + 文字 | 论文自称 | 复杂度论断有图示；"大复合物"未给统一规模定义 |
| 无条件生成的可设计性（实验验证） | 湿实验（split-GFP、CD、DSC） | 论文自称 | 42 个条件设计里 40 个（95%）可溶表达；Tm 64–78 °C；无第三方复现 |
| 几何条件"按构造正确" | 论证 + 示例 | 可信但需注意 | 只在能量项与约束定义一致时成立；motif RMSD 阈值是人为设定 |
| 语义/语言条件有效 | 示例图 + 分类器概率 | 论文自称，且**作者自认不完美** | Ig fold 拓扑不典型、β-barrel 不闭合；"高频类别更准" |
| VHH CDR loop 重设计可行 | 示例（Fig. 3） | 论文自称 | 无实验验证；对我们是最相关的一条，但只能算概念验证 |
| "几分钟/消费级 GPU" | README 文字 | 作者自称 | 无卡型、步数、链长基准 |

- **可复现所需材料**：代码（Apache-2.0）+ 四个权重（需 key）+ 一个 Colab 即可跑通；训练复现需 PDB/UniProt/Pfam 快照，成本远高于推理。
- **一致性检查**：论文说"joint model of structure and sequence"，代码里是**两个独立网络**（backbone 扩散 + design 自回归），"joint"指联合使用而非单塔联合建模——写方法对比时不要照抄"joint"。【代码／配置核实】

---

## 6. 实践与应用

- **适用**：无条件主链生成 / motif scaffolding（substructure）/ 对称组装 / 给定主链的序列-侧链打包（design 模式）/ 折叠类或语言提示的探索性生成 / **抗体 CDR 区重设计的概念验证**。
- **不适用**：含配体、核酸、金属的复合物（纯蛋白）；需要 MSA 级精度的高同源单体预测（那是预测器的工作）；商用（权重许可禁止）。
- **关键旋钮**：采样步数、温度（低温更"听话"但多样性塌缩）、Conditioner 权重、`protein_length` / `lengths`。
- **与我们 VNAR 工作的接口**：把框架区作为 substructure 固定、CDR3 段掩码，用 Chroma 出主链 → design 网络出序列与 χ 角 → 外接 AF3/RFAA 或 Rosetta 打分。风险点：Chroma 对**长 CDR3（>15 aa）** 与**非典型二硫键模式**没有专门训练，生成分布可能偏向常规抗体。【推断】

---

## 7. 疑问与讨论

- **Q**：为什么论文强调 correlated diffusion 让中间态"像塌缩聚合物"，而不是直接用 Rg 约束？
  - **A**：约束在采样里是"软"的，噪声协方差是"硬"的——它把链拓扑写进了先验，任何一步去噪都不会离开"聚合物流形"太远。代价是前向过程不再是各向同性，闭式解更复杂（代码用 MVN + Brownian 核处理）。【核实：代码注释 + 论文】
- **Q**：条件化是"零样本"吗？
  - **A**：几何条件（motif、对称、形状）基本可以视为零样本；语义条件不是——它依赖额外训练的 ProClass / ProCap 网络，**换一个新的语义约束就要新训一个分类器**。这是 Chroma 条件接口最容易被误读的地方。【核实：论文】
- **Q**：为什么没有 Chroma-large？
  - **A**：公开材料里只发布了一个规模的主链权重。矩阵里若出现 "Chroma-large"，应视为待核实项而非事实。

---

## 8. 学习心得

- 把"扩散模型改先验"这件事做成一篇 Nature，说明**前向过程的设计空间被严重低估**：大多数工作只在网络上加归纳偏置，Chroma 是在噪声分布上加物理。
- 条件接口的"统一为能量项"很优雅，但优雅的代价被低估了：能量项不可微、分类器不准、约束互斥时，整个框架退化成"调温度"。论文对语义条件的不完美是诚实的，这一点值得学。
- 写对比表时，Chroma 的"全原子"必须加限定：主链逐原子 + χ 角重建，**不支持非蛋白组分**，否则会和 RFAA/AF3 混为一谈。

---

## 9. 参考资料

- Ingraham J.B., Baranov M., Costello Z., et al. *Illuminating protein space with a programmable generative model.* Nature 623, 1070–1078 (2023). DOI: 10.1038/s41586-023-06728-8
- 代码：https://github.com/generatebio/chroma （Apache-2.0）
- 权重与许可：https://chroma-weights.generatebiomedicines.com/ （学术非商用，需 API key）
- 发布模型清单：`chroma/constants/named_models.py`（四个 `*_v1.0.pt`）
- 协方差实现：`chroma/layers/structure/mvn.py`（`BackboneMVNGlobular`，`covariance_model="brownian"`，`ν=2/5`）
- 数据切分：Zenodo 10.5281/zenodo.8285077（CC-BY 4.0，逐条 PDB ID）
- 仓库内相关笔记：[AlphaFold3](../AlphaFold3/README.md)、[RFAA](../RFAA/README.md)、专题 [01 §2.2 / §6.1 / §6.2](../../专题/01_全原子表示与生成范式.md)
