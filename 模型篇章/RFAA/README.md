---
model: RFAA（RoseTTAFold All-Atom）
aliases: [RFAA, RF-AA, RF2AA]
family: 结构预测
granularity: 残基 token（蛋白/核酸）+ 非聚合物逐原子（46 种元素 token、4 种键级、3D 轨手性符号）
backbone: 沿用 RF2 三轨（1D/2D/3D）；32 main + 4 extra + 4 ref block（configuration 值），结构模块为 SE3Transformer（1 层 / 32 通道 / degree 2）；聚合物走 Cα／P 刚性 frame，重原子只做平移更新
generative: 非生成式（单次前向 + 4 次 recycle）；同仓库另有 RFdiffusion-AA（由 RFAA 微调得到的扩散生成器），二者不可混为一谈
conditioning: MSA｜模板（n_templ=4，pdb100_2021Mar03）｜序列（蛋白 20 种 + 核酸 8 种）｜配体/金属/修饰残基的原子图（元素 + 键级 + 手性）｜蛋白-配体共价键（邻接矩阵中的独立 bond token）
data_scope: PDB 蛋白-小分子 121,800 结构 / 5,662 簇；蛋白-金属 112,546 / 5,324 簇；共价修饰 12,689 / 1,099 簇；30% 序列同一性聚类；补充 CSD 小分子晶体；"原子化"增广；训练裁剪 256 → 375 token；验证集取 10% 蛋白序列簇；PDB 快照日期论文未写【未核实】
status: 学习中
verified_at: 2026-11-10
---

# RoseTTAFold All-Atom（RFAA）

> **状态**：学习中（架构、数据、许可证已核实；推理墙钟时间与 PDB 快照日期未核实）
> **记录人**：团队共同学习
> **最后更新**：2026-11-10

---

## 1. 学习背景与目标

- **为什么学**：RFAA 是"三轨 + frame"路线在全原子问题上的最后一代表现最好的公开模型，也是本仓库里唯一一个**同时**给出预测器（RFAA）和生成器（RFdiffusion-AA）权重、且许可证为 BSD 的 Baker 系全原子模型。它承接 [RFD4-Proteina](../RFD4-Proteina/README.md) 的谱系上游，不搞清楚 RFAA 的 frame 表示与 FAPE 损失，就无法说清 RFD4-Proteina 为什么要换成"锚点 + 局部隐变量 + 自编码器"。
- **本次目标**：
  1. 把"全原子"拆到化学组分粒度，说明它到底覆盖了什么、没覆盖什么；
  2. 把与 AF3 的差异落在**架构**和**训练数据**两点上，而不是停在"都是全原子"；
  3. 分清 RFAA（预测器）与 RFdiffusion-AA（生成器）——仓库、权重、许可证都不同；
  4. 记录仓库现状（许可证、权重可得性、依赖库体积），为复现估算成本。
- **一句话判断**：RFAA 的贡献是**在不放弃残基 frame 的前提下**，把非聚合物成分降级为"可自由平移的重原子 + 原子图 token"，从而用一个 RF2 主干同时吃蛋白、核酸、配体、金属、共价修饰；它相对 AF3 的差别不在"是否全原子"，而在**结构模块仍是 frame+FAPE 的迭代 refinement，而不是 atom-level 扩散采样**。【推断，依据 §3 与论文方法】

---

## 2. 核心概念

| 术语 | 含义 |
| ---- | ---- |
| 三轨表示 | RF2 的 1D（序列/MSA）、2D（pair）、3D（坐标）三条信息轨，每个 block 之间互相混合。【核实：论文正文】 |
| 元素 token | 非聚合物原子的 1D 输入。RFAA 在 RFNA 的 20 氨基酸 + 8 碱基字母表之外，**新增 46 种元素类型 token**，取自 PDB 中出现频率最高的元素，并明确"并非完整周期表"。【核实：论文正文 + Fig. 1B；"not the full periodic table" 在文中被明确写出】 |
| 键级 embedding | 2D 轨的原子对特征，编码单键/双键/三键/芳香四种键级，线性嵌入后在**每个 recycle 开始时**加到初始 pair 特征上。【核实：论文正文】 |
| 手性编码 | 1D/2D 表示对镜像不变，因此手性只能放在 3D 轨：以每个手性中心周围原子的**角度符号**（R/S）表示，并在每个 block 把"实际角度偏离理想值的梯度"作为下一 block 的输入特征。【核实：论文正文 + Fig. S1】 |
| 原子化（atomization） | 训练增广：随机把 3–5 个连续残基删掉序列/模板特征，改为提供原子、键、手性信息（一个 Ala 变成 5 个重原子 token），并新增 "atom-to-residue" 键 token 与相应位置编码。【核实：论文正文】 |
| 全原子 FAPE | 把 AF2 的 frame-aligned point error 推广到任意分子：以每个原子的键邻居定义坐标 frame，逐个对齐后计算周围原子误差；对小分子几何相关项加权。【核实：论文正文 + 方法】 |
| 重原子平移更新 | 蛋白残基仍由 Cα 坐标 + N-C-C frame 表示（核酸为 P + OP1-P-OP2），网络预测旋转 + 平移；**重原子只接受平移更新**，不预测其朝向 frame。【核实：论文正文 + 配置 `use_atom_frames: True`】 |
| 共价键 token | 蛋白与配体之间的共价连接不写进元素/键级体系，而是作为 bond 邻接矩阵里的**独立 token** 提供。【核实：论文 Fig. 1B 图注】 |
| RFdiffusion-AA | 在 RFAA 上做去噪任务微调得到的生成器，从配体周围的随机残基分布出发"长"出口袋；与 RFAA 是两个 checkpoint、两个仓库。【核实：论文摘要 + 两仓库 LICENSE 署名不同】 |
| recycle / block 数 | 推理默认 `MAXCYCLE: 4`；主干 32 main + 4 extra + 4 ref block。【代码／配置核实：`rf2aa/config/inference/base.yaml`】 |

---

## 3. 原理与推导

### 3.1 全原子到底覆盖哪些化学组分

| 组分 | 进入网络的形式 | 覆盖程度 | 证据 |
| ---- | ---- | ---- | ---- |
| 蛋白 | 残基 token + Cα/N-C-C frame + 显式重原子坐标 | 完整 20 种 + 非天然残基（按原子图处理） | 【核实：论文】 |
| 核酸 | 残基 token（4 DNA + 4 RNA 碱基）+ P/OP1-P-OP2 frame，可选磷酸 frame | DNA/RNA 双链、蛋白-核酸复合物 | 【核实：论文 + `use_phospate_frames_for_NA: True`】 |
| 小分子配体 | 原子图：46 元素 token + 4 种键级 + 手性 | 覆盖 PDB 常见元素子集，**不是全周期表**；氢不显式建模 | 【核实：论文】 |
| 金属离子 | 单原子（元素 token），无键级 | 覆盖；112,546 条蛋白-金属数据专门训练 | 【核实：论文】 |
| 共价修饰 | 修饰基团按原子图 + 与蛋白的共价键 token | 覆盖训练集中出现的修饰类型；46% 可在 <2.5 Å 内预测准确 | 【论文自称】 |
| 溶剂/结晶添加剂 | **被显式过滤掉** | 不建模 | 【核实：论文】 |
| 显式氢 / 质子化态 / 电荷 | 未见元素级电荷通道；论文未提 | **未覆盖** | 【推断，依据表征节只列元素/键级/手性】 |

**关键限制**：46 种元素 token 是"PDB 高频元素"的经验子集，且 1D 只有元素身份，没有 FormalCharge / 芳香性单独通道（芳香性只以"aromatic"键级体现）。这意味着带电荷的金属配位、质子化依赖的氢键网络只能靠几何与键级间接学。【推断】

### 3.2 与 AF3 的实质差异（架构层）

| 维度 | RFAA | AF3 | 差异性质 |
| ---- | ---- | ---- | ---- |
| 坐标产生方式 | 迭代 refinement：4 次 recycle × (32+4+4) block，每次预测 frame 旋转+平移与重原子位移，单次前向 | atom-level diffusion：trunk 出 pair/single 表征后，扩散头做 200 步去噪，每步重采样坐标 | **范式差异**：RFAA 是"回归 + recycle"，AF3 是"生成式采样" |
| 等变实现 | SE3Transformer（1 层/32 通道/degree 2）+ Cα frame，网络内建旋转等变 | 无显式等变：随机旋转增强 + 坐标回归 | RFAA 保留等变归纳偏置，AF3 放弃 |
| 侧链 | 显式重原子坐标参与 3D 轨，随主干一起更新 | 打包进 token 的 atom slot（上限 24），由扩散头输出 | RFAA 侧链与骨架耦合在同一 refinement 里；AF3 侧链是原子级采样的一部分 |
| 手性 | 3D 轨角度符号 + 每 block 的梯度特征 | 无显式约束，论文自报 4.4% 违规率 | RFAA 显式建模，AF3 靠数据 |
| 配体表示 | 原子图（元素/键级/手性），无参考构象 | CCD 参考构象 `ref_pos` + 原子级 token | RFAA 给"化学图"，AF3 给"参考几何" |
| MSA 处理 | RF2 式 MSA 堆栈，db 42 + 1（`d_msa=256`, `d_pair=192`） | 4-block MSA 模块，pair-weighted averaging 压缩 | RFAA 保留重量级 MSA，AF3 大幅压缩 |
| 复杂度 | 二次注意力在**残基/token**级 + 原子作为附加点 | token 内原子注意力 + token 间注意力 | 两者都避免"全原子二次注意力"，但切分方式不同 |
| 置信度 | pLDDT + PAE（论文明确两项） | pLDDT + PAE + PDE + pTM/ipTM | AF3 输出面更宽 |

AF3 侧的数值与出处见 [AlphaFold3 笔记](../AlphaFold3/README.md) 与专题 [01 §3](../../专题/01_全原子表示与生成范式.md)。【核实：两侧论文 + 配置】

**为什么这个差异重要**：RFAA 的精度上限由"4 次 recycle 的回归"决定，误差不能在采样阶段被多次尝试摊平；AF3 可以用多 seed × 多样本排序换精度，代价是 200 步 × N 样本的算力。这两条路线的**成本结构完全不同**，直接比较单次 RMSD 会掩盖这一点。【推断】

### 3.3 与 AF3 的实质差异（数据层）

| 维度 | RFAA | AF3 |
| ---- | ---- | ---- |
| 聚合物数据 | 蛋白单体 / 蛋白复合物 / 蛋白-核酸，沿用 RF2 与 RFNA 的切分 | PDB ≤2021-09-30 + AF2-Multimer v2.3 蒸馏 |
| 非聚合物数据 | PDB 蛋白-小分子 121,800 / 5,662 簇；蛋白-金属 112,546 / 5,324；共价修饰 12,689 / 1,099 | PDB 全库统一时间切分，无按模态分表 |
| 额外数据源 | **CSD（剑桥结构数据库）小分子晶体**，用于学"小分子一般性质"而非 PDB 特有分子 | 无 CSD；用 AF2 蒸馏补长尾 |
| 去偏方式 | 30% 序列同一性聚类 + 每例从有机非聚合物分子集合中**均匀采样** | 聚类 + 时间切分 + 蒸馏加权 |
| 增广 | 原子化（3–5 连续残基降为原子）+ 原子-残基键 | 随机旋转/平移、MSA 子采样等 |
| 裁剪 | 训练 256 token，微调 375 token | 按 AF3 自身 crop 策略（见 AF3 笔记） |
| 验证 | 10% 蛋白序列簇 | 时间切分后的近期 PDB |
| 时间切分 | **按簇切分，未见时间切分**；模板库停在 `pdb100_2021Mar03` | PDB 时间切分 2021-09-30，另有 2022 年蒸馏集 |

数据层最实质的差别不是规模，而是：**RFAA 用 CSD 补小分子化学先验、用"原子化"把蛋白内部原子相互作用迁移给配体相互作用；AF3 用蒸馏补结构与长尾**。这两件事解决的是不同问题，不能互相替代。【推断，依据两篇论文的方法节】

### 3.4 谱系：RFAA → RFdiffusion-AA → RFD4-Proteina

- **RF2（2023）→ RFAA（2024）**：主干直接沿用 RF2，新增非聚合物原子图与全原子 FAPE。【核实：论文 "modeled the network architecture after RF2"】
- **RFAA → RFdiffusion-AA**：不是新架构，而是把 RFAA **微调**到去噪任务上，输入改为"配体 + 随机残基分布"，输出折叠好的口袋。同属 Baker lab，但代码在 `baker-laboratory/rf_diffusion_all_atom`（LICENSE 署名 Rohith Krishna / Woody Ahern），预测器在 `baker-laboratory/RoseTTAFold-All-Atom`（署名 Rohith Krishna / Jue Wang / Frank DiMaio）。【代码／配置核实：两仓库 LICENSE 原文】
- **RFdiffusion-AA → RFD4-Proteina（2026）**：RFD4-Proteina 放弃了 RFAA 的 Cα frame + 显式重原子 refinement，改为"Cα/C1′ 锚点 + 每残基 32 维局部隐变量 + 130M 自编码器解码"，生成范式也从 SE(3) 扩散换成双模态流匹配。**两者不是同一模型的两个版本，是两条不同表征路线的产物**，比较时应对齐"给定条件 → 输出全原子结构"的接口，而不是对齐参数量。【核实：RFD4 笔记 YAML + 专题 01 §6.2】

---

## 4. 训练数据构成与切分

- **数据源与规模**（【核实：论文 "Training RFAA" 节原文】）：
  - 蛋白-小分子：**121,800** 个结构，**5,662** 个簇；
  - 蛋白-金属：**112,546** 个结构，**5,324** 个簇；
  - 共价修饰氨基酸：**12,689** 个结构，**1,099** 个簇；
  - 另加 CSD 小分子晶体结构（数量论文未在正文给出，见 SI table S4【未核实具体条目数】）。
- **清洗与去重**：过滤常见溶剂与结晶添加剂；按 **30% 序列同一性**聚类以避免过表征结构主导。【核实：论文】
- **切分与泄漏控制**：按蛋白序列簇切分，**取 10% 簇作验证**；盲测用 CAMEO 连续评测（第三方盲配体对接赛道）。**未见 PDB 时间切分**，也未见"训练 PDB 快照日期"——论文正文与 README 均未写；我查过论文正文、CSD 段落、SI 目录与仓库 README，只能确认**模板库**停在 `pdb100_2021Mar03`，这不等于训练集截止日。【未核实：训练集 PDB 快照日期】
- **覆盖与偏置**：
  - 元素覆盖受 46 token 限制，重元素/稀有金属可能落到 unknown token【推断，依据 "not the full periodic table"】；
  - 配体侧存在 PDB 药物化学偏置，论文用"低 Tanimoto（<0.5）仍可预测"来论证泛化，但这是**论文自称的泛化证据**，不是无偏证明【论文自称】；
  - 抗体/纳米抗体：训练数据来自通用 PDB，**未见 VNAR 专项增广或评测**【未核实：论文与 README 未提及】。
- **可获取性与许可**：
  - 代码 + 权重：`baker-laboratory/RoseTTAFold-All-Atom`，**BSD 3-Clause 类许可，Copyright 2024 University of Washington**，且 LICENSE 明确写"covers both the source code and model weights"。【代码／配置核实：LICENSE 原文】
  - 权重：`http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFAA_paper_weights.pt`，**wget 直接下载，无需申请、无需登录**；HTTP HEAD 返回 `Content-Length = 1,336,673,865`（≈1.34 GB），Last-Modified 2024-03-05。【核实：实际 HEAD 请求】
  - 依赖数据库：UniRef30_2020_06（46 GB）、BFD（272 GB）、pdb100_2021Mar03 模板（81 GB）。【核实：README】
  - RFdiffusion-AA 仓库 GitHub API 标记 `license: NOASSERTION`，但仓库内 LICENSE 文件本身是同样的 BSD 文本（署名不同）。**GitHub 的 license 识别与文件内容不一致，引用时以 LICENSE 文件为准并注明。**【代码／配置核实】

---

## 5. 宣称与可验证性

| 宣称 | 证据类型（消融 / 曲线 / 表格 / 仅文字） | 核查状态 | 备注 |
| ---- | ---- | ---- | ---- |
| 覆盖蛋白、核酸、小分子、金属、共价修饰五类组分 | 图 + 文字 + 分任务评测 | 已核实 | 五类均有专门数据与示例命令 |
| 非聚合物原子用 46 种元素 token | 表格（table S5）+ 正文 | 已核实 | 明确"非完整周期表" |
| 键级四类（单/双/三/芳香）+ 手性放 3D 轨 | 正文 + Fig. S1 | 已核实 | 注意：不是"R/S token" |
| 46% 共价修饰预测 <2.5 Å RMSD | 表格/统计 | 论文自称 | 未见第三方复现 |
| 蛋白单体精度与 AF2 相当 | 对照表 | 论文自称 | 未做与 AF3 的同口径对比 |
| CAMEO 盲测：43% 目标 PAE-interaction<10，其中 77% 配体 RMSD<2 Å | 曲线 + 文字 | 论文自称 | CAMEO 为滚动盲测，可作为弱独立证据 |
| 低同源（BLAST e>1）与低配体相似（Tanimoto<0.5）仍准确 | 分箱统计 | 论文自称 | 未给样本量 |
| RFdiffusion-AA 设计 heme 结合蛋白，晶体结构 Cα RMSD 0.86 Å | 实验（晶体学） | 论文自称 | 有 PDB 号可查，但未逐条核对 |
| 仓库权重可直接下载、许可证覆盖权重 | 文件 + 许可文本 | 已核实 | 见 §4 |
| 推理墙钟时间 | — | **未核实** | 论文正文、README、config 均未给；只能给结构成本：4 recycle × 40 block，`MAXSEQ 1024`、`n_templ 4`，另加 hhblits/BLAST 搜库（库合计 ≈399 GB） |

- **可复现所需材料**：代码（BSD）、权重（1.34 GB，公开）、三个搜库文件（合计 ≈399 GB）、单卡 GPU（README 给 Docker + `--gpus all`，未给显存下限）。**显存下限【未核实】**。
- **与官方文档 / 论文的一致性**：README 的组分覆盖、示例命令与论文一致；差异点在于README 未提训练数据规模，需回论文。仓库最后一次 push 为 2024-07-09（RFAA 主仓库），RFdiffusion-AA 仓库 push 停在 2024-07-09 但 2026-10 仍有 star 活动——**代码已进入维护停滞期**【代码／配置核实：GitHub API】。

---

## 6. 实践与应用

- **适用场景**：需要"序列/化学式 → 全原子复合物结构"且**要求单次确定性输出**（不走采样）的任务；配体对接的盲测初筛；蛋白-金属位点；共价修饰残基建模。
- **不适用/需谨慎**：需要多候选排序的场景（RFAA 只有 4 次 recycle，没有 AF3 式的多样本采样）；需要显式电荷/质子化态的场景；大规模筛选（每次都要跑模板搜索）。
- **关键参数**（上游仓库路径 `rf2aa/config/inference/base.yaml`，本地未克隆）：
  - `MAXCYCLE: 4` —— recycle 次数，直接决定精度/耗时；
  - `n_templ: 4`、`MAXSEQ: 1024`、`MAXLAT: 128` —— 模板与序列上限；
  - `use_chiral_l1: True`、`use_lj_l1: True` —— 手性与范德华项损失，关掉会改变物理合理性；
  - `use_atom_frames: True` —— 决定重原子是否带 frame（当前实现为重原子仅平移）。
- **与 VNAR/纳米抗体工作的关系**：RFAA 可作为"抗原-纳米抗体复合物"的**基线预测器**，但没有 VNAR 专项训练与评测；把它当作通用复合物预测器使用，需自行构建 VNAR-抗原测试集。【推断】

---

## 7. 疑问与讨论

- **Q**：RFAA 和 RFdiffusion-AA 是同一个模型吗？
  - **A**：不是。RFAA 是预测器（给定组分 → 结构，单次前向 + recycle）；RFdiffusion-AA 是在 RFAA 权重上做去噪微调得到的生成器（给定配体 + 随机残基 → 口袋）。两者代码仓库、LICENSE 署名、权重文件都不同。专题 01 §6.2 把两者写成一行"RFdiffusion-AA / RFAA"，并在"生成范式"列填了扩散——这条需要订正，已记入 work 下的待订正清单。
- **Q**：为什么手性不能放在 1D/2D？
  - **A**：因为 1D/2D 表示对镜像反射不变，R 与 S 在序列和原子对特征上完全同构；只能靠 3D 轨的角度符号打破这一对称。【核实：论文】
- **Q**：RFAA 的"46 种元素"够用吗？
  - **A**：对 PDB 常见元素够用；对含稀有金属、卤素重同位素、硼酸类片段的体系可能落到 unknown。是否影响精度论文未做消融。【推断 + 未核实】

---

## 8. 学习心得

- RFAA 值得学的地方不是"它也做全原子"，而是**它证明了不必为了全原子而放弃 frame**：把非聚合物降级为"可自由平移的点 + 图 token"，就能在 RF2 主干上复用全部聚合物归纳偏置。这与 AF3"连聚合物也原子化"是两种正交的工程取舍。
- "全原子"这个词在 RFAA 里是有边界的：无显式氢、无电荷通道、元素子集化。写对比表时若只写"全原子"三个字，会把这三条限制全部抹掉。
- 数据层最容易被忽略的一条是 **CSD 增广 + 原子化**：前者让模型见 PDB 之外的小分子几何，后者把"蛋白内部原子相互作用"迁移成"蛋白-配体原子相互作用"的训练信号。这比"堆 PDB 条目数"更能解释它对低 Tanimoto 配体的泛化。

---

## 9. 参考资料

- Krishna R., Wang J., Ahern W., et al. *Generalized biomolecular modeling and design with RoseTTAFold All-Atom.* Science 384, 6693 (2024). DOI: 10.1126/science.adl2528
- 论文 PDF（IPD 公开副本）：https://www.ipd.uw.edu/publication-pdfs/326/91362ad5c3c9def48ef01ea5f46d142c/science.adl2528.pdf
- 预测器仓库：https://github.com/baker-laboratory/RoseTTAFold-All-Atom （LICENSE: BSD，UW 2024）
- 生成器仓库：https://github.com/baker-laboratory/rf_diffusion_all_atom （LICENSE 文件为 BSD 文本；GitHub 识别为 NOASSERTION）
- 推理配置：`rf2aa/config/inference/base.yaml`（`MAXCYCLE: 4`、`n_templ: 4`、`use_atom_frames: True`）
- 仓库内相关笔记：[AlphaFold3](../AlphaFold3/README.md)、[RFD4-Proteina](../RFD4-Proteina/README.md)、专题 [01 §2.2 / §6.1 / §6.2](../../专题/01_全原子表示与生成范式.md)
- 本次核查产物：`d:\postdoc_work\文献学习\2026-11-10\rfaa-chroma-boltzgen-notes\work\`
