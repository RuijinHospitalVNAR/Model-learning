---
model: TorchCraft
aliases: [TorchCraft, TorchX, TorchFold-TorchCraft]
family: 序列/结构设计（binder 设计 / 预测器反演）
granularity: 全原子坐标（继承冻结 AF3；但优化变量是序列 logits，坐标不参与反传更新）
backbone: 冻结 AlphaFold3（TorchFold 实现）+ 可训练序列 logits；无自有生成网络
generative: 非生成式（test-time 反传优化 / predictor inversion；无扩散、无流匹配、无自回归）
conditioning: 靶标全原子结构 + 可编辑残基掩码（固定框架 / CDR 限定 / hotspot）+ IgLM 序列先验 + 接触/几何/置信度约束
data_scope: 无训练数据；仅用 AF3 预训练权重、IgLM、ProteinMPNN 参考组成分布、SAbDab/OASis 评估参考；实验验证 4 靶 minibinder + 4 靶 VHH
status: 已完成
verified_at: 2026-11-10
---

# TorchCraft

> **状态**：已完成
> **记录人**：模型学习笔记（AI 辅助整理）
> **最后更新**：2026-11-10

---

## 1. 学习背景与目标

TorchCraft 是昌平实验室（Changping Laboratory）2026 年的预印本（arXiv 2609.19770v1，2026-09-17），标题是 *Unified binder design by inverting an all-atom structure predictor*。它被列进全原子对比矩阵，但**它不是生成模型**：

> 冻结 AF3，把待设计序列写成一个可训练 logits 矩阵，用 AF3 输出的置信度/接触/几何量做损失，反传梯度改序列。

这类方法通常叫 **predictor inversion**（预测器反演）或 test-time optimization。与 RFdiffusion / BoltzGen / IgGM 那些"训练一个生成器"的路线是**并列而非同类**。

本次目标：

1. 定名并确认它确实属于蛋白/全原子设计谱系（结论：属于，但属于"设计范式"而非"生成范式"）；
2. 把三轴填清楚，尤其是 generative 轴不能硬塞进"扩散/流匹配"词表；
3. 判断它是否应留在矩阵的"生成模型"子表里（结论：应保留在总表，但需要单独分组或加一列"是否需训练"）；
4. 评估它对 VNAR / 纳米抗体设计的可迁移性——这是本批五篇里唯一有**实验验证 VHH** 的工作。

**材料边界**：依据 arXiv 全文（PDF 转文本，work 目录 `torchcraft_2609.19770.txt`）；代码仓库 `github.com/Mingchenchen/TorchX` 只确认了 URL 存在（论文给出），**未克隆、未跑通**；许可证未核实。

---

## 2. 核心概念

| 术语 | 含义 |
| ---- | ---- |
| predictor inversion | 不改预测器权重，只优化输入（这里是序列），让预测器输出满足设计目标 |
| TorchFold | 昌平实验室的 AF3 实现框架；TorchCraft 在其上跑冻结 AF3 权重 |
| TorchScore | 只打分不 refold 的评估器：给坐标、用同一套 AF3 权重算置信度 |
| sequence logits | 形状 L×20 的可训练矩阵；softmax 后是每位置氨基酸分布 |
| soft warm-up | 阶段 0：先在连续概率空间走若干步，避免一开始就离散化 |
| temperature annealing | 阶段 2：逐步降低 softmax 温度 τ，让分布变尖 |
| straight-through | 阶段 3：前向用 one-hot，反向用 softmax 梯度，实现离散序列上的继续优化 |
| distogram contact loss | 用 AF3 预测的距离分布（而非最终坐标距离）构造可微接触损失 |
| hotspot | 靶标上指定的关键残基；接触损失会优先奖励 binder 与 hotspot 的接触 |
| CDR-restricted IgLM loss | 只对可编辑 CDR 施加 IgLM 的局部 infilling 交叉熵，框架与其余 CDR 作上下文 |
| framework-conditioned VHH | 固定 VHH 框架序列，只优化三个 CDR |
| α-helix loss | 用 (i, i+3) 接触概率奖励或抑制螺旋（权重可正可负） |

---

## 3. 原理与推导

### 3.1 基本假设

TorchCraft 的假设只有一句：**一个足够好的全原子结构预测器，其内部置信度与几何输出已经隐含了"什么样的序列能结合这个靶标"**。因此不需要再训一个生成器，只要把这个隐含知识通过梯度反传"倒推"成序列即可。

这与 BindCraft（AF2 反演）、BoltzDesign（Boltz 反演）、Germinal（抗体专用反演）同源；TorchCraft 的增量是**统一到 AF3 的全原子表示**，并覆盖 minibinder / VHH / 环肽 / 小分子结合蛋白四种形态。

### 3.2 优化对象与流程

```text
输入：靶标全原子结构（蛋白/核酸/小分子/离子）+ 可编辑位置定义 + 约束
变量：binder 链的序列 logits  L ∈ R^{L×20}
循环：
  1. logits → softmax（或 one-hot）→ 序列
  2. 序列 + 靶标送入冻结 AF3 → 坐标 + pLDDT/pTM/ipTM/PAE/PDE/distogram
  3. 计算损失 L_total
  4. 反传到 logits，更新
输出：argmax 序列 + AF3 预测复合物结构
```

四个阶段（附录 A.1.2–A.1.3）【核实】：

| 阶段 | 名称 | 作用 |
| ---- | ---- | ---- |
| 0 | soft warm-up | 连续概率下先走几步，稳定梯度 |
| 1 | logits → probabilities | 从 logits 过渡到概率表示 |
| 2 | temperature annealing | τ 从 τ₀ 退火到 τ_min，分布变尖 |
| 3 | straight-through | 前向 one-hot，反向用 softmax 梯度 |

任务配置不同：minibinder 与 VHH 用"温度退火 + straight-through"；环肽用全部四阶段；小分子任务先用**无配体 warm-up**，再逐步引入界面项 `L(t) = L_scaffold + w_interface(t)·L_interface`。

### 3.3 损失函数

按类别归纳（附录 A.1.4）【核实】：

**置信度类**：pLDDT、pTM、ipTM、PDE、PAE、iPAE——直接对 AF3 的置信头做梯度上升/下降。

**接触类**：

- 用 **distogram** 而不是最终坐标距离构造接触损失（更平滑、可微）；
- intra-chain binder contact：鼓励 binder 自身形成足够内部接触，偏向紧凑折叠；
- inter-chain target–binder contact：鼓励跨链接触；若给 hotspot，则优先奖励 binder 与 hotspot 的接触；
- 每个 anchor 残基只保留损失最低的一部分伙伴残基，再取表现最好的比例平均——即"奖励一小撮高概率接触"，避免被平均稀释。

**结构先验类**：α-helix loss（奖励 i 与 i+3 在 6 Å 内），权重可为负以抑制螺旋。

**序列先验类**：

- 氨基酸组成正则：把平均残基分布对齐到 ProteinMPNN 重设计样本的参考分布；
- CDR-restricted IgLM loss：对每个 CDR，用"框架 + 其余 CDR + 当前 CDR 的 argmax"构造 IgLM 的 infilling 上下文，取 IgLM 局部条件分布作教师，对当前 CDR 分布算交叉熵。

注意 IgLM 那一步的梯度处理：上下文用 argmax 离散化（不回传），但被优化 CDR 的分布保持可微，这样既用上抗体序列先验，又不引入离散化断梯度【核实：Algorithm 3】。

### 3.4 三轴判定

| 轴 | 取值 | 判定依据 |
| ---- | ---- | ---- |
| 表征粒度 | 全原子坐标（继承 AF3） | AF3 输出原子级坐标与 distogram；但**优化变量是序列 logits**，坐标只是前向计算结果【核实：2.1】 |
| 生成范式 | **非生成式**：test-time 反传优化 / predictor inversion | 无训练、无采样链、无噪声调度；"hallucination" 指迭代优化而非生成【核实：摘要 + 2.1】 |
| 条件接口 | 靶标全原子结构 + 可编辑残基掩码 + hotspot + 序列先验 + 几何约束 | 通过 mask 决定哪些位置可编辑；VHH 固定框架只放开 CDR【核实：2.2–2.4】 |

**为什么 generative 轴不能填"扩散"**：TorchCraft 不做"从噪声生成结构"，也不学数据分布；它是在给定靶标下做**单目标优化**。对比表的"test-time 反传优化（非生成式）"是准确描述，但它**超出了受控词表**（扩散/流匹配/自回归/非生成式），严格说应填"非生成式"，再在备注里写"test-time 优化"。

### 3.5 它应不应该留在矩阵里

**结论：保留在总表，但不能和生成模型混在一张子表里比较。**

理由：

- **属于该谱系**：它确实用全原子表示、确实做 binder 设计、确实产出可实验验证的序列，和 RFdiffusion+ProteinMPNN 是同一任务的不同解法；
- **不属于"生成模型"**：没有训练集、没有分布学习、没有采样；把它和 IgGM/BoltzGen 放同一列比"生成范式"是范畴错误；
- **建议**：总表保留，另设"设计范式"列（生成 vs 反演 vs 前向 hallucination），或在生成模型子表中单列一行并标注"参照系"。

### 3.6 关键缺陷：评估器与设计器同源

TorchCraft 的计算通过率用 **TorchScore** 评估——**同一套 AF3 权重**，只打分不 refold。这意味着：

- 设计时被优化的置信度，评估时又被当作质量指标 → **自洽偏差**；
- 作者自己在讨论里承认："predictor-derived pass rates are sensitive to the evaluator and should not be interpreted as experimental success rates"【核实：Discussion】；
- 缓解手段是同时报 Rosetta interface score 分布，但这仍不是独立实验。

这一点是读它所有"计算 pass rate 领先"结论时必须要打的折扣。

---

## 4. 训练数据构成与切分

### 数据源与规模

**TorchCraft 本身没有训练数据**——这是它与本批其他四篇最根本的差别。

它依赖的外部资源【核实：正文 + 附录】：

| 资源 | 用途 |
| ---- | ---- |
| AlphaFold3 预训练权重 | 冻结预测器，提供坐标与置信度 |
| IgLM | 抗体序列先验（CDR-restricted CE 的教师） |
| ProteinMPNN 重设计样本 | 估计氨基酸组成参考分布（组成正则） |
| SAbDab | IgLM likelihood 的天然抗体参照、CDR 序列 logo 对照 |
| OASis / BioPhi | humanness 评分 |
| PDB | 结合构象多样性对照（target-aligned Cα RMSD 聚类） |
| AbMPNN / LigandMPNN | 可选的 post hoc 重设计基线 |

### 清洗、切分与泄漏控制

不适用（无训练）。但有两个**间接的数据依赖**需要记录：

1. **AF3 权重里的数据切点**决定了它能"认识"哪些靶标类似物；若靶标与 AF3 训练集同源，反演会更容易成功——论文未做此分层分析【未核实：正文与附录未见 AF3 训练集重叠分析】；
2. **IgLM 的训练分布**（OAS 系人类抗体）决定了 CDR 先验的偏向；用于 VHH 时已经跨了物种/格式，论文未量化这一偏移【未核实】。

### 覆盖与偏置

- **实验验证只覆盖 minibinder 与 VHH**：环肽与小分子结合蛋白只有计算基准【核实：Discussion 局限段】；
- **每类 4 个靶**：样本量很小，"4/4 有结合"是 campaign 级结论，不是 hit rate；
- **靶标偏置**：IFNAR2、IL-7Rα、PDGFRβ、PD-L1、BHRF1、EFNA5、S100A4 都是常见测试靶，不能代表困难靶（如多次跨膜、高度糖基化）；
- **无 post hoc 重设计**：实验测的序列直接来自 TorchCraft 输出，这一点是加分项。

### 可获取性与许可

- 代码：`https://github.com/Mingchenchen/TorchX`（论文给出），**许可证未核实**【未核实：未克隆仓库】；
- 权重：**不含自有权重**，但依赖 AF3 权重，须遵守 AF3 权重条款（非商用等），具体条款本次未核对【未核实】；
- IgLM 权重另有权重许可，同样未核实【未核实】。

---

## 5. 宣称与可验证性

| 宣称 | 证据类型 | 核查状态 | 备注 |
| ---- | ---- | ---- | ---- |
| 冻结 AF3、只优化序列 logits，无训练 | 方法 + 算法伪代码 | 已核实 | 摘要明写 "frozen all-atom predictor" |
| minibinder：IFNAR2 60.2 / IL-7Rα 8.55 / PDGFRβ 2.04 / PD-L1 23.3 nM | 实验（BLI，多浓度） | 已核实（论文口径） | 每靶从 3,000 个设计中筛选；只报代表克隆，未给 hit rate |
| VHH：BHRF1 24.3 / EFNA5 122 / PDGFRβ 268 / S100A4 354 nM | 实验（BLI） | 已核实（论文口径） | 靶-Kd 对应按正文顺序；图中只标 24.3 与 354 两个 |
| 实验序列未经 post hoc 重设计 | 正文明确 | 已核实 | 作者强调两次 |
| 计算基准：12 蛋白靶 × 4 长度（75/120/160/200）对比 BoltzGen / RFdiffusion | 图（Fig. 2f） | 论文自称 | 正文未给具体通过率数值，只在图中【未核实：无可引数字】 |
| VHH：15 靶 × 5 框架，对比 RFantibody / BoltzGen / Germinal | 图（Fig. 3i） | 论文自称 | 同上，数值未在正文给出 |
| 环肽：8 靶，对比 RFpeptides / BoltzGen | 图（Fig. 4c） | 论文自称 | 结果依赖靶标与环化方式，非全面领先 |
| 小分子结合蛋白设计 | 仅计算 | 论文自称 | 无实验 |
| "统一框架可覆盖四种 binder 形态" | 多任务演示 | 部分核实 | 实验只覆盖两种；另两种是计算 |
| 计算通过率可代表实验成功率 | — | **存疑** | 作者自己否认；且评估器与设计器同源 |

**可复现所需材料**：TorchX 代码 + AF3 权重 + IgLM 权重 + GPU（单靶需多次 AF3 前向/反向，成本不低）。论文未给单靶耗时与显存【未核实：正文与附录未见】。

**与官方文档/论文的一致性**：对比表三轴中 "granularity=冻结 AF3 全原子"、"conditioning=序列 logits + CDR 限定 IgLM" 与论文一致；"generative=test-time 反传优化（非生成式）"事实正确但**超出受控词表**，且表中把它与扩散模型并列比较"生成范式"，属分类错位，已记入待订正清单。

---

## 6. 实践与应用

**适用场景**

- 有明确靶标结构、想要少量高置信候选的 binder 设计（minibinder / VHH）；
- 需要固定框架、只改 CDR 的抗体工程（framework-conditioned VHH）；
- 需要引入非蛋白组分（小分子、核酸、离子）的界面设计；
- 想复用 AF3 先验、不想训新模型的团队。

**不适用 / 需谨慎**

- 需要大规模多样本生成（它是单目标优化，多样性靠多次随机初始化）；
- 没有靶标实验结构（AF3 对未见过靶标的预测误差会直接传导）；
- 需要严格 hit rate 估计的项目（每靶只报代表克隆）；
- 商用：AF3 权重条款 + TorchX 许可证都需先确认。

**实操要点**

1. **可编辑位置的定义比损失权重更重要**：VHH 固定框架只放开 CDR，是最有效的降搜索空间手段；
2. **组成正则很关键**：不加正则时序列组成偏离天然分布，表达差；作者用 ProteinMPNN 参考分布对齐后，两个 PDGFRβ campaign 的表达提升；
3. **IgLM 先验要限定在 CDR**：全序列引导（Germinal 式）会把先验摊到框架上，反而稀释 CDR 的可信度；
4. **不要用 TorchScore 单独下结论**：至少配 Rosetta 界面分或独立 refold。

**未实测**：本次未运行 TorchX，以上为读论文后的判断。

---

## 7. 疑问与讨论

- **Q**：TorchCraft 算不算"模型"？
  - **A**：它是一套**优化流程/框架**，不是一个训练出来的模型。真正的"模型"是冻结的 AF3。所以写综述时应归到"设计方法"而非"生成模型"。

- **Q**：为什么它能把 binder 设计统一到四种形态？
  - **A**：因为 AF3 的输入本来就是"任意分子组成 + 链 + 可掩码位置"，TorchCraft 只是把"哪些位置可改、加什么损失"做成配置。换形态不需要换网络，这是全原子表示的直接红利。

- **Q**：4/4 靶都有结合，是不是说明成功率很高？
  - **A**：不能这么读。每靶从 3,000 个设计里筛出代表克隆做 BLI，"有结合"是定性结论；没有报"多少个设计里有多少个结合"，也没有独立实验室复现。真正可信的是"能做出 nM 级 binder"这件事本身。

- **Q**：它和 BoltzGen / RFdiffusion 的本质差别？
  - **A**：那两者训练生成器、从噪声采样结构再定序列；TorchCraft 不生成结构，直接优化序列让冻结预测器"相信"这个复合物成立。前者探索空间大、多样性好；后者单点精度可能高，但容易过拟合预测器的偏好（自洽偏差）。

- **Q**：对 VNAR 可迁移吗？
  - **A**：本批五篇里它最接近可用。机制上是"固定框架 + 优化 CDR"，VNAR 只要有可用的 V-NAR 框架序列与 CDR 边界定义，就能直接套；IgLM 先验可以换成 VNAR 专用先验或直接关掉。风险有三：(1) VNAR 的 CDR3 常含非典型半胱氨酸与超长环，IgLM 的人类抗体先验会拉偏；(2) AF3 对 VNAR-抗原复合物的预测精度未见专项评估；(3) 评估仍是同源打分。结论：**流程可迁移，先验需替换，精度待验证**【推断】。

---

## 8. 学习心得

- 这篇最大的价值是**提醒我们"全原子"不等于"生成"**。冻结一个全原子预测器，把它当可微打分器用，也是一条完整的设计路线。
- 它的工程细节（distogram 接触损失、只奖励 top 分位接触、α-helix 权重可正可负、组成正则）比架构更值得抄，因为这些是决定湿实验成败的东西。
- 同时它也是一个警示案例：**用同一个模型设计与评估**，数字一定好看，但说服力有限。作者诚实承认了这一点，值得肯定。
- 对矩阵维护的启示：把"是否需要训练"作为一列加进去，能立刻把 TorchCraft 与扩散/流匹配模型区分开，避免读者误读。

---

## 9. 参考资料

- 论文：TorchCraft Team. *TorchCraft: Unified binder design by inverting an all-atom structure predictor.* arXiv:2609.19770v1（2026-09-17），Changping Laboratory
- 项目页：https://torchx-cpl.github.io；代码：https://github.com/Mingchenchen/TorchX
- 前置方法：BindCraft（AF2 反演，Pacesa et al., 2025）；BoltzDesign；Germinal（抗体反演）；BoltzGen；RFdiffusion / RFpeptides；ProteinMPNN / LigandMPNN / AbMPNN
- 序列先验：IgLM（Shuai, Ruffolo & Gray, Cell Systems 2023）
- 评测参考：SAbDab；OASis / BioPhi humanness
- 本仓库关联：`专题/01_全原子表示与生成范式.md` §6.1、§6.2；`专题/04_抗体与VNAR可迁移点.md`
