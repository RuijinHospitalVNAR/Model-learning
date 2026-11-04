# 推理时优化机制对比：AnewOmni 抗体 pipeline × RFD4-Proteina 奖励搜索

> **涉及的模型**：AnewOmni（抗体/纳米抗体设计 pipeline）· RFD4-Proteina（reward-guided search）
> **记录人**：团队共同学习
> **最后更新**：2026-11-04（追加订正说明；正文未改）
> **材料来源**：本地学习对话 + 两份官方文档（AnewOmni `docs/antibody-pipeline.md`、RFD4-Proteina 官网 concepts/search，均已于 2026-11-03 核实）

---

## 订正说明（2026-11-04 追加，原文保留可追溯）

核查发现旧笔记把两个模型混为一谈：2025 年的 **Proteina**（arXiv 2503.00710，ICLR 2025 Oral）与 2026 年 `rfd4-proteina.org` 上的**另一个模型**（Baker Lab × NVIDIA GenAIR，接口含 `bb_ca` + `local_latents`，reward guided search，AF3/OF3 oracle refold）。以下四条曾记在 Proteina 名下的"公认属性"**归属被证伪**（不是说机制在 2026 年模型上也不存在）：

| 被证伪的结论 | 核查结果 |
| --- | --- |
| 主链显式 + 侧链压成 32 维 latent | arXiv 2503.00710 与官方仓库 `NVIDIA-BioNeMo/proteina` 均 0 命中。Proteina 是 **Cα 单点表示**（`x∈R^{3L}`，Atom37 只有 index 1 非零），无侧链 latent、无解码器。 |
| PackBench 侧链打包误差约 0.82 Å | 两处一手来源均无此数字，也无 PackBench 评测记录。 |
| 解码受 Rosetta 能量函数显式监督 | 全文 "Rosetta" 仅 1 次且出现在基线名 "RoseTTAFold"；仓库无 rosetta 模块；可设计性链路用 ESMFold。该说法对应的是 2026 年模型的微调阶段（可微能量损失，非 conditioner）。 |
| beam search / reward guided search 属于 Proteina | 该机制属于 2026 年模型（内部 recycling + 外部 checkpoint beam search、oracle refold）。Proteina 本身不含搜索或奖励环节。 |

**证据**：arXiv 2503.00710 全文与 `NVIDIA-BioNeMo/proteina` 仓库逐项检索，"32 维 latent""0.82""Rosetta""beam search" 均 0 命中；而 `rfd4-proteina.org` 的 `concepts/metric-guidance`、`concepts/search`、`concepts/metrics` 与论文 PDF 片段可以逐条对应上述机制。2026 年模型确实有 `local_latents` 接口，但"32 维""0.82 Å"这两个具体数值在两处一手来源中均未见披露。

**本文件状态**：正文写于 2026-11-03，标题与行文沿用 "RFD4-Proteina" 旧称，但正文并未断言上述四条；它把 RFD4-Proteina 描述为 `rfd4-proteina.org` 的 reward guided search，方向与本订正一致，故正文不改写，仅作旧称传播源的标注。真正的错误内容落在另一篇，需按下表重写：

| 文件 | 需重写的段落 |
| --- | --- |
| `联合分析/全原子模型解题思路归类分析.md` | §3 路线 ② 标题「连续隐空间压缩（RFD4-Proteina / AnewOmni）」；路线 ② 正文「主链保持显式坐标（锚原子 N/CA/C/O），仅把侧链压成 32 维 latent … Rosetta 能量函数显式监督 … PackBench 侧链打包误差约 0.82 Å」；对比表中「全原子解题思路：主链显式+侧链 32D latent」「物理约束：**显式**（Rosetta 监督）」「侧链精度：SOTA（~0.82 Å PackBench）」「RFD4-Proteina：主链量坐标 + 侧链压 32D」四行 |

处理原则：原文一律保留以便追溯；新结论以本说明与 [`模型篇章/Proteina/README.md`](../模型篇章/Proteina/README.md) §1.1、§3.7 为准。

---

## 1. 分析目的

区分两类容易混淆的"推理时循环"机制：**把输出重新作为输入**（输出回灌）与**奖励引导搜索**（reward-guided search），澄清 AnewOmni 抗体 pipeline 与 RFD4-Proteina 各自属于哪类（或兼具），并辨析"具有推理时优化机制"与"增加计算量能持续提高设计质量"这两个不同层面的判断。

## 2. 机制事实（均经官方文档核实）

### 2.1 AnewOmni 抗体/纳米抗体 pipeline（bytedance/AnewOmni）

迭代式生成 + cofold 校正，循环发生在**多轮完整设计任务之间**：

1. 从抗体/纳米抗体框架库（framework_lib，官方筛选了相互作用主要由 CDR 贡献的框架）初始化；
2. 与靶标做复合物结构预测（cofold，后端可选 Protenix 默认 / Boltz2）；
3. 对候选加权排序；
4. 以 top 姿态为条件，重新设计指定 CDR；
5. 对新 binder 再次 cofold、评分，进入下一轮，直至迭代上限。

**排序加权项（默认键，可经 `ranking_weights` 覆盖）**：`metrics.bs_overlap`（表位覆盖）、`metrics.contact_cdr_ratio`（CDR 接触比例）、`cplx_confidences.cofold_iptm`、`cplx_confidences.cofold_binder_plddt`、`cplx_confidences.cofold_normalized_ipae`（界面预测误差）、`generative_confidences.normalized_cdr_design_likelihood`（生成似然）、`metrics.normalized_scRMSD_cdr`（CDR 结构自洽性）。

**关键工程细节**：
- `--n_beam 10`（默认每轮选 10 个进 CDR 设计）+ `--n_loser_up 2`（另从剩余池**随机**抽 2 个）——即默认 top 8 + 2 随机，显式的探索机制，对冲纯贪心选择的多样性坍缩；
- `--max_num_iterations` 默认 60，每轮生成 50 个候选——**停机是预算上限，不是收敛判据**；
- cofold 后端默认 `protenix_mini_esm_v0.5.0`（无 MSA/模板场景表现更好），评分器是外部结构预测模型。

### 2.2 RFD4-Proteina 的两种循环

- **内部 recycling**：生成过程中主干反复更新内部表示，low/medium/high thinking 对应不同循环计算预算，**不依赖外部评分反馈**。
- **外部 reward-guided search**：beam search 式——沿生成轨迹设若干**检查点（checkpoint）**，每个存活候选分支出若干子代，奖励打分后剪枝保留强者。检查点之间自包含（上一点的评分信息不带入下一点）；末检查点可选。
  - **奖励两类**：structure-only（直接读生成结构的几何指标，如界面氢键/疏水接触数，便宜，可每个分支都算）vs **oracle-based**（独立结构预测模型 AF3/OF3 对候选序列重折叠，如 pLDDT，最贵但更接近湿实验依赖的量）；一次运行可混用——早期检查点用便宜的粗筛，后期用贵的精判。
  - **continuation criteria**：与奖励排序独立的原始指标阈值——若某输入的所有入选候选都不过阈，则整个输入停止采样（"放弃无望输入"），是对"排名只挑相对最优、不管绝对好坏"的补充；
  - **novelty-guided continuation**：以"与已接受设计档案不重复"作为继续条件，把搜索推向未探索区域；
  - **非末检查点可二选一**：对本步干净预测快速近似评分，或从该状态完整 rollout 后再评真实成品。

## 3. 对照表

| 比较点 | AnewOmni 抗体 pipeline | RFD4 奖励搜索 |
|---|---|---|
| 选择的对象 | 已完成生成并经过 cofold 的候选及其结合姿态 | 生成过程中的中间轨迹/状态 |
| 下一轮做什么 | 根据选中姿态重新设计 CDR | 从保留的中间状态继续分支、生成 |
| 反馈发生位置 | 多轮完整设计之间 | 一条生成轨迹的多个检查点 |
| 评分来源 | 外部 cofold 模型置信度 + 几何指标加权 | structure-only 几何 或 oracle（AF3/OF3 refold），可分层混用 |
| 探索机制 | n_loser_up 随机捞非 top 候选 | 分支宽度 + novelty 引导继续 |
| 放弃机制 | 无（仅预算上限） | continuation criteria 阈值弃疗 |
| 共同点 | 评分影响下一轮生成的起点或条件 | 评分影响哪些轨迹继续推进 |

**判定**：AnewOmni 该 pipeline **兼具**"输出回灌"（上轮候选作为下轮设计条件）与"评分引导"（加权排序决定谁进入下轮），属于**广义评分引导的推理时优化**；区别的本质在于**评分后如何继续搜索**（重新播种设计 vs. 轨迹分支剪枝），而非评分时结构是否完整（RFD4 两种评分时机都支持）。

## 4. 更细的分类轴（三维）

1. **反馈来源**：内部自洽（RFD4 recycling）vs. 外部评分（AnewOmni cofold 置信度、RFD4 oracle reward）；
2. **反馈作用位置**：单条生成轨迹内（RFD4 checkpoint）vs. 完整生成轮次间（AnewOmni 迭代）；
3. **选择对象**：中间轨迹/状态（RFD4）vs. 完整候选及其姿态（AnewOmni）。

## 5. 核心辨析：机制存在 ≠ 计算扩展收益

**"具有推理时优化机制"是算法结构事实（可从文档/代码证明）；"增加计算量能持续提高设计质量"是实证命题（需消融/质量-计算量曲线证明）。两者不能互推。**文档层面的证据：

1. **代理奖励风险**：AnewOmni 评分项全是 cofold 模型的置信度/几何代理（ipTM、pLDDT、ipAE、scRMSD…），非真实亲和力；轮次越多，代理被过度优化（Goodhart）的风险越大；
2. **评分器偏差被承认**：RFD4 官方对奖励的分层措辞（oracle"更接近湿实验所依赖的"但最贵）本身就承认 structure-only 与真实目标间有系统偏差；
3. **设计者在主动对冲"多轮≠单调提升"**：AnewOmni 的 ε-greedy 式 `n_loser_up`、RFD4 的 novelty 引导，都是针对贪心多轮搜索副作用（多样性坍缩、重复命中）的工程补丁——机制存在的同时，其失效模式也被预设了；
4. **无收敛判据**：两者都以预算/阈值停机，没有"质量不再提升即停止"的判据；质量对计算量的响应曲线必须实测。

**对综述的写作建议**：推理时优化应分两节——(a) 机制分类（本文 §3–§4 框架）；(b) 测试时缩放的实证证据（各模型报告的 success rate vs. 采样数/轮数/beam 宽度曲线，如 Mol-1 README 中 DockQ vs. seed 数的 5–95 分位带即是此类证据的范例）。引用时避免用 (a) 冒充 (b)。

## 6. 事实性备注

- AnewOmni pipeline 描述与 `bytedance/AnewOmni` 仓库 `docs/antibody-pipeline.md`（main 分支，2026-11-03 抓取）逐条一致；`n_loser_up`、迭代预算、ranking 键名为文档新增细节，原对话未提及；
- RFD4-Proteina 搜索机制与官网 `concepts/search.html` 一致；recycling/thinking 预算的表述来自官网 workflow 页，未逐字复核，引用前建议再查；
- RFD4-Proteina 的公开论文出处仍待补全（见《全原子模型解题思路归类分析》§8）。

## 7. 参考资料

- AnewOmni 抗体 pipeline：<https://github.com/bytedance/AnewOmni/blob/main/docs/antibody-pipeline.md>
- RFD4-Proteina 奖励搜索概念页：<https://rfd4-proteina.org/latest/concepts/search.html>
- 关联篇章：[全原子模型解题思路归类分析](全原子模型解题思路归类分析.md)、[Mol-1](../模型篇章/Mol-1/README.md)（其 README 的 DockQ vs. seed 数曲线即测试时缩放实证范例）
