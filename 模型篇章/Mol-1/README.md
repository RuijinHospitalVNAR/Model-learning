# Mol-1（Open Frontier Biomolecular Intelligence）

> **状态**：学习中
> **记录人**：团队共同学习
> **最后更新**：2026-10-28
> **仓库**：<https://github.com/molence/Mol-1> · **出品方**：Molence Bio · **许可证**：Apache 2.0

---

## 1. 学习背景与目标

Mol-1 是 Molence Bio 开源的**生物分子智能模型**，面向复合物级别的结构预测：输入可包含蛋白质链、DNA、RNA、小分子配体、离子，并支持共价键与残基修饰的描述。它衍生自字节跳动的 [Protenix](https://github.com/bytedance/Protenix)，而 Protenix 建立在 AlphaFold 3 生态（AlphaFold 3、OpenFold、ColabFold）之上。

- **为什么学**：Mol-1 在抗体–抗原、蛋白–配体/分子胶、蛋白–多肽复合物预测上提供了完整开源实现与基准（仓库自带的示例就包含由 JSON 模板驱动的纳米抗体案例），对我们理解和使用结构预测模型有直接参考价值。
- **本次学习目标**：
  1. 跑通安装 → 数据下载 → 示例预测的完整流程；
  2. 掌握输入 JSON 与输出 CIF/置信度文件的格式与解读；
  3. 理解 MSA / 模板搜索管线与显存控制策略；
  4. 了解微调路径（用自己的 mmCIF 数据）。

## 2. 核心概念

| 术语 | 含义 |
| ---- | ---- |
| job / sequences | 输入 JSON 中的一个预测任务，`sequences` 内列出 proteinChain、dnaSequence、rnaSequence、ligand、ion 等条目 |
| token | 模型处理的基本单元；显存占用随 token 数接近二次方增长 |
| MSA | 多序列比对；蛋白 MSA 默认经 ColabFold MMseqs2 公共服务在线搜索（`--use_msa`，需联网），RNA MSA 默认关闭 |
| template | 结构模板；用 HMMER 在 `search_database/` 中搜索，对齐需 `kalign`，命中结构首次使用时从 PDBe 下载 |
| plddt | 逐残基置信度，**量程 0–100** |
| chain_plddt | 逐链置信度，**量程 0–1**（注意与 plddt 量纲不同，容易踩坑） |
| PAE | 预测对齐误差矩阵，**按 token 而非残基编号索引** |
| ranking_score | 输出样本排序依据；同一 seed 内 `sample_0` 即排名最高的结构 |
| DockQ / lDDT-PLI | 复合物预测常用评价指标；PoseBusters 用于合理性检查 |
| CIF | 输出结构文件格式（mmCIF），每个样本附带一个置信度 JSON |

## 3. 原理与架构

- **来源与谱系**：Mol-1 由 Protenix 衍生，保留了 AF3 生态的方法体系（MSA/模板信息 + 成对表示 + 扩散式结构生成 + 置信度头）；每个衍生文件保留了上游版权头，修改过的文件标注了所来自的上游版本。
- **推理流程**（依据 README 归纳）：
  1. 输入 JSON 描述复合物组成（可自带 MSA/模板路径，缺什么就自动搜什么）；
  2. 蛋白 MSA 走 MMseqs2 服务，模板走 HMMER + kalign 对齐；
  3. 模型主干预测 → 扩散采样生成多个样本 → 输出 CIF + 置信度；
  4. `--need_atom_confidence true` 时额外输出逐 token / 逐原子的 `full_data` JSON（含 PAE 矩阵）。
- **待深入**：网络结构细节、训练配方见仓库自带技术报告 `assets/report.pdf`，列为下次学习材料。

## 4. 实践与应用

### 4.1 安装与数据

```bash
git clone https://github.com/molence/Mol-1 && cd Mol-1
pip install -e ".[cuda]"        # NVIDIA GPU；CPU 用 .[cpu]；CUDA < 13.0 用 .[cuda12]
```

- 首次在 CUDA 机器上运行会编译 fused LayerNorm 扩展（看似卡住实为编译，之后按环境缓存）。
- 参考数据 ~0.3 GB，模型权重 `mol-1-base.pt` ~2.2 GB：

```bash
export MOL_ROOT_DIR="./data"
bash scripts/database/download_mol_data.sh
mkdir -p ${MOL_ROOT_DIR}/checkpoint
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt \
  -O ${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt
```

### 4.2 最简预测

```bash
# 同源二聚体 + 两个配体的官方示例（644 tokens，约需 14 GB 显存）
mol pred -i examples/examples_with_template/example_9fm7.json -o ./output
```

结果落在 `output/<job_name>/seed_<seed>/predictions/`，为 CIF 结构 + 置信度 JSON；默认不用模板（`--use_template false`），用模板需装 `hmmer` 与 `kalign`（conda 注意 `kalign3=3.3.5` 钉版本）。

### 4.3 命令一览

| 命令 | 作用 |
| ---- | ---- |
| `mol pred` | 推理预测 |
| `mol json` | 将 PDB/CIF 结构转为 Mol-1 输入 JSON |
| `mol msa` | 蛋白 MSA 搜索 |
| `mol prep` | 补齐输入缺失的 MSA 与模板（`--use_rna_msa true` 时含 RNA MSA） |
| `mol doctor` | 环境自检（只读、不加载模型，报 bug 前先跑） |

### 4.4 显存与大批量技巧

- 峰值显存随 token 数接近二次方增长；默认设置在 95 GiB GPU 上可跑约 **1,500 tokens**。
- 更大体系（如 FoldBench 中 2,456 / 2,532 tokens 的输入）需 large-input 设置：更小 chunk、`--trimul_kernel torch`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 显存富余时可用 `--infer_setting.sample_diffusion_chunk_size 5` 让 5 个样本一起去噪，速度更快。
- 显存不够时，报错信息会直接指出该调哪个参数。

### 4.5 微调

```bash
# 结构 → bioassembly 字典 + 链/界面索引表
python3 scripts/prepare_training_data.py -i finetune_data/structures.txt \
  -b finetune_data/bioassembly -o finetune_data/indices.csv \
  -c "$MOL_ROOT_DIR/common/clusters-by-entity-40.txt"
# 蛋白 MSA（与推理同源搜索服务）
python3 scripts/prepare_finetune_msa.py -b finetune_data/bioassembly -o finetune_data/msa

CHECKPOINT_PATH="${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt" bash finetune_demo.sh \
  --max_steps 6 --checkpoint_interval 3
```

- 微调从 checkpoint 起步：权重严格加载，**优化器与调度器全新开始，EMA 从加载权重初始化**——这与"连同优化器状态恢复训练"是刻意区分的两件事。
- 全量微调数据（预计算 MSA + 模板命中）为 9.8 GB 压缩 / 46 GB 解压：`download_mol_data.sh --full`。

## 5. 疑问与讨论

> 待本地学习对话后补充精选问答。当前带入学的问题：

- **Q1**：plddt（0–100）与 chain_plddt（0–1）为何采用不同量纲？使用时如何统一？
- **Q2**：PAE 矩阵按 token 索引，配体/离子/修饰残基各占多少 token？
- **Q3**：`mol-1-base` 与潜在后续权重的关系？技术报告中如何描述训练数据构成？
- **Q4**：对纳米抗体（VNAR 等单域抗体）体系，JSON 模板驱动流程相比纯 MSA 流程收益有多大？

## 6. 学习心得

- 仓库文档工程质量高：从 `mol doctor` 环境自检、显存不足时报错即给调参建议，到 checkpoints 严格加载（形状不匹配直接拒绝），对复现和排障友好。
- "置信度两套量纲 + PAE 按 token 索引"是官方文档明确提醒的两个易错点，写脚本解析输出时务必先看 `docs/output_format.md`。

## 7. 参考资料

- 仓库：<https://github.com/molence/Mol-1>（技术报告：`assets/report.pdf`）
- 官方文档：[安装](https://github.com/molence/Mol-1/blob/main/docs/installation.md) · [输入 JSON 格式](https://github.com/molence/Mol-1/blob/main/docs/infer_json_format.md) · [输出格式](https://github.com/molence/Mol-1/blob/main/docs/output_format.md) · [显存控制](https://github.com/molence/Mol-1/blob/main/docs/inference_memory.md) · [微调](https://github.com/molence/Mol-1/blob/main/docs/finetune.md) · [MSA/模板管线](https://github.com/molence/Mol-1/blob/main/docs/msa_template_pipeline.md) · [数据源清单](https://github.com/molence/Mol-1/blob/main/docs/data_sources.md)
- 上游：[Protenix](https://github.com/bytedance/Protenix)（AlphaFold 3 / OpenFold / ColabFold 生态）
