# 实验记录与复现说明

## 1. 目录约定

- 实验根目录：`/root/autodl-tmp/cc-bishe-experiments`
- 数据缓存：`/root/autodl-tmp/cc-bishe-cache/full_v1/dataset_cache.pt`
- 实验总表：
  - `/root/autodl-tmp/cc-bishe-experiments/summary.csv`
  - `/root/autodl-tmp/cc-bishe-experiments/summary.md`
- 合成威胁情报目录：
  - `/root/autodl-tmp/cc-bishe-threat-intel/synthetic_train_v1`
  - `/root/autodl-tmp/cc-bishe-threat-intel/synthetic_train_v2_top8ports`
  - `/root/autodl-tmp/cc-bishe-threat-intel/synthetic_train_v3_ip_only`
- 融合扫参与分析目录：
  - `/root/autodl-tmp/cc-bishe-experiments/_fusion_studies/exp05_concat_synthintel_v1`
  - `/root/autodl-tmp/cc-bishe-experiments/_fusion_studies/exp04_weighted_sampler_synthintel_v1`
  - `/root/autodl-tmp/cc-bishe-experiments/_fusion_studies/exp05_concat_synthintel_v2_top8ports`
  - `/root/autodl-tmp/cc-bishe-experiments/_fusion_studies/exp05_concat_synthintel_v3_ip_only`

每个完整实验目录统一包含：

- `resolved_config.yaml`：实验最终配置
- `metadata.json`：运行时间、主机、Python、Git commit
- `run_status.json`：训练/评估耗时与退出码
- `logs/train.log`：完整训练日志
- `logs/evaluate.log`：完整评估日志
- `artifacts/train_history.json`：逐 epoch 指标
- `artifacts/train_summary.json`：训练摘要
- `artifacts/evaluation_metrics.json`：测试指标摘要
- `artifacts/classification_report_*.json`：结构化分类报告
- `reports/classification_report_*.txt`：文本版分类报告
- `figures/*.png`：训练曲线、混淆矩阵、指标对比、标签分布

每个融合扫参与分析目录统一包含：

- `resolved_study_config.yaml`：分析配置
- `metadata.json`：来源 checkpoint 与类别信息
- `threat_intel_match_summary.json`：IOC 覆盖率统计
- `strategy_metrics.csv/json`：不同融合策略与 alpha 的指标结果
- `report.md`：关键结论摘要

## 2. 基础配置与脚本

- 基础实验配置：`config.experiment_base.yaml`
- 单实验运行：`run_experiment.py`
- 批量矩阵运行：`run_experiment_matrix.py`
- 实验汇总：`summarize_experiments.py`
- 合成威胁情报生成：`generate_synthetic_threat_intel.py`
- 融合策略扫参与复用 checkpoint 分析：`analyze_fusion_strategies.py`

补充实现调整：

- `src/models/threat_intel.py`
  现在会跳过 `metadata.json` 这类非 IOC JSON，避免把元数据误计入情报条目。
- `src/models/decision_fusion.py`
  新增 `adaptive_weighted_average`，便于后续做“仅在情报信息量较强时再增强”的对照实验。

## 3. 本轮完整实验矩阵

1. `exp01_baseline_cross_attention`
   交叉注意力多模态基线
2. `exp02_class_weight_ce`
   类别加权交叉熵
3. `exp03_focal_class_weight`
   Focal Loss + 类别权重
4. `exp04_weighted_sampler_ce`
   WeightedRandomSampler + 交叉熵
5. `exp05_ablation_concat_no_attention`
   消融 Cross-Attention，改为直接拼接
6. `exp06_ablation_traffic_only`
   仅使用流量模态
7. `exp07_tuned_concat_weighted_sampler`
   拼接融合 + WeightedRandomSampler
8. `exp08_ablation_log_only`
   仅使用日志模态
9. `exp09_concat_synthintel_weighted_alpha07`
   `concat` + 合成威胁情报 + 加权平均 `alpha=0.7`
10. `exp10_concat_synthintel_soft_voting`
    `concat` + 合成威胁情报 + 软投票
11. `exp11_weighted_sampler_synthintel_alpha07`
    `weighted_sampler` + 合成威胁情报 + 加权平均 `alpha=0.7`

## 4. 当前关键结果

### 4.1 纯深度学习模型

- 最优纯深度学习准确率与 `macro_f1`：
  `exp05_ablation_concat_no_attention`
  - `dl_accuracy = 0.965346`
  - `dl_balanced_accuracy = 0.702475`
  - `dl_f1_macro = 0.747809`
- 最优纯深度学习 `balanced_accuracy`：
  `exp04_weighted_sampler_ce`
  - `dl_accuracy = 0.899445`
  - `dl_balanced_accuracy = 0.840404`
  - `dl_f1_macro = 0.546269`

### 4.2 威胁情报融合后

- 最优融合后准确率与 `macro_f1`：
  `exp10_concat_synthintel_soft_voting`
  - `fused_accuracy = 0.999402`
  - `fused_balanced_accuracy = 0.773844`
  - `fused_f1_macro = 0.798046`
- 最优融合后 `balanced_accuracy`：
  `exp11_weighted_sampler_synthintel_alpha07`
  - `fused_accuracy = 0.899505`
  - `fused_balanced_accuracy = 0.888111`
  - `fused_f1_macro = 0.552334`

## 5. 论文可直接使用的结论

1. 当前实现中，简单拼接优于交叉注意力。
   `exp05` 相比 `exp01` 在保持准确率的同时明显提高了 `macro_f1`，说明当前 Cross-Attention 结构并未带来预期收益。

2. 多模态优于任一单模态。
   `traffic_only` 与 `log_only` 都弱于 `concat` 多模态，说明流量与日志确实存在互补信息。

3. 不平衡优化里，`WeightedRandomSampler` 最稳定。
   `exp04` 的 `balanced_accuracy=0.840404`，明显优于类权重与 Focal+类权重方案。

4. 仅提高威胁情报接入比例并不会自动增益。
   `exp09` 与 `exp05` 的结果完全一致，说明在当前模型输出下，`weighted_average(alpha=0.7)` 还不足以改变最终决策。

5. 决策层融合权重非常关键。
   `exp10` 显示 `soft_voting` 后结果显著提升；后续融合扫参进一步确认它与 `weighted_average(alpha=0.5)` 完全一致，因此收益主要来自“降低 DL 权重到 0.5”。

6. 不平衡优化与威胁情报融合可以叠加改善少数类召回。
   `exp11` 将 `balanced_accuracy` 从 `0.840404` 提高到 `0.888111`，虽然整体准确率没有像 `exp10` 那样跃升，但在少数类检测能力上更强。

## 6. 融合扫参与附加分析

基于已有最佳 checkpoint 做复用评估，不重复训练，仅对决策层融合做系统对照。

### 6.1 `exp05` + synthetic v1

输出目录：
`/root/autodl-tmp/cc-bishe-experiments/_fusion_studies/exp05_concat_synthintel_v1`

关键结论：

- `soft_voting` 与 `weighted_average(alpha=0.5)` 结果完全一致。
- `alpha=0.6` 已能带来明显增益，`alpha=0.5` 达到当前最优。
- `adaptive_weighted_average` 没有带来额外收益。
- `dempster_shafer` 略优于 `soft_voting`，但提升幅度非常小。

### 6.2 `exp04` + synthetic v1

输出目录：
`/root/autodl-tmp/cc-bishe-experiments/_fusion_studies/exp04_weighted_sampler_synthintel_v1`

关键结论：

- `dempster_shafer`、`soft_voting`、`weighted_average(alpha=0.5~0.7)` 都优于 `dl_only`。
- 最优 `balanced_accuracy` 约为 `0.888178`，说明威胁情报对少数类召回提升是成立的。
- `adaptive_weighted_average` 仍然没有额外收益。

### 6.3 IOC 覆盖率风险说明

融合扫参与分析显示，合成 IOC 并非覆盖全部测试样本，但对恶意类具有极高覆盖率：

- `synthetic_train_v1`
- `synthetic_train_v2_top8ports`
- `synthetic_train_v3_ip_only`

在当前数据切分方式下，对测试集的总体 `match_rate` 分别约为：

- `v1 = 0.1346`
- `v2 = 0.1339`
- `v3 = 0.1292`

但它们对攻击类样本的命中率几乎均为 `1.0`，对 Benign 的命中率仅约 `0.7%` 到 `1.3%`。

这说明：

1. 数据集中的攻击 IP / 目的 IP / 高风险端口在训练集与测试集之间高度重复。
2. 基于训练集生成的模拟 IOC 更像是在模拟“持久存在的外部威胁情报”，而不是完全独立的新知识源。
3. 因此 `exp10` 这类极强结果可以作为“上界实验”或“威胁情报有效性示范”，但正文需要明确说明其潜在乐观性，不能直接解释为完全无泄漏的泛化能力提升。

建议在论文中这样表述：

- 本文构造的合成威胁情报用于模拟外部 IOC 服务在已知恶意基础设施长期存在时的增强效果。
- 由于原始数据集采用随机划分，指示器在训练/测试间存在重复，融合结果应视为带有先验共享条件下的性能上界。

## 7. 当前推荐写法

如果正文强调不同目标，可分别选用：

- 强调整体检测性能：
  `exp05_ablation_concat_no_attention`
- 强调不平衡数据优化：
  `exp04_weighted_sampler_ce`
- 强调威胁情报融合增益上界：
  `exp10_concat_synthintel_soft_voting`
- 强调少数类检测与情报融合协同：
  `exp11_weighted_sampler_synthintel_alpha07`

## 8. 复现命令

单实验：

```bash
cd /root/bishe/cc-bishe
/root/miniconda3/bin/python run_experiment.py \
  --config config.experiment_base.yaml \
  --name exp05_ablation_concat_no_attention \
  --experiments-root /root/autodl-tmp/cc-bishe-experiments
```

整组实验：

```bash
cd /root/bishe/cc-bishe
/root/miniconda3/bin/python run_experiment_matrix.py \
  --config config.experiment_base.yaml \
  --experiments-root /root/autodl-tmp/cc-bishe-experiments
```

重新汇总：

```bash
cd /root/bishe/cc-bishe
/root/miniconda3/bin/python summarize_experiments.py \
  --experiments-root /root/autodl-tmp/cc-bishe-experiments
```

生成合成威胁情报：

```bash
cd /root/bishe/cc-bishe
/root/miniconda3/bin/python generate_synthetic_threat_intel.py \
  --config config.experiment_base.yaml \
  --output-dir /root/autodl-tmp/cc-bishe-threat-intel/synthetic_train_v1 \
  --max-port-entries 64
```

融合扫参与分析：

```bash
cd /root/bishe/cc-bishe
/root/miniconda3/bin/python analyze_fusion_strategies.py \
  --source-experiment-dir /root/autodl-tmp/cc-bishe-experiments/exp05_ablation_concat_no_attention \
  --threat-intel-dir /root/autodl-tmp/cc-bishe-threat-intel/synthetic_train_v1 \
  --output-dir /root/autodl-tmp/cc-bishe-experiments/_fusion_studies/exp05_concat_synthintel_v1
```
