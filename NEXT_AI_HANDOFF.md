# NEXT AI Handoff

本文件用于给下一次接手本项目的 AI / 开发者快速建立上下文，避免重复摸索。

## 1. 项目目标

本项目是毕业设计实验项目，论文主题是：

> 基于多源数据融合的网络攻击检测方法研究

当前研究主线：

- 流量特征 + 日志特征 + 威胁情报三类信息融合
- 基于深度学习的多分类网络攻击检测
- 解决数据不平衡问题
- 在可复现、可追踪的实验流水线基础上持续扩展对比实验

## 2. 当前最重要的结论

### 纯深度学习模型

- 当前最优纯 DL 方案：
  `exp05_ablation_concat_no_attention`
  - `dl_accuracy = 0.965346`
  - `dl_balanced_accuracy = 0.702475`
  - `dl_f1_macro = 0.747809`

- 当前最优少数类召回方案：
  `exp04_weighted_sampler_ce`
  - `dl_accuracy = 0.899445`
  - `dl_balanced_accuracy = 0.840404`
  - `dl_f1_macro = 0.546269`

### 威胁情报融合后

- 当前最优融合后准确率 / macro-F1：
  `exp10_concat_synthintel_soft_voting`
  - `fused_accuracy = 0.999402`
  - `fused_balanced_accuracy = 0.773844`
  - `fused_f1_macro = 0.798046`

- 当前最优融合后 balanced accuracy：
  `exp11_weighted_sampler_synthintel_alpha07`
  - `fused_accuracy = 0.899505`
  - `fused_balanced_accuracy = 0.888111`
  - `fused_f1_macro = 0.552334`

## 3. 结构结论

- 当前实现里，`concat` 融合优于 `cross_attention`
- `traffic_only` 和 `log_only` 都弱于多模态
- 不平衡优化中，`WeightedRandomSampler` 最稳
- `class_weight` 与 `focal + class_weight` 会明显伤害整体准确率
- 威胁情报增强是否有效，强依赖决策层融合策略

## 4. 威胁情报相关结论

目前真实外部 threat intel API 还没稳定接入，因此使用本地生成的合成情报代替。

已有生成脚本：

- `generate_synthetic_threat_intel.py`

已有版本：

- `synthetic_train_v1`
- `synthetic_train_v2_top8ports`
- `synthetic_train_v3_ip_only`

融合扫参与分析结果在：

- `/root/autodl-tmp/cc-bishe-experiments/_fusion_studies`

重要发现：

- `soft_voting` 与 `weighted_average(alpha=0.5)` 在当前实现中等价
- `adaptive_weighted_average` 目前没有带来额外收益
- 合成情报对总体测试集命中率约 `13%`，但对攻击类几乎 `100%` 命中
- 这意味着当前结果更适合作为“共享 IOC 先验条件下的增强上界”

## 5. 路径约定

代码仓根目录：

- `/root/bishe/cc-bishe`

实验目录：

- `/root/autodl-tmp/cc-bishe-experiments`

数据缓存：

- `/root/autodl-tmp/cc-bishe-cache/full_v1/dataset_cache.pt`

合成威胁情报：

- `/root/autodl-tmp/cc-bishe-threat-intel`

论文素材：

- `/root/bishe/cc-bishe/thesis_materials`

资料保存仓本地副本：

- `/root/bishe-ziliao-baocun`

## 6. 关键脚本

- `run_experiment.py`
  单实验统一入口
- `run_experiment_matrix.py`
  批量实验矩阵
- `summarize_experiments.py`
  汇总实验结果
- `generate_synthetic_threat_intel.py`
  生成模拟情报库
- `analyze_fusion_strategies.py`
  复用已有 checkpoint 做决策层融合扫参

## 7. 可复现实验记录

每个实验目录通常包含：

- `resolved_config.yaml`
- `metadata.json`
- `run_status.json`
- `logs/train.log`
- `logs/evaluate.log`
- `artifacts/train_history.json`
- `artifacts/train_summary.json`
- `artifacts/evaluation_metrics.json`
- `reports/classification_report_*.txt`
- `figures/*.png`
- `checkpoints/best_model.pth`
- `processed/preprocessor.pkl`

如果需要继续实验，优先保留这些文件，不要误删。

## 8. 已知限制

- `dataset_cache.pt` 很大，未上传 GitHub，但本地仍保存在 `/root/autodl-tmp`
- 当前结果依赖随机切分，攻击基础设施在训练/测试间存在重叠
- 因此 threat intel 融合提升不能直接视为完全独立外部知识带来的泛化提升

## 9. 下一步最值得做的事

优先级建议如下：

1. 做更严格的数据划分
   例如按攻击基础设施、时间段、IP/端口分组切分，降低训练测试泄漏

2. 构造更真实的 threat intel 设置
   例如只保留训练集外来源、引入 API wrapper、或人为设定时效与可信度

3. 在 `exp05` 和 `exp04` 这两条线上继续扩展
   - `exp05` 代表当前最强结构
   - `exp04` 代表当前最强少数类优化方案

4. 若论文优先落地
   直接以 `thesis_materials/main.tex` 为底稿迁移到学校模板

## 10. 启动建议

如果是下一次实验 AI：

- 先看 `EXPERIMENTS.md`
- 再看 `/root/autodl-tmp/cc-bishe-experiments/summary.md`
- 然后确认 `dataset_cache.pt` 是否还在
- 最后决定是否继续新增实验矩阵，还是先修正数据划分策略

## 11. 一句总结

当前阶段最可靠的论文结论是：

- 多模态优于单模态
- 拼接融合优于当前实现的交叉注意力
- WeightedRandomSampler 是当前最稳的不平衡优化方法
- 威胁情报融合可以显著提升结果，但当前设定更接近“已知 IOC 上界增强”，后续必须补更严格验证
