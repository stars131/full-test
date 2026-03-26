"""决策层融合模块 - 融合DL模型输出与威胁情报评分"""

import numpy as np
from typing import Optional


class DecisionFusion:
    """决策层融合：合并深度学习模型与威胁情报的预测结果

    支持三种融合策略：
    - weighted_average: 加权平均法
    - adaptive_weighted_average: 根据信息量自适应调整权重
    - dempster_shafer: Dempster-Shafer证据理论
    - soft_voting: 软投票法
    """

    def __init__(self, strategy: str = "weighted_average", alpha: float = 0.8):
        """
        Args:
            strategy: 融合策略
            alpha: DL模型预测的权重（仅weighted_average使用）
        """
        if strategy not in (
            "weighted_average",
            "adaptive_weighted_average",
            "dempster_shafer",
            "soft_voting",
        ):
            raise ValueError(f"不支持的融合策略: {strategy}")
        self.strategy = strategy
        self.alpha = alpha

    def fuse(
        self,
        dl_probs: np.ndarray,
        threat_scores: np.ndarray,
    ) -> np.ndarray:
        """融合DL预测概率和威胁情报评分

        Args:
            dl_probs: (batch, num_classes) DL模型的softmax输出
            threat_scores: (batch, num_classes) 威胁情报评分

        Returns:
            (batch, num_classes) 融合后的概率分布
        """
        if self.strategy == "weighted_average":
            return self._weighted_average(dl_probs, threat_scores)
        elif self.strategy == "adaptive_weighted_average":
            return self._adaptive_weighted_average(dl_probs, threat_scores)
        elif self.strategy == "dempster_shafer":
            return self._dempster_shafer(dl_probs, threat_scores)
        elif self.strategy == "soft_voting":
            return self._soft_voting(dl_probs, threat_scores)

    def _weighted_average(
        self, dl_probs: np.ndarray, threat_scores: np.ndarray
    ) -> np.ndarray:
        """加权平均法: P_final = α * P_dl + (1-α) * P_threat"""
        fused = self.alpha * dl_probs + (1 - self.alpha) * threat_scores
        # 归一化
        row_sums = fused.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return fused / row_sums

    def _adaptive_weighted_average(
        self, dl_probs: np.ndarray, threat_scores: np.ndarray
    ) -> np.ndarray:
        """根据威胁情报分布尖锐程度自适应调整DL权重

        当威胁情报接近均匀分布时，保持DL输出不被稀释；
        当威胁情报更集中时，逐步降低DL权重，最小降至alpha。
        """
        num_classes = threat_scores.shape[1]
        uniform_max = 1.0 / num_classes
        max_scores = threat_scores.max(axis=1, keepdims=True)
        strength = (max_scores - uniform_max) / max(1e-12, 1.0 - uniform_max)
        strength = np.clip(strength, 0.0, 1.0)

        dl_weight = 1.0 - (1.0 - self.alpha) * strength
        fused = dl_weight * dl_probs + (1.0 - dl_weight) * threat_scores

        row_sums = fused.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return fused / row_sums

    def _dempster_shafer(
        self, dl_probs: np.ndarray, threat_scores: np.ndarray
    ) -> np.ndarray:
        """Dempster-Shafer证据理论融合

        将两个源的概率视为mass函数，使用DS组合规则合并。
        """
        batch_size, num_classes = dl_probs.shape
        results = np.zeros_like(dl_probs)

        for i in range(batch_size):
            m1 = dl_probs[i]  # mass function 1
            m2 = threat_scores[i]  # mass function 2

            # 计算冲突系数 K
            K = 0.0
            combined = np.zeros(num_classes)

            for j in range(num_classes):
                for k in range(num_classes):
                    if j == k:
                        combined[j] += m1[j] * m2[k]
                    else:
                        K += m1[j] * m2[k]

            # 归一化（去除冲突）
            if K < 1.0:
                combined = combined / (1 - K)
            else:
                # 完全冲突时回退到均匀分布
                combined = np.ones(num_classes) / num_classes

            results[i] = combined

        return results

    def _soft_voting(
        self, dl_probs: np.ndarray, threat_scores: np.ndarray
    ) -> np.ndarray:
        """软投票法：简单平均"""
        fused = (dl_probs + threat_scores) / 2.0
        row_sums = fused.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return fused / row_sums

    def predict(
        self,
        dl_probs: np.ndarray,
        threat_scores: np.ndarray,
    ) -> np.ndarray:
        """融合后取argmax得到最终预测类别

        Returns:
            (batch,) 预测类别索引
        """
        fused = self.fuse(dl_probs, threat_scores)
        return np.argmax(fused, axis=1)
