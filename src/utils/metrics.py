"""评估指标模块"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """计算全面的评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        average: 多分类平均方式

    Returns:
        包含各项指标的字典
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average=average, zero_division=0
        ),
    }

    # 同时计算macro指标
    metrics["precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """打印并返回分类报告"""
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print(report)
    return report


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """计算混淆矩阵"""
    return confusion_matrix(y_true, y_pred)
