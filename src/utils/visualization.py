"""可视化模块 - 训练曲线、混淆矩阵、评估结果可视化"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 非交互式后端

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class Visualizer:
    """可视化工具"""

    def __init__(
        self,
        output_dir: str = "results/figures",
        dpi: int = 150,
        figsize: tuple = (10, 8),
    ):
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize
        os.makedirs(output_dir, exist_ok=True)

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = [
            "SimHei", "DejaVu Sans", "Arial"
        ]
        plt.rcParams["axes.unicode_minus"] = False

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accs: List[float],
        val_accs: List[float],
        save_name: str = "training_curves.png",
    ):
        """绘制训练曲线（loss和accuracy）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        epochs = range(1, len(train_losses) + 1)

        # Loss曲线
        ax1.plot(epochs, train_losses, "b-", label="Train Loss")
        ax1.plot(epochs, val_losses, "r-", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy曲线
        ax2.plot(epochs, train_accs, "b-", label="Train Acc")
        ax2.plot(epochs, val_accs, "r-", label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"训练曲线已保存: {save_path}")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_name: str = "confusion_matrix.png",
        normalize: bool = True,
    ):
        """绘制混淆矩阵"""
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_display = cm.astype(float) / row_sums
            fmt = ".2f"
            title = "Normalized Confusion Matrix"
        else:
            cm_display = cm
            fmt = "d"
            title = "Confusion Matrix"

        fig, ax = plt.subplots(figsize=self.figsize)

        if HAS_SEABORN:
            sns.heatmap(
                cm_display,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
            )
        else:
            im = ax.imshow(cm_display, cmap="Blues")
            plt.colorbar(im, ax=ax)
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    val = cm_display[i, j]
                    text = f"{val:{fmt}}" if isinstance(val, float) else str(val)
                    ax.text(j, i, text, ha="center", va="center")
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"混淆矩阵已保存: {save_path}")

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_name: str = "metrics_comparison.png",
    ):
        """对比不同方法的评估指标

        Args:
            metrics_dict: {"方法名": {"accuracy": ..., "f1": ...}}
        """
        methods = list(metrics_dict.keys())
        metric_names = ["accuracy", "precision", "recall", "f1"]

        x = np.arange(len(metric_names))
        width = 0.8 / len(methods)

        fig, ax = plt.subplots(figsize=self.figsize)

        for i, method in enumerate(methods):
            values = [metrics_dict[method].get(m, 0) for m in metric_names]
            offset = (i - len(methods) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=method)
            # 在柱子上标注数值
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xlabel("Metrics")
        ax.set_ylabel("Score")
        ax.set_title("Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"指标对比图已保存: {save_path}")

    def plot_label_distribution(
        self,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: str = "label_distribution.png",
    ):
        """绘制标签分布图"""
        unique, counts = np.unique(labels, return_counts=True)

        if class_names:
            names = [class_names[i] for i in unique]
        else:
            names = [str(u) for u in unique]

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.barh(names, counts, color="steelblue")

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Count")
        ax.set_title("Label Distribution")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"标签分布图已保存: {save_path}")
