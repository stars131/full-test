"""预处理模块 - 归一化、编码、数据集划分"""

import os
import pickle
from typing import Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class NIDSDataset(Dataset):
    """网络入侵检测数据集"""

    def __init__(
        self,
        traffic_features: np.ndarray,
        log_features: np.ndarray,
        labels: np.ndarray,
    ):
        self.traffic = torch.FloatTensor(traffic_features)
        self.log = torch.FloatTensor(log_features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.traffic[idx], self.log[idx], self.labels[idx]


class Preprocessor:
    """数据预处理器：归一化、编码、划分"""

    def __init__(self):
        self.traffic_scaler = StandardScaler()
        self.log_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.num_classes: int = 0
        self.class_names: list = []

    def fit_transform(
        self,
        traffic: np.ndarray,
        log: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_seed: int = 42,
    ) -> Dict[str, NIDSDataset]:
        """对数据进行归一化、编码和划分

        Args:
            traffic: 流量特征矩阵
            log: 日志特征矩阵
            labels: 原始标签数组
            test_size: 测试集比例
            val_size: 验证集比例
            random_seed: 随机种子

        Returns:
            包含 'train', 'val', 'test' 的数据集字典
        """
        # 编码标签
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        self.class_names = list(self.label_encoder.classes_)
        print(f"类别数量: {self.num_classes}")
        print(f"类别名称: {self.class_names}")

        # 划分数据集 8:1:1 (保留原始索引)
        indices = np.arange(len(encoded_labels))
        (
            traffic_train, traffic_temp,
            log_train, log_temp,
            y_train, y_temp,
            idx_train, idx_temp,
        ) = train_test_split(
            traffic, log, encoded_labels, indices,
            test_size=test_size + val_size,
            random_state=random_seed,
            stratify=encoded_labels,
        )

        val_ratio = val_size / (test_size + val_size)
        (
            traffic_val, traffic_test,
            log_val, log_test,
            y_val, y_test,
            idx_val, idx_test,
        ) = train_test_split(
            traffic_temp, log_temp, y_temp, idx_temp,
            test_size=1 - val_ratio,
            random_state=random_seed,
            stratify=y_temp,
        )

        # 保存划分索引
        self.train_indices = idx_train
        self.val_indices = idx_val
        self.test_indices = idx_test

        print(f"训练集: {len(y_train)}, 验证集: {len(y_val)}, 测试集: {len(y_test)}")

        # 归一化（仅在训练集上fit）
        traffic_train = self.traffic_scaler.fit_transform(traffic_train)
        traffic_val = self.traffic_scaler.transform(traffic_val)
        traffic_test = self.traffic_scaler.transform(traffic_test)

        log_train = self.log_scaler.fit_transform(log_train)
        log_val = self.log_scaler.transform(log_val)
        log_test = self.log_scaler.transform(log_test)

        return {
            "train": NIDSDataset(traffic_train, log_train, y_train),
            "val": NIDSDataset(traffic_val, log_val, y_val),
            "test": NIDSDataset(traffic_test, log_test, y_test),
        }

    def create_dataloaders(
        self,
        datasets: Dict[str, NIDSDataset],
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> Dict[str, TorchDataLoader]:
        """创建PyTorch DataLoader"""
        return {
            "train": TorchDataLoader(
                datasets["train"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
            ),
            "val": TorchDataLoader(
                datasets["val"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            "test": TorchDataLoader(
                datasets["test"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        }

    def save(self, path: str):
        """保存预处理器状态"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "traffic_scaler": self.traffic_scaler,
            "log_scaler": self.log_scaler,
            "label_encoder": self.label_encoder,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """加载预处理器状态"""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.traffic_scaler = state["traffic_scaler"]
        self.log_scaler = state["log_scaler"]
        self.label_encoder = state["label_encoder"]
        self.num_classes = state["num_classes"]
        self.class_names = state["class_names"]

    def inverse_label(self, encoded: np.ndarray) -> np.ndarray:
        """将编码后的标签还原"""
        return self.label_encoder.inverse_transform(encoded)
