"""预处理模块 - 归一化、编码、数据集划分"""

import os
import pickle
from typing import Dict

import numpy as np
import torch
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    Dataset,
    WeightedRandomSampler,
)
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
        self.traffic = torch.as_tensor(traffic_features, dtype=torch.float32)
        self.log = torch.as_tensor(log_features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

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
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.train_labels = None
        self.val_labels = None
        self.test_labels = None
        self.train_class_counts: list[int] = []
        self.val_class_counts: list[int] = []
        self.test_class_counts: list[int] = []
        self.split_sizes: Dict[str, int] = {}

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
        self.train_labels = y_train
        self.val_labels = y_val
        self.test_labels = y_test
        self.train_class_counts = np.bincount(
            y_train, minlength=self.num_classes
        ).tolist()
        self.val_class_counts = np.bincount(
            y_val, minlength=self.num_classes
        ).tolist()
        self.test_class_counts = np.bincount(
            y_test, minlength=self.num_classes
        ).tolist()
        self.split_sizes = {
            "train": len(y_train),
            "val": len(y_val),
            "test": len(y_test),
        }

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
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        use_weighted_sampler: bool = False,
    ) -> Dict[str, TorchDataLoader]:
        """创建PyTorch DataLoader"""
        train_sampler = None
        shuffle = True

        if use_weighted_sampler:
            class_counts = np.array(self.train_class_counts, dtype=np.float64)
            class_counts = np.where(class_counts == 0, 1.0, class_counts)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[self.train_labels]
            train_sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

        common_loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            common_loader_kwargs["persistent_workers"] = persistent_workers
            common_loader_kwargs["prefetch_factor"] = prefetch_factor

        return {
            "train": TorchDataLoader(
                datasets["train"],
                shuffle=shuffle,
                sampler=train_sampler,
                drop_last=False,
                **common_loader_kwargs,
            ),
            "val": TorchDataLoader(
                datasets["val"],
                shuffle=False,
                **common_loader_kwargs,
            ),
            "test": TorchDataLoader(
                datasets["test"],
                shuffle=False,
                **common_loader_kwargs,
            ),
        }

    def save(self, path: str):
        """保存预处理器状态"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = self.get_state()
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """加载预处理器状态"""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.load_state(state)

    def get_state(self) -> dict:
        """导出预处理器状态"""
        return {
            "traffic_scaler": self.traffic_scaler,
            "log_scaler": self.log_scaler,
            "label_encoder": self.label_encoder,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "train_labels": self.train_labels,
            "val_labels": self.val_labels,
            "test_labels": self.test_labels,
            "train_class_counts": self.train_class_counts,
            "val_class_counts": self.val_class_counts,
            "test_class_counts": self.test_class_counts,
            "split_sizes": self.split_sizes,
        }

    def load_state(self, state: dict):
        """从字典加载预处理器状态"""
        self.traffic_scaler = state["traffic_scaler"]
        self.log_scaler = state["log_scaler"]
        self.label_encoder = state["label_encoder"]
        self.num_classes = state["num_classes"]
        self.class_names = state["class_names"]
        self.train_indices = state.get("train_indices")
        self.val_indices = state.get("val_indices")
        self.test_indices = state.get("test_indices")
        self.train_labels = state.get("train_labels")
        self.val_labels = state.get("val_labels")
        self.test_labels = state.get("test_labels")
        self.train_class_counts = state.get("train_class_counts", [])
        self.val_class_counts = state.get("val_class_counts", [])
        self.test_class_counts = state.get("test_class_counts", [])
        self.split_sizes = state.get("split_sizes", {})

    def inverse_label(self, encoded: np.ndarray) -> np.ndarray:
        """将编码后的标签还原"""
        return self.label_encoder.inverse_transform(encoded)
