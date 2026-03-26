"""数据缓存 - 复用全量特征工程与数据集划分结果"""

from __future__ import annotations

import os

import pandas as pd
import torch

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import NIDSDataset, Preprocessor


def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "dataset_cache.pt")


def _build_payload(
    raw_dir: str,
    file_pattern: str,
    processed_dir: str,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> tuple[dict, DataLoader]:
    loader = DataLoader(raw_dir)
    data = loader.load(file_pattern)

    engineer = FeatureEngineer()
    engineer.fit(data)
    traffic_features, log_features = engineer.transform(data)
    labels = data[loader.label_col].values

    preprocessor = Preprocessor()
    datasets = preprocessor.fit_transform(
        traffic_features,
        log_features,
        labels,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
    )

    os.makedirs(processed_dir, exist_ok=True)
    preprocessor.save(os.path.join(processed_dir, "preprocessor.pkl"))

    payload = {
        "preprocessor_state": preprocessor.get_state(),
        "traffic_dim": traffic_features.shape[1],
        "log_dim": log_features.shape[1],
        "label_distribution": loader.get_label_distribution(data).to_dict(),
        "network_info": {},
        "datasets": {
            split: {
                "traffic": datasets[split].traffic,
                "log": datasets[split].log,
                "labels": datasets[split].labels,
            }
            for split in ("train", "val", "test")
        },
    }

    if "src_ip" in data.columns:
        payload["network_info"]["src_ip"] = data["src_ip"].astype(str).tolist()
    if "dst_ip" in data.columns:
        payload["network_info"]["dst_ip"] = data["dst_ip"].astype(str).tolist()
    if "dst_port" in data.columns:
        payload["network_info"]["dst_port"] = (
            pd.to_numeric(data["dst_port"], errors="coerce")
            .fillna(0)
            .astype(int)
            .tolist()
        )
    return payload, loader


def load_or_prepare_datasets(data_config: dict) -> tuple[dict, Preprocessor, dict]:
    """加载或构建缓存后的数据集"""
    cache_dir = data_config.get("cache_dir")
    cache_path = _cache_path(cache_dir) if cache_dir else None

    if cache_path and os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"加载缓存数据集: {cache_path}")
    else:
        payload, _ = _build_payload(
            raw_dir=data_config["raw_dir"],
            file_pattern=data_config.get("file_pattern", "*.csv"),
            processed_dir=data_config["processed_dir"],
            test_size=data_config.get("test_size", 0.1),
            val_size=data_config.get("val_size", 0.1),
            random_seed=data_config.get("random_seed", 42),
        )
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(payload, cache_path)
            print(f"缓存数据集已保存: {cache_path}")

    preprocessor = Preprocessor()
    preprocessor.load_state(payload["preprocessor_state"])

    datasets = {
        split: NIDSDataset(
            payload["datasets"][split]["traffic"],
            payload["datasets"][split]["log"],
            payload["datasets"][split]["labels"],
        )
        for split in ("train", "val", "test")
    }
    metadata = {
        "traffic_dim": payload["traffic_dim"],
        "log_dim": payload["log_dim"],
        "label_distribution": payload["label_distribution"],
        "network_info": payload.get("network_info", {}),
    }
    return datasets, preprocessor, metadata
