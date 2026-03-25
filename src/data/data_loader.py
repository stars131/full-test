"""数据加载模块 - 加载CIC-IDS-2018 CSV数据"""

import os
import glob
import pandas as pd
import numpy as np


class DataLoader:
    """加载CIC-IDS-2018处理后的CSV数据集"""

    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir
        self.label_col = "label"

    def load(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """加载CSV文件，支持多文件合并"""
        csv_files = sorted(glob.glob(os.path.join(self.raw_dir, file_pattern)))
        if not csv_files:
            raise FileNotFoundError(
                f"在 {self.raw_dir} 中未找到匹配 '{file_pattern}' 的CSV文件"
            )

        dfs = []
        for f in csv_files:
            print(f"加载文件: {os.path.basename(f)}")
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        print(f"合并后数据集大小: {data.shape}")

        # 自动检测标签列
        self.label_col = self._detect_label_col(data)
        print(f"检测到标签列: {self.label_col}")

        # 处理缺失值和无穷值
        data = self._clean_data(data)

        return data

    def _detect_label_col(self, data: pd.DataFrame) -> str:
        """自动检测标签列"""
        candidates = ["label", "Label", "LABEL", "attack_cat", "Attack"]
        for col in candidates:
            if col in data.columns:
                return col
        for col in data.columns:
            if "label" in col.lower():
                return col
        raise ValueError(f"未找到标签列，可用列: {list(data.columns)}")

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值和无穷值"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        nan_count = data[numeric_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"填充 {nan_count} 个缺失/无穷值为0")
            data[numeric_cols] = data[numeric_cols].fillna(0)
        return data

    def get_label_distribution(self, data: pd.DataFrame) -> pd.Series:
        """获取标签分布"""
        return data[self.label_col].value_counts()

    def get_network_indicators(self, data: pd.DataFrame) -> dict:
        """提取网络指标（IP、端口）用于威胁情报查询"""
        result = {}
        if "src_ip" in data.columns:
            result["src_ips"] = data["src_ip"].tolist()
        if "dst_ip" in data.columns:
            result["dst_ips"] = data["dst_ip"].tolist()
        if "dst_port" in data.columns:
            result["dst_ports"] = data["dst_port"].astype(int).tolist()
        return result
