"""数据加载模块 - 加载CIC-IDS-2018 CSV数据"""

import fnmatch
import glob
import os

import numpy as np
import pandas as pd


class DataLoader:
    """加载CIC-IDS-2018处理后的CSV数据集"""

    def __init__(
        self,
        raw_dir: str,
        full_dataset: bool = False,
        exclude_patterns: list[str] | None = None,
    ):
        self.raw_dir = raw_dir
        self.full_dataset = full_dataset
        self.exclude_patterns = exclude_patterns or []
        self.label_col = "label"
        self.loaded_files: list[str] = []

    def list_csv_files(self, file_pattern: str = "*.csv") -> list[str]:
        csv_files = sorted(glob.glob(os.path.join(self.raw_dir, file_pattern)))
        csv_files = [f for f in csv_files if os.path.isfile(f)]

        excluded = []
        kept = []
        for path in csv_files:
            name = os.path.basename(path)
            if self._should_exclude(name):
                excluded.append(path)
            else:
                kept.append(path)

        if self.full_dataset and excluded:
            names = ", ".join(os.path.basename(f) for f in excluded)
            print(f"全量模式已排除采样文件: {names}")

        if self.full_dataset:
            sampled = [f for f in kept if self._looks_sampled(os.path.basename(f))]
            if sampled:
                names = ", ".join(os.path.basename(f) for f in sampled)
                raise ValueError(f"全量模式禁止使用采样CSV: {names}")

        return kept

    def load(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """加载CSV文件，支持多文件合并"""
        csv_files = self.list_csv_files(file_pattern)
        if not csv_files:
            raise FileNotFoundError(
                f"在 {self.raw_dir} 中未找到匹配 '{file_pattern}' 的CSV文件"
            )

        self.loaded_files = csv_files
        print(f"将加载CSV文件数: {len(csv_files)}")

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

    def _should_exclude(self, file_name: str) -> bool:
        lower_name = file_name.lower()
        if self.full_dataset and self._looks_sampled(lower_name):
            return True
        return any(fnmatch.fnmatch(file_name, pattern) for pattern in self.exclude_patterns)

    def _looks_sampled(self, file_name: str) -> bool:
        return "sample" in file_name.lower() or "sampled" in file_name.lower()

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
