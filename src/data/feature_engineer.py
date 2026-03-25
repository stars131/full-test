"""特征工程模块 - 将CIC-IDS-2018特征分为流量特征和日志/上下文特征两个模态

CIC-IDS-2018数据集的特征天然分为：
- 流量模态（Traffic）：流级别统计特征（包大小、字节数、速率等）
- 日志/上下文模态（Log/Context）：时间序列特征（IAT、delta、active/idle等）
"""

from typing import Tuple, List
import pandas as pd
import numpy as np


# CIC-IDS-2018 流量模态特征（流级统计：包大小、字节、头部、速率、标志）
TRAFFIC_FEATURES = [
    # 基础流统计
    "duration",
    "packets_count",
    "fwd_packets_count",
    "bwd_packets_count",
    "total_payload_bytes",
    "fwd_total_payload_bytes",
    "bwd_total_payload_bytes",
    # 载荷字节统计
    "payload_bytes_max",
    "payload_bytes_min",
    "payload_bytes_mean",
    "payload_bytes_std",
    "payload_bytes_median",
    "fwd_payload_bytes_max",
    "fwd_payload_bytes_min",
    "fwd_payload_bytes_mean",
    "fwd_payload_bytes_std",
    "fwd_payload_bytes_median",
    "bwd_payload_bytes_max",
    "bwd_payload_bytes_min",
    "bwd_payload_bytes_mean",
    "bwd_payload_bytes_std",
    "bwd_payload_bytes_median",
    # 头部字节统计
    "total_header_bytes",
    "max_header_bytes",
    "min_header_bytes",
    "mean_header_bytes",
    "fwd_total_header_bytes",
    "fwd_max_header_bytes",
    "fwd_min_header_bytes",
    "fwd_mean_header_bytes",
    "bwd_total_header_bytes",
    "bwd_max_header_bytes",
    "bwd_min_header_bytes",
    "bwd_mean_header_bytes",
    # 段大小
    "fwd_avg_segment_size",
    "bwd_avg_segment_size",
    "avg_segment_size",
    # 窗口
    "fwd_init_win_bytes",
    "bwd_init_win_bytes",
    # 速率
    "bytes_rate",
    "fwd_bytes_rate",
    "bwd_bytes_rate",
    "packets_rate",
    "fwd_packets_rate",
    "bwd_packets_rate",
    "down_up_rate",
    # Bulk统计
    "avg_fwd_bytes_per_bulk",
    "avg_fwd_packets_per_bulk",
    "avg_fwd_bulk_rate",
    "avg_bwd_bytes_per_bulk",
    "avg_bwd_packets_bulk_rate",
    "avg_bwd_bulk_rate",
    # TCP标志计数
    "fin_flag_counts",
    "psh_flag_counts",
    "urg_flag_counts",
    "ece_flag_counts",
    "syn_flag_counts",
    "ack_flag_counts",
    "cwr_flag_counts",
    "rst_flag_counts",
    "fwd_fin_flag_counts",
    "fwd_psh_flag_counts",
    "fwd_urg_flag_counts",
    "fwd_syn_flag_counts",
    "fwd_ack_flag_counts",
    "fwd_rst_flag_counts",
    "bwd_fin_flag_counts",
    "bwd_psh_flag_counts",
    "bwd_urg_flag_counts",
    "bwd_syn_flag_counts",
    "bwd_ack_flag_counts",
    "bwd_rst_flag_counts",
]

# CIC-IDS-2018 日志/上下文模态特征（时间序列：IAT、delta time/len、active/idle）
LOG_FEATURES = [
    # 包间到达时间 (IAT)
    "packets_IAT_mean",
    "packet_IAT_std",
    "packet_IAT_max",
    "packet_IAT_min",
    "packet_IAT_total",
    "packets_IAT_median",
    "fwd_packets_IAT_mean",
    "fwd_packets_IAT_std",
    "fwd_packets_IAT_max",
    "fwd_packets_IAT_min",
    "fwd_packets_IAT_total",
    "fwd_packets_IAT_median",
    "bwd_packets_IAT_mean",
    "bwd_packets_IAT_std",
    "bwd_packets_IAT_max",
    "bwd_packets_IAT_min",
    "bwd_packets_IAT_total",
    "bwd_packets_IAT_median",
    # Active/Idle时间
    "active_min",
    "active_max",
    "active_mean",
    "active_std",
    "active_median",
    "idle_min",
    "idle_max",
    "idle_mean",
    "idle_std",
    "idle_median",
    # 子流
    "subflow_fwd_packets",
    "subflow_bwd_packets",
    "subflow_fwd_bytes",
    "subflow_bwd_bytes",
    # 握手与delta time
    "delta_start",
    "handshake_duration",
    "mean_packets_delta_time",
    "std_packets_delta_time",
    "median_packets_delta_time",
    "mean_bwd_packets_delta_time",
    "std_bwd_packets_delta_time",
    "median_bwd_packets_delta_time",
    "mean_fwd_packets_delta_time",
    "std_fwd_packets_delta_time",
    "median_fwd_packets_delta_time",
    # Delta len（包大小差异序列统计）
    "mean_packets_delta_len",
    "std_packets_delta_len",
    "median_packets_delta_len",
    "mean_bwd_packets_delta_len",
    "std_bwd_packets_delta_len",
    "median_bwd_packets_delta_len",
    "mean_fwd_packets_delta_len",
    "std_fwd_packets_delta_len",
    "median_fwd_packets_delta_len",
    # 头部delta len
    "mean_header_bytes_delta_len",
    "std_header_bytes_delta_len",
    "median_header_bytes_delta_len",
    "mean_bwd_header_bytes_delta_len",
    "std_bwd_header_bytes_delta_len",
    "median_bwd_header_bytes_delta_len",
    "mean_fwd_header_bytes_delta_len",
    "std_fwd_header_bytes_delta_len",
    "median_fwd_header_bytes_delta_len",
    # 载荷delta len
    "mean_payload_bytes_delta_len",
    "std_payload_bytes_delta_len",
    "median_payload_bytes_delta_len",
    "mean_bwd_payload_bytes_delta_len",
    "std_bwd_payload_bytes_delta_len",
    "median_bwd_payload_bytes_delta_len",
    "mean_fwd_payload_bytes_delta_len",
    "std_fwd_payload_bytes_delta_len",
    "median_fwd_payload_bytes_delta_len",
    # 标志百分比
    "fin_flag_percentage_in_total",
    "psh_flag_percentage_in_total",
    "syn_flag_percentage_in_total",
    "ack_flag_percentage_in_total",
    "rst_flag_percentage_in_total",
]

# 非特征列（需要排除）
NON_FEATURE_COLS = [
    "flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port",
    "protocol", "label", "handshake_state",
]


class FeatureEngineer:
    """将CIC-IDS-2018数据特征分为流量模态和日志模态"""

    def __init__(self):
        self.traffic_features: List[str] = []
        self.log_features: List[str] = []

    def fit(self, data: pd.DataFrame) -> "FeatureEngineer":
        """根据数据集实际列名匹配特征分组"""
        available_cols = set(data.columns)

        self.traffic_features = [
            f for f in TRAFFIC_FEATURES if f in available_cols
        ]
        self.log_features = [f for f in LOG_FEATURES if f in available_cols]

        # 将未匹配的数值特征分配到合适的模态
        matched = set(self.traffic_features + self.log_features)
        numeric_cols = set(data.select_dtypes(include=[np.number]).columns)
        exclude = set(NON_FEATURE_COLS)

        for col in sorted(numeric_cols - matched - exclude):
            if any(kw in col.lower() for kw in [
                "iat", "delta", "active", "idle", "subflow", "handshake"
            ]):
                self.log_features.append(col)
            else:
                self.traffic_features.append(col)

        # 过滤掉实际包含非数值数据的列
        numeric_cols = set(data.select_dtypes(include=[np.number]).columns)
        self.traffic_features = [
            f for f in self.traffic_features if f in numeric_cols
        ]
        self.log_features = [
            f for f in self.log_features if f in numeric_cols
        ]

        print(f"流量特征数量: {len(self.traffic_features)}")
        print(f"日志/上下文特征数量: {len(self.log_features)}")

        return self

    def transform(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """提取两个模态的特征矩阵"""
        # 确保只使用数值列
        traffic = data[self.traffic_features].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).values.astype(np.float32)
        log = data[self.log_features].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).values.astype(np.float32)
        return traffic, log

    def get_feature_dims(self) -> Tuple[int, int]:
        """返回两个模态的特征维度"""
        return len(self.traffic_features), len(self.log_features)
