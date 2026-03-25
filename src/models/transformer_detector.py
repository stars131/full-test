"""Transformer检测模型 - 基于多模态融合的网络攻击检测"""

import torch
import torch.nn as nn
from src.data.fusion import MultiModalFusion


class TransformerDetector(nn.Module):
    """基于Transformer的网络攻击检测模型

    架构：
    输入 → [流量特征, 日志特征]
         → Cross-Attention融合
         → Transformer Encoder
         → Classification Head
         → 攻击类别概率
    """

    def __init__(
        self,
        traffic_dim: int,
        log_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes

        # 多模态融合模块
        self.fusion = MultiModalFusion(
            traffic_dim=traffic_dim,
            log_dim=log_dim,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification Head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self, traffic: torch.Tensor, log: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            traffic: (batch, traffic_dim) 流量特征
            log: (batch, log_dim) 日志特征

        Returns:
            (batch, num_classes) 攻击类别logits
        """
        # 特征融合
        fused = self.fusion(traffic, log)  # (batch, d_model)

        # Transformer编码 (需要序列维度)
        fused = fused.unsqueeze(1)  # (batch, 1, d_model)
        encoded = self.transformer_encoder(fused)  # (batch, 1, d_model)

        # Global Average Pooling (这里序列长度为1，等价于squeeze)
        pooled = encoded.mean(dim=1)  # (batch, d_model)

        # 分类
        logits = self.classifier(pooled)  # (batch, num_classes)

        return logits

    def predict_proba(
        self, traffic: torch.Tensor, log: torch.Tensor
    ) -> torch.Tensor:
        """返回softmax概率"""
        logits = self.forward(traffic, log)
        return torch.softmax(logits, dim=-1)
