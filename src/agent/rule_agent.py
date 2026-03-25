"""规则引擎Agent - 基于规则模板的检测与解释"""

from typing import Dict, Any, Optional

import numpy as np
import torch

from src.agent.base_agent import BaseAgent
from src.models.transformer_detector import TransformerDetector
from src.models.threat_intel import ThreatIntelScorer
from src.models.decision_fusion import DecisionFusion
from src.data.preprocessor import Preprocessor


class RuleAgent(BaseAgent):
    """规则引擎Agent

    使用Transformer模型进行推理，基于规则模板生成解释。
    """

    def __init__(
        self,
        model: TransformerDetector,
        preprocessor: Preprocessor,
        threat_scorer: ThreatIntelScorer,
        decision_fusion: DecisionFusion,
        device: str = "cpu",
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.threat_scorer = threat_scorer
        self.decision_fusion = decision_fusion
        self.device = device
        self.model.to(device)
        self.model.eval()

    def analyze(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用Transformer模型推理"""
        traffic = flow_data["traffic_features"]
        log = flow_data["log_features"]

        # 转为tensor
        if isinstance(traffic, np.ndarray):
            traffic = torch.FloatTensor(traffic)
        if isinstance(log, np.ndarray):
            log = torch.FloatTensor(log)

        # 确保batch维度
        if traffic.dim() == 1:
            traffic = traffic.unsqueeze(0)
        if log.dim() == 1:
            log = log.unsqueeze(0)

        traffic = traffic.to(self.device)
        log = log.to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(traffic, log)

        probs_np = probs.cpu().numpy()
        pred_idx = np.argmax(probs_np, axis=1)
        pred_labels = self.preprocessor.inverse_label(pred_idx)

        return {
            "predicted_class": pred_labels[0],
            "predicted_index": int(pred_idx[0]),
            "probabilities": probs_np[0],
            "confidence": float(probs_np[0].max()),
        }

    def query_threat_intel(
        self, indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """查询本地威胁情报JSON"""
        scores = self.threat_scorer.score(
            src_ip=indicators.get("src_ip"),
            dst_ip=indicators.get("dst_ip"),
            dst_port=indicators.get("dst_port"),
        )

        # 检查是否命中威胁情报
        is_uniform = np.allclose(
            scores, np.ones_like(scores) / len(scores), atol=0.01
        )

        return {
            "threat_scores": scores,
            "matched": not is_uniform,
            "indicators": indicators,
        }

    def make_decision(
        self,
        dl_result: Dict[str, Any],
        ti_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """使用决策层融合模块做最终判断"""
        dl_probs = dl_result["probabilities"].reshape(1, -1)
        ti_scores = ti_result["threat_scores"].reshape(1, -1)

        fused_probs = self.decision_fusion.fuse(dl_probs, ti_scores)
        final_pred_idx = np.argmax(fused_probs, axis=1)
        final_pred_label = self.preprocessor.inverse_label(final_pred_idx)

        return {
            "final_class": final_pred_label[0],
            "final_index": int(final_pred_idx[0]),
            "fused_probabilities": fused_probs[0],
            "final_confidence": float(fused_probs[0].max()),
            "dl_predicted": dl_result["predicted_class"],
            "threat_intel_matched": ti_result["matched"],
        }

    def explain(self, decision: Dict[str, Any]) -> str:
        """基于规则模板生成解释文本"""
        final_class = decision["final_class"]
        confidence = decision["final_confidence"]
        dl_pred = decision["dl_predicted"]
        ti_matched = decision["threat_intel_matched"]

        # 判断是否为攻击
        is_attack = final_class.lower() not in ("benign", "normal", "legitimate")

        lines = []

        if is_attack:
            lines.append(f"[警告] 检测到潜在网络攻击: {final_class}")
            lines.append(f"置信度: {confidence:.2%}")
        else:
            lines.append(f"[正常] 流量判定为正常: {final_class}")
            lines.append(f"置信度: {confidence:.2%}")

        lines.append(f"深度学习模型预测: {dl_pred}")

        if ti_matched:
            lines.append("威胁情报: 命中已知威胁指标")
            if dl_pred != final_class:
                lines.append(
                    f"注意: 融合威胁情报后预测从 {dl_pred} 调整为 {final_class}"
                )
        else:
            lines.append("威胁情报: 未命中已知威胁指标")

        # 概率分布Top-3
        probs = decision["fused_probabilities"]
        class_names = self.preprocessor.class_names
        top3_idx = np.argsort(probs)[::-1][:3]
        lines.append("概率分布 Top-3:")
        for idx in top3_idx:
            lines.append(f"  - {class_names[idx]}: {probs[idx]:.4f}")

        return "\n".join(lines)
