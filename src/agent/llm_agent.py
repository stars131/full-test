"""LLM Agent - 基于Claude API的智能检测与解释"""

import os
from typing import Dict, Any, Optional

import numpy as np
import torch

from src.agent.base_agent import BaseAgent
from src.models.transformer_detector import TransformerDetector
from src.models.threat_intel import ThreatIntelScorer
from src.models.decision_fusion import DecisionFusion
from src.data.preprocessor import Preprocessor


class LLMAgent(BaseAgent):
    """LLM Agent

    核心决策和解释通过Claude API完成。
    DL模型推理和威胁情报查询仍在本地执行，
    将结果发送给LLM进行综合分析和自然语言解释。
    """

    def __init__(
        self,
        model: TransformerDetector,
        preprocessor: Preprocessor,
        threat_scorer: ThreatIntelScorer,
        decision_fusion: DecisionFusion,
        api_key: Optional[str] = None,
        llm_model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        device: str = "cpu",
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.threat_scorer = threat_scorer
        self.decision_fusion = decision_fusion
        self.device = device
        self.llm_model = llm_model
        self.max_tokens = max_tokens

        self.model.to(device)
        self.model.eval()

        # 初始化Anthropic客户端
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = None
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("警告: anthropic包未安装，LLM功能不可用")

    def _call_llm(self, prompt: str) -> str:
        """调用Claude API"""
        if self.client is None:
            return "[LLM不可用] 请设置ANTHROPIC_API_KEY环境变量并安装anthropic包"

        message = self.client.messages.create(
            model=self.llm_model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def analyze(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用DL模型推理 + LLM分析"""
        traffic = flow_data["traffic_features"]
        log = flow_data["log_features"]

        if isinstance(traffic, np.ndarray):
            traffic = torch.FloatTensor(traffic)
        if isinstance(log, np.ndarray):
            log = torch.FloatTensor(log)

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

        # 构建LLM分析提示
        class_names = self.preprocessor.class_names
        prob_str = ", ".join(
            f"{name}: {p:.4f}" for name, p in zip(class_names, probs_np[0])
        )

        prompt = f"""你是一个网络安全分析专家。深度学习模型对一条网络流量进行了分类，结果如下：

预测类别: {pred_labels[0]}
置信度: {probs_np[0].max():.4f}
各类别概率分布: {prob_str}

请简要分析这个检测结果的可信度，并指出需要关注的风险点。用中文回答，控制在100字以内。"""

        llm_analysis = self._call_llm(prompt)

        return {
            "predicted_class": pred_labels[0],
            "predicted_index": int(pred_idx[0]),
            "probabilities": probs_np[0],
            "confidence": float(probs_np[0].max()),
            "llm_analysis": llm_analysis,
        }

    def query_threat_intel(
        self, indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """查询wxqb API威胁情报"""
        scores = self.threat_scorer.score(
            src_ip=indicators.get("src_ip"),
            dst_ip=indicators.get("dst_ip"),
            dst_port=indicators.get("dst_port"),
        )

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
        """使用LLM做最终决策"""
        dl_probs = dl_result["probabilities"].reshape(1, -1)
        ti_scores = ti_result["threat_scores"].reshape(1, -1)

        # 先用决策融合模块获取基础融合结果
        fused_probs = self.decision_fusion.fuse(dl_probs, ti_scores)
        fused_pred_idx = np.argmax(fused_probs, axis=1)
        fused_pred_label = self.preprocessor.inverse_label(fused_pred_idx)

        # LLM综合分析
        class_names = self.preprocessor.class_names
        dl_prob_str = ", ".join(
            f"{name}: {p:.4f}"
            for name, p in zip(class_names, dl_result["probabilities"])
        )
        ti_prob_str = ", ".join(
            f"{name}: {p:.4f}"
            for name, p in zip(class_names, ti_result["threat_scores"])
        )

        prompt = f"""你是一个网络安全分析专家，需要综合深度学习模型和威胁情报做出最终判断。

深度学习模型结果:
- 预测: {dl_result['predicted_class']}，置信度: {dl_result['confidence']:.4f}
- 概率分布: {dl_prob_str}

威胁情报结果:
- 命中: {"是" if ti_result['matched'] else "否"}
- 评分分布: {ti_prob_str}

融合算法结果: {fused_pred_label[0]}

请给出你的最终判断和理由。用中文回答，控制在150字以内。"""

        llm_decision = self._call_llm(prompt)

        return {
            "final_class": fused_pred_label[0],
            "final_index": int(fused_pred_idx[0]),
            "fused_probabilities": fused_probs[0],
            "final_confidence": float(fused_probs[0].max()),
            "dl_predicted": dl_result["predicted_class"],
            "threat_intel_matched": ti_result["matched"],
            "llm_decision": llm_decision,
        }

    def explain(self, decision: Dict[str, Any]) -> str:
        """使用LLM生成自然语言解释"""
        prompt = f"""你是一个网络安全分析专家，请对以下检测结果生成一份简洁的检测报告。

最终判定: {decision['final_class']}
置信度: {decision['final_confidence']:.2%}
深度学习模型预测: {decision['dl_predicted']}
威胁情报命中: {"是" if decision['threat_intel_matched'] else "否"}

LLM分析意见: {decision.get('llm_decision', 'N/A')}

请生成一份结构化的中文检测报告，包括：
1. 检测结论
2. 风险等级（高/中/低）
3. 分析依据
4. 建议措施

控制在200字以内。"""

        return self._call_llm(prompt)
