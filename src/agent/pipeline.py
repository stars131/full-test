"""Agent编排管道 - 根据配置选择Agent类型，批量处理流量数据"""

from typing import Dict, Any, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.agent.base_agent import BaseAgent
from src.agent.rule_agent import RuleAgent
from src.agent.llm_agent import LLMAgent
from src.models.transformer_detector import TransformerDetector
from src.models.threat_intel import ThreatIntelScorer
from src.models.decision_fusion import DecisionFusion
from src.data.preprocessor import Preprocessor


class AgentPipeline:
    """Agent编排管道

    根据配置创建相应的Agent，支持批量处理流量数据。
    """

    def __init__(
        self,
        model: TransformerDetector,
        preprocessor: Preprocessor,
        threat_scorer: ThreatIntelScorer,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.config = config
        self.device = device

        # 创建决策融合模块
        fusion_config = config.get("fusion", {})
        self.decision_fusion = DecisionFusion(
            strategy=fusion_config.get("strategy", "weighted_average"),
            alpha=fusion_config.get("alpha", 0.8),
        )

        # 根据配置创建Agent
        agent_config = config.get("agent", {})
        mode = agent_config.get("mode", "rule")

        if mode == "rule":
            self.agent = RuleAgent(
                model=model,
                preprocessor=preprocessor,
                threat_scorer=threat_scorer,
                decision_fusion=self.decision_fusion,
                device=device,
            )
        elif mode == "llm":
            llm_config = agent_config.get("llm", {})
            self.agent = LLMAgent(
                model=model,
                preprocessor=preprocessor,
                threat_scorer=threat_scorer,
                decision_fusion=self.decision_fusion,
                api_key=llm_config.get("api_key", ""),
                llm_model=llm_config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=llm_config.get("max_tokens", 1024),
                device=device,
            )
        else:
            raise ValueError(f"不支持的Agent模式: {mode}")

        print(f"Agent模式: {mode}")

    def process_single(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单条流量数据"""
        return self.agent.run(flow_data)

    def process_batch(
        self,
        traffic_batch: np.ndarray,
        log_batch: np.ndarray,
        src_ips: Optional[List[str]] = None,
        dst_ips: Optional[List[str]] = None,
        dst_ports: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """批量处理流量数据

        Args:
            traffic_batch: (N, traffic_dim) 流量特征矩阵
            log_batch: (N, log_dim) 日志特征矩阵
            src_ips: 源IP列表
            dst_ips: 目的IP列表
            dst_ports: 目的端口列表

        Returns:
            检测结果列表
        """
        results = []
        n = len(traffic_batch)

        for i in tqdm(range(n), desc="Agent检测"):
            flow_data = {
                "traffic_features": traffic_batch[i],
                "log_features": log_batch[i],
                "src_ip": src_ips[i] if src_ips else None,
                "dst_ip": dst_ips[i] if dst_ips else None,
                "dst_port": dst_ports[i] if dst_ports else None,
            }
            result = self.process_single(flow_data)
            results.append(result)

        return results

    def generate_report(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """汇总检测结果生成报告

        Args:
            results: 检测结果列表

        Returns:
            汇总报告
        """
        total = len(results)
        if total == 0:
            return {"total": 0, "message": "无数据"}

        # 统计各类别数量
        class_counts: Dict[str, int] = {}
        attack_count = 0

        for r in results:
            final_class = r["decision"]["final_class"]
            class_counts[final_class] = class_counts.get(final_class, 0) + 1

            is_attack = final_class.lower() not in (
                "benign", "normal", "legitimate"
            )
            if is_attack:
                attack_count += 1

        # 平均置信度
        avg_confidence = np.mean(
            [r["decision"]["final_confidence"] for r in results]
        )

        report = {
            "total_flows": total,
            "attack_flows": attack_count,
            "benign_flows": total - attack_count,
            "attack_ratio": attack_count / total,
            "class_distribution": class_counts,
            "average_confidence": float(avg_confidence),
        }

        return report
