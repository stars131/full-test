"""Agent基类 - 定义检测Agent的统一接口"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseAgent(ABC):
    """网络攻击检测Agent基类"""

    @abstractmethod
    def analyze(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析流量数据，返回DL模型检测结果

        Args:
            flow_data: 包含流量特征和日志特征的字典

        Returns:
            包含预测类别、概率分布等信息的字典
        """
        pass

    @abstractmethod
    def query_threat_intel(
        self, indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """查询威胁情报

        Args:
            indicators: 包含src_ip, dst_ip, dst_port等指标

        Returns:
            威胁情报评分结果
        """
        pass

    @abstractmethod
    def make_decision(
        self,
        dl_result: Dict[str, Any],
        ti_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """综合DL结果和威胁情报做出最终决策

        Args:
            dl_result: 深度学习模型检测结果
            ti_result: 威胁情报查询结果

        Returns:
            最终决策结果
        """
        pass

    @abstractmethod
    def explain(self, decision: Dict[str, Any]) -> str:
        """生成决策解释文本

        Args:
            decision: 最终决策结果

        Returns:
            自然语言解释
        """
        pass

    def run(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """完整检测pipeline

        Args:
            flow_data: 流量数据，包含特征和网络指标

        Returns:
            完整的检测报告
        """
        # Step 1: DL模型分析
        dl_result = self.analyze(flow_data)

        # Step 2: 查询威胁情报
        indicators = {
            "src_ip": flow_data.get("src_ip"),
            "dst_ip": flow_data.get("dst_ip"),
            "dst_port": flow_data.get("dst_port"),
        }
        ti_result = self.query_threat_intel(indicators)

        # Step 3: 决策融合
        decision = self.make_decision(dl_result, ti_result)

        # Step 4: 生成解释
        explanation = self.explain(decision)

        return {
            "dl_result": dl_result,
            "threat_intel": ti_result,
            "decision": decision,
            "explanation": explanation,
        }
