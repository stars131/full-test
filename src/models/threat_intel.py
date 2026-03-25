"""威胁情报评分模块 - 支持本地JSON和HTTP API两种查询方式"""

import os
import json
from typing import Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np


class ThreatIntelScorer:
    """威胁情报评分器

    支持两种模式:
    - local: 从本地JSON文件读取威胁情报
    - api: 通过HTTP API查询（模拟真实外部API调用）
    """

    def __init__(
        self,
        threat_intel_dir: str,
        class_names: List[str],
        api_url: Optional[str] = None,
    ):
        self.threat_intel_dir = threat_intel_dir
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.api_url = api_url  # e.g. "http://127.0.0.1:5000"
        self.intel_db: Dict = {}

        if not api_url:
            self._load_local_intel()

    def _load_local_intel(self):
        """加载所有威胁情报JSON文件"""
        if not os.path.exists(self.threat_intel_dir):
            print(f"威胁情报目录不存在: {self.threat_intel_dir}，使用空数据库")
            return

        for fname in os.listdir(self.threat_intel_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.threat_intel_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.intel_db.update(data)

        print(f"加载威胁情报条目: {len(self.intel_db)}")

    def _query_api(self, indicator: str) -> Optional[Dict]:
        """通过HTTP API查询威胁情报"""
        if not self.api_url:
            return None

        try:
            if indicator.startswith("port:"):
                url = f"{self.api_url}/api/v1/query/port"
                port = int(indicator.split(":")[1])
                body = json.dumps({"port": port}).encode("utf-8")
            else:
                url = f"{self.api_url}/api/v1/query/ip"
                body = json.dumps({"ip": indicator}).encode("utf-8")

            req = Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")

            with urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("found", False):
                    return result
        except (URLError, ConnectionError, TimeoutError):
            pass

        return None

    def _query_indicator(self, indicator: str) -> Optional[Dict]:
        """查询单个指标（优先API，fallback到本地）"""
        if self.api_url:
            result = self._query_api(indicator)
            if result:
                return result

        # 本地查询
        if indicator in self.intel_db:
            return self.intel_db[indicator]

        return None

    def score(
        self,
        src_ip: Optional[str] = None,
        dst_ip: Optional[str] = None,
        dst_port: Optional[int] = None,
    ) -> np.ndarray:
        """根据流量指标查询威胁情报，返回威胁评分向量"""
        scores = np.ones(self.num_classes) / self.num_classes  # 均匀先验

        indicators = []
        if src_ip:
            indicators.append(src_ip)
        if dst_ip:
            indicators.append(dst_ip)
        if dst_port is not None:
            indicators.append(f"port:{dst_port}")

        matched_entries = []
        for indicator in indicators:
            entry = self._query_indicator(indicator)
            if entry:
                matched_entries.append(entry)

        if not matched_entries:
            return scores

        for entry in matched_entries:
            risk_score = entry.get("risk_score", 0.5)
            confidence = entry.get("confidence", 0.5)
            attack_types = entry.get("attack_types", [])
            weight = risk_score * confidence

            for attack_type in attack_types:
                for i, cls_name in enumerate(self.class_names):
                    if self._match_attack_type(attack_type, cls_name):
                        scores[i] += weight

        total = scores.sum()
        if total > 0:
            scores = scores / total

        return scores

    def _match_attack_type(self, attack_type: str, class_name: str) -> bool:
        """模糊匹配攻击类型和类别名称"""
        at = attack_type.lower().replace("_", " ").replace("-", " ")
        cn = class_name.lower().replace("_", " ").replace("-", " ")
        return at in cn or cn in at

    def batch_score(
        self,
        src_ips: Optional[List[str]] = None,
        dst_ips: Optional[List[str]] = None,
        dst_ports: Optional[List[int]] = None,
        batch_size: int = 0,
    ) -> np.ndarray:
        """批量评分"""
        n = batch_size or len(src_ips or dst_ips or dst_ports or [])
        if n == 0:
            return np.ones((1, self.num_classes)) / self.num_classes

        results = []
        for i in range(n):
            src = src_ips[i] if src_ips and i < len(src_ips) else None
            dst = dst_ips[i] if dst_ips and i < len(dst_ips) else None
            port = dst_ports[i] if dst_ports and i < len(dst_ports) else None
            results.append(self.score(src, dst, port))

        return np.array(results)
