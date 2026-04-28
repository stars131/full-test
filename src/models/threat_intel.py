"""wxqb API威胁情报评分模块"""

import json
import time
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


HIGH_RISK_KEYWORDS = {
    "malware",
    "botnet",
    "c2",
    "command and control",
    "ransomware",
    "trojan",
    "backdoor",
    "apt",
    "phishing",
    "exploit",
    "scanner",
    "bruteforce",
    "brute force",
    "ddos",
    "spam",
    "miner",
    "cryptominer",
    "blacklist",
    "malicious",
}

MEDIUM_RISK_KEYWORDS = {
    "suspicious",
    "proxy",
    "tor",
    "vpn",
    "anonymizer",
    "compromised",
    "abuse",
    "reputation",
}

RISKY_PORT_BONUS = {
    22: 0.05,
    23: 0.08,
    135: 0.05,
    139: 0.05,
    445: 0.08,
    1433: 0.05,
    3306: 0.05,
    5432: 0.05,
    6379: 0.05,
    8000: 0.03,
    8080: 0.03,
    8443: 0.03,
    3389: 0.08,
}


class ThreatIntelScorer:
    """通过wxqb FastAPI服务查询IOC并映射为类别概率分布。"""

    def __init__(
        self,
        threat_intel_dir: str,
        class_names: List[str],
        api_url: Optional[str] = None,
        timeout_seconds: int = 5,
        use_search_fallback: bool = True,
        cache_ttl_seconds: int = 3600,
    ):
        self.threat_intel_dir = threat_intel_dir
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.api_url = api_url.rstrip("/") if api_url else None
        self.timeout_seconds = timeout_seconds
        self.use_search_fallback = use_search_fallback
        self.cache_ttl_seconds = cache_ttl_seconds
        self.intel_db: Dict = {}
        self.api_available = False
        self._cache: Dict[str, Tuple[float, List[Dict]]] = {}
        self._last_error: Optional[str] = None
        self._diagnostics = {
            "api_url": self.api_url,
            "api_available": False,
            "queries": 0,
            "lookup_queries": 0,
            "search_queries": 0,
            "matched_indicators": 0,
            "matched_records": 0,
            "errors": 0,
            "cache_hits": 0,
            "last_error": None,
        }

        if self.api_url:
            self._check_api_health()
        else:
            self._set_error("threat_intel_api.url未配置，威胁情报评分返回中性分布")

    def _set_error(self, message: str):
        self._last_error = message
        self._diagnostics["last_error"] = message
        self._diagnostics["errors"] += 1

    def _check_api_health(self):
        """检查wxqb API是否可用。"""
        try:
            data = self._get_json("/health", timeout=min(self.timeout_seconds, 3))
            if data.get("status") == "ok":
                self.api_available = True
                self._diagnostics["api_available"] = True
                print(f"wxqb威胁情报API连接成功: {self.api_url}")
            else:
                self._set_error(f"wxqb威胁情报API健康检查异常: {data}")
        except (URLError, ConnectionError, TimeoutError, json.JSONDecodeError, OSError) as e:
            self._set_error(f"wxqb威胁情报API不可用: {e}")
            print(self._last_error)

    def _get_json(
        self,
        path: str,
        params: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ):
        query = f"?{urlencode(params)}" if params else ""
        with urlopen(
            f"{self.api_url}{path}{query}",
            timeout=timeout or self.timeout_seconds,
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _parse_records(self, data) -> List[Dict]:
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if not isinstance(data, dict):
            return []

        for key in ("matches", "results"):
            records = data.get(key)
            if isinstance(records, list):
                return [r for r in records if isinstance(r, dict)]

        if "indicator" in data and "type" in data:
            return [data]
        return []

    def _normalize_record(self, record: Dict) -> Dict:
        tags = record.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.replace(";", ",").split(",") if t.strip()]
        elif not isinstance(tags, list):
            tags = []

        return {
            "indicator": str(record.get("indicator") or ""),
            "type": str(record.get("type") or ""),
            "pulse_id": str(record.get("pulse_id") or ""),
            "pulse_name": record.get("pulse_name") or "",
            "description": record.get("description") or "",
            "created": record.get("created") or "",
            "modified": record.get("modified") or "",
            "tags": [str(t) for t in tags],
            "source": "wxqb",
        }

    def _api_query_ioc(self, value: str) -> List[Dict]:
        """通过wxqb /api/v1/lookup 精确查询IOC。"""
        if not self.api_available:
            return []

        self._diagnostics["queries"] += 1
        self._diagnostics["lookup_queries"] += 1
        try:
            data = self._get_json("/api/v1/lookup", {"indicator": value})
            return [self._normalize_record(r) for r in self._parse_records(data)]
        except (URLError, ConnectionError, TimeoutError, json.JSONDecodeError, OSError) as e:
            self._set_error(f"lookup失败 {value}: {e}")
            return []

    def _api_search(self, query: str, limit: int = 5) -> List[Dict]:
        """通过wxqb /api/v1/search 模糊搜索IOC。"""
        if not self.api_available or not self.use_search_fallback:
            return []
        if len(query) < 2:
            return []

        self._diagnostics["queries"] += 1
        self._diagnostics["search_queries"] += 1
        try:
            data = self._get_json(
                "/api/v1/search",
                {"q": query, "limit": limit, "offset": 0},
            )
            return [self._normalize_record(r) for r in self._parse_records(data)]
        except (URLError, ConnectionError, TimeoutError, json.JSONDecodeError, OSError) as e:
            self._set_error(f"search失败 {query}: {e}")
            return []

    def _query_indicator_records(self, indicator: str) -> List[Dict]:
        now = time.time()
        cached = self._cache.get(indicator)
        if cached and now - cached[0] < self.cache_ttl_seconds:
            self._diagnostics["cache_hits"] += 1
            return cached[1]

        records = self._api_query_ioc(indicator)
        if not records:
            records = self._api_search(indicator, limit=5)

        self._cache[indicator] = (now, records)
        if records:
            self._diagnostics["matched_indicators"] += 1
            self._diagnostics["matched_records"] += len(records)
        return records

    def _query_indicator(self, indicator: str) -> Optional[Dict]:
        records = self._query_indicator_records(indicator)
        return self._aggregate_api_records(records) if records else None

    def _record_text(self, record: Dict) -> str:
        parts = [
            record.get("type", ""),
            record.get("pulse_name", ""),
            record.get("description", ""),
            " ".join(record.get("tags", [])),
        ]
        return " ".join(parts).lower()

    def _extract_attack_types(self, records: List[Dict]) -> List[str]:
        attack_types = []
        for record in records:
            attack_types.extend(record.get("tags", []))
            for field in ("type", "pulse_name", "description"):
                value = record.get(field)
                if value:
                    attack_types.append(str(value))

        text = " ".join(self._record_text(r) for r in records)
        for keyword in HIGH_RISK_KEYWORDS | MEDIUM_RISK_KEYWORDS:
            if keyword in text:
                attack_types.append(keyword)

        return list({a for a in attack_types if a})

    def _aggregate_api_records(self, records: List[Dict]) -> Optional[Dict]:
        """将wxqb记录聚合为内部评分格式。"""
        if not records:
            return None

        count = len(records)
        pulse_count = len({r.get("pulse_id") for r in records if r.get("pulse_id")})
        type_set = {str(r.get("type") or "").lower() for r in records}
        text = " ".join(self._record_text(r) for r in records)

        risk = 0.35
        if count >= 4:
            risk = 0.65
        elif count >= 2:
            risk = 0.50

        if pulse_count >= 2:
            risk += 0.10
        if type_set & {"ip", "ipv4", "ipv6", "domain", "hostname", "url"}:
            risk += 0.05

        high_hits = sum(1 for keyword in HIGH_RISK_KEYWORDS if keyword in text)
        medium_hits = sum(1 for keyword in MEDIUM_RISK_KEYWORDS if keyword in text)
        risk += min(high_hits * 0.12, 0.30)
        risk += min(medium_hits * 0.05, 0.15)
        risk = min(risk, 1.0)

        confidence = min(0.50 + count * 0.08 + pulse_count * 0.05, 0.95)

        return {
            "risk_score": risk,
            "attack_types": self._extract_attack_types(records),
            "confidence": confidence,
            "source": "wxqb",
            "match_count": count,
            "records": records,
        }

    def score(
        self,
        src_ip: Optional[str] = None,
        dst_ip: Optional[str] = None,
        dst_port: Optional[int] = None,
    ) -> np.ndarray:
        """根据流量指标查询wxqb威胁情报，返回威胁评分向量。"""
        scores = np.ones(self.num_classes) / self.num_classes
        if not self.api_available:
            return scores

        indicators = []
        if src_ip:
            indicators.append((str(src_ip), 0.9))
        if dst_ip:
            indicators.append((str(dst_ip), 1.0))

        matched_entries = []
        for indicator, indicator_weight in indicators:
            entry = self._query_indicator(indicator)
            if entry:
                entry["indicator_weight"] = indicator_weight
                matched_entries.append(entry)

        if not matched_entries:
            return scores

        port_bonus = self._port_bonus(dst_port)
        for entry in matched_entries:
            risk_score = min(entry.get("risk_score", 0.5) + port_bonus, 1.0)
            confidence = entry.get("confidence", 0.5)
            indicator_weight = entry.get("indicator_weight", 1.0)
            attack_types = entry.get("attack_types", [])
            weight = risk_score * confidence * indicator_weight

            for attack_type in attack_types:
                for i, cls_name in enumerate(self.class_names):
                    if self._match_attack_type(attack_type, cls_name):
                        scores[i] += weight

        total = scores.sum()
        if total > 0:
            scores = scores / total

        return scores

    def _port_bonus(self, dst_port: Optional[int]) -> float:
        if dst_port is None:
            return 0.0
        try:
            return RISKY_PORT_BONUS.get(int(dst_port), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _match_attack_type(self, attack_type: str, class_name: str) -> bool:
        """模糊匹配攻击类型和类别名称。"""
        at = (attack_type or "").lower().replace("_", " ").replace("-", " ")
        cn = (class_name or "").lower().replace("_", " ").replace("-", " ")
        if not at or not cn:
            return False
        return at in cn or cn in at

    def batch_score(
        self,
        src_ips: Optional[List[str]] = None,
        dst_ips: Optional[List[str]] = None,
        dst_ports: Optional[List[int]] = None,
        batch_size: int = 0,
    ) -> np.ndarray:
        """批量评分。"""
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

    def get_diagnostics(self) -> Dict:
        diagnostics = dict(self._diagnostics)
        diagnostics["api_available"] = self.api_available
        diagnostics["cache_size"] = len(self._cache)
        diagnostics["local_fallback_entries"] = 0
        return diagnostics
