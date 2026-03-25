"""威胁情报API模拟服务器

提供RESTful API接口，模拟真实的外部威胁情报查询服务。
支持IP查询和端口查询。

启动方式：
    python threat_intel_api.py [--port 5000]

API端点：
    GET  /api/v1/health             - 健康检查
    POST /api/v1/query/ip           - 查询IP威胁情报
    POST /api/v1/query/port         - 查询端口威胁情报
    POST /api/v1/query/batch        - 批量查询
"""

import os
import json
import argparse
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Dict, Any


# 加载威胁情报数据库
INTEL_DB: Dict[str, Any] = {}


def load_intel_db(intel_dir: str = "data/threat_intel"):
    """加载威胁情报JSON"""
    global INTEL_DB
    for fname in os.listdir(intel_dir):
        if fname.endswith(".json"):
            with open(os.path.join(intel_dir, fname), "r", encoding="utf-8") as f:
                INTEL_DB.update(json.load(f))
    print(f"已加载 {len(INTEL_DB)} 条威胁情报")


class ThreatIntelHandler(BaseHTTPRequestHandler):
    """威胁情报API请求处理器"""

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/v1/health":
            self._send_json({
                "status": "ok",
                "entries": len(INTEL_DB),
                "timestamp": time.time(),
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/v1/query/ip":
            body = self._read_body()
            ip = body.get("ip", "")
            result = self._query_indicator(ip)
            self._send_json(result)

        elif path == "/api/v1/query/port":
            body = self._read_body()
            port = body.get("port")
            result = self._query_indicator(f"port:{port}")
            self._send_json(result)

        elif path == "/api/v1/query/batch":
            body = self._read_body()
            results = []
            for indicator in body.get("indicators", []):
                results.append(self._query_indicator(indicator))
            self._send_json({"results": results})

        else:
            self._send_json({"error": "Not found"}, 404)

    def _query_indicator(self, indicator: str) -> dict:
        """查询单个指标"""
        if indicator in INTEL_DB:
            entry = INTEL_DB[indicator]
            return {
                "indicator": indicator,
                "found": True,
                "risk_score": entry["risk_score"],
                "attack_types": entry["attack_types"],
                "confidence": entry["confidence"],
                "description": entry.get("description", ""),
            }
        return {
            "indicator": indicator,
            "found": False,
            "risk_score": 0.0,
            "attack_types": [],
            "confidence": 0.0,
            "description": "未在威胁情报库中找到",
        }

    def log_message(self, format, *args):
        """简化日志输出"""
        print(f"[ThreatIntel API] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="威胁情报API模拟服务器")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    parser.add_argument(
        "--intel-dir", type=str, default="data/threat_intel",
        help="威胁情报JSON目录",
    )
    args = parser.parse_args()

    load_intel_db(args.intel_dir)

    server = HTTPServer(("127.0.0.1", args.port), ThreatIntelHandler)
    print(f"威胁情报API服务启动: http://127.0.0.1:{args.port}")
    print("API端点:")
    print(f"  GET  http://127.0.0.1:{args.port}/api/v1/health")
    print(f"  POST http://127.0.0.1:{args.port}/api/v1/query/ip")
    print(f"  POST http://127.0.0.1:{args.port}/api/v1/query/port")
    print(f"  POST http://127.0.0.1:{args.port}/api/v1/query/batch")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务关闭")
        server.server_close()


if __name__ == "__main__":
    main()
