"""触发wxqb威胁情报API同步并检查服务状态。"""

from __future__ import annotations

import argparse
import json
from urllib.error import URLError
from urllib.request import Request, urlopen


def request_json(url: str, method: str = "GET", timeout: int = 30) -> dict:
    request = Request(url, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="同步wxqb威胁情报API数据")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="wxqb API base URL，例如 http://127.0.0.1:8000 或服务器地址",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="调用 POST /api/v1/sync 从OTX同步数据",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="请求超时时间（秒）",
    )
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    try:
        health = request_json(f"{api_url}/health", timeout=min(args.timeout, 10))
        print("health:", json.dumps(health, ensure_ascii=False))

        if args.sync:
            result = request_json(
                f"{api_url}/api/v1/sync",
                method="POST",
                timeout=args.timeout,
            )
            print("sync:", json.dumps(result, ensure_ascii=False))

        stats = request_json(f"{api_url}/api/v1/stats", timeout=min(args.timeout, 10))
        print("stats:", json.dumps(stats, ensure_ascii=False))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"wxqb API请求失败: {exc}") from exc


if __name__ == "__main__":
    main()
