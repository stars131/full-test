"""旧本地JSON威胁情报生成脚本已停用。"""

from __future__ import annotations


def main():
    raise SystemExit(
        "威胁情报已切换为wxqb API模式，不再生成本地JSON模拟情报。"
        "请启动 D:/毕设相关/wxqb 服务，并在 threat_intel_api.url 中配置API地址。"
    )


if __name__ == "__main__":
    main()
