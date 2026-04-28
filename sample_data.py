"""采样数据生成入口已停用。

当前项目默认且仅支持 BCCC-CSE-CIC-IDS-2018 全量训练流程：
    python prepare_data.py --download
    python train.py
"""

import sys


def main() -> int:
    print("采样数据生成已停用。请使用全量数据流程：")
    print("  python prepare_data.py --download")
    print("  python train.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
