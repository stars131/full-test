"""BCCC-CSE-CIC-IDS-2018 全量数据集完整性检查脚本

用法:
    python check_dataset.py
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ZIP_DIR = "/root/autodl-tmp/cc-bishe-full/zips"
RAW_DIR = "/root/autodl-tmp/cc-bishe-full/raw"
DATASET_URL = "https://bccc.laps.yorku.ca/BCCC-CSE-CIC-IDS-2018/"

EXPECTED_ZIPS = {
    "Wednesday_14_02_2018.zip": ["benign", "BF_FTP", "BF_SSH"],
    "Thursday_15_02_2018.zip": ["benign", "DoS_Golden_Eye", "DoS_Slowloris"],
    "Friday-16-02-2018.zip": ["benign(x3)", "dos_hulk(x2)", "dos_slowhttp(x2)"],
    "Tuesday_20_02_2018.zip": ["benign", "loic_http"],
    "Wednesday_21_02_2018.zip": ["benign(x4)", "hoic(x2)"],
    "Thursday-22-02-2018.zip": ["benign", "BF_web", "BF_XSS", "SQL_Injection"],
    "Friday-23-02-2018.zip": ["benign", "BF_web", "BF_XSS", "SQL_Injection"],
    "Wednesday-28-02-2018.zip": ["benign"],
    "Thursday-01-03-2018.zip": ["benign", "infiltration"],
    "Friday-02-03-2018.zip": ["benign", "bot"],
}


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def is_sampled_csv(file_name: str) -> bool:
    lower_name = file_name.lower()
    return "sample" in lower_name or "sampled" in lower_name


def check_outer_zips():
    print(f"\n[1/3] 外层ZIP检查 ({ZIP_DIR})")
    print("-" * 60)

    if not os.path.exists(ZIP_DIR):
        print("  目录不存在")
        return [], list(EXPECTED_ZIPS.keys())

    present, missing = [], []
    for zip_name, contents in EXPECTED_ZIPS.items():
        path = os.path.join(ZIP_DIR, zip_name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            present.append((zip_name, size))
            print(
                f"  [OK]   {zip_name:<32s} {human_size(size):>10s}  "
                f"[{', '.join(contents)}]"
            )
        else:
            missing.append(zip_name)
            print(
                f"  [MISS] {zip_name:<32s} {'-':>10s}  "
                f"[{', '.join(contents)}]"
            )
    return present, missing


def check_extracted_csvs():
    print(f"\n[2/3] 已提取CSV检查 ({RAW_DIR})")
    print("-" * 60)

    if not os.path.exists(RAW_DIR):
        print("  目录不存在")
        return [], []

    csv_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".csv"))
    raw_csvs = [f for f in csv_files if not is_sampled_csv(f)]
    sampled_csvs = [f for f in csv_files if is_sampled_csv(f)]

    if not csv_files:
        print("  (无CSV文件)")
    else:
        for f in csv_files:
            size = os.path.getsize(os.path.join(RAW_DIR, f))
            marker = "[SAMPLED]" if is_sampled_csv(f) else "[RAW]    "
            print(f"  {marker} {f:<45s} {human_size(size):>10s}")

    return raw_csvs, sampled_csvs


def classify_status(n_present: int, n_total: int, n_csv: int, sampled_count: int) -> str:
    if sampled_count > 0:
        return "SAMPLED_PRESENT"
    if n_present == 0:
        return "NOT_DOWNLOADED"
    if n_present < n_total:
        return "PARTIAL_ZIP"
    if n_csv == 0:
        return "ZIP_ONLY"
    if n_csv < 10:
        return "PARTIAL_EXTRACTED"
    return "FULL_EXTRACTED"


def print_next_steps(status: str, missing: list[str], sampled_csvs: list[str]) -> None:
    print()
    print("=" * 60)
    print("下一步操作建议")
    print("=" * 60)

    if status == "SAMPLED_PRESENT":
        print("\n[状态] 检测到采样CSV，默认全量训练不会使用这些文件。")
        for name in sampled_csvs:
            print(f"  - {name}")
        print("请确认 raw 目录中同时存在全量CSV，然后运行: python train.py")
        return

    if status == "NOT_DOWNLOADED":
        print("\n[状态] 尚未下载数据集")
        print(f"下载源: {DATASET_URL}")
        print("执行: python prepare_data.py --download")
        return

    if status == "PARTIAL_ZIP":
        print(f"\n[状态] 部分ZIP缺失 ({len(missing)} 个):")
        for z in missing:
            print(f"  - {z}")
        print("补齐并处理: python prepare_data.py --download")
        return

    if status == "ZIP_ONLY":
        print("\n[状态] 全量ZIP已就绪，尚未提取CSV")
        print("执行: python prepare_data.py --no-download")
        return

    if status == "PARTIAL_EXTRACTED":
        print("\n[状态] CSV数量偏少，可能只提取了部分数据")
        print("重新处理全量数据: python prepare_data.py --download")
        return

    if status == "FULL_EXTRACTED":
        print("\n[状态] 全量数据集已提取")
        print("开始训练: python train.py")


def main():
    print("=" * 60)
    print("BCCC-CSE-CIC-IDS-2018 全量数据集完整性检查")
    print("=" * 60)

    present, missing = check_outer_zips()
    raw_csvs, sampled_csvs = check_extracted_csvs()

    status = classify_status(
        n_present=len(present),
        n_total=len(EXPECTED_ZIPS),
        n_csv=len(raw_csvs),
        sampled_count=len(sampled_csvs),
    )

    print(f"\n[3/3] 状态汇总")
    print("-" * 60)
    print(f"  外层ZIP:    {len(present)}/{len(EXPECTED_ZIPS)}")
    print(f"  全量CSV:    {len(raw_csvs)}")
    print(f"  采样CSV:    {len(sampled_csvs)}")
    print(f"  整体状态:   {status}")

    print_next_steps(status, missing, sampled_csvs)
    print()

    return 0 if status == "FULL_EXTRACTED" else 1


if __name__ == "__main__":
    sys.exit(main())
