"""数据准备脚本 - 从CIC-IDS-2018嵌套ZIP中提取CSV并采样"""

import os
import io
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path


DATASET_DIR = "data/数据集BCCC-CSE-CIC-IDS-2018"
OUTPUT_DIR = "data/raw"

# 选择代表性子集（含多种攻击类型，避免数据过大）
SELECTED_ZIPS = [
    "Wednesday_14_02_2018.zip",       # Brute Force FTP/SSH
    "Thursday_15_02_2018.zip",        # DoS GoldenEye/Slowloris
    "Friday-23-02-2018.zip",          # BF Web/XSS/SQL Injection
    "Friday-02-03-2018.zip",          # Bot
]

# 每类最大采样数（控制数据规模）
MAX_SAMPLES_PER_FILE = 20000


def extract_nested_zip(outer_zip_path: str, output_dir: str):
    """从嵌套ZIP中提取CSV"""
    extracted = []
    outer = zipfile.ZipFile(outer_zip_path)

    for name in outer.namelist():
        if not name.endswith(".zip"):
            continue

        inner_name = name.split("/")[-1]
        print(f"  提取: {inner_name}")

        # 解压到临时目录再读取（避免压缩方法不支持问题）
        import tempfile, subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            # 先把内层zip写到临时目录
            inner_zip_path = os.path.join(tmpdir, inner_name)
            with open(inner_zip_path, "wb") as f:
                f.write(outer.read(name))

            # 用PowerShell解压内层zip
            extract_dir = os.path.join(tmpdir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            subprocess.run(
                ["powershell", "-Command",
                 f"Expand-Archive -Path '{inner_zip_path}' -DestinationPath '{extract_dir}' -Force"],
                capture_output=True, check=True
            )

            # 读取CSV
            for csv_file in Path(extract_dir).rglob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, low_memory=False)
                    if df.empty or "label" not in df.columns:
                        print(f"    {csv_file.name}: 跳过（空或无label列）")
                        continue
                    print(f"    {csv_file.name}: {len(df)} rows, label={df['label'].unique()}")
                    extracted.append(df)
                except Exception as e:
                    print(f"    {csv_file.name}: 读取失败 - {e}")

    return extracted


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_dfs = []

    for zip_name in SELECTED_ZIPS:
        zip_path = os.path.join(DATASET_DIR, zip_name)
        if not os.path.exists(zip_path):
            print(f"跳过（未找到）: {zip_name}")
            continue

        print(f"\n处理: {zip_name}")
        dfs = extract_nested_zip(zip_path, OUTPUT_DIR)
        all_dfs.extend(dfs)

    if not all_dfs:
        print("未提取到任何数据！")
        return

    # 合并所有数据
    full_data = pd.concat(all_dfs, ignore_index=True)
    print(f"\n合并后总数据: {full_data.shape}")
    print(f"标签分布:\n{full_data['label'].value_counts()}")

    # 按类别分层采样
    sampled_dfs = []
    for label, group in full_data.groupby("label"):
        n = min(len(group), MAX_SAMPLES_PER_FILE)
        sampled = group.sample(n=n, random_state=42)
        sampled_dfs.append(sampled)
        print(f"  {label}: {len(group)} -> {n}")

    sampled_data = pd.concat(sampled_dfs, ignore_index=True)
    sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = os.path.join(OUTPUT_DIR, "cic_ids_2018_sampled.csv")
    sampled_data.to_csv(output_path, index=False)
    print(f"\n采样后数据保存至: {output_path}")
    print(f"最终数据集大小: {sampled_data.shape}")
    print(f"标签分布:\n{sampled_data['label'].value_counts()}")


if __name__ == "__main__":
    main()
