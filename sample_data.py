"""从已提取的CSV中采样，创建可用于训练的平衡数据集"""

import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
OUTPUT = "data/raw/cic_ids_2018_sampled.csv"

# 每类最大采样数
MAX_PER_CLASS = 10000

def main():
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv") and "sampled" not in f]

    all_dfs = []
    for f in sorted(csv_files):
        path = os.path.join(RAW_DIR, f)
        print(f"读取: {f}")
        # 对大文件只读取前N行
        n_rows = None
        file_size = os.path.getsize(path)
        if file_size > 100 * 1024 * 1024:  # >100MB
            n_rows = 50000
            print(f"  文件过大({file_size // 1024 // 1024}MB)，仅读取前{n_rows}行")
        df = pd.read_csv(path, low_memory=False, nrows=n_rows)
        print(f"  行数: {len(df)}, 标签: {df['label'].unique()}")
        all_dfs.append(df)

    data = pd.concat(all_dfs, ignore_index=True)
    print(f"\n合并后: {data.shape}")
    print(f"标签分布:\n{data['label'].value_counts()}")

    # 分层采样
    sampled = []
    for label, group in data.groupby("label"):
        n = min(len(group), MAX_PER_CLASS)
        s = group.sample(n=n, random_state=42)
        sampled.append(s)
        print(f"  {label}: {len(group)} -> {n}")

    result = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    result.to_csv(OUTPUT, index=False)
    print(f"\n保存至: {OUTPUT}")
    print(f"最终: {result.shape}")
    print(f"标签分布:\n{result['label'].value_counts()}")

if __name__ == "__main__":
    main()
