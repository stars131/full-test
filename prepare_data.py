"""BCCC-CSE-CIC-IDS-2018 全量数据准备脚本

默认流程会下载完整数据集的 10 个外层 ZIP，并从嵌套 ZIP 中提取 CSV：
    python prepare_data.py --download

服务器默认目录：
    ZIP: /root/autodl-tmp/cc-bishe-full/zips
    CSV: /root/autodl-tmp/cc-bishe-full/raw

小范围验证：
    python prepare_data.py --only Wednesday_14_02_2018.zip --download
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urljoin
from urllib.request import urlopen

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATASET_URL = "https://bccc.laps.yorku.ca/BCCC-CSE-CIC-IDS-2018/"
ZIP_DIR = "/root/autodl-tmp/cc-bishe-full/zips"
OUTPUT_DIR = "/root/autodl-tmp/cc-bishe-full/raw"

ALL_ZIPS = [
    "Wednesday_14_02_2018.zip",
    "Thursday_15_02_2018.zip",
    "Friday-16-02-2018.zip",
    "Tuesday_20_02_2018.zip",
    "Wednesday_21_02_2018.zip",
    "Thursday-22-02-2018.zip",
    "Friday-23-02-2018.zip",
    "Wednesday-28-02-2018.zip",
    "Thursday-01-03-2018.zip",
    "Friday-02-03-2018.zip",
]

SAMPLED_KEYWORDS = ("sample", "sampled")


def safe_join(base: Path, member_name: str) -> Path:
    target = (base / member_name).resolve()
    base_resolved = base.resolve()
    if base_resolved != target and base_resolved not in target.parents:
        raise ValueError(f"ZIP成员路径不安全: {member_name}")
    return target


def validate_zip_members(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            safe_join(output_dir, member.filename)


def extract_zip_with_python(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = safe_join(output_dir, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def extract_zip_with_external_tool(zip_path: Path, output_dir: Path) -> bool:
    tool = shutil.which("7z") or shutil.which("7za")
    if tool:
        cmd = [tool, "x", "-y", f"-o{output_dir}", str(zip_path)]
    else:
        tool = shutil.which("unzip")
        if not tool:
            return False
        cmd = [tool, "-o", str(zip_path), "-d", str(output_dir)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip().splitlines()
        first_err = err[0] if err else "unknown"
        raise RuntimeError(f"外部解压失败: {zip_path.name}: {first_err[:200]}")
    return True


def extract_zip_safe(zip_path: Path, output_dir: Path) -> None:
    try:
        extract_zip_with_python(zip_path, output_dir)
    except NotImplementedError as exc:
        validate_zip_members(zip_path, output_dir)
        print(f"  Python不支持该ZIP压缩格式，尝试外部解压工具: {zip_path.name}")
        if not extract_zip_with_external_tool(zip_path, output_dir):
            raise RuntimeError(
                "当前ZIP可能使用Deflate64，Python标准库无法解压。"
                "请在服务器安装 unzip 或 p7zip 后重试。"
            ) from exc


def download_zip(zip_name: str, dataset_url: str, zip_dir: Path, force: bool) -> Path:
    if zip_name not in ALL_ZIPS:
        raise ValueError(f"不允许下载未声明的数据集文件: {zip_name}")

    zip_dir.mkdir(parents=True, exist_ok=True)
    target = zip_dir / zip_name
    if target.exists() and target.stat().st_size > 0 and not force:
        print(f"[SKIP] ZIP已存在: {target}")
        return target

    url = urljoin(dataset_url.rstrip("/") + "/", quote(zip_name))
    partial = target.with_suffix(target.suffix + ".part")
    print(f"[DOWN] {zip_name} <- {url}")

    try:
        with urlopen(url, timeout=60) as resp, open(partial, "wb") as f:
            shutil.copyfileobj(resp, f, length=1024 * 1024)
    except (URLError, TimeoutError, OSError) as exc:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"下载失败: {zip_name}: {exc}") from exc

    if partial.stat().st_size == 0:
        partial.unlink()
        raise RuntimeError(f"下载结果为空: {zip_name}")

    partial.replace(target)
    print(f"[SAVE] {target} ({target.stat().st_size / 1024 / 1024:.1f}MB)")
    return target


def ensure_zips(
    zip_names: list[str],
    dataset_url: str,
    zip_dir: Path,
    download: bool,
    force_download: bool,
    workers: int,
) -> list[Path]:
    if not download:
        missing = [name for name in zip_names if not (zip_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                "缺少数据集ZIP，请加 --download 下载: " + ", ".join(missing)
            )
        return [zip_dir / name for name in zip_names]

    results: dict[str, Path] = {}
    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_zip,
                zip_name,
                dataset_url,
                zip_dir,
                force_download,
            ): zip_name
            for zip_name in zip_names
        }
        for future in as_completed(futures):
            zip_name = futures[future]
            results[zip_name] = future.result()
    return [results[name] for name in zip_names]


def copy_csvs_from_tree(tmpdir: Path, output_dir: Path) -> list[Path]:
    saved = []
    for csv_path in sorted(tmpdir.rglob("*.csv")):
        if any(keyword in csv_path.name.lower() for keyword in SAMPLED_KEYWORDS):
            print(f"  [SKIP] 采样文件: {csv_path.name}")
            continue
        dst = output_dir / csv_path.name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  [SKIP] 已存在: {dst.name}")
        else:
            shutil.copy2(csv_path, dst)
            print(f"  [SAVE] {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f}MB)")
        saved.append(dst)
    return saved


def process_outer_zip(outer_zip_path: Path, output_dir: Path) -> list[Path]:
    print(f"\n处理: {outer_zip_path.name}")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        extract_zip_safe(outer_zip_path, tmpdir)

        inner_zips = sorted(tmpdir.rglob("*.zip"))
        print(f"  找到 {len(inner_zips)} 个内层ZIP")
        for inner_zip in inner_zips:
            inner_output = inner_zip.with_suffix("")
            inner_output.mkdir(parents=True, exist_ok=True)
            print(f"  解压: {inner_zip.name}")
            extract_zip_safe(inner_zip, inner_output)

        return copy_csvs_from_tree(tmpdir, output_dir)


def write_manifest(zip_paths: list[Path], csv_paths: list[Path], output_dir: Path) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "BCCC-CSE-CIC-IDS-2018",
        "mode": "full",
        "zips": [
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for path in zip_paths
        ],
        "csvs": [
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for path in sorted(set(csv_paths), key=lambda p: p.name)
        ],
    }
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest已保存: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载并提取 BCCC-CSE-CIC-IDS-2018 全量CSV"
    )
    parser.add_argument(
        "--dataset-url",
        default=DATASET_URL,
        help=f"数据集下载根地址 (默认: {DATASET_URL})",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="缺失ZIP时自动下载 (默认: 启用，可用 --no-download 关闭)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="重新下载已存在的ZIP",
    )
    parser.add_argument(
        "--zip-dir",
        default=ZIP_DIR,
        help=f"ZIP保存目录 (默认: {ZIP_DIR})",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help=f"CSV输出目录 (默认: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="并发下载线程数 (默认: 2)",
    )
    parser.add_argument(
        "--only",
        choices=ALL_ZIPS,
        help="只处理一个外层ZIP，用于验证",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="兼容旧命令；当前脚本默认就是全量模式",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_dir = Path(args.zip_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_zips = [args.only] if args.only else ALL_ZIPS

    print("=" * 60)
    print("BCCC-CSE-CIC-IDS-2018 全量数据准备")
    print("=" * 60)
    print(f"下载地址: {args.dataset_url}")
    print(f"ZIP目录:  {zip_dir}")
    print(f"CSV目录:  {output_dir}")
    print(f"目标ZIP:  {len(target_zips)} 个")
    print("提示: 全量数据解压、缓存和训练需要较大磁盘与内存空间。")

    zip_paths = ensure_zips(
        target_zips,
        args.dataset_url,
        zip_dir,
        args.download,
        args.force_download,
        args.workers,
    )

    all_csvs = []
    for zip_path in zip_paths:
        all_csvs.extend(process_outer_zip(zip_path, output_dir))

    write_manifest(zip_paths, all_csvs, output_dir)

    print()
    print("=" * 60)
    print(f"处理完成: {len(set(all_csvs))} 个CSV位于 {output_dir}")
    print("下一步: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
