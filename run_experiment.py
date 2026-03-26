"""单次实验运行器 - 统一保存配置、日志、指标和元数据"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy

import yaml

from src.utils.experiment import (
    collect_runtime_metadata,
    prepare_experiment_config,
    save_json,
    save_yaml,
)


def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def stream_to_console_and_file(command: list[str], log_path: str, cwd: str) -> int:
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.wait()
        return process.returncode


def main():
    parser = argparse.ArgumentParser(description="运行单次完整实验")
    parser.add_argument("--config", required=True, help="基础配置文件路径")
    parser.add_argument("--name", required=True, help="实验名称")
    parser.add_argument(
        "--overrides",
        help="额外覆盖配置的YAML文件路径",
    )
    parser.add_argument(
        "--experiments-root",
        default="/root/autodl-tmp/cc-bishe-experiments",
        help="实验根目录",
    )
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.overrides:
        with open(args.overrides, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
        config = deep_update(deepcopy(config), overrides)

    resolved_config, experiment_dir = prepare_experiment_config(
        config,
        experiment_name=args.name,
        experiments_root=args.experiments_root,
    )

    resolved_config_path = os.path.join(experiment_dir, "resolved_config.yaml")
    save_yaml(resolved_config_path, resolved_config)

    metadata = collect_runtime_metadata(repo_dir)
    metadata.update(
        {
            "experiment_name": args.name,
            "base_config": os.path.abspath(args.config),
            "overrides_config": (
                os.path.abspath(args.overrides) if args.overrides else None
            ),
            "resolved_config": resolved_config_path,
        }
    )
    save_json(os.path.join(experiment_dir, "metadata.json"), metadata)

    logs_dir = resolved_config["experiment"]["logs_dir"]
    run_status = {
        "experiment_name": args.name,
        "started_at": time.time(),
    }

    train_cmd = [
        sys.executable,
        "train.py",
        "--config",
        resolved_config_path,
    ]
    train_log = os.path.join(logs_dir, "train.log")
    train_start = time.perf_counter()
    train_code = stream_to_console_and_file(train_cmd, train_log, repo_dir)
    run_status["train_exit_code"] = train_code
    run_status["train_seconds"] = time.perf_counter() - train_start
    save_json(os.path.join(experiment_dir, "run_status.json"), run_status)
    if train_code != 0:
        raise SystemExit(train_code)

    eval_cmd = [
        sys.executable,
        "evaluate.py",
        "--config",
        resolved_config_path,
    ]
    eval_log = os.path.join(logs_dir, "evaluate.log")
    eval_start = time.perf_counter()
    eval_code = stream_to_console_and_file(eval_cmd, eval_log, repo_dir)
    run_status["evaluate_exit_code"] = eval_code
    run_status["evaluate_seconds"] = time.perf_counter() - eval_start
    run_status["finished_at"] = time.time()
    save_json(os.path.join(experiment_dir, "run_status.json"), run_status)

    if eval_code != 0:
        raise SystemExit(eval_code)


if __name__ == "__main__":
    main()
