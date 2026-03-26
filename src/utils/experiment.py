"""实验记录与元数据工具"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import yaml


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_yaml(path: str, payload: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def save_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_jsonl(path: str, payload: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit(repo_dir: str) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            text=True,
        )
        return output.strip()
    except Exception:
        return None


def collect_runtime_metadata(repo_dir: str) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": os.path.realpath(os.sys.executable),
        "git_commit": get_git_commit(repo_dir),
    }


def prepare_experiment_config(
    base_config: dict[str, Any],
    experiment_name: str,
    experiments_root: str,
) -> tuple[dict[str, Any], str]:
    cfg = deepcopy(base_config)
    exp_dir = os.path.join(experiments_root, experiment_name)
    artifacts_dir = os.path.join(exp_dir, "artifacts")
    logs_dir = os.path.join(exp_dir, "logs")
    figures_dir = os.path.join(exp_dir, "figures")
    processed_dir = os.path.join(exp_dir, "processed")
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    reports_dir = os.path.join(exp_dir, "reports")

    for path in (
        exp_dir,
        artifacts_dir,
        logs_dir,
        figures_dir,
        processed_dir,
        checkpoints_dir,
        reports_dir,
    ):
        ensure_dir(path)

    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "name": experiment_name,
            "root_dir": experiments_root,
            "dir": exp_dir,
            "artifacts_dir": artifacts_dir,
            "logs_dir": logs_dir,
            "reports_dir": reports_dir,
        }
    )

    cfg.setdefault("data", {})
    cfg["data"]["processed_dir"] = processed_dir
    cfg.setdefault("training", {})
    cfg["training"]["checkpoint_dir"] = checkpoints_dir
    cfg["training"]["log_dir"] = logs_dir
    cfg.setdefault("visualization", {})
    cfg["visualization"]["output_dir"] = figures_dir

    return cfg, exp_dir
