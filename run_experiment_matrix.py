"""批量运行实验矩阵"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml


EXPERIMENTS = [
    {
        "name": "exp01_baseline_cross_attention",
        "overrides": {},
    },
    {
        "name": "exp02_class_weight_ce",
        "overrides": {
            "imbalance": {
                "use_class_weights": True,
                "loss": "cross_entropy",
            }
        },
    },
    {
        "name": "exp03_focal_class_weight",
        "overrides": {
            "imbalance": {
                "use_class_weights": True,
                "loss": "focal",
                "focal_gamma": 2.0,
            }
        },
    },
    {
        "name": "exp04_weighted_sampler_ce",
        "overrides": {
            "imbalance": {
                "use_weighted_sampler": True,
                "loss": "cross_entropy",
            }
        },
    },
    {
        "name": "exp05_ablation_concat_no_attention",
        "overrides": {
            "model": {
                "fusion_strategy": "concat",
            }
        },
    },
    {
        "name": "exp06_ablation_traffic_only",
        "overrides": {
            "model": {
                "fusion_strategy": "traffic_only",
            }
        },
    },
    {
        "name": "exp08_ablation_log_only",
        "overrides": {
            "model": {
                "fusion_strategy": "log_only",
            }
        },
    },
    {
        "name": "exp09_concat_wxqb_weighted_alpha07",
        "overrides": {
            "model": {
                "fusion_strategy": "concat",
            },
            "threat_intel_api": {
                "url": "http://127.0.0.1:8000",
            },
            "fusion": {
                "strategy": "weighted_average",
                "alpha": 0.7,
            },
        },
    },
    {
        "name": "exp10_concat_wxqb_soft_voting",
        "overrides": {
            "model": {
                "fusion_strategy": "concat",
            },
            "threat_intel_api": {
                "url": "http://127.0.0.1:8000",
            },
            "fusion": {
                "strategy": "soft_voting",
                "alpha": 0.5,
            },
        },
    },
    {
        "name": "exp11_weighted_sampler_wxqb_alpha07",
        "overrides": {
            "imbalance": {
                "use_weighted_sampler": True,
                "loss": "cross_entropy",
            },
            "threat_intel_api": {
                "url": "http://127.0.0.1:8000",
            },
            "fusion": {
                "strategy": "weighted_average",
                "alpha": 0.7,
            },
        },
    },
]


def main():
    parser = argparse.ArgumentParser(description="运行实验矩阵")
    parser.add_argument(
        "--config",
        default="config.experiment_base.yaml",
        help="基础配置文件",
    )
    parser.add_argument(
        "--experiments-root",
        default="/root/autodl-tmp/cc-bishe-experiments",
    )
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    overrides_dir = os.path.join(args.experiments_root, "_overrides")
    os.makedirs(overrides_dir, exist_ok=True)

    for experiment in EXPERIMENTS:
        name = experiment["name"]
        completed_marker = os.path.join(
            args.experiments_root,
            name,
            "artifacts",
            "evaluation_metrics.json",
        )
        if os.path.exists(completed_marker):
            print(f"跳过已完成实验: {name}")
            continue

        overrides_path = os.path.join(overrides_dir, f"{name}.yaml")
        with open(overrides_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                experiment["overrides"],
                f,
                allow_unicode=True,
                sort_keys=False,
            )

        command = [
            sys.executable,
            "run_experiment.py",
            "--config",
            args.config,
            "--name",
            name,
            "--overrides",
            overrides_path,
            "--experiments-root",
            args.experiments_root,
        ]
        subprocess.run(command, cwd=repo_dir, check=True)

    subprocess.run(
        [
            sys.executable,
            "summarize_experiments.py",
            "--experiments-root",
            args.experiments_root,
        ],
        cwd=repo_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
