"""汇总实验结果到CSV与Markdown"""

from __future__ import annotations

import argparse
import csv
import os

import yaml


def safe_load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_load_json(path: str) -> dict:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows(experiments_root: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(experiments_root)):
        exp_dir = os.path.join(experiments_root, name)
        if not os.path.isdir(exp_dir):
            continue

        metrics_path = os.path.join(exp_dir, "artifacts", "evaluation_metrics.json")
        train_path = os.path.join(exp_dir, "artifacts", "train_summary.json")
        config_path = os.path.join(exp_dir, "resolved_config.yaml")
        status_path = os.path.join(exp_dir, "run_status.json")

        if not all(
            os.path.exists(path)
            for path in (metrics_path, train_path, config_path, status_path)
        ):
            continue

        metrics = safe_load_json(metrics_path)
        train_summary = safe_load_json(train_path)
        config = safe_load_yaml(config_path)
        status = safe_load_json(status_path)

        row = {
            "experiment": name,
            "fusion_strategy": config.get("model", {}).get(
                "fusion_strategy", "cross_attention"
            ),
            "decision_fusion": config.get("fusion", {}).get(
                "strategy", "weighted_average"
            ),
            "decision_alpha": config.get("fusion", {}).get("alpha"),
            "loss": config.get("imbalance", {}).get("loss", "cross_entropy"),
            "use_class_weights": config.get("imbalance", {}).get(
                "use_class_weights", False
            ),
            "use_weighted_sampler": config.get("imbalance", {}).get(
                "use_weighted_sampler", False
            ),
            "threat_intel_dir": config.get("data", {}).get("threat_intel_dir"),
            "threat_intel_entries": metrics.get("threat_intel_entries"),
            "batch_size": config.get("data", {}).get("batch_size"),
            "num_workers": config.get("data", {}).get("num_workers"),
            "precision": config.get("runtime", {}).get("precision", "fp32"),
            "epochs_completed": train_summary.get("epochs_completed"),
            "best_epoch": train_summary.get("best_epoch"),
            "best_val_loss": train_summary.get("best_val_loss"),
            "train_seconds": round(train_summary.get("train_seconds", 0.0), 2),
            "eval_seconds": round(status.get("evaluate_seconds", 0.0), 2),
            "dl_accuracy": metrics["dl_metrics"]["accuracy"],
            "dl_balanced_accuracy": metrics["dl_metrics"]["balanced_accuracy"],
            "dl_f1": metrics["dl_metrics"]["f1"],
            "dl_f1_macro": metrics["dl_metrics"]["f1_macro"],
            "fused_accuracy": metrics["fused_metrics"]["accuracy"],
            "fused_balanced_accuracy": metrics["fused_metrics"][
                "balanced_accuracy"
            ],
            "fused_f1": metrics["fused_metrics"]["f1"],
            "fused_f1_macro": metrics["fused_metrics"]["f1_macro"],
        }
        rows.append(row)
    return rows


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, rows: list[dict]):
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="汇总实验结果")
    parser.add_argument(
        "--experiments-root",
        default="/root/autodl-tmp/cc-bishe-experiments",
    )
    args = parser.parse_args()

    rows = collect_rows(args.experiments_root)
    os.makedirs(args.experiments_root, exist_ok=True)
    write_csv(os.path.join(args.experiments_root, "summary.csv"), rows)
    write_markdown(os.path.join(args.experiments_root, "summary.md"), rows)


if __name__ == "__main__":
    main()
