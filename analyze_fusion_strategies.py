"""复用已有checkpoint做决策层融合扫参与对比分析"""

from __future__ import annotations

import argparse
import csv
import os
from contextlib import nullcontext

import numpy as np
import torch
import yaml

from src.data.cache import load_or_prepare_datasets
from src.models.decision_fusion import DecisionFusion
from src.models.threat_intel import ThreatIntelScorer
from src.models.transformer_detector import TransformerDetector
from src.utils.experiment import save_json, save_text, save_yaml, utc_now_iso
from src.utils.metrics import evaluate_metrics


def build_autocast_context(runtime_config: dict, device: torch.device):
    if device.type != "cuda":
        return nullcontext

    precision = runtime_config.get("precision", "fp32").lower()
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        return nullcontext

    def autocast_context():
        return torch.amp.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=True,
        )

    return autocast_context


def load_source_config(source_experiment_dir: str) -> dict:
    config_path = os.path.join(source_experiment_dir, "resolved_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_strategy_specs(alpha_grid: list[float]) -> list[dict]:
    specs = [{"name": "dl_only", "strategy": "dl_only", "alpha": None}]
    for alpha in alpha_grid:
        alpha_label = str(alpha).replace(".", "")
        specs.append(
            {
                "name": f"weighted_average_alpha_{alpha_label}",
                "strategy": "weighted_average",
                "alpha": alpha,
            }
        )
        specs.append(
            {
                "name": f"adaptive_weighted_average_alpha_{alpha_label}",
                "strategy": "adaptive_weighted_average",
                "alpha": alpha,
            }
        )

    specs.extend(
        [
            {
                "name": "soft_voting",
                "strategy": "soft_voting",
                "alpha": 0.5,
            },
            {
                "name": "dempster_shafer",
                "strategy": "dempster_shafer",
                "alpha": None,
            },
        ]
    )
    return specs


def collect_predictions(
    config: dict,
    source_experiment_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data_config = dict(config["data"])
    runtime_config = config.get("runtime", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_context = build_autocast_context(runtime_config, device)
    pin_memory = data_config.get("pin_memory", device.type == "cuda")
    non_blocking = pin_memory and device.type == "cuda"

    datasets, preprocessor, dataset_metadata = load_or_prepare_datasets(data_config)
    dataloaders = preprocessor.create_dataloaders(
        datasets,
        batch_size=data_config.get("batch_size", 256),
        num_workers=data_config.get("num_workers", 0),
        pin_memory=pin_memory,
        persistent_workers=data_config.get("persistent_workers", False),
        prefetch_factor=data_config.get("prefetch_factor", 2),
        use_weighted_sampler=False,
    )

    checkpoint_path = os.path.join(
        source_experiment_dir,
        "checkpoints",
        "best_model.pth",
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = TransformerDetector(
        traffic_dim=checkpoint["traffic_dim"],
        log_dim=checkpoint["log_dim"],
        num_classes=checkpoint["num_classes"],
        **{
            k: v
            for k, v in checkpoint["model_config"].items()
            if k != "num_classes"
        },
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    network_info = dataset_metadata.get("network_info", {})
    test_indices = preprocessor.test_indices
    test_src_ips = (
        np.array(network_info.get("src_ip", []), dtype=object)[test_indices]
        if "src_ip" in network_info
        else None
    )
    test_dst_ips = (
        np.array(network_info.get("dst_ip", []), dtype=object)[test_indices]
        if "dst_ip" in network_info
        else None
    )
    test_dst_ports = (
        np.array(network_info.get("dst_port", []), dtype=np.int64)[test_indices]
        if "dst_port" in network_info
        else None
    )

    threat_api_config = config.get("threat_intel_api", {})
    threat_scorer = ThreatIntelScorer(
        data_config.get("threat_intel_dir", "data/threat_intel"),
        preprocessor.class_names,
        api_url=threat_api_config.get("url"),
        timeout_seconds=threat_api_config.get("timeout_seconds", 5),
        use_search_fallback=threat_api_config.get("use_search_fallback", True),
        cache_ttl_seconds=threat_api_config.get("cache_ttl_seconds", 3600),
    )

    all_probs = []
    all_labels = []
    all_ti_scores = []
    sample_idx = 0
    with torch.no_grad():
        for traffic_batch, log_batch, label_batch in dataloaders["test"]:
            traffic_batch = traffic_batch.to(device, non_blocking=non_blocking)
            log_batch = log_batch.to(device, non_blocking=non_blocking)

            with autocast_context():
                probs = model.predict_proba(traffic_batch, log_batch)
            probs_np = probs.cpu().numpy()
            batch_size = probs_np.shape[0]
            ti_scores = np.zeros_like(probs_np)
            for i in range(batch_size):
                src = test_src_ips[sample_idx] if test_src_ips is not None else None
                dst = test_dst_ips[sample_idx] if test_dst_ips is not None else None
                port = (
                    int(test_dst_ports[sample_idx])
                    if test_dst_ports is not None
                    else None
                )
                ti_scores[i] = threat_scorer.score(src, dst, port)
                sample_idx += 1

            all_probs.append(probs_np)
            all_labels.append(label_batch.numpy())
            all_ti_scores.append(ti_scores)

    y_true = np.concatenate(all_labels)
    dl_probs = np.concatenate(all_probs)
    ti_scores = np.concatenate(all_ti_scores)
    return y_true, dl_probs, ti_scores, {
        "class_names": preprocessor.class_names,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_val_acc": checkpoint["val_acc"],
        "threat_intel_api": threat_scorer.get_diagnostics(),
        "threat_intel_entries": 0,
    }


def build_match_summary(
    y_true: np.ndarray,
    ti_scores: np.ndarray,
    class_names: list[str],
) -> dict:
    num_classes = len(class_names)
    uniform_max = 1.0 / num_classes
    matched_mask = ti_scores.max(axis=1) > (uniform_max + 1e-6)

    class_totals = {name: 0 for name in class_names}
    class_matches = {name: 0 for name in class_names}
    for label in y_true:
        class_totals[class_names[int(label)]] += 1
    for is_matched, label in zip(matched_mask, y_true):
        if is_matched:
            class_matches[class_names[int(label)]] += 1

    match_rate_by_class = {
        name: (
            class_matches[name] / class_totals[name]
            if class_totals[name]
            else 0.0
        )
        for name in class_names
    }

    return {
        "matched_samples": int(matched_mask.sum()),
        "total_samples": int(len(matched_mask)),
        "match_rate": float(matched_mask.mean()),
        "class_totals": class_totals,
        "class_matches": class_matches,
        "class_match_rate": match_rate_by_class,
        "max_score_mean": float(ti_scores.max(axis=1).mean()),
        "max_score_mean_matched": (
            float(ti_scores[matched_mask].max(axis=1).mean())
            if matched_mask.any()
            else 0.0
        ),
    }


def run_study(
    source_experiment_dir: str,
    output_dir: str,
    alpha_grid: list[float],
    threat_intel_api_url: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    config = load_source_config(source_experiment_dir)
    if threat_intel_api_url:
        config.setdefault("threat_intel_api", {})["url"] = threat_intel_api_url

    resolved = dict(config)
    resolved.setdefault("fusion_study", {})
    resolved["fusion_study"].update(
        {
            "source_experiment_dir": os.path.abspath(source_experiment_dir),
            "threat_intel_api_url": config.get("threat_intel_api", {}).get("url"),
            "output_dir": os.path.abspath(output_dir),
            "alpha_grid": alpha_grid,
            "timestamp_utc": utc_now_iso(),
        }
    )
    save_yaml(os.path.join(output_dir, "resolved_study_config.yaml"), resolved)

    y_true, dl_probs, ti_scores, metadata = collect_predictions(
        config=config,
        source_experiment_dir=source_experiment_dir,
    )
    match_summary = build_match_summary(
        y_true=y_true,
        ti_scores=ti_scores,
        class_names=metadata["class_names"],
    )
    save_json(os.path.join(output_dir, "threat_intel_match_summary.json"), match_summary)

    strategy_rows = []
    for spec in get_strategy_specs(alpha_grid):
        if spec["strategy"] == "dl_only":
            y_pred = dl_probs.argmax(axis=1)
        else:
            fusion = DecisionFusion(
                strategy=spec["strategy"],
                alpha=spec["alpha"] if spec["alpha"] is not None else 0.8,
            )
            y_pred = fusion.predict(dl_probs, ti_scores)

        metrics = evaluate_metrics(y_true, y_pred, metadata["class_names"])
        row = {
            "name": spec["name"],
            "strategy": spec["strategy"],
            "alpha": spec["alpha"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1": metrics["f1"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
        }
        strategy_rows.append(row)

    strategy_rows.sort(key=lambda item: item["f1_macro"], reverse=True)

    with open(
        os.path.join(output_dir, "strategy_metrics.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(strategy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(strategy_rows)

    save_json(os.path.join(output_dir, "strategy_metrics.json"), strategy_rows)
    save_json(os.path.join(output_dir, "metadata.json"), metadata)

    top_lines = [
        f"source_experiment_dir: {os.path.abspath(source_experiment_dir)}",
        f"threat_intel_api_url: {metadata['threat_intel_api'].get('api_url')}",
        f"threat_intel_api_available: {metadata['threat_intel_api'].get('api_available')}",
        f"match_rate: {match_summary['match_rate']:.6f}",
        "",
        "Top strategies by f1_macro:",
    ]
    for row in strategy_rows[:8]:
        top_lines.append(
            f"- {row['name']}: "
            f"acc={row['accuracy']:.6f}, "
            f"balanced_acc={row['balanced_accuracy']:.6f}, "
            f"f1_macro={row['f1_macro']:.6f}"
        )
    save_text(os.path.join(output_dir, "report.md"), "\n".join(top_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="融合策略扫参与分析")
    parser.add_argument(
        "--source-experiment-dir",
        required=True,
        help="已有实验目录，例如 /root/autodl-tmp/cc-bishe-experiments/exp05_ablation_concat_no_attention",
    )
    parser.add_argument(
        "--threat-intel-api-url",
        default=None,
        help="wxqb威胁情报API地址，例如 http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="分析结果输出目录",
    )
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
        help="weighted/adaptive weighted average的alpha网格",
    )
    args = parser.parse_args()

    run_study(
        source_experiment_dir=args.source_experiment_dir,
        output_dir=args.output_dir,
        alpha_grid=args.alphas,
        threat_intel_api_url=args.threat_intel_api_url,
    )


if __name__ == "__main__":
    main()
