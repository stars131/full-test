"""评估入口 - 加载模型，在测试集上全面评估"""

import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch
import yaml

from src.data.cache import load_or_prepare_datasets
from src.models.decision_fusion import DecisionFusion
from src.models.threat_intel import ThreatIntelScorer
from src.models.transformer_detector import TransformerDetector
from src.utils.experiment import save_json, save_text, save_yaml
from src.utils.metrics import (
    evaluate_metrics,
    get_classification_report_dict,
    get_confusion_matrix,
    print_classification_report,
)
from src.utils.visualization import Visualizer


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


def evaluate(config_path: str = "config.yaml"):
    """评估主流程"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    train_config = config["training"]
    fusion_config = config["fusion"]
    vis_config = config.get("visualization", {})
    runtime_config = config.get("runtime", {})
    experiment_config = config.get("experiment", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    autocast_context = build_autocast_context(runtime_config, device)
    pin_memory = data_config.get("pin_memory", device.type == "cuda")
    non_blocking = pin_memory and device.type == "cuda"

    # ===== 加载数据 =====
    print("\n加载数据...")
    datasets, preprocessor, dataset_metadata = load_or_prepare_datasets(
        data_config
    )
    network_info = dataset_metadata.get("network_info", {})
    dataloaders = preprocessor.create_dataloaders(
        datasets,
        batch_size=data_config.get("batch_size", 256),
        num_workers=data_config.get("num_workers", 0),
        pin_memory=pin_memory,
        persistent_workers=data_config.get("persistent_workers", False),
        prefetch_factor=data_config.get("prefetch_factor", 2),
        use_weighted_sampler=False,
    )

    # ===== 加载模型 =====
    print("\n加载模型...")
    checkpoint_path = os.path.join(
        train_config.get("checkpoint_dir", "checkpoints"),
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
    print(
        "模型加载自 epoch "
        f"{checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.4f}"
    )

    visualizer = Visualizer(
        output_dir=vis_config.get("output_dir", "results/figures"),
        dpi=vis_config.get("dpi", 150),
        figsize=tuple(vis_config.get("figsize", [10, 8])),
    )

    # ===== 评估1: 仅DL模型 =====
    print("\n" + "=" * 50)
    print("评估1: 仅深度学习模型")
    print("=" * 50)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for traffic_batch, log_batch, label_batch in dataloaders["test"]:
            traffic_batch = traffic_batch.to(device, non_blocking=non_blocking)
            log_batch = log_batch.to(device, non_blocking=non_blocking)

            with autocast_context():
                probs = model.predict_proba(traffic_batch, log_batch)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(label_batch.numpy())

    y_true = np.concatenate(all_labels)
    y_pred_dl = np.concatenate(all_preds)

    dl_metrics = evaluate_metrics(y_true, y_pred_dl, preprocessor.class_names)
    print("\n仅DL模型指标:")
    for key, value in dl_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n分类报告:")
    dl_report_text = print_classification_report(
        y_true, y_pred_dl, preprocessor.class_names
    )
    dl_report_dict = get_classification_report_dict(
        y_true, y_pred_dl, preprocessor.class_names
    )

    cm = get_confusion_matrix(y_true, y_pred_dl)
    visualizer.plot_confusion_matrix(
        cm,
        preprocessor.class_names,
        save_name="confusion_matrix_dl.png",
    )

    # ===== 评估2: DL + 威胁情报融合 =====
    print("\n" + "=" * 50)
    print("评估2: DL + 威胁情报融合")
    print("=" * 50)

    threat_scorer = ThreatIntelScorer(
        data_config.get("threat_intel_dir", "data/threat_intel"),
        preprocessor.class_names,
        api_url=config.get("threat_intel_api", {}).get("url"),
    )
    decision_fusion = DecisionFusion(
        strategy=fusion_config.get("strategy", "weighted_average"),
        alpha=fusion_config.get("alpha", 0.8),
    )

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

    all_fused_preds = []
    sample_idx = 0

    with torch.no_grad():
        for traffic_batch, log_batch, _ in dataloaders["test"]:
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

            fused_preds = decision_fusion.predict(probs_np, ti_scores)
            all_fused_preds.append(fused_preds)

    y_pred_fused = np.concatenate(all_fused_preds)
    fused_metrics = evaluate_metrics(
        y_true, y_pred_fused, preprocessor.class_names
    )
    print("\nDL+威胁情报融合指标:")
    for key, value in fused_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n分类报告:")
    fused_report_text = print_classification_report(
        y_true, y_pred_fused, preprocessor.class_names
    )
    fused_report_dict = get_classification_report_dict(
        y_true, y_pred_fused, preprocessor.class_names
    )

    cm_fused = get_confusion_matrix(y_true, y_pred_fused)
    visualizer.plot_confusion_matrix(
        cm_fused,
        preprocessor.class_names,
        save_name="confusion_matrix_fused.png",
    )

    # ===== 指标对比 =====
    print("\n" + "=" * 50)
    print("指标对比")
    print("=" * 50)

    metrics_comparison = {
        "DL Only": dl_metrics,
        "DL + Threat Intel": fused_metrics,
    }
    visualizer.plot_metrics_comparison(metrics_comparison)
    visualizer.plot_label_distribution(y_true, preprocessor.class_names)

    artifacts_dir = experiment_config.get(
        "artifacts_dir",
        train_config.get("log_dir", train_config.get("checkpoint_dir", ".")),
    )
    reports_dir = experiment_config.get("reports_dir", artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    evaluation_summary = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_val_acc": checkpoint["val_acc"],
        "class_names": preprocessor.class_names,
        "threat_intel_dir": data_config.get("threat_intel_dir"),
        "decision_fusion_strategy": fusion_config.get(
            "strategy", "weighted_average"
        ),
        "decision_fusion_alpha": fusion_config.get("alpha"),
        "dl_metrics": dl_metrics,
        "fused_metrics": fused_metrics,
        "threat_intel_entries": len(threat_scorer.intel_db),
    }
    save_json(
        os.path.join(artifacts_dir, "evaluation_metrics.json"),
        evaluation_summary,
    )
    save_json(
        os.path.join(artifacts_dir, "classification_report_dl.json"),
        dl_report_dict,
    )
    save_json(
        os.path.join(artifacts_dir, "classification_report_fused.json"),
        fused_report_dict,
    )
    save_json(
        os.path.join(artifacts_dir, "confusion_matrix_dl.json"),
        cm.tolist(),
    )
    save_json(
        os.path.join(artifacts_dir, "confusion_matrix_fused.json"),
        cm_fused.tolist(),
    )
    save_text(
        os.path.join(reports_dir, "classification_report_dl.txt"),
        dl_report_text,
    )
    save_text(
        os.path.join(reports_dir, "classification_report_fused.txt"),
        fused_report_text,
    )
    save_yaml(os.path.join(artifacts_dir, "resolved_config.yaml"), config)

    print(
        "\n评估完成！结果已保存至:",
        vis_config.get("output_dir", "results/figures"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估网络攻击检测模型")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    args = parser.parse_args()
    evaluate(args.config)
