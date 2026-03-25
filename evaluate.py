"""评估入口 - 加载模型，在测试集上全面评估"""

import os
import argparse

import yaml
import numpy as np
import pandas as pd
import torch

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import Preprocessor
from src.models.transformer_detector import TransformerDetector
from src.models.threat_intel import ThreatIntelScorer
from src.models.decision_fusion import DecisionFusion
from src.utils.metrics import evaluate_metrics, print_classification_report, get_confusion_matrix
from src.utils.visualization import Visualizer


def evaluate(config_path: str = "config.yaml"):
    """评估主流程"""

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    train_config = config["training"]
    fusion_config = config["fusion"]
    vis_config = config.get("visualization", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ===== 加载数据 =====
    print("\n加载数据...")
    loader = DataLoader(data_config["raw_dir"])
    data = loader.load(data_config.get("file_pattern", "*.csv"))

    engineer = FeatureEngineer()
    engineer.fit(data)
    traffic_features, log_features = engineer.transform(data)
    traffic_dim, log_dim = engineer.get_feature_dims()
    labels = data[loader.label_col].values

    # 保留IP信息用于威胁情报查询
    network_info = {}
    if "src_ip" in data.columns:
        network_info["src_ip"] = data["src_ip"].values
    if "dst_ip" in data.columns:
        network_info["dst_ip"] = data["dst_ip"].values
    if "dst_port" in data.columns:
        network_info["dst_port"] = pd.to_numeric(
            data["dst_port"], errors="coerce"
        ).fillna(0).astype(int).values

    preprocessor = Preprocessor()
    datasets = preprocessor.fit_transform(
        traffic_features, log_features, labels,
        test_size=data_config.get("test_size", 0.1),
        val_size=data_config.get("val_size", 0.1),
        random_seed=data_config.get("random_seed", 42),
    )
    dataloaders = preprocessor.create_dataloaders(
        datasets,
        batch_size=data_config.get("batch_size", 256),
        num_workers=data_config.get("num_workers", 0),
    )

    # ===== 加载模型 =====
    print("\n加载模型...")
    checkpoint_dir = train_config.get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = TransformerDetector(
        traffic_dim=checkpoint["traffic_dim"],
        log_dim=checkpoint["log_dim"],
        num_classes=checkpoint["num_classes"],
        **{k: v for k, v in checkpoint["model_config"].items()
           if k != "num_classes"},
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"模型加载自 epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.4f}")

    # ===== 可视化工具 =====
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
            traffic_batch = traffic_batch.to(device)
            log_batch = log_batch.to(device)

            probs = model.predict_proba(traffic_batch, log_batch)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(label_batch.numpy())

    y_true = np.concatenate(all_labels)
    y_pred_dl = np.concatenate(all_preds)

    dl_metrics = evaluate_metrics(y_true, y_pred_dl, preprocessor.class_names)
    print(f"\n仅DL模型指标:")
    for k, v in dl_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n分类报告:")
    print_classification_report(y_true, y_pred_dl, preprocessor.class_names)

    cm = get_confusion_matrix(y_true, y_pred_dl)
    visualizer.plot_confusion_matrix(
        cm, preprocessor.class_names, save_name="confusion_matrix_dl.png"
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

    # 获取测试集对应的IP信息用于威胁情报查询
    test_indices = preprocessor.test_indices
    test_src_ips = network_info.get("src_ip", np.array([]))[test_indices] if "src_ip" in network_info else None
    test_dst_ips = network_info.get("dst_ip", np.array([]))[test_indices] if "dst_ip" in network_info else None
    test_dst_ports = network_info.get("dst_port", np.array([]))[test_indices] if "dst_port" in network_info else None

    all_fused_preds = []
    all_dl_probs = []
    sample_idx = 0

    with torch.no_grad():
        for traffic_batch, log_batch, label_batch in dataloaders["test"]:
            traffic_batch = traffic_batch.to(device)
            log_batch = log_batch.to(device)

            probs = model.predict_proba(traffic_batch, log_batch)
            probs_np = probs.cpu().numpy()
            all_dl_probs.append(probs_np)

            batch_size = probs_np.shape[0]
            ti_scores = np.zeros_like(probs_np)

            for i in range(batch_size):
                src = test_src_ips[sample_idx] if test_src_ips is not None else None
                dst = test_dst_ips[sample_idx] if test_dst_ips is not None else None
                port = int(test_dst_ports[sample_idx]) if test_dst_ports is not None else None
                ti_scores[i] = threat_scorer.score(src, dst, port)
                sample_idx += 1

            fused_preds = decision_fusion.predict(probs_np, ti_scores)
            all_fused_preds.append(fused_preds)

    y_pred_fused = np.concatenate(all_fused_preds)

    fused_metrics = evaluate_metrics(
        y_true, y_pred_fused, preprocessor.class_names
    )
    print(f"\nDL+威胁情报融合指标:")
    for k, v in fused_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n分类报告:")
    print_classification_report(y_true, y_pred_fused, preprocessor.class_names)

    cm_fused = get_confusion_matrix(y_true, y_pred_fused)
    visualizer.plot_confusion_matrix(
        cm_fused, preprocessor.class_names,
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

    # 标签分布
    visualizer.plot_label_distribution(
        y_true, preprocessor.class_names
    )

    print("\n评估完成！结果已保存至:", vis_config.get("output_dir", "results/figures"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估网络攻击检测模型")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    args = parser.parse_args()
    evaluate(args.config)
