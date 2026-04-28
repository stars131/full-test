"""Agent模式演示 - 交互式展示规则Agent和LLM Agent的检测能力"""

import os
import argparse
import sys

import yaml
import numpy as np
import torch

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import Preprocessor
from src.models.transformer_detector import TransformerDetector
from src.models.threat_intel import ThreatIntelScorer
from src.agent.pipeline import AgentPipeline


def load_components(config_path: str, device: str = "cpu"):
    """加载所有组件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    train_config = config["training"]

    # 加载数据以获取特征信息
    print("加载数据集...")
    loader = DataLoader(
        data_config["raw_dir"],
        full_dataset=data_config.get("full_dataset", False),
        exclude_patterns=data_config.get("exclude_patterns", []),
    )
    data = loader.load(data_config.get("file_pattern", "*.csv"))

    engineer = FeatureEngineer()
    engineer.fit(data)
    traffic_features, log_features = engineer.transform(data)
    traffic_dim, log_dim = engineer.get_feature_dims()
    labels = data[loader.label_col].values
    network_indicators = loader.get_network_indicators(data)

    preprocessor = Preprocessor()
    datasets = preprocessor.fit_transform(
        traffic_features, log_features, labels,
        test_size=data_config.get("test_size", 0.1),
        val_size=data_config.get("val_size", 0.1),
        random_seed=data_config.get("random_seed", 42),
    )

    # 加载模型
    print("加载模型...")
    checkpoint_dir = train_config.get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"错误: 未找到模型文件 {checkpoint_path}")
        print("请先运行 python train.py 训练模型")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TransformerDetector(
        traffic_dim=checkpoint["traffic_dim"],
        log_dim=checkpoint["log_dim"],
        num_classes=checkpoint["num_classes"],
        **{k: v for k, v in checkpoint["model_config"].items()
           if k != "num_classes"},
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 威胁情报
    threat_api_config = config.get("threat_intel_api", {})
    threat_scorer = ThreatIntelScorer(
        data_config.get("threat_intel_dir", "data/threat_intel"),
        preprocessor.class_names,
        api_url=threat_api_config.get("url"),
        timeout_seconds=threat_api_config.get("timeout_seconds", 5),
        use_search_fallback=threat_api_config.get("use_search_fallback", True),
        cache_ttl_seconds=threat_api_config.get("cache_ttl_seconds", 3600),
    )

    test_indices = preprocessor.test_indices
    test_network_indicators = {
        "src_ips": np.array(network_indicators.get("src_ips", []), dtype=object)[test_indices]
        if "src_ips" in network_indicators
        else None,
        "dst_ips": np.array(network_indicators.get("dst_ips", []), dtype=object)[test_indices]
        if "dst_ips" in network_indicators
        else None,
        "dst_ports": np.array(network_indicators.get("dst_ports", []), dtype=object)[test_indices]
        if "dst_ports" in network_indicators
        else None,
    }

    return config, model, preprocessor, threat_scorer, datasets, engineer, test_network_indicators


def demo_single(pipeline, datasets, preprocessor, engineer, network_indicators):
    """单样本检测演示"""
    test_dataset = datasets["test"]
    n = len(test_dataset)

    print(f"\n测试集共 {n} 条数据")
    print("输入样本索引 (0-{}) 进行检测，输入 'q' 退出，输入 'r' 随机选取".format(n - 1))

    while True:
        user_input = input("\n> 请输入样本索引: ").strip()

        if user_input.lower() == "q":
            break
        elif user_input.lower() == "r":
            idx = np.random.randint(0, n)
        else:
            try:
                idx = int(user_input)
                if idx < 0 or idx >= n:
                    print(f"索引超出范围 [0, {n - 1}]")
                    continue
            except ValueError:
                print("请输入有效的数字索引")
                continue

        traffic, log, label = test_dataset[idx]

        # 获取真实标签
        true_label = preprocessor.inverse_label(np.array([label.item()]))[0]

        print(f"\n{'=' * 60}")
        print(f"样本索引: {idx}")
        print(f"真实标签: {true_label}")
        print(f"{'=' * 60}")

        flow_data = {
            "traffic_features": traffic.numpy(),
            "log_features": log.numpy(),
        }
        if network_indicators.get("src_ips") is not None:
            flow_data["src_ip"] = network_indicators["src_ips"][idx]
        if network_indicators.get("dst_ips") is not None:
            flow_data["dst_ip"] = network_indicators["dst_ips"][idx]
        if network_indicators.get("dst_ports") is not None:
            flow_data["dst_port"] = network_indicators["dst_ports"][idx]

        result = pipeline.process_single(flow_data)

        print(f"\n--- 检测结果 ---")
        print(result["explanation"])
        print(f"\n真实标签: {true_label}")

        correct = result["decision"]["final_class"] == true_label
        print(f"判定: {'正确 ✓' if correct else '错误 ✗'}")


def demo_batch(pipeline, datasets, preprocessor, network_indicators, num_samples=20):
    """批量检测演示"""
    test_dataset = datasets["test"]
    n = min(num_samples, len(test_dataset))

    print(f"\n随机选取 {n} 条数据进行批量检测...")

    indices = np.random.choice(len(test_dataset), n, replace=False)

    traffic_batch = []
    log_batch = []
    true_labels = []
    src_ips = []
    dst_ips = []
    dst_ports = []

    for idx in indices:
        traffic, log, label = test_dataset[idx]
        traffic_batch.append(traffic.numpy())
        log_batch.append(log.numpy())
        true_labels.append(label.item())
        if network_indicators.get("src_ips") is not None:
            src_ips.append(network_indicators["src_ips"][idx])
        if network_indicators.get("dst_ips") is not None:
            dst_ips.append(network_indicators["dst_ips"][idx])
        if network_indicators.get("dst_ports") is not None:
            dst_ports.append(network_indicators["dst_ports"][idx])

    traffic_batch = np.array(traffic_batch)
    log_batch = np.array(log_batch)

    results = pipeline.process_batch(
        traffic_batch,
        log_batch,
        src_ips=src_ips or None,
        dst_ips=dst_ips or None,
        dst_ports=dst_ports or None,
    )
    report = pipeline.generate_report(results)

    print(f"\n{'=' * 60}")
    print("批量检测报告")
    print(f"{'=' * 60}")
    print(f"总流量数: {report['total_flows']}")
    print(f"攻击流量: {report['attack_flows']}")
    print(f"正常流量: {report['benign_flows']}")
    print(f"攻击比例: {report['attack_ratio']:.2%}")
    print(f"平均置信度: {report['average_confidence']:.4f}")
    print(f"\n类别分布:")
    for cls, count in report["class_distribution"].items():
        print(f"  {cls}: {count}")

    # 计算准确率
    correct = 0
    for i, result in enumerate(results):
        true_label = preprocessor.inverse_label(
            np.array([true_labels[i]])
        )[0]
        pred_label = result["decision"]["final_class"]
        if pred_label == true_label:
            correct += 1

    print(f"\n批量检测准确率: {correct}/{n} = {correct/n:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Agent模式演示")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["rule", "llm"],
        help="Agent模式 (覆盖config中的设置)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="批量检测模式",
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="批量检测样本数",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    (
        config,
        model,
        preprocessor,
        threat_scorer,
        datasets,
        engineer,
        network_indicators,
    ) = load_components(args.config, device)

    # 覆盖Agent模式
    if args.mode:
        config["agent"]["mode"] = args.mode

    print(f"\n当前Agent模式: {config['agent']['mode']}")

    pipeline = AgentPipeline(
        model=model,
        preprocessor=preprocessor,
        threat_scorer=threat_scorer,
        config=config,
        device=device,
    )

    if args.batch:
        demo_batch(pipeline, datasets, preprocessor, network_indicators, args.num_samples)
    else:
        demo_single(pipeline, datasets, preprocessor, engineer, network_indicators)


if __name__ == "__main__":
    main()
