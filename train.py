"""训练入口 - 加载数据、构建模型、训练并保存"""

import os
import sys
import argparse
import time

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import Preprocessor
from src.models.transformer_detector import TransformerDetector
from src.utils.visualization import Visualizer


def train(config_path: str = "config.yaml"):
    """训练主流程"""

    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["training"]
    vis_config = config.get("visualization", {})

    # 设置随机种子
    seed = data_config.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ===== Step 1: 加载数据 =====
    print("\n" + "=" * 50)
    print("Step 1: 加载数据")
    print("=" * 50)

    loader = DataLoader(data_config["raw_dir"])
    data = loader.load(data_config.get("file_pattern", "*.csv"))
    print(f"标签分布:\n{loader.get_label_distribution(data)}")

    # ===== Step 2: 特征工程 =====
    print("\n" + "=" * 50)
    print("Step 2: 特征工程")
    print("=" * 50)

    engineer = FeatureEngineer()
    engineer.fit(data)
    traffic_features, log_features = engineer.transform(data)
    traffic_dim, log_dim = engineer.get_feature_dims()
    print(f"流量特征维度: {traffic_dim}, 日志特征维度: {log_dim}")

    labels = data[loader.label_col].values

    # ===== Step 3: 预处理 =====
    print("\n" + "=" * 50)
    print("Step 3: 预处理与数据集划分")
    print("=" * 50)

    preprocessor = Preprocessor()
    datasets = preprocessor.fit_transform(
        traffic_features,
        log_features,
        labels,
        test_size=data_config.get("test_size", 0.1),
        val_size=data_config.get("val_size", 0.1),
        random_seed=seed,
    )

    dataloaders = preprocessor.create_dataloaders(
        datasets,
        batch_size=data_config.get("batch_size", 256),
        num_workers=data_config.get("num_workers", 0),
    )

    num_classes = preprocessor.num_classes

    # 保存预处理器
    os.makedirs(data_config["processed_dir"], exist_ok=True)
    preprocessor.save(
        os.path.join(data_config["processed_dir"], "preprocessor.pkl")
    )

    # ===== Step 4: 构建模型 =====
    print("\n" + "=" * 50)
    print("Step 4: 构建Transformer检测模型")
    print("=" * 50)

    model = TransformerDetector(
        traffic_dim=traffic_dim,
        log_dim=log_dim,
        num_classes=num_classes,
        d_model=model_config.get("d_model", 128),
        nhead=model_config.get("nhead", 8),
        num_layers=model_config.get("num_layers", 4),
        dim_feedforward=model_config.get("dim_feedforward", 256),
        dropout=model_config.get("dropout", 0.1),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # ===== Step 5: 训练 =====
    print("\n" + "=" * 50)
    print("Step 5: 训练模型")
    print("=" * 50)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.get("learning_rate", 0.001),
        weight_decay=train_config.get("weight_decay", 0.01),
    )
    epochs = train_config.get("epochs", 50)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = train_config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    patience = train_config.get("patience", 10)
    best_val_loss = float("inf")
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for traffic_batch, log_batch, label_batch in dataloaders["train"]:
            traffic_batch = traffic_batch.to(device)
            log_batch = log_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            outputs = model(traffic_batch, log_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * label_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(label_batch).sum().item()
            total += label_batch.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # --- Validation ---
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for traffic_batch, log_batch, label_batch in dataloaders["val"]:
                traffic_batch = traffic_batch.to(device)
                log_batch = log_batch.to(device)
                label_batch = label_batch.to(device)

                outputs = model(traffic_batch, log_batch)
                loss = criterion(outputs, label_batch)

                running_loss += loss.item() * label_batch.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(label_batch).sum().item()
                total += label_batch.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Early stopping & checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "traffic_dim": traffic_dim,
                    "log_dim": log_dim,
                    "num_classes": num_classes,
                    "model_config": model_config,
                },
                os.path.join(checkpoint_dir, "best_model.pth"),
            )
            print(f"  -> 保存最优模型 (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ===== Step 6: 可视化 =====
    print("\n" + "=" * 50)
    print("Step 6: 保存训练曲线")
    print("=" * 50)

    visualizer = Visualizer(
        output_dir=vis_config.get("output_dir", "results/figures"),
        dpi=vis_config.get("dpi", 150),
        figsize=tuple(vis_config.get("figsize", [10, 8])),
    )
    visualizer.plot_training_curves(
        train_losses, val_losses, train_accs, val_accs
    )

    print("\n训练完成！")
    print(f"最优验证Loss: {best_val_loss:.4f}")
    print(f"模型保存至: {checkpoint_dir}/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练网络攻击检测模型")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    args = parser.parse_args()
    train(args.config)
