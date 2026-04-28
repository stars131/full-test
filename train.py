"""训练入口 - 加载数据、构建模型、训练并保存"""

import argparse
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.cache import load_or_prepare_datasets
from src.data.preprocessor import Preprocessor
from src.models.losses import FocalLoss
from src.models.transformer_detector import TransformerDetector
from src.utils.experiment import save_json, save_yaml
from src.utils.visualization import Visualizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(runtime_config: dict, device: torch.device):
    if device.type != "cuda":
        return None, nullcontext, False

    if runtime_config.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if runtime_config.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    precision = runtime_config.get("precision", "fp32").lower()
    use_amp = precision in {"fp16", "bf16"}
    autocast_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(precision)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and precision == "fp16",
    )

    def autocast_context():
        if not use_amp:
            return nullcontext()
        return torch.amp.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=True,
        )

    return scaler, autocast_context, use_amp


def build_criterion(
    imbalance_config: dict,
    train_class_counts: list[int],
    device: torch.device,
) -> tuple[nn.Module, list[float] | None]:
    loss_name = imbalance_config.get("loss", "cross_entropy").lower()
    use_class_weights = imbalance_config.get("use_class_weights", False)

    class_weights = None
    class_weights_list = None
    if use_class_weights:
        counts = np.array(train_class_counts, dtype=np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        power = imbalance_config.get("class_weight_power", 1.0)
        weights = counts.sum() / (len(counts) * counts)
        weights = np.power(weights, power)
        class_weights_list = weights.tolist()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    if loss_name == "focal":
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=imbalance_config.get("focal_gamma", 2.0),
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    return criterion, class_weights_list


def train(config_path: str = "config.yaml"):
    """训练主流程"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["training"]
    vis_config = config.get("visualization", {})
    runtime_config = config.get("runtime", {})
    imbalance_config = config.get("imbalance", {})
    experiment_config = config.get("experiment", {})

    seed = data_config.get("random_seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    scaler, autocast_context, _ = configure_runtime(runtime_config, device)
    pin_memory = data_config.get("pin_memory", device.type == "cuda")
    non_blocking = pin_memory and device.type == "cuda"

    # ===== Step 1: 加载数据 =====
    print("\n" + "=" * 50)
    print("Step 1: 加载数据")
    print("=" * 50)
    if data_config.get("full_dataset", False):
        print("全量数据训练模式: BCCC-CSE-CIC-IDS-2018")
        print(f"原始CSV目录: {data_config['raw_dir']}")
        print("如目录为空，请先运行: python prepare_data.py --download")

    datasets, preprocessor, dataset_metadata = load_or_prepare_datasets(
        data_config
    )
    label_distribution = dataset_metadata["label_distribution"]
    print(f"标签分布:\n{label_distribution}")

    # ===== Step 2: 特征工程 =====
    print("\n" + "=" * 50)
    print("Step 2: 特征工程")
    print("=" * 50)
    traffic_dim = dataset_metadata["traffic_dim"]
    log_dim = dataset_metadata["log_dim"]
    print(f"流量特征维度: {traffic_dim}, 日志特征维度: {log_dim}")

    # ===== Step 3: 预处理 =====
    print("\n" + "=" * 50)
    print("Step 3: 预处理与数据集划分")
    print("=" * 50)
    print(
        "训练集: "
        f"{preprocessor.split_sizes['train']}, "
        f"验证集: {preprocessor.split_sizes['val']}, "
        f"测试集: {preprocessor.split_sizes['test']}"
    )

    dataloaders = preprocessor.create_dataloaders(
        datasets,
        batch_size=data_config.get("batch_size", 256),
        num_workers=data_config.get("num_workers", 0),
        pin_memory=pin_memory,
        persistent_workers=data_config.get("persistent_workers", False),
        prefetch_factor=data_config.get("prefetch_factor", 2),
        use_weighted_sampler=imbalance_config.get(
            "use_weighted_sampler", False
        ),
    )

    num_classes = preprocessor.num_classes
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
        fusion_strategy=model_config.get("fusion_strategy", "cross_attention"),
    ).to(device)

    if device.type == "cuda" and runtime_config.get("compile", False):
        model = torch.compile(model)

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

    criterion, class_weights = build_criterion(
        imbalance_config=imbalance_config,
        train_class_counts=preprocessor.train_class_counts,
        device=device,
    )
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
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    epoch_records = []
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for traffic_batch, log_batch, label_batch in dataloaders["train"]:
            traffic_batch = traffic_batch.to(device, non_blocking=non_blocking)
            log_batch = log_batch.to(device, non_blocking=non_blocking)
            label_batch = label_batch.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                outputs = model(traffic_batch, log_batch)
                loss = criterion(outputs, label_batch)

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                traffic_batch = traffic_batch.to(
                    device, non_blocking=non_blocking
                )
                log_batch = log_batch.to(device, non_blocking=non_blocking)
                label_batch = label_batch.to(device, non_blocking=non_blocking)

                with autocast_context():
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
        epoch_time = time.perf_counter() - epoch_start
        epoch_records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_time,
                "samples_per_second": total / epoch_time if epoch_time > 0 else 0,
            }
        )

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
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

    # ===== Step 6: 可视化与记录 =====
    print("\n" + "=" * 50)
    print("Step 6: 保存训练曲线")
    print("=" * 50)

    visualizer = Visualizer(
        output_dir=vis_config.get("output_dir", "results/figures"),
        dpi=vis_config.get("dpi", 150),
        figsize=tuple(vis_config.get("figsize", [10, 8])),
    )
    visualizer.plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
    )

    artifacts_dir = experiment_config.get(
        "artifacts_dir",
        train_config.get("log_dir", checkpoint_dir),
    )
    os.makedirs(artifacts_dir, exist_ok=True)

    train_seconds = time.perf_counter() - train_start
    split_summary = {
        "class_names": preprocessor.class_names,
        "train_class_counts": preprocessor.train_class_counts,
        "val_class_counts": preprocessor.val_class_counts,
        "test_class_counts": preprocessor.test_class_counts,
        "split_sizes": preprocessor.split_sizes,
        "label_distribution_raw": {
            str(k): int(v) for k, v in label_distribution.items()
        },
    }
    train_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": max(val_accs) if val_accs else 0.0,
        "epochs_completed": len(epoch_records),
        "train_seconds": train_seconds,
        "model_total_params": total_params,
        "model_trainable_params": trainable_params,
        "loss_name": imbalance_config.get("loss", "cross_entropy"),
        "use_class_weights": imbalance_config.get("use_class_weights", False),
        "use_weighted_sampler": imbalance_config.get(
            "use_weighted_sampler", False
        ),
        "class_weights": class_weights,
    }

    save_json(os.path.join(artifacts_dir, "train_history.json"), epoch_records)
    save_json(os.path.join(artifacts_dir, "train_summary.json"), train_summary)
    save_json(os.path.join(artifacts_dir, "split_summary.json"), split_summary)
    save_yaml(os.path.join(artifacts_dir, "resolved_config.yaml"), config)

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
