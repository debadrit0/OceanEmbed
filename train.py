#!/usr/bin/env python3
"""
OceanEmbed Model Training Pipeline
----------------------------------
Trains the dual deep-learning architectures:
  1. OceanEmbed Fast (32D Latent Baseline CNN)
  2. OceanEmbed V2 Embedding (64D Latent Spatial CNN with Target Standardization)

Checkpoints are saved to:
  checkpoints/oceanembed_fast_2023.pt
  checkpoints/oceanembed_embedding_v2_2023.pt

Usage:
    # Train both models for 10 epochs (auto-creates dataset if needed):
    python train.py --epochs 10

    # Train only fast model:
    python train.py --model fast --epochs 5

    # Train with GPU if available:
    python train.py --device cuda
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model_fast import OceanEmbedFast
from src.model_embed import OceanEmbedEmbeddingModel


def get_data(data_dir):
    """Load processed patch datasets, or auto-generate if missing."""
    train_path = Path(data_dir) / "train_patches.npz"
    val_path = Path(data_dir) / "val_patches.npz"

    if not train_path.exists() or not val_path.exists():
        print(f"[INFO] Processed dataset not found in '{data_dir}'. Auto-generating sample patches...")
        from create_dataset import generate_synthetic_patches, DEPTHS
        (train_X, train_y, train_c), (val_X, val_y, val_c) = generate_synthetic_patches(num_samples=2500)
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(train_path, X=train_X, y=train_y, coords=train_c, depths=DEPTHS)
        np.savez_compressed(val_path, X=val_X, y=val_y, coords=val_c, depths=DEPTHS)
        print(f"[✓] Saved generated dataset to {data_dir}")

    train_data = np.load(train_path)
    val_data = np.load(val_path)

    X_train, y_train = train_data["X"].astype(np.float32), train_data["y"].astype(np.float32)
    X_val, y_val = val_data["X"].astype(np.float32), val_data["y"].astype(np.float32)
    depths = train_data["depths"].astype(np.float32)

    return (X_train, y_train), (X_val, y_val), depths


def train_fast_model(train_loader, val_loader, device, epochs=10, lr=1e-3, out_dir="checkpoints"):
    """Train OceanEmbedFast baseline model."""
    print("\n" + "=" * 65)
    print("TRAINING: OceanEmbed Fast (32D Latent Baseline)")
    print("=" * 65)

    model = OceanEmbedFast().to(device)
    criterion = nn.L1Loss()  # Direct MAE loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_mae += torch.abs(pred - batch_y).sum().item()
        val_mae /= (len(val_loader.dataset) * 15)

        print(f"  Epoch {epoch:2d}/{epochs:2d} | Train MAE: {train_loss:.4f}°C | Val MAE: {val_mae:.4f}°C")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out_file = Path(out_dir) / "oceanembed_fast_2023.pt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "val_mae": best_val_mae, "epochs": epochs}, out_file)
    print(f"\n[SUCCESS] OceanEmbed Fast saved to {out_file} (Best Val MAE: {best_val_mae:.4f}°C)")
    return best_val_mae


def train_v2_model(train_loader, val_loader, target_mean, target_std, depths, device, epochs=15, lr=1e-3, out_dir="checkpoints"):
    """Train OceanEmbed V2 Embedding Model with target normalization."""
    print("\n" + "=" * 65)
    print("TRAINING: OceanEmbed V2 Embedding (64D Latent + Physics Priors)")
    print("=" * 65)

    model = OceanEmbedEmbeddingModel().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t_mean_t = torch.from_numpy(target_mean).to(device)
    t_std_t = torch.from_numpy(target_std).to(device)

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Normalize targets
            y_norm = (batch_y - t_mean_t) / t_std_t

            optimizer.zero_grad()
            emb = model.encode(batch_x)
            pred_norm = model.reconstruction_head(emb)
            loss = criterion(pred_norm, y_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation (evaluate in real °C)
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_norm = model.reconstruction_head(model.encode(batch_x))
                pred_real = pred_norm * t_std_t + t_mean_t
                val_mae += torch.abs(pred_real - batch_y).sum().item()
        val_mae /= (len(val_loader.dataset) * len(depths))

        print(f"  Epoch {epoch:2d}/{epochs:2d} | Train SmoothL1: {train_loss:.4f} | Val MAE: {val_mae:.4f}°C")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out_file = Path(out_dir) / "oceanembed_embedding_v2_2023.pt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "model_state": best_state,
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
        "depths": depths.tolist(),
        "val_mae": best_val_mae,
        "epochs": epochs,
        "embedding_dim": model.EMBEDDING_DIM
    }
    torch.save(checkpoint_data, out_file)
    print(f"\n[SUCCESS] OceanEmbed V2 saved to {out_file} (Best Val MAE: {best_val_mae:.4f}°C)")
    return best_val_mae


def main():
    parser = argparse.ArgumentParser(description="OceanEmbed Model Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed .npz patches")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--model", choices=["all", "fast", "v2"], default="all", help="Model(s) to train")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: 'cpu', 'cuda', or 'auto'")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using compute device: {device}")

    # Load data
    (X_train, y_train), (X_val, y_val), depths = get_data(args.data_dir)
    print(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples.")

    # Target statistics for V2 model
    target_mean = np.mean(y_train, axis=0)
    target_std = np.std(y_train, axis=0)
    target_std[target_std < 1e-4] = 1.0  # Guard against division by zero

    # DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Train requested models
    if args.model in ("all", "fast"):
        train_fast_model(train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr, out_dir=args.output_dir)

    if args.model in ("all", "v2"):
        train_v2_model(
            train_loader, val_loader,
            target_mean=target_mean, target_std=target_std, depths=depths,
            device=device, epochs=args.epochs, lr=args.lr, out_dir=args.output_dir
        )

    print("\n" + "=" * 65)
    print("ALL REQUESTED TRAININGS COMPLETED SUCCESSFULLY")
    print("You can now run 'python app.py' to launch the interactive platform.")
    print("=" * 65)


if __name__ == "__main__":
    main()
