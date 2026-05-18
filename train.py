from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import ModelNetDenoiseDataset, load_processed_split, split_iterator
from utils.io import ensure_dir


def require_jittor():
    try:
        import jittor as jt
    except ImportError as exc:
        raise ImportError("Jittor is required for training") from exc
    return jt


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCT-style point cloud denoiser with Jittor")
    parser.add_argument("--data-dir", default="data/processed", help="Directory containing train.npz/test.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--noise-type", default="gaussian", choices=["gaussian", "uniform", "mixed"])
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--noise-range", type=float, default=0.03)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    jt = require_jittor()

    from models import PCTDenoiser
    from utils.losses import chamfer_distance

    train_points, train_labels = load_processed_split(Path(args.data_dir) / "train.npz")
    train_dataset = ModelNetDenoiseDataset(
        train_points,
        train_labels,
        noise_type=args.noise_type,
        noise_std=args.noise_std,
        noise_range=args.noise_range,
    )

    model = PCTDenoiser()
    optimizer = jt.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = ensure_dir(args.save_dir)

    for epoch in range(1, args.epochs + 1):
        losses = []
        epoch_seed = (args.seed + epoch) % (2**31 - 1)
        for noisy, clean, _ in split_iterator(train_dataset, batch_size=args.batch_size, shuffle=True, seed=epoch_seed):
            noisy_jt = jt.array(noisy)
            clean_jt = jt.array(clean)
            pred = model(noisy_jt)
            loss = chamfer_distance(pred, clean_jt)
            optimizer.step(loss)
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch [{epoch}/{args.epochs}] train_cd={mean_loss:.6f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = save_dir / f"pct_denoiser_epoch_{epoch}.pkl"
            jt.save(model.state_dict(), str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
