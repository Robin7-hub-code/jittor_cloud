from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import ModelNetDenoiseDataset, load_processed_split, split_iterator
from utils import ensure_dir, evaluate_pair, save_vis_bundle


def require_jittor():
    try:
        import jittor as jt
    except ImportError as exc:
        raise ImportError("Jittor is required for evaluation") from exc
    return jt


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate point cloud denoiser")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pkl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--noise-type", default="gaussian", choices=["gaussian", "uniform", "mixed"])
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--noise-range", type=float, default=0.03)
    parser.add_argument("--with-emd", action="store_true")
    parser.add_argument("--save-vis-dir", default="outputs/vis_bundles")
    parser.add_argument("--save-metrics", default="outputs/metrics.json")
    parser.add_argument("--max-save", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jt = require_jittor()

    from models import PCTDenoiser

    test_points, test_labels = load_processed_split(Path(args.data_dir) / "test.npz")
    test_dataset = ModelNetDenoiseDataset(
        test_points,
        test_labels,
        noise_type=args.noise_type,
        noise_std=args.noise_std,
        noise_range=args.noise_range,
    )

    model = PCTDenoiser()
    state = jt.load(args.checkpoint)
    model.load_state_dict(state)
    model.eval()

    save_vis_dir = ensure_dir(args.save_vis_dir)
    cd_scores, emd_scores = [], []
    saved = 0

    for noisy, clean, labels in split_iterator(test_dataset, batch_size=args.batch_size, shuffle=False, seed=0):
        noisy_jt = jt.array(noisy)
        pred = model(noisy_jt).numpy()

        for i in range(pred.shape[0]):
            metrics = evaluate_pair(pred[i], clean[i], with_emd=args.with_emd)
            cd_scores.append(metrics["cd"])
            if args.with_emd:
                emd_scores.append(metrics["emd"])

            if saved < args.max_save:
                residual = noisy[i] - pred[i]
                save_vis_bundle(
                    save_vis_dir / f"sample_{saved:04d}_label_{int(labels[i])}.npz",
                    {
                        "clean": clean[i],
                        "noisy": noisy[i],
                        "denoised": pred[i],
                        "residual": residual,
                    },
                )
                saved += 1

    result = {
        "cd_mean": float(np.mean(cd_scores)) if cd_scores else float("nan"),
        "cd_std": float(np.std(cd_scores)) if cd_scores else float("nan"),
        "num_samples": len(cd_scores),
    }
    if args.with_emd:
        arr = np.array(emd_scores, dtype=np.float32)
        valid = arr[~np.isnan(arr)]
        result["emd_mean"] = float(valid.mean()) if len(valid) else float("nan")
        result["emd_std"] = float(valid.std()) if len(valid) else float("nan")

    out_metrics = Path(args.save_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved visualization bundles to: {save_vis_dir}")


if __name__ == "__main__":
    main()
