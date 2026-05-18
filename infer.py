from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.io import save_vis_bundle


def require_jittor():
    try:
        import jittor as jt
    except ImportError as exc:
        raise ImportError("Jittor is required for inference") from exc
    return jt


def parse_args():
    parser = argparse.ArgumentParser(description="Denoise a single point cloud sample")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help=".npy or .npz with key 'points'")
    parser.add_argument("--output", default="outputs/infer_result.npz")
    return parser.parse_args()


def load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        pts = np.load(path)
    elif path.suffix == ".npz":
        npz = np.load(path)
        if "points" in npz:
            pts = npz["points"]
        else:
            first_key = list(npz.keys())[0]
            pts = npz[first_key]
    else:
        raise ValueError("Input must be .npy or .npz")
    if pts.ndim == 2:
        pts = pts[None, ...]
    return pts.astype(np.float32)


def main() -> None:
    args = parse_args()
    jt = require_jittor()

    from models import PCTDenoiser

    points = load_points(Path(args.input))
    model = PCTDenoiser()
    state = jt.load(args.checkpoint)
    model.load_state_dict(state)
    model.eval()

    denoised = model(jt.array(points)).numpy()
    residual = points - denoised

    save_vis_bundle(args.output, {"noisy": points[0], "denoised": denoised[0], "residual": residual[0]})
    print(f"Saved inference bundle to: {args.output}")


if __name__ == "__main__":
    main()
