from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization") from exc
    return plt


def _scatter(ax, points: np.ndarray, title: str, color=None):
    if color is None:
        color = points[:, 2]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=2, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def parse_angles(text: str) -> Iterable[Tuple[int, int]]:
    pairs = []
    for part in text.split(";"):
        e, a = part.split(",")
        pairs.append((int(e), int(a)))
    return pairs


def save_views(bundle_path: Path, out_prefix: Path, angles: Iterable[Tuple[int, int]]) -> None:
    plt = _require_matplotlib()
    data = np.load(bundle_path)
    clean = data.get("clean")
    noisy = data["noisy"]
    denoised = data["denoised"]
    residual = data["residual"]
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for idx, (elev, azim) in enumerate(angles):
        fig = plt.figure(figsize=(16, 4))
        axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]
        if clean is not None:
            _scatter(axes[0], clean, "clean")
        else:
            _scatter(axes[0], noisy, "clean(unavailable)")
        _scatter(axes[1], noisy, "noisy")
        _scatter(axes[2], denoised, "denoised")
        residual_mag = np.sqrt((residual**2).sum(axis=-1))
        _scatter(axes[3], residual, "residual", color=residual_mag)

        for ax in axes:
            ax.view_init(elev=elev, azim=azim)

        fig.tight_layout()
        out_path = out_prefix.parent / f"{out_prefix.name}_view_{idx:02d}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")


def interactive_show(bundle_path: Path) -> None:
    plt = _require_matplotlib()
    data = np.load(bundle_path)
    clean = data.get("clean")
    noisy = data["noisy"]
    denoised = data["denoised"]
    residual = data["residual"]

    fig = plt.figure(figsize=(16, 4))
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]
    if clean is not None:
        _scatter(axes[0], clean, "clean")
    else:
        _scatter(axes[0], noisy, "clean(unavailable)")
    _scatter(axes[1], noisy, "noisy")
    _scatter(axes[2], denoised, "denoised")
    residual_mag = np.sqrt((residual**2).sum(axis=-1))
    _scatter(axes[3], residual, "residual", color=residual_mag)

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize clean/noisy/denoised/residual point clouds")
    parser.add_argument("--bundle", required=True, help=".npz from eval.py/infer.py")
    parser.add_argument("--save-prefix", default="outputs/compare")
    parser.add_argument("--angles", default="30,45;30,135;60,45", help="elev,azim pairs separated by ';'")
    parser.add_argument("--interactive", action="store_true", help="Open matplotlib interactive window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_path = Path(args.bundle)
    out_prefix = Path(args.save_prefix)
    angles = parse_angles(args.angles)

    save_views(bundle_path, out_prefix, angles)
    if args.interactive:
        interactive_show(bundle_path)


if __name__ == "__main__":
    main()
