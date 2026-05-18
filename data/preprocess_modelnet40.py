import argparse
from pathlib import Path

import numpy as np


def _read_split(split_files):
    import h5py

    points, labels = [], []
    for f in split_files:
        with h5py.File(f, "r") as h5f:
            points.append(h5f["data"][:].astype(np.float32))
            labels.append(h5f["label"][:].astype(np.int64).reshape(-1))
    return np.concatenate(points, axis=0), np.concatenate(labels, axis=0)


def _normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=1, keepdims=True)
    centered = points - centroid
    max_radius = np.sqrt((centered**2).sum(axis=2)).max(axis=1, keepdims=True)[..., None]
    safe_radius = np.maximum(max_radius, 1e-8)
    return centered / safe_radius


def preprocess(raw_dir: Path, output_dir: Path, num_points: int = 1024) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_files = sorted(raw_dir.glob("ply_data_train*.h5"))
    test_files = sorted(raw_dir.glob("ply_data_test*.h5"))

    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Cannot find ModelNet40 .h5 files under {raw_dir}. "
            "Run data/download_modelnet40.py first."
        )

    train_points, train_labels = _read_split(train_files)
    test_points, test_labels = _read_split(test_files)

    train_points = _normalize_unit_sphere(train_points[:, :num_points, :])
    test_points = _normalize_unit_sphere(test_points[:, :num_points, :])

    np.savez_compressed(output_dir / "train.npz", points=train_points, labels=train_labels)
    np.savez_compressed(output_dir / "test.npz", points=test_points, labels=test_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ModelNet40 for denoising")
    parser.add_argument(
        "--raw-dir",
        default="data/raw/modelnet40_ply_hdf5_2048",
        help="Directory containing raw ModelNet40 HDF5 files",
    )
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for processed splits")
    parser.add_argument("--num-points", type=int, default=1024, help="Point count per sample")
    args = parser.parse_args()

    preprocess(Path(args.raw_dir), Path(args.output_dir), num_points=args.num_points)
    print(f"Saved processed train/test splits to {args.output_dir}")


if __name__ == "__main__":
    main()
