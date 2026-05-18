import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

MODELNET40_H5_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"


def download_and_extract(output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "modelnet40_ply_hdf5_2048.zip"
    extract_dir = output_dir / "modelnet40_ply_hdf5_2048"

    if extract_dir.exists() and not force:
        return extract_dir

    if force:
        if zip_path.exists():
            zip_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

    if not zip_path.exists():
        urllib.request.urlretrieve(MODELNET40_H5_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ModelNet40 HDF5 point cloud dataset")
    parser.add_argument("--output-dir", default="data/raw", help="Target directory")
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract")
    args = parser.parse_args()

    extract_dir = download_and_extract(Path(args.output_dir), force=args.force)
    print(f"ModelNet40 downloaded/extracted at: {extract_dir}")


if __name__ == "__main__":
    main()
