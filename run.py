#!/usr/bin/env python3
"""Minimal orchestration: photographs -> COLMAP SfM -> gsplat PLY.

PyCOLMAP reconstructs camera poses in-process; the upstream gsplat example\nis launched as a subprocess. This
repository deliberately does not vendor or reimplement either project.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def check_images(images: Path) -> None:
    if not images.is_dir():
        raise ValueError(f"Image directory does not exist: {images}")
    count = sum(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in images.iterdir())
    if count < 3:
        raise ValueError(f"Found {count} images in {images}; need at least 3 (20-50 strongly recommended).")
    print(f"Found {count} image(s) in {images}.", flush=True)


def load_calibration(path: Path, image_size: tuple[int, int]) -> str:
    """Convert OpenCV calibration JSON to COLMAP's 8-parameter OPENCV model."""
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = (int(data["image_width"]), int(data["image_height"]))
    if expected != image_size:
        raise ValueError(f"Calibration resolution {expected} differs from image {image_size}")
    K = data["camera_matrix"]
    dist = data["distortion_coefficients"]
    if len(dist) < 4 or (len(dist) > 5 and any(abs(float(v)) > 1e-9 for v in dist[5:])):
        raise ValueError("Expected 4/5 coefficient OpenCV distortion without nonzero higher coefficients")
    vals = [K[0][0], K[1][1], K[0][2], K[1][2], *dist[:4]]
    if not all(math.isfinite(float(v)) for v in vals) or vals[0] <= 0 or vals[1] <= 0:
        raise ValueError("Invalid calibration parameters")
    return ",".join(str(float(v)) for v in vals)


def reconstruct(images: Path, dataset: Path, calibration: Path | None = None) -> None:
    """Create a conventional COLMAP dataset: images/, sparse/0/, database.db."""
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("PyCOLMAP is missing. Install it with: python -m pip install pycolmap") from exc

    destination = dataset / "images"
    database = dataset / "database.db"
    sparse = dataset / "sparse"
    destination.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    if images.resolve() != destination.resolve():
        for source in images.iterdir():
            if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
                target = destination / source.name
                if not target.exists():
                    shutil.copy2(source, target)

    if database.exists():
        raise FileExistsError(
            f"{database} already exists. Use a fresh --output directory to avoid "
            "mixing reconstructions (or pass --skip-colmap to use existing poses)."
        )

    print("Extracting features with PyCOLMAP (CPU)...", flush=True)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    mapping_options = None
    if calibration is not None:
        from PIL import Image
        first_image = next(p for p in sorted(destination.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS)
        with Image.open(first_image) as im:
            size = im.size
        params = load_calibration(calibration, size)
        reader.camera_model = "OPENCV"
        reader.camera_params = params
        mapping_options = pycolmap.IncrementalPipelineOptions()
        mapping_options.ba_refine_focal_length = False
        mapping_options.ba_refine_principal_point = False
        mapping_options.ba_refine_extra_params = False
        mapping_options.mapper.abs_pose_refine_focal_length = False
        mapping_options.mapper.abs_pose_refine_extra_params = False
        print(f"Fixed calibrated OPENCV camera: {params}", flush=True)
    pycolmap.extract_features(
        database_path=database,
        image_path=destination,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        device=pycolmap.Device.cpu,
    )
    print("Matching image features with PyCOLMAP (CPU)...", flush=True)
    pycolmap.match_exhaustive(database_path=database, device=pycolmap.Device.cpu)
    print("Estimating camera poses with PyCOLMAP...", flush=True)
    mapping_kwargs = dict(database_path=database, image_path=destination, output_path=sparse)
    if mapping_options is not None:
        mapping_kwargs["options"] = mapping_options
    reconstructions = pycolmap.incremental_mapping(**mapping_kwargs)
    if not reconstructions:
        raise RuntimeError("PyCOLMAP returned no reconstruction. Try more overlapping, textured photos.")

    # gsplat expects a conventional sparse/0/ COLMAP model. Keep the largest
    # reconstructed model, regardless of which numeric ID PyCOLMAP assigned.
    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    model = sparse / "0"
    model.mkdir(parents=True, exist_ok=True)
    best.write(model)
    print(f"Registered {best.num_reg_images()} images.", flush=True)

    model = sparse / "0"
    if not model.is_dir() or not (model / "cameras.bin").exists() and not (model / "cameras.txt").exists():
        raise RuntimeError(
            f"COLMAP did not create a sparse model at {model}. Try more overlapping,"
            " sharp images of a well-textured stationary scene."
        )
    print(f"Sparse reconstruction: {model}", flush=True)


def train(dataset: Path, output: Path, *, gsplat_examples: Path, steps: int, factor: int) -> None:
    trainer = gsplat_examples / "simple_trainer.py"
    if not trainer.is_file():
        raise FileNotFoundError(
            f"Missing {trainer}. Clone the upstream gsplat repository and set "
            "--gsplat-examples to its examples directory (see README)."
        )
    if not (dataset / "sparse" / "0").is_dir() or not (dataset / "images").is_dir():
        raise ValueError(f"Not a COLMAP dataset (expected images/ and sparse/0/): {dataset}")
    output.mkdir(parents=True, exist_ok=True)
    run_command([
        sys.executable, str(trainer), "default",
        "--data_dir", str(dataset),
        "--result_dir", str(output),
        "--data_factor", str(factor),
        "--max_steps", str(steps),
        "--save_ply",
        "--ply_steps", str(steps),
        "--disable_viewer",
        "--disable_video",
    ], cwd=gsplat_examples)
    ply_files = sorted(output.rglob("*.ply"))
    if not ply_files:
        raise RuntimeError(
            f"Training finished but no PLY found in {output}. Check the trainer "
            "output and upstream gsplat CLI options."
        )
    print("\nExported Gaussian PLY file(s):", flush=True)
    for ply in ply_files:
        print(f"  {ply}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/test01/images"))
    parser.add_argument("--calibration", type=Path, help="Calibrated intrinsics JSON from ChArUco")
    parser.add_argument("--output", type=Path, default=Path("output/test01"))
    parser.add_argument("--gsplat-examples", type=Path, default=Path("third_party/gsplat/examples"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--factor", type=int, default=4, help="Training image downsample factor")
    parser.add_argument("--skip-colmap", action="store_true", help="Reuse output/COLMAP data")
    parser.add_argument("--colmap-only", action="store_true", help="Do not train a splat")
    args = parser.parse_args()

    if args.steps < 1 or args.factor < 1:
        parser.error("--steps and --factor must be positive")
    images = args.images.expanduser().resolve()
    output = args.output.expanduser().resolve()
    examples = args.gsplat_examples.expanduser().resolve()
    dataset = output / "dataset"

    try:
        if not args.skip_colmap:
            check_images(images)
            reconstruct(images, dataset, args.calibration.expanduser().resolve() if args.calibration else None)
        if not args.colmap_only:
            train(dataset, output / "training", gsplat_examples=examples,
                  steps=args.steps, factor=args.factor)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
