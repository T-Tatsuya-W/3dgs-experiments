#!/usr/bin/env python3
"""Minimal orchestration: photographs -> COLMAP SfM -> gsplat PLY.

COLMAP and the upstream gsplat example are executed as subprocesses. This
repository deliberately does not vendor or reimplement either project.
"""
from __future__ import annotations

import argparse
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


def reconstruct(images: Path, dataset: Path, *, colmap: str) -> None:
    """Create a conventional COLMAP dataset: images/, sparse/0/, database.db."""
    if not shutil.which(colmap) and not Path(colmap).is_file():
        raise RuntimeError(f"COLMAP executable not found: {colmap}")

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

    run_command([colmap, "feature_extractor",
                 "--database_path", str(database),
                 "--image_path", str(destination),
                 "--ImageReader.single_camera", "1",
                 "--ImageReader.camera_model", "SIMPLE_RADIAL"])
    run_command([colmap, "exhaustive_matcher", "--database_path", str(database)])
    run_command([colmap, "mapper", "--database_path", str(database),
                 "--image_path", str(destination),
                 "--output_path", str(sparse)])

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
    parser.add_argument("--output", type=Path, default=Path("output/test01"))
    parser.add_argument("--gsplat-examples", type=Path, default=Path("third_party/gsplat/examples"))
    parser.add_argument("--colmap", default="colmap")
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
            reconstruct(images, dataset, colmap=args.colmap)
        if not args.colmap_only:
            train(dataset, output / "training", gsplat_examples=examples,
                  steps=args.steps, factor=args.factor)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
