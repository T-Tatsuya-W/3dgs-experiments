#!/usr/bin/env python3
"""Run calibrated 10 cm stereo -> COLMAP dataset -> (optionally) gsplat training."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output/binocular"))
    p.add_argument("--max-width", type=int, default=1200)
    p.add_argument("--max-points", type=int, default=12000)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--geometry-only", action="store_true", help="Stop after producing a validated COLMAP dataset")
    args = p.parse_args()
    output = args.output.resolve()
    stereo = output / "stereo"
    dataset = output / "dataset"
    pose = HERE / "pose.example.json"
    if not args.left.is_file() or not args.right.is_file() or not args.calibration.is_file():
        p.error("Left photo, right photo, and calibration JSON must all exist.")
    if args.steps < 1 or args.max_width < 200 or args.max_points < 50:
        p.error("Check steps, max-width and max-points")
    if (dataset / "sparse" / "0").exists():
        p.error(f"Dataset already exists: {dataset}. Choose a fresh --output to avoid stale data.")
    run(HERE / "stereo.py", "--left", args.left.resolve(), "--right", args.right.resolve(),
        "--calibration", args.calibration.resolve(), "--pose", pose, "--output", stereo,
        "--max-width", args.max_width)
    run(HERE / "export_colmap.py", "--stereo", stereo, "--output", dataset,
        "--max-points", args.max_points)
    if args.geometry_only:
        print(f"Geometry ready at {dataset}; inspect {stereo / 'disparity.png'} and points.ply")
        return
    run(ROOT / "scripts" / "patch_gsplat_v153.py")
    run(HERE / "patch_two_view.py")
    run(ROOT / "run.py", "--skip-colmap", "--output", output, "--steps", args.steps, "--factor", 1)
    print(f"Done. Check {output / 'training'} for Gaussian PLY files.")


if __name__ == "__main__":
    main()
