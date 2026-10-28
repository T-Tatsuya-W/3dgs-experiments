#!/usr/bin/env python3
"""One-command two-photo stereo -> COLMAP -> pure-PyTorch Gaussian PLY workflow."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def execute(python, script, *args):
    command = [str(python), str(script), *map(str, args)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-dir", type=Path, default=Path("data/stereo"),
                   help="Directory containing left.jpg and right.jpg")
    p.add_argument("--left", type=Path, help="Override left photo path")
    p.add_argument("--right", type=Path, help="Override right photo path")
    p.add_argument("--calibration", type=Path, default=Path("calibration/olympus_intrinsics.json"))
    p.add_argument("--output", type=Path, default=Path("output/pure_binocular"))
    p.add_argument("--baseline-m", type=float, default=0.10, help="Camera 2 rightward translation in metres")
    p.add_argument("--stereo-width", type=int, default=1200)
    p.add_argument("--disparities", type=int, default=128, help="Stereo disparity search width (multiple of 16)")
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--max-stereo-points", type=int, default=12000)
    p.add_argument("--train-width", type=int, default=240)
    p.add_argument("--gaussians", type=int, default=500)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--geometry-only", action="store_true")
    p.add_argument("--python-geometry", type=Path, default=Path(".venv/Scripts/python.exe"),
                   help="Python with OpenCV, NumPy, SciPy, PyCOLMAP")
    p.add_argument("--python-training", type=Path, default=Path(".venv-puregs/Scripts/python.exe"),
                   help="Python with CUDA PyTorch, NumPy, Pillow, plyfile")
    args = p.parse_args()
    if args.baseline_m <= 0 or args.steps <= 0 or args.gaussians <= 0 or args.train_width < 32:
        p.error("Baseline, steps, gaussians and training width must be positive")
    pair = args.pair_dir.resolve()
    left = (args.left or pair / "left.jpg").resolve()
    right = (args.right or pair / "right.jpg").resolve()
    calibration = args.calibration.resolve()
    output = args.output.resolve()
    for file in (left, right, calibration):
        if not file.is_file():
            p.error(f"Missing input: {file}")
    geom_python = args.python_geometry.resolve()
    train_python = args.python_training.resolve()
    if not geom_python.is_file():
        p.error(f"Missing geometry Python interpreter: {geom_python}")
    if not args.geometry_only and not train_python.is_file():
        p.error(f"Missing training Python interpreter: {train_python}")
    if output.exists() and any(output.iterdir()):
        p.error(f"Output directory is not empty: {output}. Use a new --output to avoid stale results.")
    output.mkdir(parents=True, exist_ok=True)
    pose = output / "pose.json"
    pose.write_text(json.dumps({
        "camera2_center_in_camera1_m": [args.baseline_m, 0.0, 0.0],
        "camera1_to_camera2_rotation_xyz_deg": [0.0, 0.0, 0.0]
    }, indent=2), encoding="utf-8")
    stereo = output / "stereo"
    dataset = output / "dataset"
    execute(geom_python, HERE / "stereo.py", "--left", left, "--right", right,
            "--calibration", calibration, "--pose", pose, "--output", stereo,
            "--max-width", args.stereo_width, "--num-disparities", args.disparities,
            "--block-size", args.block_size)
    execute(geom_python, HERE / "export_colmap.py", "--stereo", stereo,
            "--output", dataset, "--max-points", args.max_stereo_points)
    if args.geometry_only:
        print(f"Geometry finished: {stereo / 'points.ply'}")
        return
    training = output / "training"
    execute(train_python, HERE / "train_pure_torch.py", "--dataset", dataset,
            "--output", training, "--width", args.train_width,
            "--points", args.gaussians, "--steps", args.steps, "--lr", args.lr)
    ply = training / "gaussians.ply"
    if not ply.is_file():
        raise RuntimeError(f"Trainer exited without producing {ply}")
    print(f"Binary Gaussian PLY: {ply}")
    print(f"Preview renders: {training / 'render_1.png'} and render_2.png")


if __name__ == "__main__":
    main()
