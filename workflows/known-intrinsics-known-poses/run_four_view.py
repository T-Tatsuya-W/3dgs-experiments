#!/usr/bin/env python3
"""Four parallel camera positions on a 10 cm square -> two stereo pairs -> four-view Gaussian training.

No unknown-pose SfM; two horizontal stereo depth maps are fused in one known-pose
coordinate frame. Not a joint 4-view stereo matcher or cross-row feature fusion.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAMES = ("image00", "image01", "image10", "image11")


def run(python, script, *args):
    cmd = [str(python), str(script), *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def lines(path):
    return [line for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def build_dataset(out, top, bottom, vertical):
    """Merge two identical-orientation rectified stereo COLMAP datasets.

    World frame = rectified top-left optical camera frame. The bottom row
    offset is expressed in that frame via stereoRectify's R1.
    """
    a = json.loads((top / "stereo_meta.json").read_text())
    b = json.loads((bottom / "stereo_meta.json").read_text())
    R1a, R1b = np.asarray(a["R1_rectify"]), np.asarray(b["R1_rectify"])
    if not np.allclose(R1a, R1b, atol=1e-5):
        raise ValueError("Top and bottom rectifications differ; cannot merge frames without reprojecting.")
    offset = R1a @ np.array([0., vertical, 0.])
    sources = (top, bottom)
    images = out / "images"
    sparse = out / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    cam_rows, img_rows, pts_rows = [], [], []
    point_offset = 0
    for row, src in enumerate(sources):
        local = src.parent / ("top_dataset" if row == 0 else "bottom_dataset") / "sparse" / "0"
        if not local.exists():
            raise FileNotFoundError(local)
        for idx, name in enumerate(("left.png", "right.png")):
            shutil.copyfile(src / name, images / f"{NAMES[row*2+idx]}.png")
        # Reuse calibrated rectified intrinsics; assign unique COLMAP camera IDs.
        for entry in lines(local / "cameras.txt"):
            c = entry.split()
            new_cam_id = row * 2 + int(c[0])
            cam_rows.append(" ".join([str(new_cam_id), *c[1:]]))
        img_entries = lines(local / "images.txt")
        if len(img_entries) != 4:
            raise ValueError(f"Expected two images + two observation lines in {local}")
        for idx in range(2):
            head = img_entries[idx * 2].split()
            q = np.asarray(list(map(float, head[1:5])))
            t = np.asarray(list(map(float, head[5:8])))
            # world -> row-camera: R x + t_local - R*row_offset
            qw,qx,qy,qz = q
            R = np.array([
                [1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)]
            ])
            if row:
                t -= R @ offset
            new_id = row*2+idx+1
            img_rows.append(" ".join([str(new_id), *head[1:5],
                                      *(f"{v:.15g}" for v in t),
                                      str(new_id), f"{NAMES[new_id-1]}.png"]))
            obs = img_entries[idx*2+1].split()
            for j in range(2, len(obs), 3):
                if int(obs[j]) > 0:
                    obs[j] = str(int(obs[j]) + point_offset)
            img_rows.append(" ".join(obs))
        local_points = lines(local / "points3D.txt")
        for record in local_points:
            x = record.split()
            xyz = np.asarray(list(map(float, x[1:4]))) + (offset if row else 0)
            track = x[8:]
            for j in range(0, len(track), 2):
                track[j] = str(int(track[j]) + row*2)
            pts_rows.append(" ".join([str(int(x[0])+point_offset),
                                       *(f"{v:.12g}" for v in xyz),
                                       *x[4:8], *track]))
        point_offset += len(local_points)
    (sparse/"cameras.txt").write_text("\n".join(cam_rows)+"\n")
    (sparse/"images.txt").write_text("\n".join(img_rows)+"\n")
    (sparse/"points3D.txt").write_text("\n".join(pts_rows)+"\n")
    import pycolmap
    rec = pycolmap.Reconstruction(str(sparse))
    if rec.num_reg_images() != 4 or not rec.points3D:
        raise RuntimeError("Merged model validation failed")
    print(f"Validated four-view model with {rec.num_reg_images()} registered images and {len(rec.points3D)} points.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-dir", type=Path, default=Path("data/four_view/scene01"))
    p.add_argument("--calibration", type=Path, default=Path("calibration/olympus_intrinsics.json"))
    p.add_argument("--output", type=Path, default=Path("output/four_view_01"))
    p.add_argument("--spacing-m", type=float, default=0.10)
    p.add_argument("--stereo-width", type=int, default=1200)
    p.add_argument("--disparities", type=int, default=128)
    p.add_argument("--max-stereo-points", type=int, default=12000)
    p.add_argument("--train-width", type=int, default=240)
    p.add_argument("--gaussians", type=int, default=500)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--geometry-only", action="store_true")
    p.add_argument("--python-geometry", type=Path, default=Path(".venv/Scripts/python.exe"))
    p.add_argument("--python-training", type=Path, default=Path(".venv-puregs/Scripts/python.exe"))
    args = p.parse_args()
    if args.spacing_m <= 0:
        p.error("spacing-m must be positive")
    files = [args.pair_dir / (name+".jpg") for name in NAMES]
    for path in [*files, args.calibration, args.python_geometry]:
        if not path.is_file():
            p.error(f"Missing file: {path}")
    if not args.geometry_only and not args.python_training.is_file():
        p.error(f"Missing training interpreter: {args.python_training}")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error("Output folder is not empty; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    pose = output / "pose.json"
    pose.write_text(json.dumps({
        "camera2_center_in_camera1_m":[args.spacing_m,0,0],
        "camera1_to_camera2_rotation_xyz_deg":[0,0,0]
    }))
    for name, pair in (("top", files[:2]), ("bottom", files[2:])):
        stereo = output / (name+"_stereo")
        run(args.python_geometry, HERE/"stereo.py", "--left", pair[0].resolve(),
            "--right", pair[1].resolve(), "--calibration", args.calibration.resolve(),
            "--pose", pose, "--output", stereo, "--max-width", args.stereo_width,
            "--num-disparities", args.disparities)
        run(args.python_geometry, HERE/"export_colmap.py", "--stereo", stereo,
            "--output", output/(name+"_dataset"), "--max-points", args.max_stereo_points)
    # Dataset merge requires pycolmap. Execute in the geometry venv via a subprocess.
    run(args.python_geometry, HERE/"merge_four_views.py",
        "--top", output/"top_stereo", "--bottom", output/"bottom_stereo",
        "--output", output/"dataset", "--spacing-m", args.spacing_m)
    if args.geometry_only:
        print(f"Four-view dataset available: {output/'dataset'}")
        return
    run(args.python_training, HERE/"train_pure_torch.py", "--dataset", output/"dataset",
        "--output", output/"training", "--width", args.train_width,
        "--points", args.gaussians, "--steps", args.steps, "--lr", args.lr)
    print(f"Gaussian PLY: {output/'training'/'gaussians.ply'}")


if __name__ == "__main__":
    main()
