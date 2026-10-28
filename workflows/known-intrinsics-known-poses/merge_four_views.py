#!/usr/bin/env python3
"""Merge two rectified stereo models into a four-view known-pose COLMAP model."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
NAMES = ("image00", "image01", "image10", "image11")

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
    p.add_argument("--top", type=Path, required=True)
    p.add_argument("--bottom", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--spacing-m", type=float, default=0.10)
    args = p.parse_args()
    if args.spacing_m <= 0:
        p.error("spacing must be positive")
    build_dataset(args.output, args.top, args.bottom, args.spacing_m)


if __name__ == "__main__":
    main()
