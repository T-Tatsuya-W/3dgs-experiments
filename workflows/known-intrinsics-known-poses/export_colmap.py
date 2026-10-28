#!/usr/bin/env python3
"""Convert rectified stereo output to a COLMAP-format dataset for two-view gsplat.

Writes a valid *text* COLMAP sparse model; the patched gsplat v1.5.3 loader
uses PyCOLMAP Reconstruction to read it. No SfM initialization is required.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np


def quat_from_rotation(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (w, x, y, z)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stereo", type=Path, default=Path("output/stereo"))
    p.add_argument("--output", type=Path, default=Path("output/two_view/dataset"))
    p.add_argument("--max-points", type=int, default=12000)
    p.add_argument("--min-depth", type=float, default=0.15)
    p.add_argument("--max-depth", type=float, default=20.0)
    args = p.parse_args()
    if args.max_points < 10 or args.min_depth <= 0 or args.max_depth <= args.min_depth:
        p.error("Check max-points and depth range")
    meta = json.loads((args.stereo / "stereo_meta.json").read_text(encoding="utf8"))
    with np.load(args.stereo / "stereo_data.npz") as data:
        points, valid, rgb, disparity = (data[k] for k in ("points", "valid", "rgb", "disparity"))
        points, valid, rgb, disparity = (v.copy() for v in (points, valid, rgb, disparity))
    h, w = valid.shape
    yy, xx = np.nonzero(valid & (points[:, :, 2] >= args.min_depth) & (points[:, :, 2] <= args.max_depth))
    xx_right = xx.astype(np.float64) - disparity[yy, xx]
    keep = (xx_right >= 0) & (xx_right < w - 1)
    yy, xx, xx_right = yy[keep], xx[keep], xx_right[keep]
    if len(xx) > args.max_points:
        # Evenly subsample across image locations; avoids creating a million Gaussians.
        selection = np.linspace(0, len(xx) - 1, args.max_points, dtype=np.int64)
        yy, xx, xx_right = yy[selection], xx[selection], xx_right[selection]
    if len(xx) < 50:
        raise RuntimeError(f"Only {len(xx)} usable stereo points. Inspect disparity and camera pose first.")
    xyz, cols = points[yy, xx], rgb[yy, xx]
    dataset = args.output
    images, sparse = dataset / "images", dataset / "sparse" / "0"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.stereo / "left.png", images / "left.png")
    shutil.copy2(args.stereo / "right.png", images / "right.png")
    P1 = np.asarray(meta["P1"], dtype=float)
    P2 = np.asarray(meta["P2"], dtype=float)
    with (sparse / "cameras.txt").open("w", encoding="utf8") as f:
        for i, P in enumerate((P1, P2), start=1):
            f.write(f"{i} PINHOLE {w} {h} {P[0,0]:.12g} {P[1,1]:.12g} {P[0,2]:.12g} {P[1,2]:.12g}\n")
    R = np.asarray(meta["R_right_from_left"], dtype=float)
    t = np.asarray(meta["t_right_from_left_m"], dtype=float)
    with (sparse / "images.txt").open("w", encoding="utf8") as f:
        f.write("1 1 0 0 0 0 0 0 1 left.png\n")
        f.write(" ".join(f"{float(x):.6f} {float(y):.6f} {i}" for i, (x,y) in
                         enumerate(zip(xx, yy), start=1)) + "\n")
        q = quat_from_rotation(R)
        f.write("2 " + " ".join(f"{v:.15g}" for v in (*q, *t)) + " 2 right.png\n")
        f.write(" ".join(f"{float(x):.6f} {float(y):.6f} {i}" for i, (x,y) in
                         enumerate(zip(xx_right, yy), start=1)) + "\n")
    with (sparse / "points3D.txt").open("w", encoding="utf8") as f:
        for i, (pt, col) in enumerate(zip(xyz, cols), start=1):
            f.write(f"{i} {pt[0]:.10g} {pt[1]:.10g} {pt[2]:.10g} "
                    f"{int(col[0])} {int(col[1])} {int(col[2])} 0 1 {i-1} 2 {i-1}\n")
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(sparse))
    if reconstruction.num_reg_images() != 2 or len(reconstruction.points3D) == 0:
        raise RuntimeError("Validation failed: sparse reconstruction has no registered stereo pair/points.")
    print(f"COLMAP dataset: {dataset} | 2 registered images, {len(reconstruction.points3D)} points")
    print("Next: python run.py --skip-colmap --output", dataset.parent, "--factor 1 --steps 1000")
    print("WARNING: stereo depth and the two-image gsplat path are experimental, not yet end-to-end tested.")


if __name__ == "__main__":
    main()
