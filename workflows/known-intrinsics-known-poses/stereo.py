#!/usr/bin/env python3
"""Two calibrated, known-pose photos -> rectified stereo disparity -> coloured point-cloud PLY.

This does NOT create a gsplat/COLMAP training dataset or Gaussian PLY.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_calibration(path, shape):
    obj = json.loads(path.read_text(encoding="utf8"))
    w, h = shape
    if (obj["image_width"], obj["image_height"]) != (w, h):
        raise ValueError(f"Calibration {path} is for {obj['image_width']}x{obj['image_height']}, not {w}x{h}")
    K = np.asarray(obj["camera_matrix"], dtype=np.float64)
    D = np.asarray(obj["distortion_coefficients"], dtype=np.float64)
    return K, D


def rotation_xyz_degrees(deg):
    x, y, z = np.radians(deg)
    Rx = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Rz = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    return Rz @ Ry @ Rx


def save_cloud(path, points, colors):
    xyz = points.reshape(-1, 3)
    rgb = colors.reshape(-1, 3)
    valid = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[valid], rgb[valid]
    if len(xyz) > 1000000:
        indices = np.linspace(0, len(xyz)-1, 1000000, dtype=np.int64)
        xyz, rgb = xyz[indices], rgb[indices]
    with path.open("wb") as out:
        header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        out.write(header.encode("ascii"))
        dtype = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),
                          ("red","u1"),("green","u1"),("blue","u1")])
        data = np.empty(len(xyz), dtype=dtype)
        for i, name in enumerate(("x", "y", "z")):
            data[name] = xyz[:, i]
        for i, name in enumerate(("red", "green", "blue")):
            data[name] = rgb[:, i]
        out.write(data.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--calibration-right", type=Path)
    p.add_argument("--pose", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output/stereo"))
    p.add_argument("--min-disparity", type=int, default=0)
    p.add_argument("--num-disparities", type=int, default=128)
    p.add_argument("--block-size", type=int, default=5)
    args = p.parse_args()
    if args.num_disparities < 16 or args.num_disparities % 16:
        p.error("--num-disparities must be a positive multiple of 16")
    if args.block_size < 3 or args.block_size % 2 != 1:
        p.error("--block-size must be odd and >= 3")
    left = cv2.imread(str(args.left))
    right = cv2.imread(str(args.right))
    if left is None or right is None:
        p.error("Could not read left/right photographs")
    if left.shape[:2] != right.shape[:2]:
        p.error("Images must have identical width and height for this first experiment")
    h, w = left.shape[:2]
    K1, D1 = load_calibration(args.calibration, (w, h))
    K2, D2 = load_calibration(args.calibration_right or args.calibration, (w, h))
    pose = json.loads(args.pose.read_text(encoding="utf8"))
    C = np.asarray(pose["camera2_center_in_camera1_m"], dtype=np.float64)
    if C.shape != (3,) or np.linalg.norm(C) < 1e-6:
        p.error("Camera 2 centre must be a nonzero 3-vector in metres")
    # Orientation of camera 2 in camera 1 coordinates; OpenCV wants 1 -> 2.
    R2_in_1 = rotation_xyz_degrees(pose["camera1_to_camera2_rotation_xyz_deg"])
    R = R2_in_1.T
    T = -R @ C
    image_size = (w, h)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    m1x, m1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    a = cv2.remap(left, m1x, m1y, cv2.INTER_LINEAR)
    b = cv2.remap(right, m2x, m2y, cv2.INTER_LINEAR)
    matcher = cv2.StereoSGBM_create(
        minDisparity=args.min_disparity, numDisparities=args.num_disparities,
        blockSize=args.block_size, P1=8*3*args.block_size**2,
        P2=32*3*args.block_size**2, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1
    )
    disparity = matcher.compute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16
    cloud = cv2.reprojectImageTo3D(disparity, Q)
    valid = (disparity > args.min_disparity) & (disparity < args.min_disparity + args.num_disparities)
    valid &= np.isfinite(cloud).all(axis=2) & (cloud[:, :, 2] > 0)
    cloud[~valid] = np.nan
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "rectified_left.png"), a)
    cv2.imwrite(str(args.output / "rectified_right.png"), b)
    disp_vis = np.clip((disparity - args.min_disparity) / args.num_disparities * 255, 0, 255).astype(np.uint8)
    disp_vis[~valid] = 0
    cv2.imwrite(str(args.output / "disparity.png"), disp_vis)
    save_cloud(args.output / "points.ply", cloud, cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    print(f"Valid stereo depths: {int(valid.sum())}/{valid.size}. Wrote {args.output / 'points.ply'}")
    print("This is a regular point cloud, not a Gaussian splat; inspect disparity and geometry.")


if __name__ == "__main__":
    main()
