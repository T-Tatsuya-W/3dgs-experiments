#!/usr/bin/env python3
"""Experimental sm_61-friendly two-view Gaussian optimisation (PyTorch only).

Uses a differentiable fixed-size screen-space Gaussian footprint and weighted
transparency accumulation, NOT depth-sorted alpha compositing. Exports a real
3D Gaussian PLY, but render fidelity is limited; intended as a first training
and SuperSplat compatibility experiment on Pascal GPUs.
"""
import argparse
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F
from PIL import Image

C0 = 0.28209479177387814


def read_colmap(folder, limit, width):
    sparse = folder / "sparse" / "0"
    cams = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            c = line.split()
            if len(c) < 8 or c[1] != "PINHOLE":
                raise ValueError("This experiment requires rectified PINHOLE cameras")
            cams[int(c[0])] = tuple(map(float, c[2:8]))
    if not cams:
        raise ValueError(f"No PINHOLE cameras found in {sparse / 'cameras.txt'}")
    raw = [x.strip() for x in (sparse / "images.txt").read_text().splitlines()
           if x.strip() and not x.startswith("#")]
    views = []
    for i in range(0, len(raw), 2):
        p = raw[i].split()
        if len(p) < 10:
            raise ValueError("Unexpected COLMAP images.txt format")
        w, h, fx, fy, cx, cy = cams[int(p[8])]
        target_w = min(int(w), width)
        target_h = max(1, round(h * target_w / w))
        sx, sy = target_w / w, target_h / h
        q = np.array(list(map(float, p[1:5])))
        qw, qx, qy, qz = q
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
        ], dtype=np.float32)
        t = np.array(list(map(float, p[5:8])), dtype=np.float32)
        img = Image.open(folder / "images" / p[9]).convert("RGB").resize(
            (target_w, target_h), Image.Resampling.BILINEAR)
        views.append(dict(R=torch.from_numpy(R), t=torch.from_numpy(t),
                          K=(fx*sx, fy*sy, cx*sx, cy*sy),
                          target=torch.from_numpy(np.asarray(img, dtype=np.float32).copy() / 255),
                          w=target_w, h=target_h))
    if len(views) != 2:
        raise ValueError(f"Expected exactly two images, got {len(views)}")
    xyz, rgb = [], []
    for line in (sparse / "points3D.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            p = line.split()
            xyz.append([float(x) for x in p[1:4]])
            rgb.append([float(x)/255 for x in p[4:7]])
    if not xyz:
        raise ValueError("No stereo points found")
    sel = np.linspace(0, len(xyz)-1, min(limit, len(xyz)), dtype=int)
    return views, np.asarray(xyz, np.float32)[sel], np.asarray(rgb, np.float32)[sel]


def render(positions, log_scales, colors, opacity_logits, view, offsets, device):
    R, t = view["R"], view["t"]
    cam = positions @ R.T + t
    z = cam[:, 2].clamp_min(1e-4)
    fx, fy, cx, cy = view["K"]
    u = fx * cam[:, 0] / z + cx
    v = fy * cam[:, 1] / z + cy
    # Project an isotropic world-space Gaussian as a screen-space circle.
    radius = (torch.exp(log_scales[:, 0]) * (fx+fy)/2 / z).clamp(0.5, 12)
    # Use a 9x9 support; floor defines patch placement, smooth weights retain
    # derivatives w.r.t. centre, world-space location, size, opacity and colour.
    ix = torch.floor(u.detach()).long()[:, None] + offsets[None, :, 0]
    iy = torch.floor(v.detach()).long()[:, None] + offsets[None, :, 1]
    good = ((cam[:, 2:3] > 0.01) & (ix >= 0) & (ix < view["w"]) &
            (iy >= 0) & (iy < view["h"]))
    weights = (torch.exp(-0.5*((ix.float()+0.5-u[:, None])**2 +
                               (iy.float()+0.5-v[:, None])**2) /
                         radius[:, None].square()) *
               torch.sigmoid(opacity_logits)[:, None]) * good.float()
    indices = (iy.clamp(0, view["h"]-1) * view["w"] +
               ix.clamp(0, view["w"]-1)).reshape(-1)
    total = torch.zeros(view["h"]*view["w"], device=device)
    summed = torch.zeros(view["h"]*view["w"], 3, device=device)
    total = total.scatter_add(0, indices, weights.reshape(-1))
    summed = summed.index_add(0, indices,
                              (weights[..., None]*torch.sigmoid(colors)[:, None, :]).reshape(-1, 3))
    coverage = 1-torch.exp(-total)
    result = summed / total[:, None].clamp_min(1e-6) * coverage[:, None] + (1-coverage[:, None])
    return result.reshape(view["h"], view["w"], 3)


def export_ply(path, xyz, log_scales, colors, logits):
    xyz = xyz.detach().cpu().numpy()
    scales = log_scales.detach().cpu().numpy()
    rgb = torch.sigmoid(colors).detach().cpu().numpy()
    op = logits.detach().cpu().numpy()
    fields = ["x", "y", "z", "nx", "ny", "nz",
              "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    vertices = np.empty(len(xyz), dtype=[(name, "<f4") for name in fields])
    values = np.column_stack((
        xyz, np.zeros_like(xyz), (rgb - 0.5) / C0, op[:, None],
        scales, np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(xyz), 1))
    )).astype(np.float32)
    for i, name in enumerate(fields):
        vertices[name] = values[:, i]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("output/binocular_test2/dataset"))
    p.add_argument("--output", type=Path, default=Path("output/puregs_test"))
    p.add_argument("--width", type=int, default=240)
    p.add_argument("--points", type=int, default=500)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.002)
    args = p.parse_args()
    if args.points < 1 or args.width < 32 or args.steps < 1:
        p.error("points, width and steps must be positive")
    if not torch.cuda.is_available():
        p.error("Requires CUDA-capable PyTorch")
    device = torch.device("cuda")
    views, points, initial_colors = read_colmap(args.dataset, args.points, args.width)
    for v in views:
        v["R"], v["t"], v["target"] = v["R"].to(device), v["t"].to(device), v["target"].to(device)
    positions = torch.nn.Parameter(torch.tensor(points, device=device))
    # 1 cm starting diameter in the stereo dataset's metre-scale coordinates.
    log_scales = torch.nn.Parameter(torch.full_like(positions, math.log(0.01)))
    colors = torch.nn.Parameter(torch.logit(torch.tensor(initial_colors, device=device).clamp(0.01, 0.99)))
    opacities = torch.nn.Parameter(torch.zeros(len(points), device=device))
    offsets = torch.tensor([(x,y) for y in range(-4,5) for x in range(-4,5)],
                           device=device, dtype=torch.long)
    opt = torch.optim.Adam([positions, log_scales, colors, opacities], lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps):
        opt.zero_grad()
        losses = []
        for view in views:
            image = render(positions, log_scales, colors, opacities, view, offsets, device)
            losses.append(F.l1_loss(image, view["target"]))
        loss = sum(losses)/len(losses)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        opt.step()
        if step == 0 or (step+1) % 10 == 0:
            print(f"step {step+1}/{args.steps}  loss {loss.item():.6f}", flush=True)
    with torch.no_grad():
        for i, view in enumerate(views):
            preview = render(positions, log_scales, colors, opacities, view, offsets, device)
            out = (preview.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(out).save(args.output / f"render_{i+1}.png")
    export_ply(args.output / "gaussians.ply", positions, log_scales, colors, opacities)
    print(f"Experimental Gaussian PLY: {args.output / 'gaussians.ply'}")
    print("NOTE: Weighted splatting lacks depth-sorted compositing and full projected covariance.")


if __name__ == "__main__":
    main()
