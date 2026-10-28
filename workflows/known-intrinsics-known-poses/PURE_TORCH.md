# One-command binocular → binary Gaussian PLY (GTX 1060)

This is the recommended *experimental* two-photo pipeline on a Pascal GTX 1060. It does not use the incompatible gsplat custom CUDA extension.

## Inputs

From repository root:

```text
data/stereo/
  left.jpg       # first image, optical centre at origin
  right.jpg      # second image, exactly 10 cm to the right, no rotation
calibration/
  olympus_intrinsics.json  # ChArUco intrinsics, matching photo resolution/lens
```

The subject must be static, well textured, and visible in both frames. The camera's **optical centre**, not necessarily tripod mount, needs to translate 10 cm. Relative rotations are assumed zero.

## Setup (once)

Have the original `.venv` available for OpenCV/PyCOLMAP/Scipy geometry and `.venv-puregs` for CUDA PyTorch training. The runner invokes **both interpreters explicitly**; activate neither or either in PowerShell. In `.venv-puregs` install `plyfile` for binary PLY output:

```powershell
.\.venv-puregs\Scripts\python.exe -m pip install plyfile
```

## One command

From project root, with the pair and calibration in the locations above:

```powershell
python workflows/known-intrinsics-known-poses/run_pure_binocular.py --output output/my_first_pair --steps 1000
```

This does:
1. Rectified OpenCV stereo and disparity -> `stereo/points.ply` (ordinary RGB point cloud)
2. PyCOLMAP-validated sparse camera/point dataset -> `dataset/`
3. Pure PyTorch optimisation using **both photos** -> `training/render_1.png`, `render_2.png`, and **binary Gaussian** `training/gaussians.ply`

Open the final *binary* Gaussian file in [SuperSplat](https://superspl.at/editor). The source image output and disparity previews help diagnose quality. **Use a new --output directory for each experiment**; the runner refuses to replace an occupied directory.

## Different pairs and settings

To keep many pairs, place them in `data/pairs/plant01/left.jpg` and `right.jpg`, then:

```powershell
python workflows/known-intrinsics-known-poses/run_pure_binocular.py --pair-dir data/pairs/plant01 --output output/plant01 --steps 1000
```

For filenames other than `left.jpg` and `right.jpg` use `--left PATH --right PATH`. Set a different calibration with `--calibration PATH`.

Start with `--geometry-only` to assess disparity and the point cloud without GPU training. The key tuning flags:

| Flag | Default | Effect |
| --- | --- | --- |
| `--baseline-m` | `0.10` | Measured camera-right translation in metres; zero rotation assumed |
| `--stereo-width` | `1200` | Downsample width for stereo matching |
| `--disparities` | `128` | Stereo search range, multiple of 16 |
| `--block-size` | `5` | StereoSGBM block size (odd >=3) |
| `--max-stereo-points` | `12000` | Number of COLMAP initial points retained |
| `--train-width` | `240` | Width for training both photos |
| `--gaussians` | `500` | Initial Gaussians, sampled from stereo point cloud |
| `--steps` | `1000` | Optimisation iterations |
| `--lr` | `0.002` | Adam learning rate |

For 6 GB GPU memory issues, try `--gaussians 200 --train-width 160`. More steps alone may not improve results: the baseline has fixed small footprints and does not implement proper depth-sorted visibility. It outputs 3D Gaussian attributes, not photorealistic multi-view reconstruction.

If using non-default environments, pass `--python-geometry PATH` and `--python-training PATH`.

## Existing results / manual conversion

The *updated* `train_pure_torch.py` writes little-endian binary PLY directly; no post-processing required. Older ASCII `gaussians.ply` files can be converted without retraining:

```powershell
@'
from plyfile import PlyData
p = PlyData.read("output/puregs_test1/gaussians.ply")
p.text = False
p.byte_order = "<"
p.write("output/puregs_test1/gaussians_binary.ply")
'@ | .\.venv-puregs\Scripts\python.exe
```

These are experimental scripts and have **not** been verified end-to-end with a newly captured pair after the new runner change. Inspect the Gaussian previews and the import result rather than treating a generated PLY as proof of reconstruction quality.

---

# Pure-PyTorch binocular Gaussian experiment (GTX 1060 / sm_61)

This experiment uses the **existing calibrated two-photo COLMAP dataset** and directly optimises a few hundred 3D Gaussians with ordinary PyTorch CUDA ops. No custom CUDA extension, gsplat wheel, GLM or Visual Studio compiler is required for this stage.

**Important limitation:** This is a *minimal differentiable Gaussian-footprint renderer*, not full 3DGS rasterisation. It uses isotropic 3D scales, a fixed 9×9 Gaussian footprint, and weighted blending **without depth-sorted alpha compositing**. The exported file contains real 3D Gaussian parameters and should be useful for viewer/interoperability tests, but visual quality, occlusion and performance may be poor.

## Run on Windows

From the repository root, after merging the PR:

```powershell
git switch main
git pull origin main
.\.venv-puregs\Scripts\Activate.ps1
python -c "import torch, numpy, PIL; print(torch.__version__, torch.cuda.get_device_name(0))"
```

If NumPy or Pillow are missing: `python -m pip install "numpy<2" pillow`.

Use the dataset already generated by the binocular pipeline (`output/binocular_test2/dataset`). It must contain `images/left.png`, `images/right.png`, and `sparse/0/{cameras,images,points3D}.txt`.

```powershell
python workflows/known-intrinsics-known-poses/train_pure_torch.py --dataset output/binocular_test2/dataset --output output/puregs_test1 --width 240 --points 500 --steps 100
```

Outputs:
- `output/puregs_test1/render_1.png` and `render_2.png` — reconstructions of the two training views
- `output/puregs_test1/gaussians.ply` — experimental Gaussian PLY for SuperSplat or another 3DGS viewer

Run from the repository root. Unlike the upstream educational project, this script doesn't import `third_party/puregs`; it was written directly against PyTorch to avoid the upstream single-pixel, non-position-differentiable renderer.

If the renderer runs but the results are mostly white or poor, that isn't necessarily a failure of camera calibration; this baseline has restricted footprints, slow optimisation, and missing correct occlusion. Compare the preview renders against both rectified training photos. If GPU OOM occurs, reduce `--width` to 160 and `--points` to 200. The initial point positions are a subset of the previously validated stereo geometry, with the COLMAP camera poses taken from the exported reconstruction.

No existing gsplat installation or training outputs are overwritten by this script; use distinct `--output` directories for each run. This is **not yet tested on the user's Windows GTX 1060 or photographs**.

The next engineering milestone, after successful renders and PLY import, is a true per-pixel anisotropic covariance and visibility-aware compositing backend (or a verified Pascal-compatible rasterizer), before interpreting this as production-quality 3DGS.
