# Four-view 10 cm square — known intrinsics, known poses

**Experiment:** four photographs with identical orientation, captured at corners of a 10 cm square, using the same calibrated camera. This is separate from the two-photo experiment.

## Capture layout

Looking *from behind* the camera towards the scene, with the camera's **+X right, +Y down, +Z forward** convention:

```text
image00.jpg  ---- 10 cm ----  image01.jpg
    |                           |
   10 cm                       10 cm
    |                           |
image10.jpg  ---- 10 cm ----  image11.jpg
```

World origin: image00 camera's optical centre. Other optical centres:
- image00 = (0, 0, 0) m
- image01 = (0.10, 0, 0) m
- image10 = (0, 0.10, 0) m
- image11 = (0.10, 0.10, 0) m

All rotations are assumed **identical**. The 10 cm is optical-centre displacement, not distance between tripod screw marks. Do not aim each camera individually at the subject; the cameras should stay parallel. Keep scene and lens settings unchanged across all shots. This geometry is idealized — unintended rotations or translations directly cause depth errors.

Put the four JPEGs at `data/four_view/scene01/image00.jpg` through `image11.jpg`. Intrinsics: `calibration/olympus_intrinsics.json`, matching image resolution, focus and focal length.

## Quick start (PowerShell, repository root)

```powershell
cd "C:\Users\Toby T Watkinson\3dgs\3dgs-experiments"
git switch main
git pull origin main
.\.venv-puregs\Scripts\Activate.ps1
python -m pip install plyfile
python workflows/known-intrinsics-known-poses/run_four_view.py --pair-dir data/four_view/scene01 --output output/four_scene01 --steps 1000
```

The runner invokes the existing `.venv` (OpenCV, PyCOLMAP, SciPy) for geometry and `.venv-puregs` (PyTorch CUDA, Pillow, plyfile) for training. You do not need to install gsplat or compile CUDA extensions for training. Use a fresh `--output` directory for every run.

Final outputs:
- `output/four_scene01/training/gaussians.ply`: binary 3D Gaussian PLY
- `output/four_scene01/training/render_1.png` ... `render_4.png`: four training-view previews
- `output/four_scene01/top_stereo/disparity.png` and `bottom_stereo/disparity.png`: stereo diagnostics
- `output/four_scene01/dataset/sparse/0/`: four registered known-pose cameras + 3D points

To inspect geometry *before* training:

```powershell
python workflows/known-intrinsics-known-poses/run_four_view.py --pair-dir data/four_view/scene01 --output output/four_scene01_geom --geometry-only
```

## Tuning

| Option | Default | Meaning |
|---|---:|---|
| `--spacing-m` | 0.10 | Both horizontal and vertical spacing in metres |
| `--stereo-width` | 1200 | Width for OpenCV stereo matching |
| `--disparities` | 128 | Disparity search span, multiple of 16 |
| `--max-stereo-points` | 12000 | Point cap **per** stereo row |
| `--train-width` | 240 | Width of all four training views |
| `--gaussians` | 500 | Initial points sampled from merged cloud |
| `--steps` | 1000 | Training iterations |
| `--lr` | 0.002 | Adam learning rate |
| `--calibration` | calibration/olympus_intrinsics.json | Calibration path |

For example, try `--gaussians 1500 --train-width 320 --steps 2000` with a new output directory. Lower values if GTX 1060 VRAM is constrained.

## How much improvement should we expect?

Four known viewpoints provide more constraints and *potential* surface coverage than two, but **do not guarantee accurate depth**. The initial geometry here comes from **two horizontal stereo reconstructions** (00↔01 and 10↔11), then the points are transformed into a common metric frame and both clouds are used to initialise Gaussians. The renderer is optimised against all four images.

This first version **does not perform joint four-view feature matching, multi-view consistency filtering, diagonal-baseline triangulation, or true depth-sorted Gaussian compositing**. Corresponding objects can therefore still yield floaters, misaligned depths or duplicate points between rows. The next improvement is to cross-validate the upper/lower/diagonal matches or jointly triangulate across all four cameras.

The current PyTorch renderer is also a simplified weighted-splat prototype, so inspect the two disparity maps and point clouds before attributing poor results solely to the number of cameras. **This workflow has not yet been run on the user's photographs.**
