# 3DGS Experiments

A small Python-driven workflow for **photographs → COLMAP camera poses → gsplat Gaussian PLY**. No Nerfstudio application required. First goal: photograph a stationary scene with one camera (e.g. Olympus PEN E-P2), then open the resulting splat in [SuperSplat](https://superspl.at/editor/).

**Status:** Initial untested scaffold. The project wraps two external tools; it does not implement Gaussian optimisation from scratch. The standard gsplat example trainer requires an NVIDIA CUDA GPU.

## What to install

Recommended host: **Ubuntu Linux**, or **WSL2 Ubuntu on Windows 11 with NVIDIA CUDA support**. On Linux you need Git, Python + venv, COLMAP command-line program, and compatible NVIDIA driver / CUDA PyTorch. Target a modern Python supported by the checked-out gsplat version (often Python 3.10+). Raspberry Pi 4B and integrated Intel GPU systems can do camera capture / some COLMAP processing, but **not the standard CUDA gsplat training**.

On Ubuntu start with:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip build-essential colmap
nvidia-smi               # should show your NVIDIA GPU
colmap -h                # should show COLMAP CLI
```

COLMAP's CLI can do SfM on CPU; a CUDA-enabled COLMAP build isn't required for this starter. If the distro package gives problems, see [COLMAP installation](https://colmap.github.io/install.html).

Clone and create an environment:

```bash
git clone https://github.com/T-Tatsuya-W/3dgs-experiments.git
cd 3dgs-experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install a **matching CUDA PyTorch build** from the official [PyTorch selector](https://pytorch.org/get-started/locally/). Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last value must be `True` for training. Then clone gsplat and set up **its example dependencies**:

```bash
mkdir -p third_party
git clone https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
cd third_party/gsplat
python -m pip install -e .
# Check examples/requirements.txt: it may PIN torch/torchvision.
# Resolve that compatibility with your installed CUDA PyTorch FIRST.
python -m pip install -r examples/requirements.txt
cd ../..
```

**Caution:** Upstream gsplat's examples requirements can pin particular torch/torchvision builds or include CUDA extensions and Git dependencies. Do not blindly run the final install if those pins conflict with your chosen PyTorch/CUDA stack. Use a compatible upstream revision and follow [gsplat's current instructions](https://docs.gsplat.studio/). This repo uses only the Python standard library itself; dependencies live in the third-party trainer.

## Photographs

For the **first test**, take around **20–50 sharp, overlapping JPEG photos** of a *stationary, textured* subject, from genuinely different camera positions. Keep zoom/focal length, focus, exposure and white balance as constant as practical. Do not just rotate the camera in place. Calibration can come later: COLMAP initially estimates shared intrinsics for a single camera.

Copy them into:

```text
3dgs-experiments/
├── README.md
├── run.py
├── .gitignore
├── data/
│   └── test01/
│       └── images/
│           ├── 001.jpg
│           ├── 002.jpg
│           └── ...
├── output/                          # generated, git-ignored
│   └── test01/
│       ├── dataset/
│       │   ├── images/
│       │   ├── database.db
│       │   └── sparse/0/            # poses, intrinsics, points
│       └── training/
│           └── ply/                 # exported .ply
└── third_party/                     # cloned upstream gsplat, git-ignored
    └── gsplat/examples/simple_trainer.py
```

The input images, training results and third-party repository are intentionally excluded from Git.

## Run

With the environment activated, in the project root:

```bash
python run.py --images data/test01/images --output output/test01
```

The wrapper does three things:

1. Validates / copies images to a self-contained COLMAP dataset.
2. Runs COLMAP `feature_extractor`, `exhaustive_matcher`, and `mapper` using a single shared `SIMPLE_RADIAL` camera model. The first reconstruction should appear at `output/test01/dataset/sparse/0/`.
3. Runs `third_party/gsplat/examples/simple_trainer.py` at factor 4 and 3,000 steps, with a requested Gaussian PLY export at the final step.

Useful commands:

```bash
# CPU-capable camera estimation ONLY, for debugging:
python run.py --colmap-only

# Reuse successful pose estimation and train again:
python run.py --skip-colmap --output output/test01

# Separate experiment, more steps and higher resolution:
python run.py --images data/test02/images --output output/test02 --steps 7000 --factor 2

# External gsplat clone:
python run.py --gsplat-examples /path/to/gsplat/examples
```

**Important:** Use a *new* `--output` directory for each new image set. The script prevents running a new COLMAP reconstruction over an existing `database.db`. COLMAP can fail with insufficient overlap, and successful COLMAP poses do not guarantee good-quality splats. gsplat's example CLI may change across upstream revisions; adapt the wrapper if that happens.

## View the result

Find `.ply` in `output/test01/training/ply/` (the script prints paths when exports exist), then import it into [SuperSplat](https://superspl.at/editor/). A Gaussian PLY stores splat attributes and needs a Gaussian-aware viewer, rather than an ordinary mesh viewer.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| No COLMAP sparse reconstruction | Sharp textured subject, 20–50 images, substantial overlap and actual camera translation |
| `colmap` not found | Install the COLMAP CLI / check your PATH |
| CUDA unavailable | NVIDIA driver, CUDA-compatible PyTorch wheel, WSL GPU passthrough |
| gsplat import/compilation error | Compatible Python, PyTorch, toolkit/compiler and upstream example requirements |
| GPU out of memory | Increase `--factor` (e.g. 4 to 8) or use smaller input photos |
| No PLY generated | Trainer logs, `--save_ply` / `--ply_steps` options for your gsplat revision |

## Later

Future experiments: saved camera intrinsics; pre-calibrated fixed multi-webcam rig extrinsics; alternative sparse-geometry initialisation; intentional Gaussian artefacts and artistic parameters. **These are not part of this first MVP.**
