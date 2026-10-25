# 3DGS Experiments

A small Python-driven workflow for **photographs → COLMAP camera poses → gsplat Gaussian PLY**. No Nerfstudio application required. First goal: photograph a stationary scene with one camera (e.g. Olympus PEN E-P2), then open the resulting splat in [SuperSplat](https://superspl.at/editor/).

**Status:** Initial untested scaffold. The project wraps two external tools; it does not implement Gaussian optimisation from scratch. The standard gsplat example trainer requires an NVIDIA CUDA GPU.

## Supported environments

The Python wrapper (`run.py`) is written with standard-library, cross-platform APIs and can run on **Ubuntu or Windows**. You still need two external components:

1. **COLMAP** for camera poses: supported on both Windows and Ubuntu, including CPU-based feature extraction/matching.
2. **gsplat's upstream example trainer** for splat optimisation/export: standard training needs an **NVIDIA CUDA-compatible GPU**, an appropriate PyTorch build and other example dependencies. Windows-native setup may require Visual Studio C++ Build Tools, a CUDA toolkit and extension compilation. **WSL2 Ubuntu is the recommended Windows training path**, not a guarantee that every gsplat revision runs natively.

You can run `--colmap-only` without installing gsplat or an NVIDIA GPU. A Raspberry Pi 4B is not a practical target for the CUDA training stage.

### Option A: Ubuntu Linux (or Ubuntu inside WSL2)

Ubuntu terminal:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential colmap
git clone https://github.com/T-Tatsuya-W/3dgs-experiments.git
cd 3dgs-experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
colmap -h
```

When running inside **WSL2 on Windows**, first install WSL from an **administrator PowerShell** on the Windows host:

```powershell
wsl --install -d Ubuntu
```

Restart if prompted, open the Ubuntu distribution, then run the Ubuntu commands above. Install/update a suitable **Windows NVIDIA driver** for WSL CUDA support; do not try to install a separate Linux display driver inside WSL. Confirm `nvidia-smi` works from WSL and PyTorch detects CUDA before training. If `colmap` has missing GUI/display libraries in your headless environment, consult [COLMAP installation](https://colmap.github.io/install.html).

### Option B: Native Windows (PowerShell)

COLMAP's **Windows distribution** is available via its [official releases/install docs](https://colmap.github.io/install.html). Download/extract it, locate `COLMAP.bat` or `colmap.exe` (depending on distribution), and verify that its command-line executable runs. Either add its directory to `PATH` or pass the executable path via `--colmap` (quote paths containing spaces).

Install **Git for Windows**, **Python for Windows** (with `py` launcher), and (if training) a compatible NVIDIA GPU driver plus the compiler/CUDA prerequisites for your gsplat revision.

```powershell
git clone https://github.com/T-Tatsuya-W/3dgs-experiments.git
cd 3dgs-experiments
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
colmap -h
```

If PowerShell blocks activating the venv, you can instead run Python explicitly with `.\.venv\Scripts\python.exe`, or set the **current process only** to allow activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `colmap` is not on `PATH`, use a fully qualified path in the run command:

```powershell
python run.py --colmap "C:\path\to\COLMAP\COLMAP.bat" --colmap-only
```

**Note:** Some Windows COLMAP bundles use a `.bat` launcher. Python's subprocess handling of batch files differs from native executables; if that path fails, prefer invoking the actual `colmap.exe`, adding its directory to `PATH`, or running the project under WSL2. Check `colmap -h` independently first.

### CUDA/PyTorch and gsplat training — both environments

If you only want to check camera reconstruction, skip this section and use `--colmap-only`.

Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for a CUDA-enabled build matching your OS, GPU/driver and the gsplat revision. The appropriate install command can differ between Ubuntu, WSL2 and native Windows. After installing:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The final value must be `True`.

Get the upstream gsplat example trainer:

**Ubuntu / WSL2:**

```bash
mkdir -p third_party
git clone https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
cd third_party/gsplat
python -m pip install -e .
# Inspect examples/requirements.txt for torch/torchvision pins first!
python -m pip install -r examples/requirements.txt
cd ../..
```

**Native Windows PowerShell:**

```powershell
New-Item -ItemType Directory -Force third_party | Out-Null
git clone https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
cd third_party/gsplat
python -m pip install -e .
# Inspect examples/requirements.txt for torch/torchvision pins first!
python -m pip install -r examples/requirements.txt
cd ../..
```

**Compatibility warning:** Do not blindly install `examples/requirements.txt`: it can pin particular PyTorch builds and CUDA extensions and may overwrite your working PyTorch. Verify compatibility against your chosen gsplat revision and follow the [gsplat docs](https://docs.gsplat.studio/). If you hit native Windows CUDA compilation problems, prefer **WSL2 Ubuntu**. This README documents the intended workflow; it is not yet an end-to-end verified Windows build.

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

## Run on Ubuntu or Windows

Use the same script and arguments on both systems. Run commands from the **repository root**, with your environment activated. Paths are resolved using Python's `pathlib`.

**Ubuntu / WSL2 (bash):**

```bash
mkdir -p data/test01/images
# Copy 20–50 JPEGs into data/test01/images/ first.
python run.py --images data/test01/images --output output/test01
```

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force data/test01/images | Out-Null
# Copy 20–50 JPEGs into data/test01/images/ first.
python run.py --images data/test01/images --output output/test01
```

For a first smoke test without any training dependencies, run `python run.py --colmap-only` after putting photographs in the input directory. After successful COLMAP poses, `python run.py --skip-colmap --output output/test01` runs the trainer separately.

The script validates/copies photos, runs COLMAP `feature_extractor`, `exhaustive_matcher` and `mapper` using a shared `SIMPLE_RADIAL` camera model, then calls the upstream gsplat `simple_trainer.py` with 3,000 steps / downsample factor 4 / a requested Gaussian PLY export.

**Commands shared across platforms:**

```text
python run.py --colmap-only
python run.py --skip-colmap --output output/test01
python run.py --images data/test02/images --output output/test02 --steps 7000 --factor 2
python run.py --gsplat-examples third_party/gsplat/examples
python run.py --help
```

Use a new `--output` directory for each image dataset; the script stops if COLMAP's database already exists. The 3,000-step preset is for experimenting, not guaranteed convergence. If the gsplat example CLI changes, the wrapper may require an update.

## View the result

Find `.ply` in `output/test01/training/ply/` (the script prints paths when exports exist), then import it into [SuperSplat](https://superspl.at/editor/). A Gaussian PLY stores splat attributes and needs a Gaussian-aware viewer, rather than an ordinary mesh viewer.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| No COLMAP sparse reconstruction | Sharp textured subject, 20–50 images, substantial overlap and actual camera translation |
| `colmap` not found | Install the COLMAP CLI / check your PATH |
| CUDA unavailable | NVIDIA driver, CUDA-enabled PyTorch build, and (on Windows) WSL2 GPU passthrough where relevant |\n| PowerShell won't activate venv | Use `Set-ExecutionPolicy -Scope Process Bypass`, or call the venv's `python.exe` directly |\n| Native Windows gsplat build fails | Check MSVC/CUDA/PyTorch compatibility; WSL2 Ubuntu is the recommended fallback |
| gsplat import/compilation error | Compatible Python, PyTorch, toolkit/compiler and upstream example requirements |
| GPU out of memory | Increase `--factor` (e.g. 4 to 8) or use smaller input photos |
| No PLY generated | Trainer logs, `--save_ply` / `--ply_steps` options for your gsplat revision |

## Later

Future experiments: saved camera intrinsics; pre-calibrated fixed multi-webcam rig extrinsics; alternative sparse-geometry initialisation; intentional Gaussian artefacts and artistic parameters. **These are not part of this first MVP.**
