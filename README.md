# 3DGS Experiments

A small Python-driven workflow for **photographs → PyCOLMAP camera poses → gsplat Gaussian PLY**. No Nerfstudio application required. First goal: photograph a stationary scene with one camera (e.g. Olympus PEN E-P2), then open the resulting splat in [SuperSplat](https://superspl.at/editor/).

**Status:** Initial untested scaffold. The project uses PyCOLMAP in Python and an upstream gsplat example trainer; it does not implement Gaussian optimisation from scratch. The standard gsplat example trainer requires an NVIDIA CUDA GPU.

## Supported environments

The reconstruction stage now uses **[PyCOLMAP](https://colmap.github.io/pycolmap/)** directly rather than the separate COLMAP desktop/CLI installation. It supports Windows and Ubuntu through Python wheels (for supported Python/platform versions). Feature extraction and matching run on CPU by default; reconstruction does not require NVIDIA hardware.

**Important:** The *gsplat training* stage is separate and still normally needs an NVIDIA CUDA GPU. If you use Windows for training, **WSL2 Ubuntu is recommended** over native Windows due to PyTorch/CUDA extension build requirements. You may run `--colmap-only` on a non-NVIDIA computer.

### Ubuntu Linux / WSL2 Ubuntu

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip build-essential
git clone https://github.com/T-Tatsuya-W/3dgs-experiments.git
cd 3dgs-experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For Windows with WSL2, install Ubuntu from an administrator PowerShell with `wsl --install -d Ubuntu`, then use the Ubuntu steps above inside that distribution. WSL2 GPU training additionally needs a compatible Windows NVIDIA driver and CUDA-enabled PyTorch.

### Native Windows PowerShell

Install Python (including the `py` launcher) and Git for Windows, then:

```powershell
git clone https://github.com/T-Tatsuya-W/3dgs-experiments.git
cd 3dgs-experiments
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` followed by the activation command, or invoke `.\.venv\Scripts\python.exe` directly.

### Confirm PyCOLMAP works (either OS)

```bash
python -c "import pycolmap; print('PyCOLMAP', pycolmap.__version__)"
```

**No standalone `colmap` executable is necessary.** If PyCOLMAP has no compatible wheel for your Python/platform, switch to a supported Python version rather than trying to build from source as the first option.

### Alternative: precompiled gsplat wheel (avoid local C++/CUDA compilation)

If you only want to **use** gsplat, prefer a compatible prebuilt wheel rather than `pip install -e .`, which compiles code from source. Official wheel index:

- [Browse available gsplat binaries](https://docs.gsplat.studio/whl/gsplat/)
- [Official PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)

**Suggested experimental target (x86-64 Windows or x86-64 Linux, not Raspberry Pi):**

| Component | Target |
| --- | --- |
| Python | **3.10**, 64-bit |
| PyTorch | **2.4.1** |
| torchvision / torchaudio | **0.19.1 / 2.4.1** |
| PyTorch CUDA runtime | **12.4** (`cu124`) |
| gsplat binary | **1.5.3** with `pt24cu124` and `cp310` |
| Platform | `win_amd64` (Windows) or `linux_x86_64` (Linux) |

These exact platform wheels are listed in the official gsplat wheel index. **Check your NVIDIA GPU supports CUDA 12.4 before adopting this older environment.** This is a *binary-availability target*, not a guarantee that the newest upstream `simple_trainer.py` and its latest example dependencies work with gsplat 1.5.3. Keep the environment separate from any current CUDA 13/PyTorch 2.9 environment. The local CUDA 13.4 toolkit does not need to match this runtime just to use the prebuilt wheel.

**PowerShell (Windows):**

```powershell
py -3.10 -m venv .venv-wheel
.\.venv-wheel\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

**Bash (Ubuntu; install Python 3.10 first if absent):**

```bash
python3.10 -m venv .venv-wheel
source .venv-wheel/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Then **download the exact platform wheel** using the [gsplat wheel index](https://docs.gsplat.studio/whl/gsplat/):

- Windows: `gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl`
- Linux: `gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl`

Install it from the download directory with a quoted filename (replace the filename for Linux):

```bash
python -m pip install "./gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl"
```

Verify the environment:

```bash
python -c "import sys, torch, gsplat; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(gsplat.__version__)"
```

**Trainer compatibility matters:** This repository currently invokes `third_party/gsplat/examples/simple_trainer.py`, which is cloned separately and can change on `main`. The old 1.5.3 wheel might be incompatible with a new trainer (new APIs, import paths, or required torch >= 2.9). For a binary-based *end-to-end* run, use the corresponding gsplat `v1.5.3` source tag for its examples (do not install that source in editable mode), inspect its requirements, and expect to adapt the wrapper CLI flags to that tagged trainer if they differ. In `third_party/gsplat`:

```bash
git fetch --tags
git checkout v1.5.3
```

Install compatible **example-only** dependencies carefully; don't blindly install an example requirements file that downgrades or overwrites the chosen PyTorch build. Our `run.py` is currently tested against neither that historical trainer nor the prebuilt wheel, so this is an **optional alternative to experiment with**, not yet the project's verified default.

**Raspberry Pi 4B:** These wheels do **not** work on the Pi. They target x86-64 Windows/Linux, while the Pi is ARM-based and has no NVIDIA CUDA GPU. The Pi can still handle capture, photo transfer, and potentially PyCOLMAP on a supported ARM build, but not this standard gsplat CUDA training. Plan to run Gaussian training on a CUDA-equipped PC.

### Source installation: CUDA/PyTorch and gsplat training — both environments

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

For the **first test**, take around **20–50 sharp, overlapping JPEG photos** of a *stationary, textured* subject, from genuinely different camera positions. Keep zoom/focal length, focus, exposure and white balance as constant as practical. Do not just rotate the camera in place. Calibration can come later: PyCOLMAP currently estimates shared intrinsics for a single camera.

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

For a first smoke test without CUDA or gsplat, run `python run.py --colmap-only` after adding photographs. After successful poses, `python run.py --skip-colmap --output output/test01` runs the trainer separately.

The script validates/copies photos, calls PyCOLMAP's `extract_features`, `match_exhaustive` and `incremental_mapping` using a shared `SIMPLE_RADIAL` camera model, then calls the upstream gsplat `simple_trainer.py` with 3,000 steps / downsample factor 4 / a requested Gaussian PLY export.

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
| `ModuleNotFoundError: pycolmap` | Activate your environment and run `python -m pip install -r requirements.txt` |
| CUDA unavailable | NVIDIA driver, CUDA-enabled PyTorch build, and (on Windows) WSL2 GPU passthrough where relevant |\n| PowerShell won't activate venv | Use `Set-ExecutionPolicy -Scope Process Bypass`, or call the venv's `python.exe` directly |\n| Native Windows gsplat build fails | Check MSVC/CUDA/PyTorch compatibility; WSL2 Ubuntu is the recommended fallback |
| gsplat import/compilation error | Compatible Python, PyTorch, toolkit/compiler and upstream example requirements |
| GPU out of memory | Increase `--factor` (e.g. 4 to 8) or use smaller input photos |
| No PLY generated | Trainer logs, `--save_ply` / `--ply_steps` options for your gsplat revision |

## Later

Future experiments: saved camera intrinsics and image calibration import (not implemented yet); pre-calibrated fixed multi-webcam rig extrinsics; alternative sparse-geometry initialisation; intentional Gaussian artefacts and artistic parameters. **These are not part of this first MVP.**
