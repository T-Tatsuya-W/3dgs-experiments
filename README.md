# 3DGS Experiments

A minimal Windows pipeline: **photos → PyCOLMAP poses → precompiled gsplat → Gaussian .ply**.

## One supported setup (Windows 11, NVIDIA GTX 1060 6 GB)

This setup is pinned to the user's currently working binary imports. **End-to-end training has not yet been run on a real dataset**; do not treat a successful package import as proof that training succeeds.

| Component | Version |
| --- | --- |
| Python | **3.10, 64-bit** |
| PyTorch | **2.4.1+cu124** |
| gsplat | **1.5.3+pt24cu124** precompiled Windows wheel |
| PyCOLMAP | **4.2.1** (reconstruction + training reader) |
| gsplat example source | tag **v1.5.3** |

No Visual Studio Build Tools or system CUDA Toolkit required for the pinned **binary** path. Keep the NVIDIA graphics driver. The example trainer is patched locally for modern PyCOLMAP and to replace a mandatory compiled SSIM dependency.

### 1. Clean start (PowerShell at project root)

```powershell
git clone https://github.com/T-Tatsuya-W/3dgs-experiments.git
cd 3dgs-experiments
# After this PR merges, or while testing this PR: git checkout fix/pinned-windows-gsplat-setup
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again. Use a **fresh** environment; do not try to repair a previously source-built gsplat installation.

### 2. Install pinned binary packages

```powershell
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install "gsplat==1.5.3" --index-url https://docs.gsplat.studio/whl/pt24cu124/ --extra-index-url https://pypi.org/simple --only-binary=gsplat
python -m pip install -r requirements.txt
python -c "import torch, gsplat, pycolmap; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), gsplat.__version__, pycolmap.__version__)"
```

Expected: torch `2.4.1+cu124`, `12.4`, `True`; gsplat `1.5.3+pt24cu124`; PyCOLMAP `4.2.1`. **Stop if CUDA reports False.** NVIDIA driver must support CUDA 12.4; `nvidia-smi` should work.

### 3. Install example-only dependencies and pin source

```powershell
python -m pip install -r requirements-trainer.txt
New-Item -ItemType Directory -Force third_party | Out-Null
git clone --branch v1.5.3 --depth 1 https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
python scripts/patch_gsplat_v153.py
python third_party/gsplat/examples/simple_trainer.py --help
```

**Never run** `pip install -e .` within `third_party/gsplat`; that invokes a CUDA/C++ build. **Do not** install the upstream `examples/requirements.txt`: it pins a different, historical PyCOLMAP implementation and CUDA-compiled extensions. Our patch uses modern PyCOLMAP's `Reconstruction` reader and substitutes pure-PyTorch SSIM for `fused_ssim`.

The trainer may still report a missing example-only dependency or version-specific API issue; please capture the traceback. This route is designed to eliminate the issues we've already encountered, but actual full training is **not yet confirmed**.

### 4. Run on a dataset

Place 20–50 overlapping, sharp JPEGs of a **stationary, textured** scene in `data/test01/images/`. Move the camera around the subject instead of rotating in place.

```powershell
python run.py --images data/test01/images --output output/test01 --colmap-only
```

If PyCOLMAP reports **no reconstruction**, this means it has not estimated usable poses. Capture the matching/mapping log and inspect photo overlap; gsplat cannot train from an empty sparse model. Use a new output folder for each attempt:

```powershell
python run.py --images data/test01/images --output output/test02 --colmap-only
```

Once poses exist, train using the same output:

```powershell
python run.py --output output/test01 --skip-colmap --steps 3000 --factor 4
```

Or run both stages with fresh output:

```powershell
python run.py --images data/test01/images --output output/test03 --steps 3000 --factor 4
```

Look for a `.ply` under `output/test01/training/`; open in [SuperSplat](https://superspl.at/editor/).

**GTX 1060 6GB:** start with `--factor 4` or `--factor 8` for reduced GPU memory use. Training will be slower than on newer GPUs.

### Troubleshooting

- `No module named imageio` or `nerfview`: make sure `python -m pip install -r requirements-trainer.txt` ran in the active venv.
- `SceneManager` import error: rerun `python scripts/patch_gsplat_v153.py`; verify the gsplat source tag is exactly `v1.5.3`.
- CUDA compiler or `cl.exe` requested: **stop**; something is building from source. Do not install editable gsplat or upstream native extensions.
- No COLMAP reconstruction: inspect matched features and the number of registered images. Without sparse poses, the training stage cannot proceed.
- Existing database: choose a fresh `--output` directory.
- PowerShell path: run all `python run.py` commands from project root.
- If wheels cannot be found: verify 64-bit Python **3.10**, `win_amd64`, and index URL before changing versions.

### Design

The project keeps the training implementation upstream (`third_party/gsplat` is ignored by Git), the **compatibility patch in `scripts/`**, and orchestration in `run.py`. The Raspberry Pi can eventually capture/transfer photographs to an NVIDIA PC for training, but cannot run this CUDA wheel. Fixed multi-camera calibration and remote capture are future projects.
