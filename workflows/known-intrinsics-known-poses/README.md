# Two-photo stereo → experimental Gaussian splat

**Status:** Experimental and **not yet tested end-to-end on the Olympus photographs**. The stereo, COLMAP exporter and two-view trainer patch are present on `main`. This PR adds a one-command runner. Camera 1 is the world origin; camera 2 is assumed **10 cm to its right with exactly parallel orientation**. If your actual capture deviates from this, geometry may be incorrect.

## Capture and environment

Use the calibrated Olympus with the same lens/focal length/focus/resolution as the saved intrinsics. Photograph a **stationary, textured scene** twice, moving the *optical centre* 10 cm right without rotation. Keep most of the scene in both frames. Camera optical centre is not necessarily the centre of the tripod screw!

Open PowerShell at the repository root after merging this PR; use your working gsplat v1.5.3 / PyCOLMAP 4.2.1 `.venv`:

```powershell
git switch main
git pull origin main
.\.venv\Scripts\Activate.ps1
python -c "import cv2, gsplat, pycolmap, torch; print(cv2.__version__, gsplat.__version__, pycolmap.__version__, torch.cuda.is_available())"
```

Put photos in `data/stereo/left.jpg` and `data/stereo/right.jpg`; calibration JSON in `calibration/olympus_intrinsics.json`. The existing [pose.example.json](pose.example.json) encodes the **10 cm / zero-rotation assumption**:

```powershell
Copy-Item workflows/known-intrinsics-known-poses/pose.example.json workflows/known-intrinsics-known-poses/pose.json
```

## Quick start: one command

Place your two photographs at `data/stereo/left.jpg` (origin) and `data/stereo/right.jpg` (camera moved exactly 10 cm right, no rotation). The runner uses the bundled `pose.example.json` directly, so no pose-file setup is necessary.

Run this first to produce stereo geometry and a PyCOLMAP-validated dataset **without running CUDA training**:

```powershell
python workflows/known-intrinsics-known-poses/run_binocular.py --left data/stereo/left.jpg --right data/stereo/right.jpg --calibration calibration/olympus_intrinsics.json --output output/binocular_test1 --geometry-only
```

Inspect `output/binocular_test1/stereo/disparity.png`, `rectified_left.png`, `rectified_right.png` and `points.ply`.

Then use a **new output path** to attempt the full Gaussian reconstruction:

```powershell
python workflows/known-intrinsics-known-poses/run_binocular.py --left data/stereo/left.jpg --right data/stereo/right.jpg --calibration calibration/olympus_intrinsics.json --output output/binocular_test2 --steps 1000
```

The runner verifies that the inputs exist, runs stereo, exports the COLMAP model, patches the gsplat v1.5.3 example, trains with both images and lets `run.py` locate the Gaussian PLY. Keep output paths unique: the runner intentionally refuses to reuse an existing sparse-model folder.

**Important:** This path bypasses COLMAP's initial-pair search, but stereo still needs textured surfaces, accurate intrinsics and accurate camera translation/orientation. A successful point cloud does not guarantee gsplat optimisation will run or produce a good scene. Read the stages below to debug each part.

## Stage 1: OpenCV stereo geometry

```powershell
python workflows/known-intrinsics-known-poses/stereo.py --left data/stereo/left.jpg --right data/stereo/right.jpg --calibration calibration/olympus_intrinsics.json --pose workflows/known-intrinsics-known-poses/pose.json --output output/stereo --max-width 1200
```

Inspect `output/stereo/rectified_left.png`, `rectified_right.png`, `disparity.png`, and `points.ply`. The last is a **plain RGB point cloud**, not a Gaussian scene. The script now also writes `left.png`, `right.png`, `stereo_data.npz` and `stereo_meta.json` for the next step.

If your scene is farther away or disparity is out of range, you can adjust `--min-disparity` and `--num-disparities` (multiple of 16). Valid stereo points need accurate intrinsics **and** relative pose; poor capture can't be repaired by changing steps.

## Stage 2: Export a COLMAP-style training dataset

```powershell
python workflows/known-intrinsics-known-poses/export_colmap.py --stereo output/stereo --output output/two_view/dataset --max-points 12000
```

The exporter creates `dataset/images/{left,right}.png`, `dataset/sparse/0/{cameras,images,points3D}.txt` with 2 known camera poses and sampled triangulated points with two-view observation tracks. It reopens the sparse model with PyCOLMAP and checks that two cameras are registered and points exist. This **does not run COLMAP feature matching or incremental mapping**.

## Stage 3: Train gsplat and export a Gaussian PLY

Our existing gsplat v1.5.3 source checkout must be prepared as in the main README (precompiled gsplat wheel, **do not run `pip install -e .`**). The known PyCOLMAP loader compatibility patch and *new* two-image split fix must both be applied:

```powershell
python scripts/patch_gsplat_v153.py
python workflows/known-intrinsics-known-poses/patch_two_view.py
python run.py --skip-colmap --output output/two_view --steps 1000 --factor 1
```

The split patch ensures that **both** photographs are train views (default gsplat split otherwise leaves one out). `--factor 1` uses the already-downsampled rectified images. The PLY should appear somewhere under `output/two_view/training/` if the trainer runs and exports correctly. Look at the printed errors if it does not.

This is exploratory sparse-view 3DGS: two photographs cannot observe the backs of objects. Expect holes and floaters. Also, the automatic stage has not been tested with your real images on the GTX 1060; the first run may reveal further trainer incompatibilities or GPU memory problems.

## Geometry conventions

Camera 1 is at `[0,0,0]`. Camera 2 optical centre is `[0.10, 0, 0]` metres in camera 1 coordinates (+X right, +Y down, +Z forward); camera 2 rotation is zero. **OpenCV stereo outputs points in rectified-left camera coordinates.** We write rectified images and *matching* rectified camera poses into COLMAP, so points and views stay in the same reference frame. Depth is metric only to the extent that the 10 cm baseline is accurate.

The exporter caps the point count to control memory on a 6 GB GTX 1060. Adjust `--max-points`, `--max-width`, and disparity settings if depth looks wrong before attempting training.
