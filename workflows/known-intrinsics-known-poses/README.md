# Known intrinsics and known relative camera poses (two-view stereo)

**Use case:** Two images with calibrated lens parameters, and known rotation + translation between the two camera positions. A single camera moved between shots is fine for a static scene.

This folder **has a runnable OpenCV stereo point-cloud experiment**. It does **not yet** export COLMAP's `cameras.bin`, `images.bin`, and `points3D.bin` or optimise a Gaussian PLY. Connecting the results to gsplat is the next implementation step.

## Prepare

1. Calibrate the camera with `scripts/calibrate_camera.py`; generate `calibration/olympus_intrinsics.json`.
2. Capture `left.jpg` and `right.jpg` of a **stationary, textured scene** with considerable overlap, preferably translating the camera sideways.
3. Copy `pose.example.json` to `pose.json` and edit the measured relative pose.
4. In your existing Python venv, install `opencv-contrib-python` and `numpy` if not installed.

## Pose convention — extremely important

We use **camera 1 optical coordinates**: +X to the camera's right, +Y down in its image, +Z forward into the scene. `camera2_center_in_camera1_m` is the physical location of camera 2's **optical centre**, in **metres**, expressed in camera 1's coordinates. It is **not** the extrinsic translation vector used by OpenCV; the script calculates that.

`camera1_to_camera2_rotation_xyz_deg` means camera 2's **orientation relative to camera 1**, encoded as successive Euler rotations **Rz @ Ry @ Rx** around X,Y,Z, in degrees. At `[0,0,0]` the cameras look exactly parallel; a movement of camera 2 10 cm to the **right** is `[0.10, 0, 0]`.

**A ruler alone cannot accurately measure rotations and optical centres.** Start with cameras held parallel on a sliding rail or tripod rig. For arbitrary handheld tilts, estimate poses from the fixed ChArUco board with `solvePnP` (not implemented in this folder yet) instead of guessing Euler angles. Errors in pose directly degrade depth.

If using **two different cameras**, add a second calibration JSON and supply it using `--calibration-right`. The example assumes both images use the same calibrated Olympus.

## Run

From the repository root:

```powershell
python workflows/known-intrinsics-known-poses/stereo.py --left data/stereo/left.jpg --right data/stereo/right.jpg --calibration calibration/olympus_intrinsics.json --pose workflows/known-intrinsics-known-poses/pose.json --output output/stereo
```

Optional for different camera intrinsics: `--calibration-right calibration/other_camera.json`.

Outputs: `rectified_left.png`, `rectified_right.png`, `disparity.png` (visualisation), `points.ply` (ordinary **coloured point cloud**, **not** a Gaussian PLY). Inspect the PLY with a point-cloud viewer such as CloudCompare, MeshLab or Open3D. It is not ready for SuperSplat as a Gaussian scene.

## Limitations and next step

StereoSGBM assumes the image pair shares visual texture and its disparities fall inside the search range. Very different viewpoints, occlusions, reflections and calibration inaccuracies cause holes or incorrect depths. The default disparity search assumes that the rectified images have positive disparity; if they do not, swap the image order or fix your pose convention. Inspect rectified pictures and disparity **before** trusting any points. `--min-disparity` and `--num-disparities` allow adjusting the search window (num must be divisible by 16).

Once a reasonable point cloud exists, the next software change is writing registered cameras, their image observations/tracks and triangulated/stereo points into a PyCOLMAP `Reconstruction`, then adapting the gsplat trainer's two-image train/eval split and optional depth regularisation. This is **not** implemented here, and we should not describe the output as train-ready until tested.
