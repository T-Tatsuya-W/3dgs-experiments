# Calibrate a camera using a printed ChArUco board

This is **intrinsic calibration** only. It estimates focal lengths, principal point and lens distortion. It does **not** determine the camera positions of later photographs; that requires separate stereo/pose estimation.

Uses OpenCV's ChArUco detector, known board corner positions, and `calibrateCamera`. [OpenCV documentation](https://docs.opencv.org/4.10.0/df/d4a/tutorial_charuco_detection.html).

## 1. Print an A4 board

From the project root, activate your virtual environment and install the calibration tools:

```powershell
python -m pip install "opencv-contrib-python>=4.8,<5" "Pillow>=10,<12"
python scripts/calibrate_camera.py board
```

Open `calibration/charuco_a4.pdf` and print on **A4, actual size/100%**, with fit-to-page/scaling disabled. The image is centred on A4; the ChArUco pattern itself is **150 × 210 mm**, with **5 × 7 squares, 30 mm each**, ArUco markers of 22 mm, and `DICT_4X4_50`.

Measure several 30 mm squares on the printout. If not 30 mm, check printer scaling. Attach paper to a *rigid, flat* board (avoid wrinkles and bends). Do not crop/rescale the pattern afterwards.

## 2. Take calibration photographs

With the Olympus, capture **20–30 sharp photos** of the printed board:
- Keep exactly the **same lens, zoom/focal length, image resolution**, and preferably focus setting that you will later use to reconstruct scenes.
- Fill various parts of the frame; include centre, corners and edges.
- Tilt the board/camera in multiple directions and use varying distances (not 30 copies of the same straight-on shot).
- Avoid glare, defocus and motion blur. Partial board visibility is okay when at least 8 ChArUco corners can be detected.
- Use original full-size JPEGs; don't crop or digitally resize them.

Put these in `data/calibration/images/`. The script doesn't require the photographs in any particular order.

## 3. Calibrate and inspect

```powershell
python scripts/calibrate_camera.py calibrate --images data/calibration/images --output calibration/olympus_intrinsics.json
```

The script prints the detected corner count for each image, rejects insufficient detections, and writes:
- intrinsic matrix **K**, in pixels
- lens distortion coefficients (OpenCV order)
- image dimensions
- overall and per-view **reprojection errors** (pixels)
- board specifications, and accepted image filenames.

Aim for diverse accepted viewpoints and low reprojection error, but **do not judge accuracy solely from RMS**. If errors are high, remove blurry images and take better-diversified shots, then rerun.

`calibration/olympus_intrinsics.json` is an output **for future reconstruction integration**. The current `run.py` does not yet accept a `--calibration` flag; adding it and/or a known-pose stereo mode is a separate step.

### Troubleshooting
- `cv2.aruco` or `CharucoDetector` missing: install `opencv-contrib-python` (not the minimal headless build); mixing multiple OpenCV wheel flavours in one environment can cause conflicts.
- Board not detected: verify the generated board, pattern version, sharpness and physical print.
- Too few usable views: acquire more oblique images with the board in different frame positions.
- Don't change focus or zoom after calibration without checking if the resulting intrinsics remain accurate.
