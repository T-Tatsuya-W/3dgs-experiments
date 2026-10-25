# Capture / reconstruction experiments

Two distinct workflows can share the intrinsics file from `scripts/calibrate_camera.py`:

- [**Known intrinsics; unknown poses**](known-intrinsics-unknown-poses/README.md): handheld photographs; need a calibrated PyCOLMAP pose-estimation mode (**planned**, not wired into `run.py` yet).
- [**Known intrinsics; known poses**](known-intrinsics-known-poses/README.md): two views with relative translation and orientation supplied in JSON; **runnable stereo-depth / coloured point-cloud proof of concept**, with gsplat dataset conversion **still to implement**.

Keep camera calibration, stereo reconstruction, and Gaussian optimisation as separate checkpoints: a point-cloud PLY is **not** a Gaussian PLY.
