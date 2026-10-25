#!/usr/bin/env python3
"""Printable A4 ChArUco target and camera intrinsics calibration.

Requires opencv-contrib-python (or an OpenCV build with cv2.aruco), numpy, Pillow.
The SAME board configuration is used for printing and for image detection.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SQUARES_X = 5
SQUARES_Y = 7
SQUARE_MM = 30.0
MARKER_MM = 22.0
DPI = 300
DICT_NAME = "DICT_4X4_50"


def board_spec():
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        raise RuntimeError("OpenCV ArUco/ChArUco unavailable. Install opencv-contrib-python.")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_MM / 1000, MARKER_MM / 1000, dictionary
    )
    return board


def make_board(output):
    from PIL import Image

    board = board_spec()
    mm_to_px = lambda mm: round(mm * DPI / 25.4)
    page_w, page_h = mm_to_px(210), mm_to_px(297)
    board_w, board_h = mm_to_px(SQUARES_X * SQUARE_MM), mm_to_px(SQUARES_Y * SQUARE_MM)
    pattern = board.generateImage((board_w, board_h), marginSize=0)
    page = Image.new("RGB", (page_w, page_h), "white")
    left, top = (page_w - board_w) // 2, (page_h - board_h) // 2
    page.paste(Image.fromarray(pattern).convert("RGB"), (left, top))
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, resolution=DPI)
    print(f"Saved A4 print target: {output}")
    print(f"Board: {SQUARES_X} x {SQUARES_Y} squares, {SQUARE_MM:g} mm each; "
          f"pattern dimensions {SQUARES_X*SQUARE_MM:g} x {SQUARES_Y*SQUARE_MM:g} mm")
    print("Print at ACTUAL SIZE / 100% on A4. Disable 'fit to page'.")
    print("Measure several printed 30 mm squares to verify scale. Mount perfectly flat.")


def calibrate(images, output, min_corners, min_views):
    board = board_spec()
    detector = cv2.aruco.CharucoDetector(board)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    paths = sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not paths:
        raise ValueError(f"No calibration photos in {images}")
    object_points, image_points, accepted = [], [], []
    image_size = None
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"SKIP unreadable: {path.name}")
            continue
        height, width = frame.shape[:2]
        if image_size is None:
            image_size = (width, height)
        elif image_size != (width, height):
            raise ValueError(f"Mixed image resolutions: {path.name} is {width}x{height}; expected {image_size}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        n = 0 if ids is None else len(ids)
        if n < min_corners:
            print(f"SKIP {path.name}: {n} corners (need {min_corners})")
            continue
        obj, img = board.matchImagePoints(corners, ids)
        object_points.append(np.asarray(obj, dtype=np.float32).reshape(-1, 3))
        image_points.append(np.asarray(img, dtype=np.float32).reshape(-1, 2))
        accepted.append(path.name)
        print(f"OK   {path.name}: {n} corners")
    if len(accepted) < min_views:
        raise RuntimeError(f"Only {len(accepted)} usable photos; need {min_views}. "
                           "Take more photos with different tilts and positions.")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    view_errors = []
    for obj, img, rv, tv in zip(object_points, image_points, rvecs, tvecs):
        reprojected, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        residual = reprojected.reshape(-1, 2) - img
        view_errors.append(float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))))
    result = {
        "camera_model": "OPENCV",
        "image_width": image_size[0], "image_height": image_size[1],
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.flatten().tolist(),
        "rms_reprojection_error_px": float(rms),
        "mean_view_reprojection_error_px": float(np.mean(view_errors)),
        "board": {
            "dictionary": DICT_NAME, "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
            "square_mm": SQUARE_MM, "marker_mm": MARKER_MM
        },
        "used_images": accepted,
        "per_view_reprojection_error_px": dict(zip(accepted, view_errors)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved calibration: {output}")
    print(f"Used {len(accepted)}/{len(paths)} photos; RMS reprojection error: {rms:.3f} px")
    print("Inspect the errors and image diversity; a low RMS alone doesn't guarantee a good calibration.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("board", help="Create a print-ready A4 ChArUco PDF")
    gen.add_argument("--output", type=Path, default=Path("calibration/charuco_a4.pdf"))
    cal = sub.add_parser("calibrate", help="Estimate camera intrinsics from photos of that board")
    cal.add_argument("--images", type=Path, required=True)
    cal.add_argument("--output", type=Path, default=Path("calibration/olympus_intrinsics.json"))
    cal.add_argument("--min-corners", type=int, default=8)
    cal.add_argument("--min-views", type=int, default=12)
    args = parser.parse_args()
    if args.command == "board":
        make_board(args.output)
    else:
        calibrate(args.images, args.output, args.min_corners, args.min_views)


if __name__ == "__main__":
    main()
