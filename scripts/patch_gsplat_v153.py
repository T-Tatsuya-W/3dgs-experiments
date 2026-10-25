#!/usr/bin/env python3
"""Adapt the exact gsplat v1.5.3 example for modern PyCOLMAP and binary-only dependencies.

Run after checking out the v1.5.3 tag. This changes only the ignored
third_party/gsplat checkout, never the pip-installed precompiled gsplat wheel.
"""
from pathlib import Path
import argparse

READER = r'''
        reconstruction = pycolmap.Reconstruction(colmap_dir)
        registered = sorted((im for im in reconstruction.images.values() if im.has_pose), key=lambda im: im.name)
        if not registered:
            raise ValueError("No registered cameras in COLMAP reconstruction")
        camtoworlds_list, camera_ids, image_names = [], [], []
        Ks_dict, params_dict, imsize_dict, mask_dict = {}, {}, {}, {}
        for im in registered:
            cam = reconstruction.cameras[im.camera_id]
            w2c = np.eye(4)
            w2c[:3, :] = im.cam_from_world().matrix()
            camtoworlds_list.append(np.linalg.inv(w2c))
            camera_ids.append(im.camera_id)
            image_names.append(im.name)
            Ks_dict[im.camera_id] = np.asarray(cam.calibration_matrix(), dtype=np.float64) / np.array([[factor, factor, factor], [factor, factor, factor], [1, 1, 1]])
            model = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
            if model in ("SIMPLE_PINHOLE", "PINHOLE"):
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif model == "SIMPLE_RADIAL":
                params = np.array([cam.params[3], 0, 0, 0], dtype=np.float32)
                camtype = "perspective"
            elif model == "RADIAL":
                params = np.array([cam.params[3], cam.params[4], 0, 0], dtype=np.float32)
                camtype = "perspective"
            elif model == "OPENCV":
                params = np.array(cam.params[4:8], dtype=np.float32)
                camtype = "perspective"
            elif model == "OPENCV_FISHEYE":
                params = np.array(cam.params[4:8], dtype=np.float32)
                camtype = "fisheye"
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")
            params_dict[im.camera_id] = params
            imsize_dict[im.camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[im.camera_id] = None
        print(f"[Parser] {len(registered)} images, {len(set(camera_ids))} cameras.")
        camtoworlds = np.stack(camtoworlds_list)
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": False}
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]
        suffix = f"_{factor}" if factor > 1 and not self.extconf["no_factor_suffix"] else ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + suffix)
        if not os.path.exists(colmap_image_dir):
            raise ValueError(f"Image folder {colmap_image_dir} does not exist.")
        if factor > 1 and not os.path.exists(image_dir):
            image_dir = _resize_image_folder(colmap_image_dir, image_dir, factor=factor)
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and image_files and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(colmap_image_dir, image_dir + "_png", factor=factor)
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        point_ids = sorted(reconstruction.points3D)
        point_id_to_idx = {pid: i for i, pid in enumerate(point_ids)}
        points = np.array([reconstruction.points3D[pid].xyz for pid in point_ids], dtype=np.float32).reshape(-1, 3)
        points_err = np.array([reconstruction.points3D[pid].error for pid in point_ids], dtype=np.float32)
        points_rgb = np.array([reconstruction.points3D[pid].color for pid in point_ids], dtype=np.uint8).reshape(-1, 3)
        if not len(points):
            raise ValueError("COLMAP reconstruction has no triangulated 3D points")
        point_indices = {}
        for pid in point_ids:
            for element in reconstruction.points3D[pid].track.elements:
                name = reconstruction.images[element.image_id].name
                point_indices.setdefault(name, []).append(point_id_to_idx[pid])
        point_indices = {name: np.array(indices, dtype=np.int32) for name, indices in point_indices.items()}
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, default=Path("third_party/gsplat/examples"))
    args = parser.parse_args()
    examples = args.examples.resolve()
    dataset = examples / "datasets" / "colmap.py"
    trainer = examples / "simple_trainer.py"
    for path in (dataset, trainer):
        if not path.is_file():
            raise SystemExit(f"Missing gsplat v1.5.3 example: {path}")
    source = dataset.read_text(encoding="utf-8")
    begin = "        manager = SceneManager(colmap_dir)"
    end = "        # Normalize the world space."
    if "reconstruction = pycolmap.Reconstruction(colmap_dir)" not in source:
        if begin not in source or end not in source:
            raise SystemExit("gsplat dataset source has changed; expected v1.5.3, aborting")
        source = source.replace("from pycolmap import SceneManager", "import pycolmap")
        source = source[:source.index(begin)] + READER + "\n" + source[source.index(end):]
        dataset.write_text(source, encoding="utf-8")
    source = trainer.read_text(encoding="utf-8")
    original = "from fused_ssim import fused_ssim"
    replacement = """from pytorch_msssim import ssim as _pure_ssim

def fused_ssim(x, y, padding="valid"):
    # Upstream uses images in [0,1]; pure torch fallback avoids compiling fused_ssim.
    return _pure_ssim(x, y, data_range=1.0, size_average=True)"""
    if original in source:
        trainer.write_text(source.replace(original, replacement), encoding="utf-8")
    elif replacement not in source:
        raise SystemExit("gsplat trainer source has changed; expected v1.5.3")
    import ast
    for path in (dataset, trainer):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print("gsplat v1.5.3 example patched for PyCOLMAP 4.x and pure-PyTorch SSIM.")

if __name__ == "__main__":
    main()
