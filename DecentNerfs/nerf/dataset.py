# dataset.py
import os
import numpy as np
from PIL import Image
import torch

from nerf_core import get_rays

def load_intrinsics(scene_dir, H, W):
    """
    Try to load intrinsics.npy if it exists.
    Otherwise, approximate fx, fy from image size.
    """
    intr_path = os.path.join(scene_dir, "intrinsics.npy")
    if os.path.exists(intr_path):
        K = np.load(intr_path)  # 3x3
        return K

    # Fallback: simple pinhole approximation
    fx = fy = 0.5 * max(H, W)
    cx = W / 2.0
    cy = H / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    return K


def load_scene(scene_dir: str,
               device="cpu",
               downscale: int = 2):
    """
    Load images and poses for a Phototourism-style scene.

    Returns a dict with:
      rays_o: (N_rays, 3)
      rays_d: (N_rays, 3)
      rgb:    (N_rays, 3)
      H, W, K
    """
    img_dir = os.path.join(scene_dir, "images")
    pose_path = os.path.join(scene_dir, "poses.npy")

    assert os.path.exists(img_dir), f"Missing images dir: {img_dir}"
    assert os.path.exists(pose_path), f"Missing poses.npy: {pose_path}"

    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    poses = np.load(pose_path)
    N = poses.shape[0]

    if N != len(image_files):
        print(f"[load_scene] WARNING: {len(image_files)} images found, but only {N} poses available.")
        print("[load_scene] Using only the first N images matched to COLMAP poses.")
        image_files = image_files[:N]



    all_rays_o = []
    all_rays_d = []
    all_rgbs = []

    H0, W0 = None, None
    for idx, fname in enumerate(image_files):
        path = os.path.join(img_dir, fname)
        img = Image.open(path).convert("RGB")
        W, H = img.size

        if downscale > 1:
            img = img.resize((W // downscale, H // downscale), Image.LANCZOS)
            W, H = img.size

        if H0 is None:
            H0, W0 = H, W

        img_t = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)
        img_t = img_t.reshape(-1, 3)  # (H*W, 3)

        c2w = poses[idx]  # (4,4)
        K = load_intrinsics(scene_dir, H, W)

        rays_o, rays_d = get_rays(H, W, K, c2w, device=device)  # each (H*W, 3)

        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_rgbs.append(img_t.to(device))

    rays_o = torch.cat(all_rays_o, dim=0)  # (N_imgs*H*W, 3)
    rays_d = torch.cat(all_rays_d, dim=0)  # (N_imgs*H*W, 3)
    rgb = torch.cat(all_rgbs, dim=0)       # (N_imgs*H*W, 3)

    print(f"[load_scene] Loaded {len(image_files)} images from {scene_dir}")
    print(f"[load_scene] Rays: {rays_o.shape}, RGB: {rgb.shape}, H={H0}, W={W0}")

    return {
        "rays_o": rays_o,
        "rays_d": rays_d,
        "rgb": rgb,
        "H": H0,
        "W": W0,
        "K": K
    }


class NeRFRaysDataset(torch.utils.data.Dataset):
    """
    Simple dataset wrapper over rays and RGB values.
    """

    def __init__(self, rays_o, rays_d, rgb):
        super().__init__()
        assert rays_o.shape == rays_d.shape == rgb.shape
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.rgb = rgb

    def __len__(self):
        return self.rays_o.shape[0]

    def __getitem__(self, idx):
        return self.rays_o[idx], self.rays_d[idx], self.rgb[idx]
