"""
render_debug_image.py

Render a single debug image from the trained DecentNeRFField model.

Steps:
  1. Load dataset (synthetic poses, intrinsics)
  2. Load trained model checkpoint
  3. Choose a pose (orbit or dataset image pose)
  4. Render an RGB image via volume rendering
  5. Save result to outputs/debug_render.png
"""

import os
import torch
import numpy as np
from PIL import Image

from nerf_backbone import DecentNeRFField
from client_dataset import FullSceneDataset
from synthetic_poses import poses_to_torch
from renderer import volume_render_rays, RenderConfig


# ----------------------------
# Config
# ----------------------------

SCENE_ROOT = "data/trevi"
MODEL_PATH = "local_debug_model.pth"
OUT_PATH = "outputs/debug_render.png"

H_RENDER = 200        # height of rendered image
W_RENDER = 200        # width of rendered image
SAMPLES = 64          # samples per ray


# ----------------------------
# Rendering utility
# ----------------------------

def render_image(field, c2w, intr, device, H, W, n_samples):
    """
    Render a full HxW image given:
      field: DecentNeRFField
      c2w:   camera-to-world (3,4)
      intr:  intrinsics [fx, fy, cx, cy]
    """

    # Build rays
    fx, fy, cx, cy = intr.tolist()

    # Create a grid of pixel coordinates
    i_coords = torch.arange(W, device=device)
    j_coords = torch.arange(H, device=device)
    jj, ii = torch.meshgrid(j_coords, i_coords, indexing="ij")  # (H,W)

    # Pixel → camera coords
    x_cam = (ii - cx) / fx
    y_cam = -(jj - cy) / fy
    z_cam = -torch.ones_like(x_cam)

    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H,W,3)
    dirs_cam = dirs_cam / (torch.norm(dirs_cam, dim=-1, keepdim=True) + 1e-9)

    # Convert to world
    R = c2w[:,:3].to(device)   # (3,3)
    t = c2w[:,3].to(device)    # (3,)

    dirs_world = (dirs_cam @ R.T).reshape(-1, 3)
    dirs_world = dirs_world / (torch.norm(dirs_world, dim=-1, keepdim=True) + 1e-9)

    rays_o = t.expand_as(dirs_world)         # (H*W,3)
    rays_d = dirs_world                      # (H*W,3)

    # Use global image index = 0 (safe for rendering)
    img_idx = torch.zeros(rays_o.shape[0], dtype=torch.long, device=device)

    render_cfg = RenderConfig(
        n_samples=n_samples,
        near=2.0,
        far=6.0,
        white_bkgd=False
    )

    with torch.no_grad():
        rgb, depth, weights, z_vals = volume_render_rays(
            field, rays_o, rays_d, img_idx, cfg=render_cfg
        )

    rgb = rgb.reshape(H, W, 3).cpu().numpy()
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb


# ----------------------------
# Main
# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # 1. Load full dataset (to access poses + intrinsics)
    full_dataset = FullSceneDataset(
        scene_root=SCENE_ROOT,
        downscale=8,
        num_clients=5,
        max_images=None
    )

    num_images = full_dataset.num_images_total
    poses_np = full_dataset.poses_np
    intr_np = full_dataset.intrinsics_np

    # Choose a camera index to render from (e.g. view #10)
    cam_id = 10
    c2w = poses_to_torch(poses_np[cam_id]).to(device)
    intr = torch.tensor(intr_np[cam_id], dtype=torch.float32, device=device)

    # 2. Create model
    field = DecentNeRFField(num_images=num_images).to(device)

    # 3. Load weights
    print("Loading model:", MODEL_PATH)
    field.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # 4. Render
    print("Rendering...")
    rgb = render_image(field, c2w, intr, device,
                       H_RENDER, W_RENDER,
                       n_samples=SAMPLES)

    # 5. Save
    os.makedirs("outputs", exist_ok=True)
    Image.fromarray(rgb).save(OUT_PATH)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
