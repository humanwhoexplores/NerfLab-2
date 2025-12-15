"""
render_video.py

Render a full NeRF orbit video using the trained DecentNeRFField.

Produces:
  outputs/nerf_video.mp4

Uses:
  - synthetic orbit poses (so camera circles the scene)
  - render_image() from Section 5.1 (copied here for clarity)
"""

import os
import torch
import numpy as np
from PIL import Image
import imageio

from nerf_backbone import DecentNeRFField
from client_dataset import FullSceneDataset
from synthetic_poses import (
    generate_orbit_poses,
    poses_to_torch,
)
from renderer import volume_render_rays, RenderConfig


# ----------------------------
# Render params
# ----------------------------

SCENE_ROOT = "data/trevi"
MODEL_PATH = "local_debug_model.pth"

H_RENDER = 200     # image height
W_RENDER = 200     # image width
SAMPLES = 64       # samples per ray

NUM_FRAMES = 120   # video length
OUT_VIDEO = "outputs/nerf_video.mp4"


# ----------------------------
# Reuse render_image from debug script
# ----------------------------

def render_image(field, c2w, intr, device, H, W, n_samples):
    fx, fy, cx, cy = intr.tolist()

    i_coords = torch.arange(W, device=device)
    j_coords = torch.arange(H, device=device)
    jj, ii = torch.meshgrid(j_coords, i_coords, indexing="ij")

    x_cam = (ii - cx) / fx
    y_cam = -(jj - cy) / fy
    z_cam = -torch.ones_like(x_cam)

    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    dirs_cam = dirs_cam / (torch.norm(dirs_cam, dim=-1, keepdim=True) + 1e-9)

    R = c2w[:, :3].to(device)
    t = c2w[:, 3].to(device)

    dirs_world = (dirs_cam @ R.T).reshape(-1, 3)
    dirs_world = dirs_world / (torch.norm(dirs_world, dim=-1, keepdim=True) + 1e-9)

    rays_o = t.expand_as(dirs_world)
    rays_d = dirs_world

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

    full_data = FullSceneDataset(
        scene_root=SCENE_ROOT,
        downscale=8,
        num_clients=5,
        max_images=None
    )

    num_images_total = full_data.num_images_total
    intr_np = full_data.intrinsics_np

    # choose intrinsics for rendering
    intr = torch.tensor(intr_np[0], dtype=torch.float32, device=device)

    # generate new orbit for video
    print("Generating orbit camera path...")
    orbit_poses = generate_orbit_poses(NUM_FRAMES)

    # Load model
    print("Loading trained model...")
    field = DecentNeRFField(num_images=num_images_total).to(device)
    field.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    field.eval()

    os.makedirs("outputs", exist_ok=True)
    frames = []

    for i in range(NUM_FRAMES):
        print(f"Rendering frame {i+1}/{NUM_FRAMES}")

        pose = poses_to_torch(orbit_poses[i]).to(device)

        rgb = render_image(
            field, pose, intr, device,
            H_RENDER, W_RENDER, SAMPLES
        )
        frames.append(rgb)

    print("Saving MP4...")
    imageio.mimwrite(OUT_VIDEO, frames, fps=30)
    print("Saved video →", OUT_VIDEO)


if __name__ == "__main__":
    main()
