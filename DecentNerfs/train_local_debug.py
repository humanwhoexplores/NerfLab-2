"""
train_local_debug.py

Single-client training to verify the DecentNeRF pipeline works
before adding Flower federation.

Steps:
  1. Load full scene dataset (synthetic poses + intrinsics)
  2. Create a single client dataset
  3. Train DecentNeRFField (global + personal)
  4. Print loss + PSNR
  5. Save model
"""

import os
import torch
import time

from nerf_backbone import DecentNeRFField
from client_dataset import FullSceneDataset
from renderer import volume_render_rays, nerf_loss, psnr_from_mse, RenderConfig


# -----------------------------
# Config
# -----------------------------

SCENE_ROOT = "data/trevi"   # path to your dataset
NUM_CLIENTS = 5             # dataset split
CLIENT_ID = 0               # which client to test
DOWNSCALE = 8
BATCH_SIZE = 1024
ITERS = 1000
LR = 5e-4
SAVE_PATH = "local_debug_model.pth"


# -----------------------------
# Main
# -----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Load full dataset
    print("Loading full dataset...")
    full_dataset = FullSceneDataset(
        scene_root=SCENE_ROOT,
        downscale=DOWNSCALE,
        num_clients=NUM_CLIENTS,
        max_images=None,       # use all images in list.txt
    )

    # 2. Create one client dataset
    print(f"Creating client {CLIENT_ID}")
    client = full_dataset.make_client(CLIENT_ID)

    # 3. Create DecentNeRF model
    num_images_total = full_dataset.num_images_total
    print("Total images:", num_images_total)

    field = DecentNeRFField(num_images=num_images_total).to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=LR)

    render_cfg = RenderConfig(
        n_samples=64,      # 64 is okay for debugging
        near=2.0,
        far=6.0,
        white_bkgd=False,
    )

    print("Starting training...")
    start_time = time.time()

    for it in range(1, ITERS + 1):
        # 4. Sample rays from client
        rays_o, rays_d, target_rgb, img_idx = client.sample_rays(BATCH_SIZE, device)

        # 5. Volume render
        rgb_pred, depth, weights, z_vals = volume_render_rays(
            field, rays_o, rays_d, img_idx, cfg=render_cfg
        )

        # 6. Loss
        loss = nerf_loss(rgb_pred, target_rgb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7. Progress
        if it % 50 == 0:
            psnr = psnr_from_mse(loss.item())
            elapsed = time.time() - start_time
            print(
                f"[Iter {it}/{ITERS}] loss={loss.item():.6f} "
                f"PSNR={psnr:.2f} dB  time={elapsed:.1f}s"
            )

    # Save model
    torch.save(field.state_dict(), SAVE_PATH)
    print(f"Saved model → {SAVE_PATH}")

    print("DONE.")


if __name__ == "__main__":
    main()
