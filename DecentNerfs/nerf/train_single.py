# train_single.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from nerf_core import TinyNeRF, sample_points, volume_render
from dataset import load_scene, NeRFRaysDataset
from render import render_orbit_video


def train_single(scene_dir: str,
                 iters: int = 2000,
                 batch_size: int = 1024,
                 num_samples: int = 32,
                 near: float = 2.0,
                 far: float = 6.0,
                 lr: float = 5e-4,
                 device: str = "cpu",
                 save_dir: str = "outputs/single"):

    os.makedirs(save_dir, exist_ok=True)

    # 1. Load data
    data = load_scene(scene_dir, device=device, downscale=4)
    rays_o = data["rays_o"]
    rays_d = data["rays_d"]
    rgb = data["rgb"]
    H, W, K = data["H"], data["W"], data["K"]

    dataset = NeRFRaysDataset(rays_o, rays_d, rgb)
    # We'll just sample randomly using DataLoader with shuffle=True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 2. Model & optimizer
    model = TinyNeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    global_step = 0
    losses = []

    # 3. Training loop
    while global_step < iters:
        for batch in loader:
            if global_step >= iters:
                break
            rays_o_b, rays_d_b, rgb_b = [x.to(device) for x in batch]

            pts, t_vals = sample_points(rays_o_b, rays_d_b,
                                        num_samples=num_samples,
                                        near=near, far=far, perturb=True)
            # pts: (B, N_samples, 3)
            rgb_pred, sigma = model(pts)  # (B, N_samples, 3), (B, N_samples, 1)
            rgb_final, _ = volume_render(rgb_pred, sigma, t_vals)  # (B, 3)

            loss = criterion(rgb_final, rgb_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            global_step += 1

            if global_step % 100 == 0:
                print(f"[train_single] Step {global_step}/{iters}, loss={loss.item():.6f}")

    # 4. Save model checkpoint
    ckpt_path = os.path.join(save_dir, "nerf_checkpoint.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "H": H,
        "W": W,
        "K": K,
    }, ckpt_path)
    print(f"[train_single] Saved checkpoint to {ckpt_path}")

    # 5. Plot loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Single-Scene TinyNeRF Training Loss")
    plt.grid(True)
    loss_plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"[train_single] Saved loss curve to {loss_plot_path}")

    # 6. Render a small orbit video
    video_path = os.path.join(save_dir, "single_nerf.mp4")
    render_orbit_video(model, data, out_video_path=video_path, device=device)
    print(f"[train_single] Saved orbit video to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to scene dir, e.g., ./data/trevi")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    train_single(
        scene_dir=args.scene,
        iters=args.iters,
        batch_size=args.batch,
        num_samples=args.samples,
        near=args.near,
        far=args.far,
        lr=args.lr,
        device=args.device,
        save_dir="outputs/single"
    )
