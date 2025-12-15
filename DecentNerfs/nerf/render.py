# render.py
import os
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio

from nerf_core import get_rays, render_rays


def create_orbit_poses(poses: np.ndarray, radius_scale: float = 1.0, n_frames: int = 60):
    """
    Create a simple orbit trajectory around the scene center.
    We approximate orbit by rotating around mean camera pose.
    """
    # Simple heuristic: take average pose as center
    center = poses[:, :3, 3].mean(axis=0)  # (3,)

    # Use one reference pose orientation
    ref = poses[0].copy()
    # move ref to center
    ref[:3, 3] = center

    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    render_poses = []

    for theta in angles:
        rot_y = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0,             1, 0,             0],
            [-np.sin(theta),0, np.cos(theta), 0],
            [0,             0, 0,             1]
        ], dtype=np.float32)

        pose = rot_y @ ref
        # push camera back a bit along -Z
        pose[2, 3] -= radius_scale
        render_poses.append(pose)

    return np.stack(render_poses, axis=0)  # (n_frames, 4, 4)


@torch.no_grad()
def render_image(model, H, W, K, c2w, device="cpu",
                 num_samples: int = 32, near: float = 2.0, far: float = 6.0):
    """
    Render one image from a given c2w pose.
    """
    rays_o, rays_d = get_rays(H, W, K, c2w, device=device)
    rgb = render_rays(model, rays_o, rays_d,
                      num_samples=num_samples, near=near, far=far, chunk=1024)
    img = rgb.reshape(H, W, 3).clamp(0.0, 1.0).cpu().numpy()
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def render_orbit_video(model,
                       data_dict,
                       out_video_path="outputs/single/single_nerf.mp4",
                       device="cpu",
                       num_samples: int = 32,
                       near: float = 2.0,
                       far: float = 6.0,
                       n_frames: int = 60):
    """
    Renders a simple orbit video using camera poses derived from training poses.
    """
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

    H = data_dict["H"]
    W = data_dict["W"]
    K = data_dict["K"]
    # We assume original training poses are in poses.npy
    scene_dir = data_dict.get("scene_dir", None)
    if scene_dir is not None:
        poses = np.load(os.path.join(scene_dir, "poses.npy"))
    else:
        # fallback: approximate from rays origins if we don't have scene_dir
        print("[render_orbit_video] Warning: scene_dir not set in data_dict, "
              "using dummy center from rays.")
        # not ideal but okay as a fallback
        rays_o = data_dict["rays_o"].cpu().numpy()
        dummy_c2w = np.eye(4, dtype=np.float32)
        dummy_c2w[:3, 3] = rays_o.mean(axis=0)
        poses = np.stack([dummy_c2w], axis=0)

    render_poses = create_orbit_poses(poses, radius_scale=1.0, n_frames=n_frames)

    frames = []
    model = model.to(device)
    model.eval()

    for i, c2w in enumerate(render_poses):
        print(f"[render_orbit_video] Rendering frame {i+1}/{n_frames}")
        img = render_image(model, H, W, K, c2w, device=device,
                           num_samples=num_samples, near=near, far=far)
        frames.append(img)

    imageio.mimsave(out_video_path, frames, fps=24)
    print(f"[render_orbit_video] Saved video to {out_video_path}")
