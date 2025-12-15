import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio.v2 as imageio


# -----------------------------
# Utils
# -----------------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # For Apple Silicon you can uncomment:
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Positional Encoding
# -----------------------------

class PositionalEncoding(nn.Module):
    """
    NeRF-style positional encoding per dimension.
    For each scalar x:
      [x, sin(2^0 x), cos(2^0 x), ..., sin(2^{L-1} x), cos(2^{L-1} x)]
    """

    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        )

    @property
    def out_dim(self) -> int:
        return 1 + 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C)
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


# -----------------------------
# 3D NeRF-W-style network
# -----------------------------

class NeRFW3D(nn.Module):
    """
    Minimal NeRF-W-ish network:

    Inputs:
      - 3D point x (world)
      - view direction d (unit)
      - per-image appearance embedding

    Outputs:
      - rgb in [0,1]^3
      - density sigma >= 0
    """

    def __init__(
        self,
        num_images: int,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 128,
        depth: int = 8,
        skips=(4,),
        app_dim: int = 32,
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(pos_freqs)
        self.dir_enc = PositionalEncoding(dir_freqs)
        self.app_embedding = nn.Embedding(num_images, app_dim)

        self.skips = set(skips)

        pts_in_dim = 3 * self.pos_enc.out_dim
        dir_in_dim = 3 * self.dir_enc.out_dim

        # MLP for density + features
        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(pts_in_dim, hidden_dim))
        for i in range(1, depth):
            if i in self.skips:
                self.pts_linears.append(nn.Linear(hidden_dim + pts_in_dim, hidden_dim))
            else:
                self.pts_linears.append(nn.Linear(hidden_dim, hidden_dim))

        self.sigma_linear = nn.Linear(hidden_dim, 1)
        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)

        # Color head: features + encoded dir + appearance
        self.rgb_linear_1 = nn.Linear(hidden_dim + dir_in_dim + app_dim, hidden_dim // 2)
        self.rgb_linear_2 = nn.Linear(hidden_dim // 2, 3)

    def forward(self, x: torch.Tensor, d: torch.Tensor, img_idx: torch.Tensor):
        """
        x:      (N, 3) world coords
        d:      (N, 3) view dirs (unit)
        img_idx:(N,)   image index
        """
        x_enc = self.pos_enc(x)
        d_enc = self.dir_enc(d)
        h = x_enc

        for i, layer in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([h, x_enc], dim=-1)
            h = F.relu(layer(h), inplace=True)

        sigma = F.relu(self.sigma_linear(h))  # (N,1)
        feat = self.feature_linear(h)         # (N,C)

        app = self.app_embedding(img_idx)     # (N, app_dim)

        h_color = torch.cat([feat, d_enc, app], dim=-1)
        h_color = F.relu(self.rgb_linear_1(h_color), inplace=True)
        rgb = torch.sigmoid(self.rgb_linear_2(h_color))

        return rgb, sigma


# -----------------------------
# Synthetic scene with orbit poses
# -----------------------------

@dataclass
class Scene3D:
    images: List[torch.Tensor]        # list of (H_i, W_i, 3)
    intrinsics: torch.Tensor          # (N,4) [fx,fy,cx,cy]
    poses: torch.Tensor               # (N,3,4) camera-to-world
    num_images: int
    H0: int
    W0: int


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build camera-to-world matrix (3x4) like OpenGL look-at.
    """
    forward = center - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)

    true_up = np.cross(right, forward)

    R = np.stack([right, true_up, forward], axis=1)  # (3,3)
    t = eye.reshape(3, 1)                            # (3,1)

    c2w = np.concatenate([R, t], axis=1)             # (3,4)
    return c2w.astype(np.float32)


def load_scene_with_synthetic_poses(
    scene_root: str,
    downscale: int = 8,
    max_images: int = 168,
) -> Scene3D:
    """
    Load images from:
      scene_root/
        list.txt
        images/*.jpg

    Ignore poses.npy; build synthetic orbit poses instead.
    """
    list_path = os.path.join(scene_root, "list.txt")
    img_root = os.path.join(scene_root, "images")

    if not os.path.exists(list_path):
        raise FileNotFoundError(f"list.txt not found at {list_path}")

    with open(list_path, "r") as f:
        names = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(names) == 0:
        raise RuntimeError("list.txt is empty")

    # Use exactly max_images entries (or fewer if list is shorter)
    names = names[:max_images]
    print(f"Using {len(names)} images from list.txt")

    images = []
    intrinsics_list = []

    for nm in names:
        if nm.startswith("images/"):
            img_path = os.path.join(scene_root, nm)
        else:
            img_path = os.path.join(img_root, nm)

        if not os.path.exists(img_path):
            print(f"⚠️ Skipping missing image: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        new_w = max(1, w // downscale)
        new_h = max(1, h // downscale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img_t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)  # (H,W,3)

        # Approx intrinsics
        focal = 0.9 * new_w
        fx = focal
        fy = focal
        cx = new_w / 2.0
        cy = new_h / 2.0

        images.append(img_t)
        intrinsics_list.append([fx, fy, cx, cy])

    if len(images) == 0:
        raise RuntimeError(f"No valid images loaded from {scene_root}")

    num_images = len(images)
    print(f"✅ Loaded {num_images} images total")

    # Synthetic orbit poses
    poses = []
    radius = 4.0
    height = 0.5
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    look_at_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for i in range(num_images):
        theta = 2.0 * math.pi * i / num_images
        eye = np.array(
            [radius * math.cos(theta), height, radius * math.sin(theta)],
            dtype=np.float32,
        )
        c2w = look_at(eye, look_at_center, up)
        poses.append(c2w)

    poses = torch.from_numpy(np.stack(poses, axis=0))          # (N,3,4)
    intrinsics = torch.tensor(intrinsics_list, dtype=torch.float32)  # (N,4)

    H0, W0, _ = images[0].shape

    return Scene3D(
        images=images,
        intrinsics=intrinsics,
        poses=poses,
        num_images=num_images,
        H0=H0,
        W0=W0,
    )


# -----------------------------
# Ray sampling & volume rendering
# -----------------------------

def sample_random_rays(scene: Scene3D, batch_size: int, device: torch.device):
    """
    Random rays from random images.
    Returns:
      rays_o: (B,3)
      rays_d: (B,3)
      target_rgb: (B,3)
      img_ids: (B,)
    """
    B = batch_size
    N = scene.num_images

    rays_o = torch.empty(B, 3, device=device)
    rays_d = torch.empty(B, 3, device=device)
    target_rgb = torch.empty(B, 3, device=device)
    img_ids = torch.empty(B, dtype=torch.long, device=device)

    poses = scene.poses.to(device)
    intr = scene.intrinsics.to(device)

    for i in range(B):
        img_idx = random.randint(0, N - 1)
        img = scene.images[img_idx]  # (H,W,3)
        H, W, _ = img.shape

        y = random.randint(0, H - 1)
        x = random.randint(0, W - 1)

        target_rgb[i] = img[y, x].to(device)
        img_ids[i] = img_idx

        fx, fy, cx, cy = intr[img_idx]

        # pixel -> camera
        # note: y axis downward → flip sign in camera coords
        x_cam = (x - cx) / fx
        y_cam = -(y - cy) / fy
        z_cam = -1.0

        dir_cam = torch.tensor([x_cam, y_cam, z_cam], dtype=torch.float32, device=device)
        dir_cam = dir_cam / (torch.norm(dir_cam) + 1e-9)

        pose = poses[img_idx]  # (3,4)
        R = pose[:, :3]
        t = pose[:, 3]

        dir_world = R @ dir_cam
        dir_world = dir_world / (torch.norm(dir_world) + 1e-9)

        rays_o[i] = t
        rays_d[i] = dir_world

    return rays_o, rays_d, target_rgb, img_ids


def volume_render_rays(
    model: NeRFW3D,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    img_ids: torch.Tensor,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    device: torch.device = None,
):
    """
    Standard NeRF volume rendering along rays.
    """
    if device is None:
        device = rays_o.device

    B = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals            # (S,)
    z_vals = z_vals.unsqueeze(0).repeat(B, 1)                # (B,S)

    # 3D points
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # (B,S,3)
    dirs = rays_d.unsqueeze(1).expand_as(pts)                               # (B,S,3)
    img_ids_exp = img_ids.unsqueeze(1).expand(B, n_samples)                 # (B,S)

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    ids_flat = img_ids_exp.reshape(-1)

    rgb_flat, sigma_flat = model(pts_flat, dirs_flat, ids_flat)  # (B*S,3), (B*S,1)
    rgb = rgb_flat.view(B, n_samples, 3)
    sigma = sigma_flat.view(B, n_samples)

    # distances between samples
    deltas = z_vals[:, 1:] - z_vals[:, :-1]               # (B,S-1)
    delta_last = 1e10 * torch.ones_like(deltas[:, :1])    # (B,1)
    deltas = torch.cat([deltas, delta_last], dim=-1)      # (B,S)

    alpha = 1.0 - torch.exp(-sigma * deltas)              # (B,S)
   # transmittance
    T = torch.cumprod(
        torch.cat([torch.ones(B, 1, device=device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]                                             # (B,S)

    weights = alpha * T                                   # (B,S)
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)     # (B,3)
    depth_map = torch.sum(weights * z_vals, dim=1)              # (B,)

    return rgb_map, depth_map


# -----------------------------
# Training
# -----------------------------

def train_nerfw_3d(
    scene_root: str,
    scene_name: str,
    iters: int = 2000,
    batch_size: int = 1024,
    lr: float = 5e-4,
    downscale: int = 8,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    max_images: int = 168,
):
    device = get_device()
    print(f"Using device: {device}")
    seed_all(123)

    scene = load_scene_with_synthetic_poses(
        scene_root=scene_root,
        downscale=downscale,
        max_images=max_images,
    )

    model = NeRFW3D(num_images=scene.num_images).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for it in range(1, iters + 1):
        rays_o, rays_d, target_rgb, img_ids = sample_random_rays(
            scene, batch_size=batch_size, device=device
        )

        pred_rgb, _ = volume_render_rays(
            model,
            rays_o,
            rays_d,
            img_ids,
            n_samples=n_samples,
            near=near,
            far=far,
            device=device,
        )

        loss = F.mse_loss(pred_rgb, target_rgb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if it == 1 or it % 50 == 0:
            psnr = -10.0 * math.log10(loss.item() + 1e-8)
            print(f"[Iter {it:05d}/{iters}] loss={loss.item():.6f}  PSNR={psnr:.2f} dB")

    out_dir = "outputs"
    ensure_dir(out_dir)
    np.save(
        os.path.join(out_dir, f"{scene_name}_3d_synth_losses.npy"),
        np.array(losses, dtype=np.float32),
    )
    print(f"✅ Saved training losses to outputs/{scene_name}_3d_synth_losses.npy")

    return model, scene


# -----------------------------
# Rendering a single image & orbit video
# -----------------------------

@torch.no_grad()
def render_image(
    model: NeRFW3D,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    H: int,
    W: int,
    img_index_for_code: int,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    device: torch.device = None,
    chunk_rays: int = 8192,
):
    if device is None:
        device = next(model.parameters()).device

    fx, fy, cx, cy = intrinsics
    fx = float(fx)
    fy = float(fy)
    cx = float(cx)
    cy = float(cy)

    ys = torch.linspace(0, H - 1, H, device=device)
    xs = torch.linspace(0, W - 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    xs_flat = xx.reshape(-1)
    ys_flat = yy.reshape(-1)
    N_rays = xs_flat.shape[0]

    # pixel -> camera
    x_cam = (xs_flat - cx) / fx
    y_cam = -(ys_flat - cy) / fy
    z_cam = -torch.ones_like(x_cam)

    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)      # (N,3)
    dirs_cam = dirs_cam / (torch.norm(dirs_cam, dim=-1, keepdim=True) + 1e-9)

    R = c2w[:, :3].to(device)  # (3,3)
    t = c2w[:, 3].to(device)   # (3,)

    dirs_world = (R @ dirs_cam.T).T
    dirs_world = dirs_world / (torch.norm(dirs_world, dim=-1, keepdim=True) + 1e-9)

    rays_o = t.expand_as(dirs_world)
    rays_d = dirs_world

    img_ids = torch.full(
        (N_rays,),
        img_index_for_code,
        dtype=torch.long,
        device=device,
    )

    rgb_all = []
    for i in range(0, N_rays, chunk_rays):
        ro = rays_o[i:i+chunk_rays]
        rd = rays_d[i:i+chunk_rays]
        ids = img_ids[i:i+chunk_rays]
        rgb_chunk, _ = volume_render_rays(
            model,
            ro,
            rd,
            ids,
            n_samples=n_samples,
            near=near,
            far=far,
            device=device,
        )
        rgb_all.append(rgb_chunk.cpu())

    rgb = torch.cat(rgb_all, dim=0)
    rgb = rgb.view(H, W, 3).clamp(0.0, 1.0).numpy()
    return (rgb * 255.0).astype(np.uint8)


@torch.no_grad()
def render_orbit_video(
    model: NeRFW3D,
    scene: Scene3D,
    scene_name: str,
    n_frames: int = 60,
    render_downscale: int = 4,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
):
    device = next(model.parameters()).device
    out_dir = "outputs"
    ensure_dir(out_dir)

    H0, W0 = scene.H0, scene.W0
    H = max(32, H0 // render_downscale)
    W = max(32, W0 // render_downscale)

    print(f"🎥 Rendering orbit video at {W}x{H}, {n_frames} frames...")

    # Use intrinsics from first image
    intr0 = scene.intrinsics[0].to(device)

    radius = 4.0
    height = 0.5
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    look_at_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    frames = []
    for f in range(n_frames):
        theta = 2.0 * math.pi * f / n_frames
        eye = np.array(
            [radius * math.cos(theta), height, radius * math.sin(theta)],
            dtype=np.float32,
        )
        c2w_np = look_at(eye, look_at_center, up)  # (3,4)
        c2w = torch.from_numpy(c2w_np).to(device)

        img = render_image(
            model,
            c2w,
            intr0,
            H,
            W,
            img_index_for_code=0,  # canonical appearance
            n_samples=n_samples,
            near=near,
            far=far,
            device=device,
        )
        frames.append(img)
        print(f"  Frame {f+1}/{n_frames} done")

    gif_path = os.path.join(out_dir, f"{scene_name}_3d_synth_orbit.gif")
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"✅ Saved orbit GIF to {gif_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3D NeRF-W-style demo with synthetic poses (ignores corrupted poses.npy)."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="trevi",
        help="Scene name under ./data (e.g. 'trevi', 'notredame')",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=2000,
        help="Training iterations",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1024,
        help="Rays per training iteration",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=8,
        help="Downscale factor for training images",
    )
    parser.add_argument(
        "--render_frames",
        type=int,
        default=60,
        help="Number of frames in rendered orbit video",
    )
    parser.add_argument(
        "--render_downscale",
        type=int,
        default=8,
        help="Downscale factor for rendering the orbit video",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=168,
        help="Maximum number of images to use from list.txt",
    )

    args = parser.parse_args()

    scene_root = os.path.join("data", args.scene)
    if not os.path.exists(scene_root):
        raise FileNotFoundError(f"Scene directory not found: {scene_root}")

    model, scene = train_nerfw_3d(
        scene_root=scene_root,
        scene_name=args.scene,
        iters=args.iters,
        batch_size=args.batch,
        lr=5e-4,
        downscale=args.downscale,
        n_samples=64,
        near=2.0,
        far=6.0,
        max_images=args.max_images,
    )

    render_orbit_video(
        model,
        scene,
        scene_name=args.scene,
        n_frames=args.render_frames,
        render_downscale=args.render_downscale,
        n_samples=64,
        near=2.0,
        far=6.0,
    )


if __name__ == "__main__":
    main()
