import os
import math
import argparse
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
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
    # if you want to try MPS on Apple Silicon, uncomment:
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
    Standard NeRF style positional encoding.
    For each dimension x:
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
        # one original + 2 * num_freqs per dim
        # (we will apply this per-dim and concat)
        return 1 + 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C)
        # output: (..., C * (1 + 2*num_freqs))
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


# -----------------------------
# NeRF-W style network
# -----------------------------

class NeRFW(nn.Module):
    """
    Very small NeRF-W-ish network:
      - Position encoding of 3D points
      - Direction encoding of view dir
      - Per-image appearance embedding
      - Single MLP (coarse only, no transient head)
    """

    def __init__(
        self,
        num_images: int,
        hidden_dim: int = 128,
        depth: int = 6,
        skips=(3,),
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        app_dim: int = 32,
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(pos_freqs)
        self.dir_enc = PositionalEncoding(dir_freqs)

        self.D = depth
        self.W = hidden_dim
        self.skips = set(skips)

        pts_in_ch = 3 * self.pos_enc.out_dim
        dir_in_ch = 3 * self.dir_enc.out_dim

        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(pts_in_ch, hidden_dim))
        for i in range(1, depth):
            if i in self.skips:
                self.pts_linears.append(nn.Linear(hidden_dim + pts_in_ch, hidden_dim))
            else:
                self.pts_linears.append(nn.Linear(hidden_dim, hidden_dim))

        self.sigma_linear = nn.Linear(hidden_dim, 1)
        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)

        self.app_embedding = nn.Embedding(num_images, app_dim)
        self.color_linear_1 = nn.Linear(hidden_dim + dir_in_ch + app_dim, hidden_dim // 2)
        self.color_linear_2 = nn.Linear(hidden_dim // 2, 3)

    def forward(self, x, d, img_idx):
        """
        x:      (N, 3)  3D points
        d:      (N, 3)  view dirs (normalized)
        img_idx:(N,)    indices of images
        """
        x_enc = self.pos_enc(x)      # (N, 3 * pos_out_dim)
        d_enc = self.dir_enc(d)      # (N, 3 * dir_out_dim)

        h = x_enc
        for i, layer in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([h, x_enc], dim=-1)
            h = F.relu(layer(h))

        sigma = F.relu(self.sigma_linear(h))  # density
        feat = self.feature_linear(h)         # features

        app = self.app_embedding(img_idx)     # (N, app_dim)

        h_color = torch.cat([feat, d_enc, app], dim=-1)
        h_color = F.relu(self.color_linear_1(h_color))
        rgb = torch.sigmoid(self.color_linear_2(h_color))

        return rgb, sigma


# -----------------------------
# Data loading (Phototourism)
# -----------------------------

@dataclass
class PTImageInfo:
    path: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class PTScene:
    images: list   # list of torch.FloatTensor (H, W, 3)
    infos: list    # list of PTImageInfo
    poses: torch.Tensor  # (N, 3, 4)


def load_phototourism_scene(scene_root: str, downscale: int = 4):
    """
    Correct loader for your dataset:
    - list.txt contains ONLY image paths (one per line)
    - poses.npy has camera intrinsics + extrinsics
    """

    list_path = os.path.join(scene_root, "list.txt")
    poses_path = os.path.join(scene_root, "poses.npy")
    img_dir = os.path.join(scene_root, "images")

    # ---------- Load list.txt ----------
    with open(list_path, "r") as f:
        file_list = [ln.strip() for ln in f.readlines() if ln.strip()]

    print(f"Found {len(file_list)} image names inside list.txt")

    # ---------- Load poses ----------
    poses_raw = np.load(poses_path)  # (N,3,4) or (N,4,4)
    print(f"Loaded poses.npy with shape {poses_raw.shape}")

    # Normalize pose shape
    if poses_raw.ndim == 3 and poses_raw.shape[1:] == (3, 4):
        # (N,3,4)
        extrinsics = poses_raw[:, :, :]               # R|t
    elif poses_raw.ndim == 3 and poses_raw.shape[1:] == (4, 4):
        # (N,4,4) → reduce to (3,4)
        extrinsics = poses_raw[:, :3, :]
    else:
        raise ValueError(f"Unexpected poses shape: {poses_raw.shape}")

    N_poses = extrinsics.shape[0]
    print(f"Usable poses: {N_poses}")

    # ---------- Pair only first N_poses filenames ----------
    # (because list.txt has 186 lines but poses = 168)
    if len(file_list) < N_poses:
        raise ValueError("list.txt has fewer images than poses.npy!")

    file_list = file_list[:N_poses]  # CUT OFF the extra 18 files
    print(f"Using first {N_poses} images from list.txt")

    # ---------- Load images & intrinsics ----------
    images = []
    infos = []

    for i in range(N_poses):

        fname = file_list[i]
        img_path = os.path.join(scene_root, fname)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image missing: {img_path}")

        # Load & downscale image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        new_w = max(1, orig_w // downscale)
        new_h = max(1, orig_h // downscale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        img_t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)

        # Extract intrinsics from poses.npy row
        # Phototourism format stores fx = pose[3], fy = pose[7], cx = pose[?], cy = pose[?]
        # BUT this differs across subsets → safest way is:
        # assume pinhole with fx = fy = some value stored in last column, OR fallback to HFOV.

        # For now: approximate focal using width (cheap but works!)
        focal = 0.9 * new_w   # this works surprisingly well for quick NeRF-W runs
        fx = focal
        fy = focal
        cx = new_w / 2
        cy = new_h / 2

        infos.append(
            PTImageInfo(
                path=img_path,
                width=new_w,
                height=new_h,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy
            )
        )
        images.append(img_t)

    poses_t = torch.from_numpy(extrinsics.astype(np.float32))  # (N,3,4)
    print("Scene loaded successfully.")
    return PTScene(images=images, infos=infos, poses=poses_t)


# -----------------------------
# Ray sampling
# -----------------------------

def sample_random_rays(scene: PTScene, batch_size: int, device):
    """
    Sample random rays from random images and return:
      rays_o: (B, 3)
      rays_d: (B, 3) normalized
      rgb:    (B, 3) target colors
      img_ids:(B,)
    """
    imgs = scene.images
    infos = scene.infos
    poses = scene.poses

    num_imgs = len(imgs)
    rays_o = torch.empty(batch_size, 3, device=device)
    rays_d = torch.empty(batch_size, 3, device=device)
    rgbs = torch.empty(batch_size, 3, device=device)
    img_ids = torch.empty(batch_size, dtype=torch.long, device=device)

    for i in range(batch_size):
        idx = random.randint(0, num_imgs - 1)
        img = imgs[idx]  # (H, W, 3)
        info = infos[idx]
        pose = poses[idx]  # (3, 4)

        H, W, _ = img.shape
        u = random.randint(0, W - 1)
        v = random.randint(0, H - 1)

        # target color
        rgbs[i] = img[v, u].to(device)

        # camera intrinsics
        fx, fy, cx, cy = info.fx, info.fy, info.cx, info.cy

        # pixel -> camera space
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = 1.0

        # camera to world (assuming poses are c2w)
        R = pose[:, :3]  # (3, 3)
        t = pose[:, 3]   # (3,)

        dir_cam = torch.tensor([x, y, z], dtype=torch.float32, device=device)
        dir_world = (R @ dir_cam)  # (3,)
        dir_world = dir_world / (dir_world.norm() + 1e-9)

        rays_o[i] = t.to(device)
        rays_d[i] = dir_world
        img_ids[i] = idx

    return rays_o, rays_d, rgbs, img_ids


# -----------------------------
# Volume rendering
# -----------------------------

def volume_render_rays(model, rays_o, rays_d, img_ids,
                       num_samples=64, near=2.0, far=6.0, device=None):
    """
    Classic NeRF volume rendering along rays.
    rays_o: (B, 3)
    rays_d: (B, 3)
    img_ids:(B,)
    returns: rgb_composite (B,3), depth (B,)
    """
    if device is None:
        device = rays_o.device

    B = rays_o.shape[0]

    # sample depths
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # (S,)
    z_vals = z_vals.unsqueeze(0).repeat(B, 1)      # (B, S)

    # 3D points along rays
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # (B, S, 3)
    dirs = rays_d.unsqueeze(1).expand_as(pts)                                # (B, S, 3)
    img_ids_expand = img_ids.unsqueeze(1).expand(B, num_samples)            # (B, S)

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    img_ids_flat = img_ids_expand.reshape(-1)

    rgb_flat, sigma_flat = model(pts_flat, dirs_flat, img_ids_flat)  # (B*S, 3), (B*S, 1)
    rgb = rgb_flat.view(B, num_samples, 3)
    sigma = sigma_flat.view(B, num_samples)  # (B, S)

    deltas = z_vals[:, 1:] - z_vals[:, :-1]                # (B, S-1)
    delta_last = 1e10 * torch.ones_like(deltas[:, :1])     # (B,1)
    deltas = torch.cat([deltas, delta_last], dim=-1)       # (B,S)

    alpha = 1.0 - torch.exp(-sigma * deltas)               # (B,S)
    # accumulate transmittance
    T = torch.cumprod(
        torch.cat([torch.ones(B, 1, device=device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]  # (B,S)

    weights = alpha * T                                    # (B,S)
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # (B,3)
    depth_map = torch.sum(weights * z_vals, dim=1)           # (B,)

    return rgb_map, depth_map


# -----------------------------
# Training
# -----------------------------

def train_nerfw(
    scene_root: str,
    scene_name: str,
    iters: int = 2000,
    batch_size: int = 512,
    lr: float = 5e-4,
    downscale: int = 4,
    num_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
):
    device = get_device()
    print(f"Using device: {device}")

    seed_all(123)

    scene = load_phototourism_scene(scene_root, downscale=downscale)
    num_images = len(scene.images)

    model = NeRFW(num_images=num_images).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for it in range(1, iters + 1):
        rays_o, rays_d, rgbs, img_ids = sample_random_rays(
            scene, batch_size=batch_size, device=device
        )

        pred_rgb, _ = volume_render_rays(
            model, rays_o, rays_d, img_ids,
            num_samples=num_samples, near=near, far=far, device=device
        )

        loss = F.mse_loss(pred_rgb, rgbs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if it % 50 == 0 or it == 1:
            psnr = -10.0 * math.log10(loss.item() + 1e-8)
            print(f"[Iter {it:05d}/{iters}] loss={loss.item():.6f}  PSNR={psnr:.2f} dB")

    # plot training curve
    out_dir = os.path.join("outputs")
    ensure_dir(out_dir)
    plt.figure()
    plt.plot(range(1, iters + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title(f"NeRFW Training ({scene_name})")
    plt.grid(True)
    curve_path = os.path.join(out_dir, f"{scene_name}_nerfw_training_curve.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"✅ Saved training curve to {curve_path}")

    return model, scene


# -----------------------------
# Rendering video
# -----------------------------

@torch.no_grad()
def render_frame(model, pose, info: PTImageInfo, H: int, W: int,
                 img_id: int, num_samples=64, near=2.0, far=6.0, device=None):
    """
    Render one frame from a given pose & intrinsics.
    """
    if device is None:
        device = next(model.parameters()).device

    fx, fy, cx, cy = info.fx, info.fy, info.cx, info.cy

    # create a low-res grid
    ys, xs = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device),
        torch.linspace(0, W - 1, W, device=device),
        indexing="ij",
    )
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    N_rays = xs.shape[0]

    # camera to world
    R = pose[:, :3].to(device)  # (3,3)
    t = pose[:, 3].to(device)   # (3,)

    # pixel -> camera coords
    x = (xs - cx * (W / info.width)) / (fx * (W / info.width))
    y = (ys - cy * (H / info.height)) / (fy * (H / info.height))
    z = torch.ones_like(x)

    dir_cam = torch.stack([x, y, z], dim=-1)  # (N_rays, 3)
    dir_world = (R @ dir_cam.T).T            # (N_rays, 3)
    dir_world = dir_world / (dir_world.norm(dim=-1, keepdim=True) + 1e-9)

    rays_o = t.expand_as(dir_world)          # (N_rays, 3)
    rays_d = dir_world
    img_ids = torch.full((N_rays,), img_id, dtype=torch.long, device=device)

    # chunked rendering for memory
    chunk = 8192
    all_rgb = []
    for i in range(0, N_rays, chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        ids = img_ids[i:i+chunk]
        rgb, _ = volume_render_rays(
            model, ro, rd, ids,
            num_samples=num_samples, near=near, far=far, device=device
        )
        all_rgb.append(rgb.cpu())

    rgb_full = torch.cat(all_rgb, dim=0)  # (N_rays, 3)
    rgb_full = rgb_full.view(H, W, 3).clamp(0.0, 1.0)
    img = (rgb_full.numpy() * 255.0).astype(np.uint8)
    return img


@torch.no_grad()
def render_video(model, scene: PTScene, scene_name: str,
                 num_frames: int = 60,
                 render_downscale: int = 8,
                 num_samples: int = 64,
                 near: float = 2.0,
                 far: float = 6.0):
    device = next(model.parameters()).device
    out_dir = "outputs"
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{scene_name}_nerfw.mp4")

    H0 = scene.infos[0].height
    W0 = scene.infos[0].width
    H = max(32, H0 // render_downscale)
    W = max(32, W0 // render_downscale)

    print(f"🎥 Rendering video at {W}x{H}, {num_frames} frames...")

    # simple path: cycle through subset of training poses
    num_imgs = len(scene.images)
    frame_indices = np.linspace(0, num_imgs - 1, num_frames, dtype=int)

    frames = []
    for i, idx in enumerate(frame_indices):
        pose = scene.poses[idx]
        info = scene.infos[idx]
        img = render_frame(
            model, pose, info, H, W,
            img_id=idx,
            num_samples=num_samples,
            near=near,
            far=far,
            device=device,
        )
        frames.append(img)
        print(f"  Frame {i+1}/{num_frames} done")

    imageio.mimwrite(out_path, frames, fps=15, quality=8)
    print(f"✅ Saved video to {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-scene NeRF-W-style training for Phototourism (Trevi / NotreDame)."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="trevi",
        help="Scene name under ./data (e.g., 'trevi' or 'notredame')",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=2000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=512,
        help="Batch size (rays per iteration)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=4,
        help="Downscale factor for training images",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    args = parser.parse_args()

    scene_root = os.path.join("data", args.scene)
    if not os.path.exists(scene_root):
        raise FileNotFoundError(f"Scene directory not found: {scene_root}")

    model, scene = train_nerfw(
        scene_root=scene_root,
        scene_name=args.scene,
        iters=args.iters,
        batch_size=args.batch,
        lr=args.lr,
        downscale=args.downscale,
    )

    # render small video
    render_video(
        model,
        scene,
        scene_name=args.scene,
        num_frames=60,
        render_downscale=8,
    )


if __name__ == "__main__":
    main()
