import os
import math
import argparse
import random

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
        # 1 (x) + 2 * num_freqs
        return 1 + 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C)
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


# -----------------------------
# 2D NeRF-W-style network
# -----------------------------

class NeRFW2D(nn.Module):
    """
    2D NeRF-W-ish network:
    - Inputs: pixel coords (x,y) in [-1,1]^2 + per-image appearance embedding.
    - Outputs: RGB in [0,1]^3.

    There is no real 3D volume here; it's a fancy MLP that learns how
    each image looks, and we use appearance embeddings to morph between views.
    """

    def __init__(
        self,
        num_images: int,
        pos_freqs: int = 10,
        hidden_dim: int = 128,
        depth: int = 5,
        app_dim: int = 32,
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(pos_freqs)
        self.app_embedding = nn.Embedding(num_images, app_dim)

        in_dim = 2 * self.pos_enc.out_dim  # (x,y) each encoded
        self.in_dim = in_dim + app_dim

        layers = []
        layers.append(nn.Linear(self.in_dim, hidden_dim))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        self.out_layer = nn.Linear(hidden_dim, 3)

        # Small init for stability
        nn.init.xavier_uniform_(self.out_layer.weight, gain=0.01)

    def forward(self, coords: torch.Tensor, img_idx: torch.Tensor) -> torch.Tensor:
        """
        coords: (N, 2) in [-1, 1]
        img_idx: (N,) long
        """
        # positional encoding
        pe = self.pos_enc(coords)  # (N, 2 * out_dim_per_dim)
        app = self.app_embedding(img_idx)  # (N, app_dim)

        x = torch.cat([pe, app], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x), inplace=True)
        rgb = torch.sigmoid(self.out_layer(x))
        return rgb

    def forward_with_app_vector(self, coords: torch.Tensor, app_vec: torch.Tensor) -> torch.Tensor:
        """
        coords: (N, 2)
        app_vec: (N, app_dim) - explicit appearance embedding (for interpolation).
        """
        pe = self.pos_enc(coords)
        x = torch.cat([pe, app_vec], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x), inplace=True)
        rgb = torch.sigmoid(self.out_layer(x))
        return rgb


# -----------------------------
# Dataset loader (Trevi / NotreDame format)
# -----------------------------

class ImageDataset2D:
    def __init__(self, scene_root: str, downscale: int = 4):
        """
        Scene structure:
          scene_root/
            list.txt   (one 'images/xxx.jpg' per line)
            images/
              *.jpg
        """
        list_path = os.path.join(scene_root, "list.txt")
        img_root = os.path.join(scene_root, "images")

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"list.txt not found at {list_path}")

        with open(list_path, "r") as f:
            names = [ln.strip() for ln in f.readlines() if ln.strip()]

        self.image_paths = []
        self.images = []

        for name in names:
            if name.startswith("images/"):
                img_path = os.path.join(scene_root, name)
            else:
                img_path = os.path.join(img_root, name)

            if not os.path.exists(img_path):
                print(f"⚠️ Skipping missing image: {img_path}")
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            new_w = max(1, w // downscale)
            new_h = max(1, h // downscale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            img_t = torch.from_numpy(
                np.array(img).astype(np.float32) / 255.0
            )  # (H, W, 3)

            self.image_paths.append(img_path)
            self.images.append(img_t)

        if len(self.images) == 0:
            raise RuntimeError(f"No valid images loaded from {scene_root}")

        self.num_images = len(self.images)
        print(f"✅ Loaded {self.num_images} images from {scene_root}")

        # we use resolution of first image as reference
        self.H, self.W, _ = self.images[0].shape

    def sample_pixels(self, batch_size: int, device: torch.device):
        """
        Sample random pixels across random images.
        Returns:
          coords_norm: (B, 2) in [-1, 1]
          targets:     (B, 3)
          img_indices: (B,)
        """
        B = batch_size
        coords = torch.empty(B, 2, device=device)
        rgbs = torch.empty(B, 3, device=device)
        img_indices = torch.empty(B, dtype=torch.long, device=device)

        for i in range(B):
            img_idx = random.randint(0, self.num_images - 1)
            img = self.images[img_idx]
            H, W, _ = img.shape

            y = random.randint(0, H - 1)
            x = random.randint(0, W - 1)

            rgb = img[y, x]

            # normalize coords to [-1,1]
            x_norm = (x / (W - 1) - 0.5) * 2.0 if W > 1 else 0.0
            y_norm = (y / (H - 1) - 0.5) * 2.0 if H > 1 else 0.0

            coords[i, 0] = x_norm
            coords[i, 1] = y_norm
            rgbs[i] = rgb.to(device)
            img_indices[i] = img_idx

        return coords, rgbs, img_indices


# -----------------------------
# Training
# -----------------------------

def train_nerfw_2d(
    scene_root: str,
    scene_name: str,
    iters: int = 2000,
    batch_size: int = 512,
    lr: float = 5e-4,
    downscale: int = 4,
):
    device = get_device()
    print(f"Using device: {device}")

    seed_all(123)

    dataset = ImageDataset2D(scene_root, downscale=downscale)
    num_images = dataset.num_images

    model = NeRFW2D(num_images=num_images).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for it in range(1, iters + 1):
        coords, rgbs, img_idx = dataset.sample_pixels(batch_size, device)

        pred = model(coords, img_idx)
        loss = F.mse_loss(pred, rgbs)

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
        os.path.join(out_dir, f"{scene_name}_2d_nerfw_losses.npy"),
        np.array(losses, dtype=np.float32),
    )
    print(f"✅ Saved training losses to outputs/{scene_name}_2d_nerfw_losses.npy")

    return model, dataset


# -----------------------------
# Rendering video (morph across appearance embeddings)
# -----------------------------

@torch.no_grad()
def render_video_2d(
    model: NeRFW2D,
    dataset: ImageDataset2D,
    scene_name: str,
    n_frames: int = 60,
    render_downscale: int = 4,
):
    device = next(model.parameters()).device
    out_dir = "outputs"
    ensure_dir(out_dir)

    # render resolution
    H0, W0 = dataset.H, dataset.W
    H = max(32, H0 // render_downscale)
    W = max(32, W0 // render_downscale)

    print(f"🎥 Rendering 2D NeRF-W video at {W}x{H}, {n_frames} frames...")

    # Precompute coordinate grid (normalized)
    ys = torch.linspace(0, H - 1, H, device=device)
    xs = torch.linspace(0, W - 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x_norm = (xx / (W - 1) - 0.5) * 2.0 if W > 1 else torch.zeros_like(xx)
    y_norm = (yy / (H - 1) - 0.5) * 2.0 if H > 1 else torch.zeros_like(yy)

    coords = torch.stack([x_norm, y_norm], dim=-1).reshape(-1, 2)  # (H*W, 2)

    # appearance embedding matrix
    app_weights = model.app_embedding.weight.data  # (N_img, app_dim)
    num_images = app_weights.shape[0]

    frames = []

    for frame in range(n_frames):
        # fractional "image index" in [0, num_images-1]
        t = frame / max(1, (n_frames - 1))
        idx_f = t * (num_images - 1)
        i0 = int(math.floor(idx_f))
        i1 = min(i0 + 1, num_images - 1)
        alpha = idx_f - i0

        app_vec = (1.0 - alpha) * app_weights[i0] + alpha * app_weights[i1]
        app_vec = app_vec.to(device)

        # expand to all pixels
        app_for_all = app_vec.unsqueeze(0).expand(coords.shape[0], -1)  # (H*W, app_dim)

        rgb = model.forward_with_app_vector(coords, app_for_all)  # (H*W,3)
        rgb_img = rgb.reshape(H, W, 3).clamp(0.0, 1.0).cpu().numpy()
        img_uint8 = (rgb_img * 255.0).astype(np.uint8)
        frames.append(img_uint8)

        print(f"  Frame {frame+1}/{n_frames} done")

    gif_path = os.path.join(out_dir, f"{scene_name}_2d_nerfw.gif")
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"✅ Saved morphing GIF to {gif_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2D NeRF-W-style morphing demo on Trevi/NotreDame Phototourism images (no poses required)."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="trevi",
        help="Scene name under ./data (e.g. 'trevi' or 'notredame')",
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
        default=512,
        help="Batch size (rays/pixels per iteration)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=4,
        help="Downscale factor for training images",
    )
    parser.add_argument(
        "--render_frames",
        type=int,
        default=60,
        help="Number of frames in output video",
    )
    parser.add_argument(
        "--render_downscale",
        type=int,
        default=4,
        help="Downscale factor when rendering video",
    )

    args = parser.parse_args()

    scene_root = os.path.join("data", args.scene)
    if not os.path.exists(scene_root):
        raise FileNotFoundError(f"Scene directory not found: {scene_root}")

    model, dataset = train_nerfw_2d(
        scene_root=scene_root,
        scene_name=args.scene,
        iters=args.iters,
        batch_size=args.batch,
        downscale=args.downscale,
    )

    render_video_2d(
        model,
        dataset,
        scene_name=args.scene,
        n_frames=args.render_frames,
        render_downscale=args.render_downscale,
    )


if __name__ == "__main__":
    main()
