# nerf_core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------
# Positional Encoding
# -------------------------

def positional_encoding(x: torch.Tensor, num_freqs: int = 6) -> torch.Tensor:
    """
    x: (..., D)
    returns: (..., D * (1 + 2 * num_freqs))
    """
    # x shape (..., D)
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device)  # (num_freqs,)
    freqs = freqs.view(*([1] * (x.dim() - 1)), -1)  # (..., 1, num_freqs) broadcast
    x_expanded = x.unsqueeze(-1) * freqs  # (..., D, num_freqs)

    sin = torch.sin(x_expanded)
    cos = torch.cos(x_expanded)

    # concat sin, cos along last dim then flatten frequency dim
    enc = torch.cat([sin, cos], dim=-1)  # (..., D, 2*num_freqs)
    enc = enc.view(*x.shape[:-1], -1)    # (..., D * 2*num_freqs)

    return torch.cat([x, enc], dim=-1)   # (..., D * (1 + 2*num_freqs))


# -------------------------
# Tiny NeRF MLP (no view dirs)
# -------------------------

class TinyNeRF(nn.Module):
    def __init__(self, num_freqs: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.num_freqs = num_freqs
        in_dim = 3 * (1 + 2 * num_freqs)  # xyz only

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_sigma = nn.Linear(hidden_dim, 1)
        self.fc_rgb = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor):
        """
        x: (..., 3) world coordinates
        """
        x_enc = positional_encoding(x, self.num_freqs)
        h = F.relu(self.fc1(x_enc))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        sigma = F.softplus(self.fc_sigma(h))  # (..., 1)
        rgb = torch.sigmoid(self.fc_rgb(h))   # (..., 3)
        return rgb, sigma


# -------------------------
# Ray Generation
# -------------------------

def get_rays(H: int, W: int, K: np.ndarray, c2w: np.ndarray, device="cpu"):
    """
    H, W: image height, width
    K: 3x3 intrinsics (fx, fy, cx, cy)
    c2w: 4x4 camera-to-world matrix
    returns:
      rays_o: (H*W, 3)
      rays_d: (H*W, 3)
    """
    # pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy"
    )
    # shape (H, W)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Convert to camera coordinates
    dirs = torch.stack([
        (i - cx) / fx,
        (j - cy) / fy,
        torch.ones_like(i)
    ], dim=-1)  # (H, W, 3)

    # Camera-to-world
    c2w_torch = torch.from_numpy(c2w).float()  # (4,4)
    R = c2w_torch[:3, :3]
    t = c2w_torch[:3, 3]

    # Rotate ray directions
    rays_d = (dirs @ R.T)  # (H, W, 3)
    # Origin is same for all rays of this camera
    rays_o = t.expand_as(rays_d)  # (H, W, 3)

    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)

    return rays_o, rays_d


# -------------------------
# Sample points along ray
# -------------------------

def sample_points(rays_o: torch.Tensor,
                  rays_d: torch.Tensor,
                  num_samples: int = 32,
                  near: float = 2.0,
                  far: float = 6.0,
                  perturb: bool = True):
    """
    Uniform sampling between near and far.

    rays_o, rays_d: (N_rays, 3)
    returns:
      pts: (N_rays, num_samples, 3)
      t_vals: (N_rays, num_samples)
    """
    N_rays = rays_o.shape[0]
    t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)
    t_vals = t_vals.expand(N_rays, num_samples)  # (N_rays, num_samples)

    if perturb:
        mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
        upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
        lower = torch.cat([t_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand(t_vals.shape, device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand

    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(-1)  # (N_rays, num_samples, 3)
    return pts, t_vals


# -------------------------
# Volume Rendering
# -------------------------

def volume_render(rgb: torch.Tensor,
                  sigma: torch.Tensor,
                  t_vals: torch.Tensor):
    """
    rgb:   (N_rays, N_samples, 3)
    sigma: (N_rays, N_samples, 1)
    t_vals: (N_rays, N_samples)
    returns:
      final_rgb: (N_rays, 3)
    """
    deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (N_rays, N_samples-1)
    # Add a dummy delta for the last sample
    delta_last = 1e-3 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_last], dim=-1)  # (N_rays, N_samples)

    # densities to alpha
    sigma = sigma.squeeze(-1)  # (N_rays, N_samples)
    alpha = 1.0 - torch.exp(-sigma * deltas)  # (N_rays, N_samples)

    # transmittance
    accum = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]  # (N_rays, N_samples)

    weights = alpha * accum  # (N_rays, N_samples)

    final_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=1)  # (N_rays, 3)
    return final_rgb, weights


# -------------------------
# Helper: Render rays in chunks
# -------------------------

@torch.no_grad()
def render_rays(model: TinyNeRF,
                rays_o: torch.Tensor,
                rays_d: torch.Tensor,
                num_samples: int = 32,
                near: float = 2.0,
                far: float = 6.0,
                chunk: int = 1024):
    """
    Render many rays in smaller chunks to avoid OOM.
    Returns: (N_rays, 3)
    """
    model.eval()
    N_rays = rays_o.shape[0]
    all_rgb = []

    for i in range(0, N_rays, chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        pts, t_vals = sample_points(ro, rd, num_samples, near, far, perturb=False)
        # pts: (chunk, N_samples, 3)
        rgb, sigma = model(pts)  # both (chunk, N_samples, 3 or 1)
        rgb_final, _ = volume_render(rgb, sigma, t_vals)
        all_rgb.append(rgb_final.cpu())

    return torch.cat(all_rgb, dim=0)  # (N_rays, 3)
