"""
renderer.py

Core NeRF volume rendering logic for DecentNeRF prototype.

Consumes:
  - rays_o, rays_d, image indices
  - DecentNeRFField (global + personal)

Produces:
  - rgb_map:  (B,3)
  - depth_map:(B,)
  - optional extras (weights, z_vals)

Used for:
  - training (per-batch forward)
  - rendering images / videos
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from nerf_backbone import DecentNeRFField


# -----------------------------
# Rendering config
# -----------------------------

@dataclass
class RenderConfig:
    n_samples: int = 64     # samples per ray
    near: float = 2.0       # near plane
    far: float = 6.0        # far plane
    white_bkgd: bool = False  # optional white background


# -----------------------------
# Ray sampling
# -----------------------------

def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    n_samples: int,
    near: float,
    far: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample 3D points along rays between near and far.

    Args:
      rays_o: (B,3) ray origins
      rays_d: (B,3) ray directions (unit)
      n_samples: number of samples per ray
      near, far: scalar floats

    Returns:
      pts:   (B, S, 3) sampled points
      z_vals:(B, S)   depth along ray
    """
    device = rays_o.device
    B = rays_o.shape[0]

    # linearly interpolate between near, far
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # (S,)
    z_vals = z_vals.unsqueeze(0).expand(B, n_samples)  # (B,S)

    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # (B,S,3)
    return pts, z_vals


# -----------------------------
# Volume rendering (batch)
# -----------------------------

def volume_render_rays(
    field: DecentNeRFField,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    img_idx: torch.Tensor,
    cfg: Optional[RenderConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Core NeRF volume rendering along rays.

    Args:
      field:   DecentNeRFField module
      rays_o:  (B,3)
      rays_d:  (B,3)
      img_idx: (B,) global image indices (for personal head)
      cfg:     RenderConfig

    Returns:
      rgb_map:   (B,3)
      depth_map: (B,)
      weights:   (B,S)   per-sample weights
      z_vals:    (B,S)   sampled depths
    """
    if cfg is None:
        cfg = RenderConfig()

    device = rays_o.device
    B = rays_o.shape[0]
    n_samples = cfg.n_samples

    # Sample points
    pts, z_vals = sample_along_rays(
        rays_o=rays_o,
        rays_d=rays_d,
        n_samples=n_samples,
        near=cfg.near,
        far=cfg.far,
    )  # (B,S,3), (B,S)

    # Expand view dirs and image indices
    dirs = rays_d.unsqueeze(1).expand_as(pts)          # (B,S,3)
    img_idx_expanded = img_idx.unsqueeze(1).expand(B, n_samples)  # (B,S)

    # Flatten for MLP
    pts_flat = pts.reshape(-1, 3)              # (B*S,3)
    dirs_flat = dirs.reshape(-1, 3)           # (B*S,3)
    idx_flat = img_idx_expanded.reshape(-1)   # (B*S,)

    # Run field
    rgb_flat, sigma_flat = field(pts_flat, dirs_flat, idx_flat)  # (B*S,3), (B*S,1)
    rgb = rgb_flat.view(B, n_samples, 3)
    sigma = sigma_flat.view(B, n_samples)  # (B,S)

    # Compute deltas between samples
    deltas = z_vals[:, 1:] - z_vals[:, :-1]           # (B,S-1)
    delta_last = 1e10 * torch.ones_like(deltas[:, :1]) # (B,1)
    deltas = torch.cat([deltas, delta_last], dim=-1)  # (B,S)

    # Convert densities to alpha
    # alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1.0 - torch.exp(-sigma * deltas)          # (B,S)

    # Compute transmittance T
    # T_i = Π_{j<i} (1 - alpha_j)
    # Use cumprod with shifted concat trick
    T = torch.cumprod(
        torch.cat(
            [torch.ones(B, 1, device=device), 1.0 - alpha + 1e-10],
            dim=-1
        ),
        dim=-1
    )[:, :-1]                                         # (B,S)

    weights = alpha * T                               # (B,S)

    # Color integration
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # (B,3)

    # Optional background color: white
    if cfg.white_bkgd:
        acc_map = torch.sum(weights, dim=-1, keepdim=True)   # (B,1)
        rgb_map = rgb_map + (1.0 - acc_map) * 1.0            # white background

    # Depth map
    depth_map = torch.sum(weights * z_vals, dim=-1)          # (B,)

    return rgb_map, depth_map, weights, z_vals


# -----------------------------
# Rendering loss helper
# -----------------------------

def nerf_loss(rgb_pred: torch.Tensor, rgb_target: torch.Tensor) -> torch.Tensor:
    """
    Simple MSE loss between predicted and ground truth RGB.
    """
    return F.mse_loss(rgb_pred, rgb_target)


def psnr_from_mse(mse: float) -> float:
    """
    Compute PSNR (dB) from scalar MSE.
    """
    if mse <= 0.0:
        return 100.0
    import math
    return -10.0 * math.log10(mse + 1e-8)
