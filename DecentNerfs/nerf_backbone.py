"""
nerf_backbone.py

Minimal NeRF-W-style backbone for DecentNeRF prototype.

Contains:
- PositionalEncoding
- GlobalNeRF: shared radiance field (no per-image embeddings)
- PersonalHead: per-image / per-client residual field
- alpha_composite: combine global + personal densities/colors
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Positional Encoding
# -----------------------------

class PositionalEncoding(nn.Module):
    """
    Classic NeRF positional encoding.

    For each scalar x:
      [x, sin(2^0 x), cos(2^0 x), ..., sin(2^{L-1} x), cos(2^{L-1} x)]

    We apply this per-dimension and concat.
    """

    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        )

    @property
    def out_dim_per_channel(self) -> int:
        # 1 (x) + 2 * num_freqs
        return 1 + 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., C)
        returns: (..., C * out_dim_per_channel)
        """
        # Ensure float
        x = x.to(self.freq_bands.dtype)

        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


# -----------------------------
# Global NeRF Field (Shared)
# -----------------------------

@dataclass
class GlobalNeRFConfig:
    pos_freqs: int = 10         # positional encoding freq count for xyz
    dir_freqs: int = 4          # positional encoding freq count for view dirs
    hidden_dim: int = 128       # MLP width
    depth: int = 8              # number of layers in position MLP
    skips: Tuple[int, ...] = (4,)  # skip connections (like original NeRF)


class GlobalNeRF(nn.Module):
    """
    Shared NeRF-style field (no per-image embeddings).

    Inputs:
      x:  (N, 3) world coordinates
      d:  (N, 3) view directions (unit)
    Outputs:
      rgb_static: (N, 3)   in [0,1]
      sigma_static: (N, 1) >= 0
      features:    (N, C)  intermediate features for personal head
    """

    def __init__(self, cfg: Optional[GlobalNeRFConfig] = None):
        super().__init__()
        self.cfg = cfg or GlobalNeRFConfig()

        self.pos_enc = PositionalEncoding(self.cfg.pos_freqs)
        self.dir_enc = PositionalEncoding(self.cfg.dir_freqs)

        pts_in_dim = 3 * self.pos_enc.out_dim_per_channel
        dir_in_dim = 3 * self.dir_enc.out_dim_per_channel

        self.skips = set(self.cfg.skips)
        self.D = self.cfg.depth
        self.W = self.cfg.hidden_dim

        # Position MLP (density + features)
        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(pts_in_dim, self.W))

        for i in range(1, self.D):
            if i in self.skips:
                self.pts_linears.append(nn.Linear(self.W + pts_in_dim, self.W))
            else:
                self.pts_linears.append(nn.Linear(self.W, self.W))

        self.sigma_linear = nn.Linear(self.W, 1)
        self.feature_linear = nn.Linear(self.W, self.W)

        # Color head: features + encoded view direction
        self.rgb_linear_1 = nn.Linear(self.W + dir_in_dim, self.W // 2)
        self.rgb_linear_2 = nn.Linear(self.W // 2, 3)

        # Initialize last color layer small for stability
        nn.init.xavier_uniform_(self.rgb_linear_2.weight, gain=0.01)

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        x: (N,3)
        d: (N,3)
        """
        # Positional encodings
        x_enc = self.pos_enc(x)  # (N, 3 * pos_out)
        d_enc = self.dir_enc(d)  # (N, 3 * dir_out)

        h = x_enc
        for i, layer in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([h, x_enc], dim=-1)
            h = F.relu(layer(h), inplace=True)

        sigma = F.relu(self.sigma_linear(h))            # (N,1)
        feat = self.feature_linear(h)                   # (N,W)

        # Color from features + view dir
        h_color = torch.cat([feat, d_enc], dim=-1)
        h_color = F.relu(self.rgb_linear_1(h_color), inplace=True)
        rgb = torch.sigmoid(self.rgb_linear_2(h_color)) # (N,3)

        return rgb, sigma, feat


# -----------------------------
# Personal Head (Local / Client-only)
# -----------------------------

@dataclass
class PersonalHeadConfig:
    app_dim: int = 32      # appearance embedding dim
    transient_dim: int = 16  # transient embedding dim
    hidden_dim: int = 64     # internal MLP size


class PersonalHead(nn.Module):
    """
    Client-local residual field.

    Each image i has:
      - appearance embedding a_i
      - transient embedding t_i   (optional / small)

    We take global features and per-image codes, and output:
      - rgb_personal: (N,3) in [0,1]
      - sigma_personal: (N,1) >= 0

    Final composition is done via alpha_composite().
    """

    def __init__(self, num_images: int, cfg: Optional[PersonalHeadConfig] = None):
        super().__init__()
        self.cfg = cfg or PersonalHeadConfig()

        self.app_embedding = nn.Embedding(num_images, self.cfg.app_dim)
        self.transient_embedding = nn.Embedding(num_images, self.cfg.transient_dim)

        in_dim = self.cfg.app_dim + self.cfg.transient_dim

        self.mlp1 = nn.Linear(in_dim, self.cfg.hidden_dim)
        self.mlp2 = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)

        self.rgb_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.sigma_head = nn.Linear(self.cfg.hidden_dim, 1)

        # Small init for stability
        nn.init.xavier_uniform_(self.rgb_head.weight, gain=0.01)
        nn.init.zeros_(self.rgb_head.bias)
        nn.init.zeros_(self.sigma_head.bias)

    def forward(self, feat: torch.Tensor, img_idx: torch.Tensor):
        """
        feat:    (N, C) features from GlobalNeRF
        img_idx: (N,)   long, image indices local to this client
        """
        a = self.app_embedding(img_idx)       # (N, app_dim)
        t = self.transient_embedding(img_idx) # (N, transient_dim)

        x = torch.cat([a, t], dim=-1)         # (N, app_dim + transient_dim)

        x = F.relu(self.mlp1(x), inplace=True)
        x = F.relu(self.mlp2(x), inplace=True)

        rgb_personal = torch.sigmoid(self.rgb_head(x))          # (N,3)
        sigma_personal = F.relu(self.sigma_head(x))             # (N,1)

        return rgb_personal, sigma_personal


# -----------------------------
# Composition: Global + Personal
# -----------------------------

def alpha_composite(
    rgb_global: torch.Tensor,
    sigma_global: torch.Tensor,
    rgb_personal: torch.Tensor,
    sigma_personal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine global + personal fields into a single color & density.

    Intuition:
      - Treat global field as "base scene" in front.
      - Treat personal field as "overlay" with transmittance of global.

    We approximate:
      T_g = exp(-sigma_global)
      rgb_out = T_g * rgb_personal + (1 - T_g) * rgb_global
      sigma_out = sigma_global + sigma_personal

    Notes:
      - This is a simplification of a full 2-layer volume compositing.
      - Good enough for prototype DecentNeRF.
    """
    # ensure shapes
    assert rgb_global.shape == rgb_personal.shape
    assert sigma_global.shape == sigma_personal.shape

    # transmittance through global
    T_g = torch.exp(-sigma_global)  # (N,1)

    rgb_out = T_g * rgb_personal + (1.0 - T_g) * rgb_global
    sigma_out = sigma_global + sigma_personal

    return rgb_out, sigma_out


# -----------------------------
# Convenience wrapper
# -----------------------------

class DecentNeRFField(nn.Module):
    """
    Convenience module that wraps:
      - GlobalNeRF (shared)
      - PersonalHead (client-local)

    On server:
      - only GlobalNeRF parameters are aggregated.

    On client:
      - both GlobalNeRF + PersonalHead are trained.
      - but only GlobalNeRF is sent back.
    """

    def __init__(self, num_images: int,
                 global_cfg: Optional[GlobalNeRFConfig] = None,
                 personal_cfg: Optional[PersonalHeadConfig] = None):
        super().__init__()
        self.global_field = GlobalNeRF(global_cfg)
        self.personal_head = PersonalHead(num_images, personal_cfg)

    def forward(self, x: torch.Tensor, d: torch.Tensor, img_idx: torch.Tensor):
        """
        Full forward:
          - run shared global field
          - run personal residual
          - alpha-composite outputs
        """
        rgb_g, sigma_g, feat = self.global_field(x, d)
        rgb_p, sigma_p = self.personal_head(feat, img_idx)
        rgb_out, sigma_out = alpha_composite(rgb_g, sigma_g, rgb_p, sigma_p)
        return rgb_out, sigma_out
