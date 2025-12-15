"""
synthetic_poses.py

Generates synthetic camera intrinsics + camera poses for DecentNeRF.
Used when COLMAP poses are not available or corrupted.

Provides:
- generate_orbit_poses()     → poses on a circular orbit
- generate_intrinsics()      → simple pinhole intrinsics
- look_at()                  → build camera-to-world matrix
- normalize()                → safe vector normalization
- poses_to_torch(), intrinsics_to_torch()
"""

from dataclasses import dataclass
import numpy as np
import torch


# -----------------------------
# Helper: normalize a vector
# -----------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector safely."""
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


# -----------------------------
# Look-at camera construction
# -----------------------------

def look_at(
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray = np.array([0.0, 1.0, 0.0])
) -> np.ndarray:
    """
    Construct a camera-to-world (c2w) matrix using a 'look-at' formulation.
    
    Args:
      eye:    (3,) camera location
      center: (3,) what the camera looks at
      up:     (3,) world up vector
    
    Returns:
      c2w: (3,4) camera-to-world matrix
    """
    forward = normalize(center - eye)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))

    R = np.stack([right, true_up, forward], axis=1)  # (3,3)
    t = eye.reshape(3, 1)
    c2w = np.concatenate([R, t], axis=1).astype(np.float32)
    return c2w


# -----------------------------
# Synthetic intrinsics
# -----------------------------

@dataclass
class SyntheticIntrinsicsConfig:
    focal_ratio: float = 0.85   # fx = fy = focal_ratio * width
    center_ratio: float = 0.5   # cx = W * center_ratio, cy = H * center_ratio


def generate_intrinsics(
    num_images: int,
    width: int,
    height: int,
    cfg: SyntheticIntrinsicsConfig = SyntheticIntrinsicsConfig()
) -> np.ndarray:
    """
    Generate simple synthetic intrinsics for each image.
    
    Returns:
      intrinsics: (N,4) each row = [fx, fy, cx, cy]
    """
    fx = cfg.focal_ratio * width
    fy = fx
    cx = cfg.center_ratio * width
    cy = cfg.center_ratio * height

    intrinsics = np.tile([fx, fy, cx, cy], (num_images, 1))
    return intrinsics.astype(np.float32)


# -----------------------------
# Synthetic orbit pose generator
# -----------------------------

@dataclass
class OrbitPoseConfig:
    radius: float = 4.0
    height: float = 0.5
    center: np.ndarray = None
    up: np.ndarray = None

    def __post_init__(self):
        if self.center is None:
            self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if self.up is None:
            self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def generate_orbit_poses(
    num_images: int,
    cfg: OrbitPoseConfig = OrbitPoseConfig()
) -> np.ndarray:
    """
    Generate camera poses placed uniformly on a circular orbit.

    Args:
      num_images: number of views
      cfg: orbit parameters (radius, height)

    Returns:
      poses: (N,3,4) camera-to-world matrices
    """
    poses = []

    for i in range(num_images):
        theta = 2.0 * np.pi * i / num_images

        eye = np.array([
            cfg.radius * np.cos(theta),
            cfg.height,
            cfg.radius * np.sin(theta)
        ], dtype=np.float32)

        c2w = look_at(eye, cfg.center, cfg.up)
        poses.append(c2w)

    poses = np.stack(poses, axis=0).astype(np.float32)
    return poses


# -----------------------------
# Torch conversion helpers
# -----------------------------

def poses_to_torch(poses: np.ndarray) -> torch.Tensor:
    """Convert (N,3,4) numpy poses to torch tensor."""
    return torch.from_numpy(poses).float()


def intrinsics_to_torch(intrinsics: np.ndarray) -> torch.Tensor:
    """Convert (N,4) numpy intrinsics to torch tensor."""
    return torch.from_numpy(intrinsics).float()
