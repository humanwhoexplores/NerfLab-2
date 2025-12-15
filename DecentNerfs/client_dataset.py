"""
client_dataset.py

Loads images + synthetic poses + intrinsics, and provides
ray sampling utilities for training a NeRF/DecentNeRF model.

This file defines:
- ClientDataset: dataset for ONE client
- FullSceneDataset: dataset wrapper that lets you split images into clients
"""

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch

from synthetic_poses import (
    generate_orbit_poses,
    generate_intrinsics,
    poses_to_torch,
    intrinsics_to_torch,
)


# -----------------------------
# Utility functions
# -----------------------------

def load_image(path: str, downscale: int = 4) -> torch.Tensor:
    """
    Loads an RGB image and downsamples it by given factor.
    Returns: torch tensor of shape (H, W, 3), float32 in [0,1]
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    w2 = max(1, w // downscale)
    h2 = max(1, h // downscale)
    img = img.resize((w2, h2), Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)  # (H, W, 3)


# -----------------------------
# Per-client dataset
# -----------------------------

@dataclass
class ClientDatasetConfig:
    downscale: int = 4
    near: float = 2.0
    far: float = 6.0


class ClientDataset:
    """
    A dataset for ONE client.

    Contains:
      - images: List[(H,W,3) torch)]
      - intrinsics: torch tensor (N, 4)
      - poses: torch tensor (N, 3,4)
      - image_indices: list of image indices owned by this client

    Provides API:
       sample_rays(batch_size) → (rays_o, rays_d, target_rgb, img_idx)
    """

    def __init__(
        self,
        scene_root: str,
        image_list: List[str],
        downscale: int,
        poses_np: np.ndarray,
        intrinsics_np: np.ndarray,
        global_image_start_index: int,
        cfg: ClientDatasetConfig = ClientDatasetConfig(),
    ):
        """
        Args:
          scene_root: e.g., "data/trevi"
          image_list: list of image filenames belonging to this client
          downscale: int
          poses_np: (M,3,4) poses for ALL images in same sorted order as list.txt
          intrinsics_np: (M,4) intrinsics for ALL images
          global_image_start_index: offset for global image numbering
        """
        self.scene_root = scene_root
        self.image_filenames = image_list
        self.downscale = downscale
        self.cfg = cfg 

        self.num_images = len(image_list)
        self.global_start = global_image_start_index  # needed for per-image embeddings

        # Load images
        self.images: List[torch.Tensor] = []
        for fname in image_list:
            full_path = os.path.join(scene_root, "images", fname)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Missing image: {full_path}")
            self.images.append(load_image(full_path, downscale))

        # Extract matching poses + intrinsics
        # They are aligned by index in full scene
        all_indices = list(range(global_image_start_index,
                                 global_image_start_index + self.num_images))
        self.poses = poses_to_torch(poses_np[all_indices])             # (N,3,4)
        self.intrinsics = intrinsics_to_torch(intrinsics_np[all_indices])  # (N,4)

    def sample_rays(self, batch_size: int, device: torch.device):
        """
        Sample random rays from this client's image subset.

        Returns:
          rays_o: (B,3)
          rays_d: (B,3)
          target_rgb: (B,3)
          img_idx_global: (B,)   # global image index (needed for embeddings)
        """
        B = batch_size

        rays_o = torch.empty(B, 3, device=device)
        rays_d = torch.empty(B, 3, device=device)
        target_rgb = torch.empty(B, 3, device=device)
        img_idx_global = torch.empty(B, dtype=torch.long, device=device)

        for i in range(B):
            # pick random image from this client
            local_img_idx = random.randint(0, self.num_images - 1)
            global_idx = self.global_start + local_img_idx

            img = self.images[local_img_idx]
            intr = self.intrinsics[local_img_idx]
            pose = self.poses[local_img_idx]

            H, W, _ = img.shape

            # sample random pixel
            y = random.randint(0, H - 1)
            x = random.randint(0, W - 1)

            target_rgb[i] = img[y, x].to(device)
            img_idx_global[i] = global_idx

            # intrinsics
            fx, fy, cx, cy = intr.tolist()

            # pixel → camera coords
            x_cam = (x - cx) / fx
            y_cam = -(y - cy) / fy  # flip camera y-axis
            z_cam = -1.0

            dir_cam = torch.tensor([x_cam, y_cam, z_cam], dtype=torch.float32, device=device)
            dir_cam = dir_cam / (torch.norm(dir_cam) + 1e-9)

            # world coords
            R = pose[:, :3].to(device)   # (3,3)
            t = pose[:, 3].to(device)    # (3,)

            dir_world = (R @ dir_cam).to(device)
            dir_world = dir_world / (torch.norm(dir_world) + 1e-9)

            rays_o[i] = t
            rays_d[i] = dir_world

        return rays_o, rays_d, target_rgb, img_idx_global


# -----------------------------
# Full dataset → split into clients
# -----------------------------

class FullSceneDataset:
    """
    Loads entire dataset (images, intrinsics, poses), then splits into per-client subsets.

    Expected structure:
       scene_root/
         list.txt
         images/*.jpg

    You provide:
      num_clients → we partition images evenly (sequential or round-robin)
    """

    def __init__(
        self,
        scene_root: str,
        downscale: int = 4,
        num_clients: int = 5,
        max_images: Optional[int] = None,
    ):
        """
        Load images + synthetic poses for full scene.

        Args:
           scene_root: e.g., "data/trevi"
           downscale: image downscaling
           num_clients: number of federated clients
           max_images: optional limit on total images
        """
        self.scene_root = scene_root
        self.downscale = downscale
        self.num_clients = num_clients

        list_file = os.path.join(scene_root, "list.txt")
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Missing {list_file}")

        with open(list_file, "r") as f:
            names_all = []
            for ln in f.readlines():
                ln = ln.strip()
                if not ln:
                    continue
                # FIX: remove leading "images/" if present
                if ln.startswith("images/"):
                    ln = ln[len("images/"):]
                names_all.append(ln)


        if max_images is not None:
            names_all = names_all[:max_images]

        self.all_images = names_all
        self.num_images_total = len(names_all)

        # Load one image to get resolution
        test_img_path = os.path.join(scene_root, "images", self.all_images[0])
        test_img = Image.open(test_img_path)
        W, H = test_img.size
        W2 = max(1, W // downscale)
        H2 = max(1, H // downscale)

        # Generate synthetic intrinsics + poses for all images
        intr_np = generate_intrinsics(self.num_images_total, W2, H2)
        poses_np = generate_orbit_poses(self.num_images_total)

        self.intrinsics_np = intr_np
        self.poses_np = poses_np

        # Partition images into clients (sequential)
        self.client_image_lists = []
        chunk = self.num_images_total // num_clients
        for c in range(num_clients):
            start = c * chunk
            end = (c + 1) * chunk if c < num_clients - 1 else self.num_images_total
            self.client_image_lists.append(self.all_images[start:end])

    def make_client(self, client_id: int) -> ClientDataset:
        """
        Return dataset for ONE client.
        """
        img_list = self.client_image_lists[client_id]
        global_start_index = sum(len(self.client_image_lists[i]) for i in range(client_id))

        return ClientDataset(
            scene_root=self.scene_root,
            image_list=img_list,
            downscale=self.downscale,
            poses_np=self.poses_np,
            intrinsics_np=self.intrinsics_np,
            global_image_start_index=global_start_index,
        )
