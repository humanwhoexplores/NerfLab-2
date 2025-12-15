"""
federated_client.py

Federated client for DecentNeRF (Flower-compatible, but works standalone too).

Responsibilities:
  - Load global model weights from server
  - Train locally on client's dataset
  - Update ONLY global model parameters (personal head stays private)
  - Return updated global parameters + metrics (loss, PSNR)
"""

import torch
import numpy as np

from nerf_backbone import DecentNeRFField
from client_dataset import ClientDataset
from renderer import volume_render_rays, nerf_loss, psnr_from_mse, RenderConfig


class DecentNeRFClient:
    """
    A federated client implementing local DecentNeRF training.

    It wraps:
      - local dataset (ClientDataset)
      - global+personal field (DecentNeRFField)
      - local optimizer

    Only GLOBAL weights are returned to the server.
    Personal head is NOT shared.
    """

    def __init__(
        self,
        client_id: int,
        dataset: ClientDataset,
        num_images_total: int,
        lr: float = 5e-4,
        iters_per_round: int = 200,
        batch_size: int = 1024,
        n_samples: int = 64,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.device = torch.device(device)

        # full model (global + personal)
        self.field = DecentNeRFField(num_images=num_images_total).to(self.device)

        # optimizer applies to BOTH global and personal components
        # (but we send only global back)
        self.optimizer = torch.optim.Adam(self.field.parameters(), lr=lr)

        self.iters_per_round = iters_per_round
        self.batch_size = batch_size

        self.render_cfg = RenderConfig(
            n_samples=n_samples,
            near=dataset.cfg.near,
            far=dataset.cfg.far,
            white_bkgd=False
        )

    # -------------------------------------------------------------
    # Utility: extract only global-field parameters
    # -------------------------------------------------------------
    def get_global_parameters(self):
        """
        Return ONLY the parameters of the global field as a list of numpy arrays.
        """
        params = []
        for p in self.field.global_field.parameters():
            params.append(p.detach().cpu().numpy())
        return params

    def set_global_parameters(self, params_list):
        """
        Load global weights received from server.
        """
        model_params = list(self.field.global_field.parameters())
        assert len(model_params) == len(params_list)

        for p, new_val in zip(model_params, params_list):
            p.data = torch.tensor(new_val, dtype=torch.float32, device=self.device)

    # -------------------------------------------------------------
    # Local training
    # -------------------------------------------------------------
    def train_one_round(self):
        """
        Runs local training for one federated round.

        Returns:
          - updated global parameters
          - metrics: {"loss": ..., "psnr": ...}
        """
        self.field.train()

        losses = []

        for it in range(self.iters_per_round):
            # sample rays
            rays_o, rays_d, target_rgb, img_idx = self.dataset.sample_rays(
                self.batch_size, device=self.device
            )

            # render
            rgb_pred, depth, weights, z_vals = volume_render_rays(
                self.field, rays_o, rays_d, img_idx, cfg=self.render_cfg
            )

            # loss
            loss = nerf_loss(rgb_pred, target_rgb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        avg_psnr = psnr_from_mse(avg_loss)

        # return updated GLOBAL parameters only
        updated_globals = self.get_global_parameters()

        metrics = {
            "client_id": self.client_id,
            "loss": avg_loss,
            "psnr": avg_psnr
        }

        return updated_globals, metrics
