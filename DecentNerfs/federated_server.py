"""
federated_server.py

DecentNeRF federated server:

- Maintains the GLOBAL NeRF field parameters (global_field only)
- Receives client-updated global parameters and metrics
- Applies (toy) secure aggregation on flattened parameter vectors
- Updates client importance weights α_k based on their performance
- Produces a new global parameter set to broadcast next round

This is a pure Python server-side helper; it does not depend on Flower
directly, but can be wrapped in a Flower strategy.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


# -----------------------------
# Helpers: flatten / unflatten
# -----------------------------

def flatten_params(params_list: List[np.ndarray]) -> Tuple[torch.Tensor, List[Tuple[int, ...]]]:
    """
    Convert a list of numpy arrays into a single 1D torch vector + shape list.
    """
    flats = []
    shapes: List[Tuple[int, ...]] = []

    for arr in params_list:
        shapes.append(arr.shape)
        t = torch.from_numpy(arr).float().reshape(-1)
        flats.append(t)

    flat = torch.cat(flats, dim=0)
    return flat, shapes


def unflatten_params(flat: torch.Tensor, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    """
    Convert a flat torch vector back into a list of numpy arrays with given shapes.
    """
    params: List[np.ndarray] = []
    idx = 0

    for shape in shapes:
        n = int(np.prod(shape))
        chunk = flat[idx:idx + n].view(shape)
        params.append(chunk.detach().cpu().numpy())
        idx += n

    return params


# -----------------------------
# "Secure" aggregator (toy)
# -----------------------------

class SecureAggregator:
    """
    Very simple secure aggregation toy model.

    Each client masks its parameter vector with Gaussian noise derived from
    a client-specific seed. The server sums masked vectors, then subtracts
    the known masks to recover an unmasked sum.

    NOTE:
      - This is *not* production-grade cryptography.
      - It is enough for a research prototype to demonstrate the idea.
    """

    def __init__(self, vector_size: int, noise_std: float = 0.01, device: str = "cpu"):
        self.vector_size = vector_size
        self.noise_std = noise_std
        self.device = torch.device(device)

    def mask(self, vec: torch.Tensor, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask a vector with Gaussian noise determined by seed.
        """
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        noise = torch.randn(self.vector_size, generator=g, device=self.device) * self.noise_std
        return vec + noise, noise

    def unmask(self, sum_masked: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Remove noise masks from aggregated sum.
        """
        if len(masks) == 0:
            return sum_masked
        stacked = torch.stack(masks, dim=0)
        return sum_masked - stacked.sum(dim=0)


# -----------------------------
# Server configuration
# -----------------------------

@dataclass
class ServerConfig:
    lr_alpha: float = 0.5           # how fast α_k adapts to client quality
    base_seed: int = 10_000         # base seed for masking
    alpha_smoothing: float = 0.5    # EMA factor for α updates


# -----------------------------
# DecentNeRF federated server
# -----------------------------

class DecentNeRFServer:
    """
    Maintains and updates the GLOBAL NeRF field parameters.

    Workflow per round:
      1. Receive {client_id: params_list} from multiple clients
      2. Receive {client_id: metrics} from multiple clients
      3. Securely aggregate (toy) and compute a weighted average using α_k
      4. Update α_k based on client performance (e.g. inverse loss)
      5. Store new global parameter vector

    The server NEVER sees personal-head parameters; only global_field params.
    """

    def __init__(
        self,
        initial_global_params: List[np.ndarray],
        client_ids: List[int],
        cfg: ServerConfig = ServerConfig(),
        device: str = "cpu"
    ):
        """
        Args:
          initial_global_params: list of numpy arrays from global_field of a freshly
                                 initialized DecentNeRF model.
          client_ids: list of known client IDs
          cfg: config for α-weight updates, masking seeds, etc.
        """
        self.device = torch.device(device)
        self.cfg = cfg

        flat, shapes = flatten_params(initial_global_params)
        self.global_flat = flat.to(self.device)
        self.shapes = shapes
        self.vector_size = self.global_flat.numel()

        n = len(client_ids)
        self.alphas: Dict[int, float] = {int(cid): 1.0 / max(n, 1) for cid in client_ids}

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def get_global_parameters(self) -> List[np.ndarray]:
        """
        Return current global parameters as list of numpy arrays.
        """
        return unflatten_params(self.global_flat, self.shapes)

    def aggregate(
        self,
        client_param_updates: Dict[int, List[np.ndarray]],
        client_metrics: Dict[int, Dict[str, float]],
    ) -> Tuple[List[np.ndarray], Dict[int, float]]:
        """
        Aggregate updates from all clients and update global params + α_k.

        Args:
          client_param_updates: {client_id: [np.array, ...]} global_field params
          client_metrics:       {client_id: {"loss":..., "psnr":...}, ...}

        Returns:
          new_global_params: list of numpy arrays (to broadcast)
          alpha_snapshot:    {client_id: α_k} after update
        """
        if not client_param_updates:
            # Nothing to aggregate
            return self.get_global_parameters(), dict(self.alphas)

        device = self.device
        client_ids = sorted(client_param_updates.keys())

        # Flatten all client parameters
        flat_updates: Dict[int, torch.Tensor] = {}
        for cid in client_ids:
            flat, shapes = flatten_params(client_param_updates[cid])
            # Shapes must match initial shapes
            if len(shapes) != len(self.shapes):
                raise ValueError("Parameter shape mismatch for client {}".format(cid))
            flat_updates[cid] = flat.to(device)

        # -----------------------------
        # Secure aggregation (toy)
        # -----------------------------
        aggregator = SecureAggregator(self.vector_size, device=device)

        masked_vecs = []
        masks = []

        for cid in client_ids:
            vec = flat_updates[cid]
            masked, noise = aggregator.mask(vec, self.cfg.base_seed + int(cid))
            masked_vecs.append(masked)
            masks.append(noise)

        sum_masked = torch.stack(masked_vecs, dim=0).sum(dim=0)
        sum_unmasked = aggregator.unmask(sum_masked, masks)  # sum of unmasked client vectors

        # If we wanted a simple uniform average:
        # uniform_avg = sum_unmasked / len(client_ids)

        # -----------------------------
        # α-weighted average
        # -----------------------------
        with torch.no_grad():
            alpha_vec = torch.tensor(
                [self.alphas.get(cid, 1.0) for cid in client_ids],
                dtype=torch.float32,
                device=device,
            )
            alpha_vec = alpha_vec / (alpha_vec.sum() + 1e-8)

            stacked = torch.stack([flat_updates[cid] for cid in client_ids], dim=0)
            weighted_avg = (alpha_vec.view(-1, 1) * stacked).sum(dim=0)

            # Update server's global_flat
            self.global_flat = weighted_avg.detach()

        # -----------------------------
        # Update α_k based on metrics
        # -----------------------------
        self._update_alphas(client_ids, client_metrics)

        alpha_snapshot = dict(self.alphas)
        new_global_params = self.get_global_parameters()
        return new_global_params, alpha_snapshot

    # ---------------------------------------------------------
    # Internal: α update logic
    # ---------------------------------------------------------

    def _update_alphas(self, client_ids: List[int], client_metrics: Dict[int, Dict[str, float]]):
        """
        Update α_k using a simple performance-based heuristic.

        We use inverse loss:

            score_k = 1 / (loss_k + eps)

        Then normalize scores to sum to 1 and blend them with the old α_k
        via EMA smoothing.
        """
        if not client_metrics:
            return

        eps = 1e-6
        scores = []

        # Compute performance scores
        for cid in client_ids:
            metrics = client_metrics.get(cid, {})
            loss = float(metrics.get("loss", 1.0))
            score = 1.0 / (loss + eps)
            scores.append(score)

        scores_t = torch.tensor(scores, dtype=torch.float32)
        scores_t = scores_t / (scores_t.sum() + 1e-8)

        # EMA-style update of α_k
        new_alphas: Dict[int, float] = {}
        for i, cid in enumerate(client_ids):
            old = self.alphas.get(cid, 1.0 / max(len(client_ids), 1))
            s = float(scores_t[i])
            updated = (1.0 - self.cfg.alpha_smoothing) * old + self.cfg.alpha_smoothing * s
            new_alphas[cid] = updated

        # Normalize α's to sum to 1
        total = sum(new_alphas.values()) + 1e-8
        for cid in new_alphas:
            new_alphas[cid] /= total

        self.alphas = new_alphas
