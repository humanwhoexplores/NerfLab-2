"""
train_federated.py

End-to-end DecentNeRF federated training driver.

What it does:
  1. Load full Trevi dataset with synthetic poses + intrinsics
  2. Initialize a global DecentNeRFField (template)
  3. Create DecentNeRFServer with initial global weights
  4. Create N DecentNeRFClient instances (one per client)
  5. For each federated round:
       - broadcast global params to all clients
       - each client trains locally (train_one_round)
       - server aggregates global params with α-weighted secure aggregator
       - server updates α_k based on client losses
       - log losses, PSNR, and α_k

Outputs:
  - logs printed per round
  - optional final global model checkpoint
"""

import os
import time
import json
from typing import Dict, List

import torch
import numpy as np

from nerf_backbone import DecentNeRFField
from client_dataset import FullSceneDataset
from federated_client import DecentNeRFClient
from federated_server import DecentNeRFServer, ServerConfig


# -----------------------------
# Config
# -----------------------------

SCENE_ROOT = "data/trevi"

NUM_CLIENTS = 5
ROUNDS = 5

DOWNSCALE = 8

ITERS_PER_ROUND = 200      # local steps per round per client
BATCH_SIZE = 512
N_SAMPLES = 64             # samples per ray

LR_CLIENT = 5e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "outputs_federated"
FINAL_MODEL_PATH = os.path.join(OUT_DIR, "decentnerf_global_final.pth")
LOG_PATH = os.path.join(OUT_DIR, "federated_log.json")


# -----------------------------
# Helper: save global model
# -----------------------------

def save_global_model(global_params: List[np.ndarray], num_images_total: int, path: str):
    """
    Build a DecentNeRFField, load given global_field params into it,
    and save its full state_dict to disk.

    Personal head will remain randomly initialized, but global_field will
    contain the federated weights.
    """
    field = DecentNeRFField(num_images=num_images_total)
    # load into global_field
    g_params = list(field.global_field.parameters())
    assert len(g_params) == len(global_params)

    for p, np_val in zip(g_params, global_params):
        p.data = torch.from_numpy(np_val).float()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(field.state_dict(), path)
    print(f"[SAVE] Global model checkpoint → {path}")


# -----------------------------
# Main federated training loop
# -----------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== DecentNeRF Federated Training ===")
    print("Scene root:", SCENE_ROOT)
    print("Device:", DEVICE)
    print(f"Clients: {NUM_CLIENTS}, Rounds: {ROUNDS}")
    print(f"Downscale: {DOWNSCALE}, iters/round/client: {ITERS_PER_ROUND}")
    print()

    device = DEVICE

    # 1. Load full dataset
    print("[1] Loading FullSceneDataset...")
    full_data = FullSceneDataset(
        scene_root=SCENE_ROOT,
        downscale=DOWNSCALE,
        num_clients=NUM_CLIENTS,
        max_images=None,
    )
    num_images_total = full_data.num_images_total
    print(f"    Total images: {num_images_total}")

    # 2. Init global model template to get parameter shapes
    print("\n[2] Initializing global DecentNeRFField template...")
    template_field = DecentNeRFField(num_images=num_images_total)
    init_global_params = [p.detach().cpu().numpy() for p in template_field.global_field.parameters()]
    client_ids = list(range(NUM_CLIENTS))

    # 3. Create server
    print("\n[3] Creating DecentNeRFServer...")
    server_cfg = ServerConfig(
        lr_alpha=0.5,
        base_seed=10000,
        alpha_smoothing=0.5,
    )
    server = DecentNeRFServer(
        initial_global_params=init_global_params,
        client_ids=client_ids,
        cfg=server_cfg,
        device=device,
    )

    # 4. Create clients
    print("\n[4] Creating DecentNeRFClient instances...")
    clients: Dict[int, DecentNeRFClient] = {}
    for cid in client_ids:
        ds = full_data.make_client(cid)
        client = DecentNeRFClient(
            client_id=cid,
            dataset=ds,
            num_images_total=num_images_total,
            lr=LR_CLIENT,
            iters_per_round=ITERS_PER_ROUND,
            batch_size=BATCH_SIZE,
            n_samples=N_SAMPLES,
            device=device,
        )
        clients[cid] = client
        print(f"    Client {cid}: {ds.num_images} images")

    print("\n[5] Starting federated rounds...")
    log = {
        "rounds": [],
        "config": {
            "scene_root": SCENE_ROOT,
            "num_clients": NUM_CLIENTS,
            "rounds": ROUNDS,
            "downscale": DOWNSCALE,
            "iters_per_round": ITERS_PER_ROUND,
            "batch_size": BATCH_SIZE,
            "n_samples": N_SAMPLES,
            "lr_client": LR_CLIENT,
        },
    }

    global_start_time = time.time()

    for r in range(1, ROUNDS + 1):
        print(f"\n=== ROUND {r}/{ROUNDS} ===")
        round_start = time.time()

        # 5.a broadcast global params to all clients
        global_params = server.get_global_parameters()
        for cid, client in clients.items():
            client.set_global_parameters(global_params)

        # 5.b each client trains locally
        client_param_updates: Dict[int, List[np.ndarray]] = {}
        client_metrics: Dict[int, Dict[str, float]] = {}

        for cid, client in clients.items():
            print(f"  [Client {cid}] local training...")
            upd, metrics = client.train_one_round()
            client_param_updates[cid] = upd
            client_metrics[cid] = metrics
            print(
                f"    -> loss={metrics['loss']:.4f}, "
                f"PSNR={metrics['psnr']:.2f} dB"
            )

        # 5.c server aggregates + updates global + α
        print("  [Server] aggregating updates...")
        new_global_params, alpha_snapshot = server.aggregate(
            client_param_updates,
            client_metrics,
        )

        # 5.d logging
        round_time = time.time() - round_start

        avg_loss = float(
            sum(m["loss"] for m in client_metrics.values()) / len(client_metrics)
        )
        avg_psnr = float(
            sum(m["psnr"] for m in client_metrics.values()) / len(client_metrics)
        )

        print(f"  [Round {r}] avg_loss={avg_loss:.4f}, avg_PSNR={avg_psnr:.2f} dB")
        print(f"  [Round {r}] α snapshot: { {cid: round(alpha_snapshot[cid], 3) for cid in alpha_snapshot} }")
        print(f"  [Round {r}] time={round_time:.1f}s")

        log_round = {
            "round": r,
            "time_sec": round_time,
            "avg_loss": avg_loss,
            "avg_psnr": avg_psnr,
            "client_metrics": client_metrics,
            "alpha": alpha_snapshot,
        }
        log["rounds"].append(log_round)

    total_time = time.time() - global_start_time
    print(f"\n=== Federated training completed in {total_time:.1f}s ===")

    # 6. Save final global model
    final_globals = server.get_global_parameters()
    save_global_model(final_globals, num_images_total, FINAL_MODEL_PATH)

    # 7. Save log
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[LOG] Federated log saved → {LOG_PATH}")


if __name__ == "__main__":
    main()
