"""
test_federated_client.py

Unit test for DecentNeRFClient (Step 6 verification)

This script:
  - Loads the full dataset
  - Creates client 0
  - Initializes a DecentNeRFClient
  - Runs ONE federated training round
  - Prints metrics (loss, PSNR)
  - Confirms that only GLOBAL params are returned
"""

import torch

from client_dataset import FullSceneDataset
from federated_client import DecentNeRFClient


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----------------------------------------
    # Load dataset
    # ----------------------------------------
    print("\nLoading FullSceneDataset...")
    full_data = FullSceneDataset(
        scene_root="data/trevi",
        downscale=8,
        num_clients=5,
        max_images=None
    )

    num_images_total = full_data.num_images_total
    print(f"Total images = {num_images_total}")

    # ----------------------------------------
    # Create client 0
    # ----------------------------------------
    print("\nCreating ClientDataset(0)...")
    client_ds0 = full_data.make_client(0)

    # ----------------------------------------
    # Create DecentNeRF federated client
    # ----------------------------------------
    client = DecentNeRFClient(
        client_id=0,
        dataset=client_ds0,
        num_images_total=num_images_total,
        lr=5e-4,
        iters_per_round=200,
        batch_size=512,
        n_samples=64,
        device=device
    )

    # ----------------------------------------
    # BEFORE training: check global params shape
    # ----------------------------------------
    globals_before = client.get_global_parameters()
    print(f"\nNumber of global param tensors: {len(globals_before)}")

    # ----------------------------------------
    # Run ONE federated round
    # ----------------------------------------
    print("\nRunning ONE federated round...")
    updated_globals, metrics = client.train_one_round()

    # ----------------------------------------
    # Check outputs
    # ----------------------------------------
    print("\n=== METRICS ===")
    print(f"Client ID: {metrics['client_id']}")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"PSNR: {metrics['psnr']:.4f} dB")

    print("\n=== Updated global params ===")
    print(f"Num arrays returned: {len(updated_globals)}")
    print(f"Sizes: {[arr.shape for arr in updated_globals]}")

    # Check that shapes match before/after
    ok = True
    for p0, p1 in zip(globals_before, updated_globals):
        if p0.shape != p1.shape:
            ok = False

    print("\nGlobal parameter shape consistency:", "OK" if ok else "MISMATCH")

    print("\nTest completed.")


if __name__ == "__main__":
    main()
