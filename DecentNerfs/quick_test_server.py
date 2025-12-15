# quick_test_server.py (optional)

import torch
from nerf_backbone import DecentNeRFField
from client_dataset import FullSceneDataset
from federated_client import DecentNeRFClient
from federated_server import DecentNeRFServer

full = FullSceneDataset("data/trevi", downscale=8, num_clients=3)
num_images = full.num_images_total

# initial model to get global params template
model0 = DecentNeRFField(num_images=num_images)
init_globals = [p.detach().cpu().numpy() for p in model0.global_field.parameters()]

server = DecentNeRFServer(initial_global_params=init_globals,
                          client_ids=[0, 1, 2])

clients = []
for cid in [0, 1, 2]:
    ds = full.make_client(cid)
    c = DecentNeRFClient(
        client_id=cid,
        dataset=ds,
        num_images_total=num_images,
        device="cpu",
        iters_per_round=50,
        batch_size=256,
    )
    # broadcast initial globals
    c.set_global_parameters(server.get_global_parameters())
    clients.append(c)

updates = {}
metrics = {}
for c in clients:
    upd, m = c.train_one_round()
    updates[c.client_id] = upd
    metrics[c.client_id] = m

new_globals, alpha_snapshot = server.aggregate(updates, metrics)
print("New α:", alpha_snapshot)
