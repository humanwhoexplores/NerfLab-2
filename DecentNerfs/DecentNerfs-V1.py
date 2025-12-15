# -------------------------------------------------------------
# DecentNerfs.py
# Minimal CPU implementation of DecentNeRFs (Phototourism Demo)
# -------------------------------------------------------------
import os, random, numpy as np
from dataclasses import dataclass
from typing import Dict, List
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# ---------- Utils ----------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def flat_tensor_like(params):
    return torch.cat([p.detach().reshape(-1) for p in params])

def assign_from_flat(params, flat):
    idx = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n

# ---------- Tiny NeRF ----------
class TinyNeRF(nn.Module):
    def __init__(self, in_dim=5, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_color = nn.Linear(hidden, 3)
        self.fc_sigma = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        rgb = torch.sigmoid(self.fc_color(h))
        sigma = F.softplus(self.fc_sigma(h))
        return rgb, sigma

# ---------- Data Loader ----------
def load_phototourism(scene_dir, downscale=8):
    """
    Load images from a Phototourism-style folder.
    Skips unreadable or incompatible files and reports count.
    """
    imgs = []
    img_dir = os.path.join(scene_dir, "images")
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])

    for fname in files:
        img_path = os.path.join(img_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            img = img.resize((max(1, w // downscale), max(1, h // downscale)))
            imgs.append(torch.tensor(np.array(img), dtype=torch.float32) / 255.0)
        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")

    poses = [torch.eye(3, 4)] * len(imgs)
    print(f"✅ Loaded {len(imgs)} valid images out of {len(files)} total from {img_dir}")
    return imgs, poses

def sample_batch(imgs, poses, batch_size=128, device="cpu"):
    """
    Randomly sample pixel coordinates from randomly selected images.
    Handles variable image resolutions safely.
    """
    n = len(imgs)
    if n == 0:
        raise RuntimeError("No images found for sampling.")

    x_inputs = torch.empty(batch_size, 5, device=device)
    rgb_targets = torch.empty(batch_size, 3, device=device)

    for i in range(batch_size):
        # pick a random image
        img = imgs[random.randint(0, n - 1)]
        H, W, _ = img.shape
        # pick random pixel within bounds
        y = random.randint(0, H - 1)
        x = random.randint(0, W - 1)
        rgb_targets[i] = img[y, x]
        # dummy NeRF input (just random noise for now)
        x_inputs[i] = torch.rand(5, device=device) * 2 - 1

    return x_inputs, rgb_targets



# ---------- Core ----------
def alpha_composite(rgb_g, sigma_g, rgb_p, sigma_p):
    T_g = torch.exp(-sigma_g)
    return T_g * rgb_p + (1 - T_g) * rgb_g


@dataclass
class ClientConfig:
    id: int
    scene_dir: str
    device: str = "cpu"
    batch_size: int = 256
    iters_per_round: int = 40
    lr: float = 1e-3
    personal_tint_strength: float = 0.15

class Client:
    def __init__(self, cfg, global_mlp, personal_mlp):
        self.cfg = cfg
        self.device = cfg.device
        self.global_mlp = global_mlp.to(cfg.device)
        self.personal_mlp = personal_mlp.to(cfg.device)
        self.opt = torch.optim.Adam(
            list(self.global_mlp.parameters()) + list(self.personal_mlp.parameters()),
            lr=cfg.lr
        )
        self.alpha = torch.tensor(1.0, device=cfg.device)
        self.personal_tint = torch.rand(3, device=cfg.device)
        self.imgs, self.poses = load_phototourism(cfg.scene_dir)

    def local_train(self, server_state):
        with torch.no_grad():
            for p, sp in zip(self.global_mlp.parameters(), server_state):
                p.copy_(sp.to(self.device))
        losses = []
        for _ in range(self.cfg.iters_per_round):
            x, y = sample_batch(self.imgs, self.poses, self.cfg.batch_size, self.device)
            y = (y + self.cfg.personal_tint_strength * self.personal_tint).clamp(0, 1)
            rgb_g, sigma_g = self.global_mlp(x)
            rgb_p, sigma_p = self.personal_mlp(x)
            loss = F.mse_loss(alpha_composite(rgb_g, sigma_g, rgb_p, sigma_p), y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        global_params = [p.detach().cpu() for p in self.global_mlp.parameters()]
        flat_global = flat_tensor_like(global_params)
        # gradient proxy
        x, y = sample_batch(self.imgs, self.poses, self.cfg.batch_size, self.device)
        y = (y + self.cfg.personal_tint_strength * self.personal_tint).clamp(0, 1)
        rgb_g, sigma_g = self.global_mlp(x)
        rgb_p, sigma_p = self.personal_mlp(x)
        loss = F.mse_loss(alpha_composite(rgb_g, sigma_g, rgb_p, sigma_p), y)
        self.opt.zero_grad()
        loss.backward()
        g_grads = [ (p.grad if p.grad is not None else torch.zeros_like(p)).detach().cpu()
            for p in self.global_mlp.parameters()]

        flat_g_grads = flat_tensor_like(g_grads)
        return {
            "client_id": torch.tensor(self.cfg.id),
            "flat_global": flat_global,
            "flat_g_grads": flat_g_grads,
            "train_loss": torch.tensor(np.mean(losses))
        }

class SecureAggregator:
    def __init__(self, vector_size):
        self.vector_size = vector_size

    def mask(self, vec, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        mask = torch.randn(self.vector_size, generator=g) * 0.01
        return vec + mask, mask

    def unmask(self, sum_masked, masks):
        return sum_masked - torch.stack(masks).sum(dim=0)

class Server:
    def __init__(self, global_mlp, lr_alpha=0.05):
        self.global_mlp = global_mlp
        self.lr_alpha = lr_alpha
        self.alpha = {}

    def get_state(self):
        return [p.detach().cpu().clone() for p in self.global_mlp.parameters()]

    def aggregate(self, packets):
        ids = [int(p["client_id"]) for p in packets]
        flats = [p["flat_global"] for p in packets]
        grads = [p["flat_g_grads"] for p in packets]
        n = flats[0].numel()
        agg = SecureAggregator(n)
        masked_f, masked_g, mf, mg = [], [], [], []
        for f, g, cid in zip(flats, grads, ids):
            mf_, maskf = agg.mask(f, 1000 + cid)
            mg_, maskg = agg.mask(g, 2000 + cid)
            masked_f.append(mf_)
            mf.append(maskf)
            masked_g.append(mg_)
            mg.append(maskg)
        sum_f = agg.unmask(sum(masked_f), mf)
        sum_g = agg.unmask(sum(masked_g), mg)
        for cid in ids:
            self.alpha.setdefault(cid, 1.0)
        alphas = torch.tensor([self.alpha[cid] for cid in ids])
        weights = alphas / (alphas.sum() + 1e-8)
        weighted = (weights.view(-1, 1) * torch.stack(flats)).sum(dim=0)
        return weighted, sum_g, ids

    def update_global(self, flat):
        assign_from_flat(list(self.global_mlp.parameters()), flat)

    def update_alphas(self, sum_g, ids, packets):
        for pkt in packets:
            cid = int(pkt["client_id"])
            gk = pkt["flat_global"]
            contrib = torch.dot(sum_g, gk) / (gk.norm() * sum_g.norm() + 1e-8)
            self.alpha[cid] = max(0.0, self.alpha[cid] - self.lr_alpha * contrib.item())
        total = sum(self.alpha.values()) + 1e-8
        for cid in ids:
            self.alpha[cid] /= total

def run(scene_dir, num_clients=5, rounds=5, iters=40, batch=256, device="cpu"):
    seed_all(123)
    server = Server(TinyNeRF().to(device))
    clients = [
        Client(
            ClientConfig(id=i, scene_dir=scene_dir, device=device,
                         iters_per_round=iters, batch_size=batch),
            TinyNeRF(),
            TinyNeRF()
        )
        for i in range(num_clients)
    ]
    avg_losses = []
    for r in range(1, rounds + 1):
        state = server.get_state()
        pkts = [c.local_train(state) for c in clients]
        new_flat, sum_g, ids = server.aggregate(pkts)
        server.update_global(new_flat)
        server.update_alphas(sum_g, ids, pkts)
        loss = torch.stack([p["train_loss"] for p in pkts]).mean().item()
        avg_losses.append(loss)
        a_snap = {cid: round(server.alpha[cid], 3) for cid in ids}
        print(f"[Round {r}] avg_loss={loss:.5f} | α={a_snap}")
    os.makedirs("outputs", exist_ok=True)
    plt.plot(range(1, rounds + 1), avg_losses, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Average Loss")
    #plt.title("DecentNerfs Prototype (Trevi Fountain)")
    plt.title("DecentNerfs Prototype (NotreDame Cathedral)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/training_curve.png", dpi=150)
    np.savetxt(
        "outputs/alpha_weights.csv",
        np.array([[k, v] for k, v in server.alpha.items()]),
        delimiter=",",
        fmt="%s"
    )
    print("✅ Saved outputs/training_curve.png and alpha_weights.csv")

if __name__ == "__main__":
    run("./data/NotreDame", num_clients=5, rounds=5, iters=40, batch=256)
