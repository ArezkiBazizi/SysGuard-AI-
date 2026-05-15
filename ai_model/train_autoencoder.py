"""
Entraîne l'Autoencoder SysGuard-AI (PyTorch).

Supporte deux modes :
  1. Données simulées (par défaut) — distribution réaliste d'application web
  2. Données réelles (CSV) — collectées via collect_falco_data.py

Architecture : 414 → 207 → 100 → 207 → 414
Perte : MSE | Optimiseur : Adam | Régularisation : Dropout (p=0.2)

Calcule deux seuils de décision :
  α (alpha) = P99 de la MSE d'entraînement   → anomalie modérée
  β (beta)  = P99.9 de la MSE d'entraînement  → anomalie critique

Usage :
  python ai_model/train_autoencoder.py [--epochs 50] [--samples 10000]
  python ai_model/train_autoencoder.py --csv ai_model/dataset/normal_traffic.csv
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import SysGuardAutoencoder, INPUT_DIM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MODEL = os.path.join(SCRIPT_DIR, "saved_model.pth")
OUTPUT_THRESHOLD = os.path.join(SCRIPT_DIR, "threshold.json")

DOMINANT_SYSCALLS = {
    "read": 0, "write": 1, "openat": 250, "close": 3,
    "epoll_wait": 225, "recvfrom": 38, "sendto": 37,
    "futex": 195, "stat": 4, "fstat": 5, "mmap": 9,
    "mprotect": 10, "brk": 12, "ioctl": 13, "access": 14,
    "socket": 34, "connect": 35, "accept": 36,
    "clone": 49, "fork": 50, "execve": 52,
    "getpid": 32, "getuid": 95, "gettid": 179,
    "clock_gettime": 221, "nanosleep": 28, "poll": 7,
    "select": 16, "pipe": 15, "dup2": 26,
}


APPLICATION_PHASES = {
    "web_serving": {
        "weight": 0.55,
        "total_range": (400, 2000),
        "profile": {
            "read": 0.13, "write": 0.12, "close": 0.11, "epoll_wait": 0.13,
            "recvfrom": 0.07, "sendto": 0.07, "futex": 0.06,
            "openat": 0.03, "stat": 0.03, "fstat": 0.02, "poll": 0.02,
            "clock_gettime": 0.015, "getpid": 0.01, "gettid": 0.01,
            "nanosleep": 0.005, "socket": 0.005, "accept": 0.005,
            "mmap": 0.003, "brk": 0.002, "ioctl": 0.003,
        },
    },
    "idle": {
        "weight": 0.15,
        "total_range": (20, 150),
        "profile": {
            "epoll_wait": 0.30, "clock_gettime": 0.15, "futex": 0.12,
            "nanosleep": 0.10, "read": 0.08, "write": 0.05,
            "poll": 0.05, "getpid": 0.03, "gettid": 0.03,
            "close": 0.02, "recvfrom": 0.02,
        },
    },
    "high_load": {
        "weight": 0.10,
        "total_range": (2000, 5000),
        "profile": {
            "read": 0.14, "write": 0.14, "close": 0.10, "epoll_wait": 0.10,
            "recvfrom": 0.08, "sendto": 0.08, "futex": 0.08,
            "openat": 0.04, "stat": 0.03, "fstat": 0.02, "poll": 0.02,
            "accept": 0.02, "socket": 0.01, "connect": 0.01,
            "clock_gettime": 0.01, "mmap": 0.005, "brk": 0.005,
        },
    },
    "db_queries": {
        "weight": 0.08,
        "total_range": (300, 1500),
        "profile": {
            "read": 0.18, "write": 0.15, "futex": 0.10,
            "recvfrom": 0.08, "sendto": 0.08, "close": 0.07,
            "openat": 0.06, "fstat": 0.04, "stat": 0.03,
            "epoll_wait": 0.05, "poll": 0.03, "mmap": 0.02,
            "clock_gettime": 0.02, "brk": 0.01,
        },
    },
    "pod_restart": {
        "weight": 0.04,
        "total_range": (500, 2500),
        "profile": {
            "clone": 0.08, "execve": 0.06, "mmap": 0.10, "mprotect": 0.08,
            "openat": 0.10, "read": 0.10, "close": 0.08, "fstat": 0.05,
            "brk": 0.04, "access": 0.03, "stat": 0.03, "write": 0.04,
            "futex": 0.04, "pipe": 0.02, "dup2": 0.02, "ioctl": 0.02,
            "getpid": 0.01, "getuid": 0.01,
        },
    },
    "cron_job": {
        "weight": 0.04,
        "total_range": (200, 1000),
        "profile": {
            "execve": 0.05, "clone": 0.04, "openat": 0.12, "read": 0.15,
            "write": 0.10, "close": 0.10, "stat": 0.06, "fstat": 0.04,
            "futex": 0.05, "mmap": 0.04, "brk": 0.03, "pipe": 0.03,
            "dup2": 0.02, "getpid": 0.02, "nanosleep": 0.03,
            "clock_gettime": 0.02,
        },
    },
    "healthcheck": {
        "weight": 0.04,
        "total_range": (50, 300),
        "profile": {
            "socket": 0.10, "connect": 0.10, "read": 0.15, "write": 0.12,
            "close": 0.12, "sendto": 0.08, "recvfrom": 0.08,
            "epoll_wait": 0.05, "futex": 0.05, "getpid": 0.03,
            "clock_gettime": 0.03, "poll": 0.03,
        },
    },
}


def generate_normal_traffic(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Genere du trafic normal multi-phases simulant 12h d'observation
    d'une application web conteneurisee (DVWA sur Kubernetes).

    Phases : web_serving (55%), idle (15%), high_load (10%),
    db_queries (8%), pod_restart (4%), cron_job (4%), healthcheck (4%).

    Chaque phase a son propre profil de syscalls et sa propre plage
    d'activite, reproduisant la variabilite temporelle reelle.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, INPUT_DIM), dtype=np.float32)

    phase_names = list(APPLICATION_PHASES.keys())
    phase_weights = [APPLICATION_PHASES[p]["weight"] for p in phase_names]

    phases_assigned = rng.choice(
        len(phase_names), size=n_samples, p=phase_weights
    )

    non_dominant = [j for j in range(INPUT_DIM) if j not in DOMINANT_SYSCALLS.values()]

    for i in range(n_samples):
        phase = APPLICATION_PHASES[phase_names[phases_assigned[i]]]
        lo, hi = phase["total_range"]
        total_calls = rng.integers(lo, hi)

        jitter = rng.uniform(0.7, 1.3)
        total_calls = int(total_calls * jitter)

        for name, ratio in phase["profile"].items():
            if name in DOMINANT_SYSCALLS:
                idx = DOMINANT_SYSCALLS[name]
            else:
                continue
            noise = rng.uniform(0.6, 1.5)
            X[i, idx] = max(0, rng.poisson(total_calls * ratio * noise))

        n_background = rng.integers(3, 30)
        bg_indices = rng.choice(
            non_dominant,
            size=min(n_background, len(non_dominant)),
            replace=False,
        )
        for idx in bg_indices:
            X[i, idx] = rng.poisson(1.5)

    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    X = X / row_sums
    return X


def load_csv_data(csv_path: str) -> np.ndarray:
    print(f"[Train] Chargement des donnees depuis {csv_path}")
    try:
        X = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    except ValueError:
        X = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != INPUT_DIM:
        raise ValueError(f"CSV contient {X.shape[1]} colonnes, attendu {INPUT_DIM}")

    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    X = X / row_sums
    print(f"[Train] {X.shape[0]} echantillons charges (dim={X.shape[1]})")
    return X


def train(model, train_loader, val_loader, epochs, device, patience=10):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"  Epoch {epoch:3d}/{epochs} — train_loss: {train_loss:.6f} — val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping à l'epoch {epoch} (patience={patience})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def compute_thresholds(model, X_np, device):
    model.eval()
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructions = model(X_tensor).cpu().numpy()

    mse_per_sample = np.mean(np.square(X_np - reconstructions), axis=1)
    alpha = float(np.percentile(mse_per_sample, 99.0))
    beta = float(np.percentile(mse_per_sample, 99.9))

    print(f"[Train] Distribution MSE sur trafic normal :")
    print(f"  - Moyenne  : {np.mean(mse_per_sample):.6f}")
    print(f"  - Mediane  : {np.median(mse_per_sample):.6f}")
    print(f"  - P95      : {np.percentile(mse_per_sample, 95):.6f}")
    print(f"  - P99 (alpha)  : {alpha:.6f}")
    print(f"  - P99.9 (beta) : {beta:.6f}")
    print(f"  - Max      : {np.max(mse_per_sample):.6f}")

    active_dims = int(np.mean(np.sum(X_np > 1e-6, axis=1)))
    print(f"  - Dims actives (moyenne) : {active_dims}/{INPUT_DIM}")

    return {
        "alpha": alpha, "beta": beta,
        "mse_mean": float(np.mean(mse_per_sample)),
        "mse_median": float(np.median(mse_per_sample)),
        "mse_p95": float(np.percentile(mse_per_sample, 95)),
        "mse_max": float(np.max(mse_per_sample)),
        "active_dims_mean": active_dims,
    }


def main():
    parser = argparse.ArgumentParser(description="Entraîne l'Autoencoder SysGuard-AI (PyTorch)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    print(f"[Train] Device : {device}")

    if args.csv:
        X = load_csv_data(args.csv)
    else:
        print(f"[Train] Generation de {args.samples} echantillons simules...")
        X = generate_normal_traffic(args.samples, seed=args.seed)

    print(f"[Train] Shape du dataset : {X.shape}")
    print(f"[Train] Ratio echantillons/features : {X.shape[0] / INPUT_DIM:.1f}x")

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    print(f"[Train] Construction de l'Autoencoder (dropout={args.dropout})...")
    model = SysGuardAutoencoder(INPUT_DIM, dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Parametres totaux : {total_params:,}")

    print(f"[Train] Entraînement ({args.epochs} epochs max, early stopping)...")
    model = train(model, train_loader, val_loader, args.epochs, device)

    os.makedirs(SCRIPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"[Train] Modele enregistre : {OUTPUT_MODEL}")

    thresholds = compute_thresholds(model, X, device)

    threshold_data = {
        "alpha": thresholds["alpha"],
        "beta": thresholds["beta"],
        "threshold": thresholds["alpha"],
        "percentile_alpha": 99.0,
        "percentile_beta": 99.9,
        "train_samples": len(X),
        "data_source": args.csv if args.csv else "simulated",
        "dropout_rate": args.dropout,
        "mse_stats": {
            "mean": thresholds["mse_mean"],
            "median": thresholds["mse_median"],
            "p95": thresholds["mse_p95"],
            "max": thresholds["mse_max"],
        },
        "active_dims_mean": thresholds["active_dims_mean"],
    }

    with open(OUTPUT_THRESHOLD, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"[Train] Seuils enregistres : {OUTPUT_THRESHOLD}")
    print(f"  alpha (P99)   = {thresholds['alpha']:.6f}")
    print(f"  beta  (P99.9) = {thresholds['beta']:.6f}")


if __name__ == "__main__":
    main()
