"""
Génère un modèle Autoencoder factice (PyTorch) pour tester le pipeline
SysGuard-AI sans jeu de données réel.

Produit : ai_model/saved_model.pth + ai_model/threshold.json

Usage :
  python ai_model/create_dummy_model.py
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

from autoencoder import SysGuardAutoencoder, INPUT_DIM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MODEL = os.path.join(SCRIPT_DIR, "saved_model.pth")
OUTPUT_THRESHOLD = os.path.join(SCRIPT_DIR, "threshold.json")


def main():
    print("Construction de l'Autoencoder (PyTorch, architecture mémoire)...")
    model = SysGuardAutoencoder(INPUT_DIM, dropout=0.2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres totaux : {total_params:,}")

    np.random.seed(42)
    n_samples = 2000
    X_np = np.random.exponential(0.5, size=(n_samples, INPUT_DIM)).astype(np.float32)
    X_np = X_np / X_np.sum(axis=1, keepdims=True)

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    print("Entraînement minimal (5 epochs) sur données factices...")
    model.train()
    for epoch in range(1, 6):
        total_loss = 0.0
        for (batch,) in loader:
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"  Epoch {epoch}/5 — loss: {avg_loss:.6f}")

    os.makedirs(SCRIPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Modèle enregistré : {OUTPUT_MODEL}")

    model.eval()
    with torch.no_grad():
        reconstructions = model(X_tensor).numpy()
    mse_per_sample = np.mean(np.square(X_np - reconstructions), axis=1)
    alpha = float(np.percentile(mse_per_sample, 99.0))
    beta = float(np.percentile(mse_per_sample, 99.9))

    threshold_data = {
        "alpha": alpha,
        "beta": beta,
        "threshold": alpha,
        "percentile_alpha": 99.0,
        "percentile_beta": 99.9,
        "train_samples": n_samples,
        "data_source": "dummy",
    }
    with open(OUTPUT_THRESHOLD, "w") as f:
        json.dump(threshold_data, f, indent=2)

    print(f"Seuils enregistres : {OUTPUT_THRESHOLD}")
    print(f"  alpha (P99)   = {alpha:.6f}")
    print(f"  beta  (P99.9) = {beta:.6f}")


if __name__ == "__main__":
    main()
